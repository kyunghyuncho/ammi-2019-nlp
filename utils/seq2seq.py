
import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from global_variables import SOS_IDX, SOS_TOKEN, EOS_IDX, EOS_TOKEN, UNK_IDX, UNK_TOKEN, PAD_IDX, PAD_TOKEN, SEP_IDX, SEP_TOKEN, device
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from beam import Beam
import math


class BagOfWords(nn.Module):
    def init_layers(self):
        for l in self.layers:
            if getattr(l, "weight", None) is not None:
                torch.nn.init.xavier_uniform_(l.weight)

    def __init__(
        self,
        input_size,
        hidden_size=512,
        reduce="sum",
        nlayers=2,
        activation="ReLU",
        dropout=0.1,
        batch_norm=False,
    ):
        super(BagOfWords, self).__init__()

        self.emb_dim = hidden_size

        self.reduce = reduce
        assert(self.reduce in ["sum", "mean", "max"]);

        self.nlayers = nlayers
        self.hidden_size = hidden_size

        self.activation = getattr(nn, activation)

        self.embedding = nn.Embedding(input_size, hidden_size, padding_idx = PAD_IDX)

        # TODO SM: add comments
        if batch_norm is True:
            self.batch_norm = nn.BatchNorm1d(self.emb_dim)
        self.layers = nn.ModuleList([nn.Linear(self.emb_dim, self.hidden_size)])

        self.layers.append(self.activation())
        self.layers.append(nn.Dropout(p=dropout))
        for i in range(self.nlayers - 2):
            self.layers.append(nn.Linear(self.hidden_size, self.hidden_size))
            self.layers.append(self.activation())
            self.layers.append(nn.Dropout(p=dropout))
        self.layers.append(nn.Linear(self.hidden_size, self.hidden_size))
        self.init_layers()

    def forward(self, x):
        postemb = self.embedding(x)

        if self.reduce == "sum":
            postemb = postemb.sum(dim=1);
        elif self.reduce == "mean":
            postemb = postemb.mean(dim=1);
        elif self.reduce == "max":
            postemb = postemb.max(dim=1)[0];

        if hasattr(self, "batch_norm"):
            x = self.batch_norm(postemb)
        else:
            x = postemb

        for l in self.layers:
            x = l(x)

        return None, x.unsqueeze(0)



class EncoderRNN(nn.Module):
    """Encodes the input context."""

    def __init__(self, vocab_size, embed_size, hidden_size, num_layers, pad_idx, dropout=0, shared_lt=None):
        """Initialize encoder.
        :param vocab_size: voc size for lt
        :param embed_size: embedding size for lt
        :param hidden_size: size of GRU hidden layers
        :param numlayers: number of GRU layers
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = nn.Dropout(p=dropout)
        
        if shared_lt is None:
            self.embedding = nn.Embedding(self.vocab_size, self.embed_size, pad_idx)
        else:
            self.embedding = shared_lt
            
        self.gru = nn.GRU(
            self.embed_size, self.hidden_size, num_layers=self.num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0,
        )
        
        
    def forward(self, text_vec, text_lens, hidden=None, use_packed=True):
        """Return encoded state.
        :param input: (batchsize x seqlen) tensor of token indices.
        :param hidden: optional past hidden state
        :param use_packed: either we pack seqs, assumed sorted if True
        """
        embedded = self.embedding(text_vec)
        embedded = self.dropout(embedded)
        if use_packed is True:
            embedded = pack_padded_sequence(embedded, text_lens, batch_first=True)
        output, hidden = self.gru(embedded, hidden)
        if use_packed is True:
            output, output_lens = pad_packed_sequence(output)
            output = output.transpose(0,1)
        
        return output, hidden



class SMAttentionLayer(nn.Module):

    def __init__(self, hidden_dim, output_dim):
        super().__init__()

        # TODO SM: add comments about l1 l2 
        self.l1 = nn.Linear(hidden_dim, output_dim, bias=False)
        self.l2 = nn.Linear(hidden_dim + output_dim, output_dim, bias=False)

    def forward(self, hidden, encoder_outs, src_lens):
        ''' hiddden: bsz x hidden_dim
        encoder_outs: bsz x sq_len x encoder dim (output_dim)
        src_lens: bsz

        x: bsz x output_dim
        attn_score: bsz x sq_len'''

        x = self.l1(hidden)
        att_score = (encoder_outs.transpose(0, 1) * x.unsqueeze(0)).sum(dim=2)

        seq_mask = self.sequence_mask(src_lens, 
                                    max_len=max(src_lens).item(), 
                                    device = hidden.device).transpose(0, 1)


        masked_att = seq_mask * att_score
        masked_att[masked_att == 0] = -1e10
        attn_scores = F.softmax(masked_att, dim=0)
        x = (attn_scores.unsqueeze(2) * encoder_outs.transpose(0, 1)).sum(dim=0)
        x = torch.tanh(self.l2(torch.cat((x, hidden), dim=1)))
        return x, attn_scores


    def sequence_mask(self, sequence_length, max_len=None, device = torch.device('cuda')):
        if max_len is None:
            max_len = sequence_length.max().item()
        batch_size = sequence_length.size(0)
        seq_range = torch.arange(0, max_len).long()
        seq_range_expand = seq_range.unsqueeze(0).repeat([batch_size, 1])
        seq_range_expand = seq_range_expand.to(device)
        seq_length_expand = (sequence_length.unsqueeze(1)
                             .expand_as(seq_range_expand))
        return (seq_range_expand < seq_length_expand).float()


class AttentionLayer(nn.Module):
    def __init__(self, decoder_hidden_size):
        super().__init__()
        attn_input_dim = decoder_hidden_size
        self.attention_merge = nn.Linear(2*decoder_hidden_size, decoder_hidden_size)
        self.attention_hidden_transform = nn.Linear(decoder_hidden_size, decoder_hidden_size)
        
    def forward(self, decoder_input, decoder_hidden, encoder_states):
        encoder_output, encoder_hidden = encoder_states
        batch_size, sequence_length, hidden_size = encoder_output.size()
        
        decoder_hidden_last_layer = decoder_hidden[-1].unsqueeze(1)  # for bmm
        
        transformed_hidden = self.attention_hidden_transform(decoder_hidden_last_layer)
        encoder_output_T = encoder_output.transpose(0,1).transpose(1,2)
        
        attn_logits = torch.bmm(transformed_hidden, encoder_output_T).squeeze(1)
        
        attn_weights = F.softmax(attn_logits, dim=1)
        
        attended_encoder_output = torch.bmm(attn_weights.unsqueeze(1), encoder_output.transpose(0,1))
        merged = torch.cat((decoder_input.squeeze(1), attended_encoder_output.squeeze(1)), 1)
        
        out = torch.tanh(self.attention_merge(merged).unsqueeze(1))
        
        return out, attn_weights


class IBMAttentionLayer(nn.Module):

    def __init__(self, dimensions, attention_type='general'):
        super().__init__()
        if attention_type not in ['dot', 'general']:
            raise ValueError('Invalid attention type selected.')

        self.attention_type = attention_type
        if self.attention_type == 'general':
            self.linear_in = nn.Linear(dimensions, dimensions, bias=False)

        self.linear_out = nn.Linear(dimensions*2, dimensions, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        self.tanh = nn.Tanh()

    def forward(self, query, context):

        query = query.transpose(0,1)
        #context = context.transpose(0,1)
        batch_size, output_len, dimensions = query.size()
        context_len = context.size(1)

        if self.attention_type == 'general':
            query = query.view(batch_size * output_len, dimensions)
            query = self.linear_in(query)
            query = query.view(batch_size, output_len, dimensions)

#         print('query: ', query.shape)
#         print('context: ', context.shape)
        attention_scores = torch.bmm(query, context.transpose(1,2).contiguous())
        attention_scores = attention_scores.view(batch_size*output_len, context_len)
        attention_weights = self.softmax(attention_scores)
        attention_weights = attention_weights.view(batch_size, output_len, context_len)

        mix = torch.bmm(attention_weights, context)
        combined = torch.cat((mix, query), dim=2)
        combined = combined.view(batch_size*output_len, 2*dimensions)

        output = self.linear_out(combined).view(batch_size, output_len, dimensions)
        output = self.tanh(output)

        return output, attention_weights


class DecoderRNN(nn.Module):
    """Generates a sequence of tokens in response to context."""

    def __init__(self, vocab_size, embed_size, hidden_size, num_layers, pad_idx, dropout=0, attention_flag = True):
        """Initialize encoder.
        :param vocab_size: voc size for lt
        :param embed_size: embedding size for lt
        :param hidden_size: size of GRU hidden layers
        :param numlayers: number of GRU layers
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = nn.Dropout(p=dropout)
        
        self.embedding = nn.Embedding(self.vocab_size, self.embed_size, pad_idx)
        self.gru = nn.GRU(
            self.embed_size, self.hidden_size, num_layers=self.num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0,
        )
        self.attention = IBMAttentionLayer(self.hidden_size) if attention_flag else None;

        self.out = nn.Linear(self.hidden_size, self.vocab_size)

    def forward(self, text_vec, decoder_hidden, encoder_states):
        """Return encoded state.
        :param input: batch_size x seqlen tensor of token indices.
        :param hidden: past (e.g. encoder or decoder) hidden state
        """
        emb = self.embedding(text_vec)
        emb = self.dropout(emb)
        seqlen = text_vec.size(1)
        
        decoder_hidden = decoder_hidden
        output = []
        attn_w_log = []
        for i in range(seqlen):
            decoder_output, decoder_hidden = self.gru(emb[:,i,:].unsqueeze(1), decoder_hidden)
            if self.attention is not None:
                decoder_output_attended, attn_weights = self.attention(decoder_hidden[-1].unsqueeze(0), encoder_states[0])
                output.append(decoder_output_attended)
                attn_w_log.append(attn_weights)
            else:
                output.append(decoder_output);
            #attn_w_log.append(attn_weights)
            
        output = torch.cat(output, dim=1).to(text_vec.device)
        scores = self.out(output)       
        return scores, decoder_hidden, attn_w_log


class seq2seq(nn.Module):
    """
    TODO IK

    """
    def __init__(self, vocab_size_encoder, vocab_size_decoder, embedding_size, encoder_type='rnn', hidden_size=64, num_layers=2, lr=0.01, 
                       pad_idx=PAD_IDX, sos_idx=SOS_IDX, eos_idx=EOS_IDX, encoder_shared_lt=False, dropout=0.0, use_cuda=True, optimizer='Adam', 
                       grad_clip=None, encoder_attention = False, self_attention = False):

        super().__init__()
        self.opts = {}
        self.opts['vocab_size_encoder'] = vocab_size_encoder
        self.opts['vocab_size_decoder'] = vocab_size_decoder
        self.opts['hidden_size'] = hidden_size
        self.opts['device'] = 'cuda' if use_cuda is True else 'cpu'
        self.opts['embedding_size'] = embedding_size
        self.opts['encoder_type'] = encoder_type
        self.opts['num_layers'] = num_layers
        self.opts['lr'] = lr
        self.opts['pad_idx'] = pad_idx
        self.opts['sos_idx'] = sos_idx
        self.opts['eos_idx'] = eos_idx
        self.opts['dropout'] = dropout
        self.opts['encoder_shared_lt'] = encoder_shared_lt
        self.opts['grad_clip'] = grad_clip
        self.opts['encoder_attention'] = encoder_attention;
        self.opts['self_attention'] = self_attention;

        assert( not (self_attention and encoder_attention) );
        
        if not self.opts['self_attention']:
            self.decoder = DecoderRNN(self.opts['vocab_size_decoder'], self.opts['embedding_size'], self.opts['hidden_size'], self.opts['num_layers'], self.opts['pad_idx'], self.opts['dropout'], self.opts['encoder_attention']);
        else:
            raise NotImplementedError # TODO SM

        if self.opts['encoder_type'] == 'rnn':
            self.encoder = EncoderRNN(self.opts['vocab_size_encoder'], self.opts['embedding_size'], self.opts['hidden_size'], self.opts['num_layers'], self.opts['pad_idx'], self.opts['dropout'], shared_lt=self.decoder.embedding if self.opts['encoder_shared_lt'] else None)
        else:
            raise NotImplementedError # TODO SM Bow...

        optim_class = getattr(optim, optimizer)

        self.optimizer = optim_class(self.parameters(), self.opts['lr'], amsgrad=True)

        self.criterion = nn.CrossEntropyLoss(ignore_index=self.opts['pad_idx'], reduction='sum')
        
        self.encoder.to(self.opts['device'])
        self.decoder.to(self.opts['device'])

        self.sos_buffer = torch.Tensor([self.opts['sos_idx']]).long().to(self.opts['device'])
        self.longest_label = 40

        self.metrics = {
                'loss': 0.0,
                'num_tokens': 0,
                }


    def reset_metrics(self):
        self.metrics['loss'] = 0
        self.metrics['num_tokens'] = 0
        
    def report_metrics(self):
        if self.metrics['num_tokens'] > 0:
            avg_loss = self.metrics['loss'] / self.metrics['num_tokens']
            ppl = math.exp(avg_loss)
            print('Loss: {}\nPPL: {}'.format(avg_loss, ppl))
            return ppl, avg_loss

    def save_model(self, filename):
        state_dict = self.state_dict()
        torch.save(state_dict, filename)

    def load_model(self, filename):
        state_dict = torch.load(filename)
        self.load_state_dict(state_dict)
        
    def train_mode(self):
        self.encoder.train()
        self.decoder.train()

    def eval_mode(self):
        self.encoder.eval()
        self.decoder.eval()
        
    def zero_grad(self):
        self.optimizer.zero_grad()

    def update_params(self):
        if self.opts['grad_clip'] is not None:
            torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), self.opts['grad_clip'])
            torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), self.opts['grad_clip'])
        self.optimizer.step()

    def scheduler_step(self, val_score, min=True):
        if min is False:
            val_score = -val_score
        self.lr_scheduler.step(val_score)

    
    def decode_forced(self, ys, encoder_states, xs_lens):
        encoder_output, encoder_hidden = encoder_states
        
        bsz = ys.size(0)
        target_length = ys.size(1)
        longest_label = max(target_length, self.longest_label)
        starts = self.sos_buffer.expand(bsz, 1).long()  # expand to batch size
        
        # Teacher forcing: Feed the target as the next input
        y_in = ys.narrow(1, 0, ys.size(1) - 1)
        decoder_input = torch.cat([starts, y_in], 1)
        decoder_output, decoder_hidden, attn_w_log = self.decoder(decoder_input, encoder_hidden, encoder_states)
        _, preds = decoder_output.max(dim=2)
        #import ipdb; ipdb.set_trace()
        
        return decoder_output, preds, attn_w_log
    
    def decode_greedy(self, encoder_states, bsz):
        encoder_output, encoder_hidden = encoder_states
        
        starts = self.sos_buffer.expand(bsz, 1)  # expand to batch size
        decoder_hidden = encoder_hidden  # no attention yet

        # greedy decoding here        
        preds = [starts]
        scores = []
        xs = starts
        _attn_w_log = []
        
        for ts in range(self.longest_label):
            decoder_output, decoder_hidden, attn_w_log = self.decoder(xs, decoder_hidden, encoder_states)
            _scores, _preds = decoder_output.max(dim=-1)
            scores.append(_scores)
            preds.append(_preds)
            _attn_w_log.append(attn_w_log)
            
            xs = _preds
            
        #import ipdb; ipdb.set_trace()
        return scores, preds, attn_w_log

    def decode_beam(self, beam_size, batch_size, encoder_states):
        dev = self.opts['device']
        beams = [ Beam() for _ in range(batch_size) ]
        decoder_input = self.sos_buffer.expand(batch_size * beam_size, 1).to(dev)
        inds = torch.arange(batch_size).to(dev).unsqueeze(1).repeat(1, beam_size).view(-1)
        
        encoder_states = self.reorder_encoder_states(encoder_states, inds)  # not reordering but expanding
        incr_state = encoder_states[1]
        
        for ts in range(self.longest_label):
            if all((b.done() for b in beams)):
                break
            score, incr_state = self.decoder(decoder_input, incr_state)
            score = score[:, -1:, :]
            score = score.view(bsz, beam_size, -1)
            score = F.log_softmax(score, dim=-1)
            
            for i, b in enumerate(beams):
                if not b.done():
                    b.advance(score[i])
                    
            incr_state_inds = torch.cat([beam_size * i + b.get_backtrack_from_current_step() for i, b in enumerate(beams)])
            incr_state = self.reorder_decoder_incremental_state(incr_state, incr_state_inds)
            selection = torch.cat([b.get_output_from_current_step() for b in beams]).unsqueeze(-1)
            decoder_input = selection
            
        for b in beams:
            b.check_finished()

        return beams

    def compute_loss(self, encoder_states, xs_lens, ys):
        decoder_output, preds, attn_w_log = self.decode_forced(ys, encoder_states, xs_lens)
        scores = decoder_output.view(-1, decoder_output.size(-1))
        loss = self.criterion(scores, ys.view(-1))
        # normalize loss per non_null num of tokens
        num_tokens = ys.ne(self.opts['pad_idx']).long().sum().item()
        # accumulate metrics
        self.metrics['loss'] += loss.item()
        self.metrics['num_tokens'] += num_tokens
        loss /= num_tokens
        
        return loss
    

    def train_step(self, batch):
        xs, ys, use_packed = batch.text_vecs, batch.label_vecs, batch.use_packed
        xs_lens, ys_lens = batch.text_lens, batch.label_lens

        if xs is None:
            return
        bsz = xs.size(0)
        
        starts = self.sos_buffer.expand(bsz, 1)  # expand to batch size
        self.zero_grad()
        self.train_mode()

        encoder_states = self.encoder(xs, xs_lens, use_packed=use_packed)
        loss = self.compute_loss(encoder_states, xs_lens, ys)
        loss.backward()
        self.update_params()


    def eval_step(self, batch, decoding_strategy='score'):
        xs, ys, use_packed = batch.text_vecs, batch.label_vecs, batch.use_packed
        xs_lens, ys_lens = batch.text_lens, batch.label_lens
            
        self.eval_mode()
        encoder_states = self.encoder(xs, xs_lens, use_packed=use_packed)
        if decoding_strategy == 'score':
            assert ys is not None
            _ = self.compute_loss(encoder_states, xs_lens, ys)
            
        if decoding_strategy == 'greedy':
            scores, preds, attn_w_log = self.decode_greedy(encoder_states, batch.text_vecs.size(0))
            return scores, preds

        if decoding_strategy == 'beam':
            beams = self.decode_beam(5, len(batch.text_lens), encoder_states)
            return beams

