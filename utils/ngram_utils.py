import numpy as np
import spacy
import random
import numpy
import itertools
from operator import itemgetter 
from glob import glob
from tqdm import tqdm_notebook, tqdm
_tqdm = tqdm_notebook
from collections import Counter
import string
import re
import more_itertools as mit  # not built-in package
from collections import Counter
import re
import pandas
import altair
import pygtrie
import global_variables as gl

# Load English tokenizer, tagger, parser, NER and word vectors
tokenizer = spacy.load('en_core_web_sm')               
punctuations = string.punctuation
# punctuations = '"#$%&\'()*+,-/:;<=>@[\\]^_`{|}~' 
TAG_RE = re.compile(r'<[^>]+>') # get rid off HTML tags from the data

def remove_tags(text):
    return TAG_RE.sub('', text)

def lower_case(parsed):
    return [token.text.lower() for token in parsed] #and (token.is_stop is False)]

def remove_punc(parsed):
    return [token.text for token in parsed if (token.text not in punctuations)]

def lower_case_remove_punc(parsed):
    return [token.text.lower() for token in parsed if (token.text not in punctuations)] #and (token.is_stop is False)]

def tokenize_dataset(dataset):
   # tokenize each sentence -- each tokenized sentence will be an element in token_dataset
    token_dataset = []
    # tokenize all words -- each token will be an item in all_tokens (in the order given by the list of sentences)
    all_tokens = []     # all the tokens -- 

    for sample in tqdm(tokenizer.pipe(dataset, disable=['parser', 'tagger', 'ner'], batch_size=512, n_threads=1)):
#         tokens = lower_case_remove_punc(sample)
        tokens = lower_case(sample)       # make words lower case
#         tokens = remove_punct(tokens)     # remove punctuation
        token_dataset.append(tokens)    
        all_tokens += tokens

    return token_dataset, all_tokens
    
class NgramLM:
    def __init__(self, tokenized_data, all_tokens, n=3, frac_vocab=0.9):

        self.n = n
        self.frac_vocab = frac_vocab
        self.vocabulary = list(set(all_tokens))
        self.num_all_tokens = len(all_tokens)
        self.raw_data = tokenized_data
        
        self.padded_data = self.pad_sentences(self.n)
        self.ngram_data = self.find_ngrams(self.n)
        
        self.vocab_ngram, self.count_ngram = self.ngram_counts(self.n)
        self.id2token, self.token2id = self.ngram_dict(self.n)
            
        self.vocab_unigram, self.count_unigram = self.ngram_counts(1)
        self.vocab_bigram, self.count_bigram = self.ngram_counts(2)
        self.vocab_trigram, self.count_trigram = self.ngram_counts(3)
        self.vocab_prev_ngram, self.count_prev_ngram = self.ngram_counts(self.n - 1)
        
        
    def pad_sentences(self, n):
        result_list = []
        for l in self.raw_data:
            padded = [gl.SOS_TOKEN for i in range((n - 1))] + l +[gl.EOS_TOKEN for i in range((n - 1))]
            result_list.append(padded)
        return result_list

    def find_ngrams(self, n):
        result_list = []
        padded_data = self.pad_sentences(n)
        for l in padded_data:
            result_list.append(list(zip(*[l[i:] for i in range(n)])))
        return result_list

    def ngram_counts(self, n):    
        ngram_data = self.find_ngrams(n)
        all_train_tokens = list(mit.flatten(ngram_data))
        
        counted_tokens = Counter(all_train_tokens)
        max_vocab_size = int(self.frac_vocab * len(counted_tokens))

        vocab, count = zip(*counted_tokens.most_common(max_vocab_size))

        return vocab, count

    def ngram_dict(self, n):        
        id2token = list(self.vocab_ngram)
        token2id = dict(zip(self.vocab_ngram, range(4, 4+len(self.vocab_ngram)))) 
        id2token = [gl.PAD_TOKEN, gl.UNK_TOKEN, gl.SOS_TOKEN, gl.EOS_TOKEN] + id2token

        token2id[gl.PAD_TOKEN] = gl.PAD_IDX 
        token2id[gl.UNK_TOKEN] = gl.UNK_IDX
        token2id[gl.SOS_TOKEN] = gl.SOS_IDX 
        token2id[gl.EOS_TOKEN] = gl.EOS_IDX

        return id2token, token2id
      
    def get_ngram_count(self, ngram):
        if ngram in self.vocab_ngram:
            ngram_idx = self.vocab_ngram.index(ngram)
            return self.count[ngram_idx] 
        else:
            return 0

    def get_ngram_prob(self, ngram):
        c = self.get_ngram_count(ngram)
        all_counts = 0
        for t in self.vocab_ngram:
            if t[:-1] == ngram[:-1]:
    #             print(t, get_ngram_count(self, t))
                all_counts += get_ngram_count(self, t)
        if all_counts > 0:
            return c / all_counts
        else:
            return 0

    def get_ngram_prob_addditive_smoothing(self, ngram, delta=0.5):
        c = self.get_ngram_count(ngram) + delta*1
        all_counts = 0
        for t in self.vocab:
            if t[:-1] == ngram[:-1]:
    #             print(t, get_ngram_count(self, t))
                all_counts += self.get_ngram_count(t)
        all_counts += delta*len(self.vocabulary)
        if all_counts > 0:
            return c / all_counts
        else:
            return 0

    def get_ngram_prob_add_one_smoothing(self, ngram):
        c = self.get_ngram_count(ngram) + 1
        all_counts = 0
        for t in self.vocab:
            if t[:-1] == ngram[:-1]:
    #             print(t, self.get_ngram_count(t))
                all_counts += self.get_ngram_count(t)
        all_counts += len(self.vocabulary)
        if all_counts > 0:
            return c / all_counts
        else:
            return 0

    def get_ngram_prob_interpolation_smoothing(self, ngram, vocab, count, prev_vocab, prev_count, alpha=0.5):
        c = self.get_ngram_count(ngram, vocab, count)
        all_counts = 0
        for t in vocab:
            if t[:-1] == ngram[:-1]:
    #             print(t, self.get_ngram_count(t, vocab, count))
                all_counts += self.get_ngram_count(t, vocab, count)
        if all_counts > 0:
            prob_ngram = c / all_counts
        else:
            prob_ngram = 0

        prev_ngram = tuple(list(ngram[1:]))
        prev_c = self.get_ngram_count(prev_ngram, prev_vocab, prev_count)
    #     print(prev_c)
        prev_all_counts = 0
        for prev_t in prev_vocab:
            if prev_t[:-1] == prev_ngram[:-1]:
    #             print(prev_t, get_ngram_count(prev_t, prev_vocab, prev_count))
                prev_all_counts += self.get_ngram_count(prev_t, prev_vocab, prev_count)
        if prev_all_counts > 0:
            prob_prev_ngram = prev_c / prev_all_counts
        else:
            0
        return alpha*(prob_ngram) + (1-alpha)*prob_prev_ngram

    def get_unigram_count(self, r):
        return np.sum([1 for i in range(len(self.vocab_unigram)) if self.count_unigram[i] == r])

    def get_bigram_count(self, r):
        return np.sum([1 for i in range(len(self.vocab_bigram)) if self.count_bigram[i] == r])

    def get_biunigram_count(self, r, token):
        cc = 0
        for other_token in self.vocab_unigram:
            bigram = tuple([token] + [other_token])
            if bigram in self.vocab_bigram:
                bigram_idx = self.vocab_bigram.index(bigram) 
                if self.count_bigram[bigram_idx] == r:
                    cc += 1
        return cc

    def get_b_bi(self):
        bbi = self.get_bigram_count(1) / (self.get_bigram_count(1) + 2 * self.get_bigram_count(2))
        return bbi

    def get_b_uni(self):
        buni = self.get_unigram_count(1) / (self.get_unigram_count(1) + 2 * self.get_unigram_count(2))
        return buni

    def get_p_uni(self, w):
        N = self.num_all_tokens

        if w in self.vocab_unigram:
            w_idx = self.vocab_unigram.index(w)
            N_w = self.count_unigram[w_idx]
        else:
            N_w = 0

        b_uni = self.get_b_uni()

        W = len(self.vocabulary)
        N_0 = self.get_unigram_count(0)
        
        p_uni = max((N_w - b_uni / N), 0) + b_uni * (W - N_0) / N * 1 / W

        return p_uni

    def get_p_bi(self, w, v):   # w given v
        if tuple([v] + [w]) in vocab_bigram:
            vw_idx = self.vocab_bigram.index(tuple([v] + [w]))
            N_vw = self.count_bigram[vw_idx]
        else:
            N_vw = 0

        if tuple([v]) in self.vocab_unigram:
            v_idx = self.vocab_unigram.index(tuple([v]))
            N_v = self.count_unigram[v_idx]
        else:
            N_v = 0  

        b_bi = self.get_b_bi()
        b_uni = self.get_b_uni()

        p_uni = self.get_p_uni(tuple([w]))

        W = len(self.vocabulary)
        N_0 = self.get_biunigram_count(0, v)


        p_bi =  max((N_vw - b_bi) / N_v,  0) + \
             b_bi * (W - N_0) / N_v * p_uni

        return p_bi


    def get_prob_sentence(self, sentence):
        padded_sentence = self.pad_sentences(sentence)  # needs a list
    #     print(padded_sentence)
        ngram_sentence = self.find_ngrams(padded_sentence)[0] # only one element in list
    #     print(ngram_sentence)
        prob = 1
        for ngram in ngram_sentence:
            prob_ngram = self.get_ngram_prob(ngram)
    #         print(ngram, prob_ngram)
            prob *= prob_ngram
        return prob

    def get_prob_distr_ngram(self, prev_tokens):
        pd = [0 for v in voc]
        for idx, token in enumerate(self.vocabulary):
    #         print("token: ", token)
    #         print("prev ngram: ", prev_tokens)
    #         print("both: ", tuple(list(prev_tokens) + [token]))
    #         print("")
            token_ngram = tuple(list(prev_tokens) + [token])
            pd[idx] = self.get_ngram_prob(token_ngram)
    #         if pd[idx] > 0 and print_nonzero_probs:
    #             print(token_ngram, " ", pd[idx])
        return pd

    def sample_from_pd(self, prev_tokens):
        pd = self.get_prob_distr_ngram(prev_tokens)
        idx_next_token = np.random.choice(len(self.vocabulary), 1, p=pd)[0]
        return self.vocabulary[idx_next_token]


    def generate_sentence(self, num_tokens):
        sentence = []
        prev_tokens = tuple(gl.SOS_TOKEN * (self.n - 1))
    #     print(prev_tokens)
        for i in range(num_tokens):
            next_token = self.sample_from_pd(prev_tokens)
    #         print(i, next_token)
    #         print(i, prev_tokens[1:])
            prev_tokens = tuple(list(prev_tokens[1:]) + [next_token])
    #         print(i, prev_tokens)
            sentence.append(next_token)
            print(' '.join(sentence))
        return ' '.join(sentence)

    def get_perplexity(self, test_sentences):
        ll = 0
        num_tokens = 0
        for s in (test_sentences):
            ll += self.get_prob_sentence([s])
            num_tokens += len(s) + 1
        ppl = np.exp(-ll/num_tokens)
        return ppl

    def _text2id(self, doc):
        return [self.token2id[t] if t in self.token2id else gl.UNK_IDX for t in doc]

    def _id2text(self, vec):
        return [self.id2token[i] for i in vec]

    def create_data_id(self, data):
        data_id = []
        for d in data:
            data_id.append(self._text2id(self, d))
        return data_id

    def create_data_id_merged(self, data):
        data_id_merged = []
        for d in data:
            for i in range(len(d) - self.n):
                data_id_merged.append((d[i:i+n], d[i+n]))
        return data_id_merged
