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

    for sample in _tqdm(tokenizer.pipe(dataset, disable=['parser', 'tagger', 'ner'], batch_size=512, n_threads=1)):
#         tokens = lower_case_remove_punc(sample)
        tokens = lower_case(sample)       # make words lower case
#         tokens = remove_punct(tokens)     # remove punctuation
        token_dataset.append(tokens)    
        all_tokens += tokens
        
    return token_dataset, all_tokens

def pad_sentences(input_list, n):
    result_list = []
    for l in input_list:
        padded = [gl.SOS_TOKEN for i in range((n - 1))] + l +[gl.EOS_TOKEN for i in range((n - 1))]
        result_list.append(padded)
    return result_list

def find_ngrams(input_list, n):
    result_list = []
    for l in input_list:
        result_list.append(list(zip(*[l[i:] for i in range(n)])))
    return result_list

def ngram_counts(data, frac_vocab=0.9):    
    all_train_tokens = list(mit.flatten(data))
    counted_tokens = Counter(all_train_tokens)
    max_vocab_size = int(frac_vocab * len(counted_tokens))

    vocab, count = zip(*counted_tokens.most_common(max_vocab_size))
    
    return vocab, count

def ngram_dict(vocab):
    id2token = list(vocab)
    token2id = dict(zip(vocab, range(4, 4+len(vocab)))) 
    id2token = [gl.PAD_TOKEN, gl.UNK_TOKEN, gl.SOS_TOKEN, gl.EOS_TOKEN] + id2token

    token2id[gl.PAD_TOKEN] = gl.PAD_IDX 
    token2id[gl.UNK_TOKEN] = gl.UNK_IDX
    token2id[gl.SOS_TOKEN] = gl.SOS_IDX 
    token2id[gl.EOS_TOKEN] = gl.EOS_IDX

    return id2token, token2id

def get_ngram_count(ngram, vocab, count):
    if ngram in vocab:
        ngram_idx = vocab.index(ngram)
        return count[ngram_idx] 
    else:
        return 0
    
def get_ngram_prob(ngram, vocab, count):
    c = get_ngram_count(ngram, vocab, count)
    all_counts = 0
    for t in vocab:
        if t[:-1] == ngram[:-1]:
#             print(t, get_ngram_count(t, vocab, count))
            all_counts += get_ngram_count(t, vocab, count)
    if all_counts > 0:
        return c / all_counts
    else:
        return 0
    
def get_ngram_prob_addditive_smoothing(ngram, vocab, count, delta=0.5):
    c = get_ngram_count(ngram, vocab, count) + delta*1
    all_counts = 0
    for t in vocab:
        if t[:-1] == ngram[:-1]:
#             print(t, get_ngram_count(t, vocab, count))
            all_counts += get_ngram_count(t, vocab, count)
    all_counts += delta*len(voc)
    if all_counts > 0:
        return c / all_counts
    else:
        return 0
    
def get_ngram_prob_add_one_smoothing(ngram, vocab, count):
    c = get_ngram_count(ngram, vocab, count) + 1
    all_counts = 0
    for t in vocab:
        if t[:-1] == ngram[:-1]:
#             print(t, get_ngram_count(t, vocab, count))
            all_counts += get_ngram_count(t, vocab, count)
    all_counts += len(voc)
    if all_counts > 0:
        return c / all_counts
    else:
        return 0
    
def get_ngram_prob_interpolation_smoothing(ngram, vocab, count, prev_vocab, prev_count, alpha=0.5):
    c = get_ngram_count(ngram, vocab, count)
    all_counts = 0
    for t in vocab:
        if t[:-1] == ngram[:-1]:
#             print(t, get_ngram_count(t, vocab, count))
            all_counts += get_ngram_count(t, vocab, count)
    if all_counts > 0:
        prob_ngram = c / all_counts
    else:
        prob_ngram = 0
    
    prev_ngram = tuple(list(ngram[1:]))
    prev_c = get_ngram_count(prev_ngram, prev_vocab, prev_count)
#     print(prev_c)
    prev_all_counts = 0
    for prev_t in prev_vocab:
        if prev_t[:-1] == prev_ngram[:-1]:
#             print(prev_t, get_ngram_count(prev_t, prev_vocab, prev_count))
            prev_all_counts += get_ngram_count(prev_t, prev_vocab, prev_count)
    if prev_all_counts > 0:
        prob_prev_ngram = prev_c / prev_all_counts
    else:
        0
    return alpha*(prob_ngram) + (1-alpha)*prob_prev_ngram

def get_unigram_count(r):
    return np.sum([1 for i in range(len(vocab_unigram)) if count_unigram[i] == r])

def get_bigram_count(r):
    return np.sum([1 for i in range(len(vocab_bigram)) if count_bigram[i] == r])

def get_biunigram_count(r, token):
    cc = 0
    for other_token in vocab_unigram:
        bigram = tuple([token] + [other_token])
        if bigram in vocab_bigram:
            bigram_idx = vocab_bigram.index(bigram) 
            if count_bigram[bigram_idx] == r:
                cc += 1
                
#     for bigram in vocab_bigram:
#         print(token, bigram[0])
#         if token == bigram[0]:
#             bigram_idx = vocab_bigram.index(bigram) 
#             if count_bigram[bigram_idx] == r:
#                 cc += 1
    return cc

def get_b_bi():
    bbi = get_bigram_count(1) / (get_bigram_count(1) + 2 * get_bigram_count(2))
    return bbi
    
def get_b_uni():
    buni = get_unigram_count(1) / (get_unigram_count(1) + 2 * get_unigram_count(2))
    return buni

def get_p_uni(w):
    if w in vocab_unigram:
        w_idx = vocab_unigram.index(w)
        N_w = count_unigram[w_idx]
    else:
        N_w = 0
        
    b_uni = get_b_uni()
    
    W = len(voc)
    N_0 = get_unigram_count(0)
    
    
    N = len(all_tokens_train) # TODO: double check the meaning of N 
    
    p_uni = max((N_w - b_uni / N), 0) + b_uni * (W - N_0) / N * 1 / W
    
    return p_uni

def get_p_bi(w, v):   # w given v
    if tuple([v] + [w]) in vocab_bigram:
        vw_idx = vocab_bigram.index(tuple([v] + [w]))
        N_vw = count_bigram[vw_idx]
    else:
        N_vw = 0
        
    if tuple([v]) in vocab_unigram:
        v_idx = vocab_unigram.index(tuple([v]))
        N_v = count_unigram[v_idx]
    else:
        N_v = 0  
        
    b_bi = get_b_bi()
    b_uni = get_b_uni()
    
    p_uni = get_p_uni(tuple([w]))
    
    W = len(voc)
    N_0 = get_biunigram_count(0, v)
    
    
    p_bi =  max((N_vw - b_bi) / N_v,  0) + \
         b_bi * (W - N_0) / N_v * p_uni
    
    return p_bi
        
    
def get_prob_sentence(sentence, vocab, count, n):
    padded_sentence = pad_sentences(sentence, n)  # needs a list
#     print(padded_sentence)
    ngram_sentence = find_ngrams(padded_sentence, n)[0] # only one element in list
#     print(ngram_sentence)
    prob = 1
    for ngram in ngram_sentence:
        prob_ngram = get_ngram_prob(ngram, vocab, count)
#         print(ngram, prob_ngram)
        prob *= prob_ngram
    return prob

def get_prob_distr_ngram(prev_tokens, vocab_ngram, count_ngram, voc, print_nonzero_probs=False):
    pd = [0 for v in voc]
    for idx, token in enumerate(voc):
#         print("token: ", token)
#         print("prev ngram: ", prev_tokens)
#         print("both: ", tuple(list(prev_tokens) + [token]))
#         print("")
        token_ngram = tuple(list(prev_tokens) + [token])
        pd[idx] = get_ngram_prob(token_ngram, vocab_ngram, count_ngram)
#         if pd[idx] > 0 and print_nonzero_probs:
#             print(token_ngram, " ", pd[idx])
    return pd

def sample_from_pd(prev_tokens, vocab_ngram, count_ngram, voc, print_nonzero_probs=False):
    pd = get_prob_distr_ngram(prev_tokens, vocab_ngram, count_ngram, voc, print_nonzero_probs=print_nonzero_probs)
    idx_next_token = np.random.choice(len(voc), 1, p=pd)[0]
    return voc[idx_next_token]
    
    
def generate_sentence(num_tokens, vocab_ngram, count_ngram, voc, n):
    sentence = []
    prev_tokens = tuple(gl.SOS_TOKEN * (n - 1))
#     print(prev_tokens)
    for i in range(num_tokens):
        next_token = sample_from_pd(prev_tokens, vocab_ngram, count_ngram, voc)
#         print(i, next_token)
#         print(i, prev_tokens[1:])
        prev_tokens = tuple(list(prev_tokens[1:]) + [next_token])
#         print(i, prev_tokens)
        sentence.append(next_token)
        print(' '.join(sentence))
    return ' '.join(sentence)

def get_perplexity(test_sentences, vocab_ngram, count_ngram):
    ll = 0
    num_tokens = 0
    for s in (test_sentences):
        ll += get_prob_sentence([s], vocab_ngram, count_ngram, n)
        num_tokens += len(s) + 1

    ppl = np.exp(-ll/num_tokens)
    return ppl

def _text2id(doc, token2id):
    return [token2id[t] if t in token2id else gl.UNK_IDX for t in doc]
    
def _id2text(vec, id2token):
    return [id2token[i] for i in vec]

def create_data_id(data, token2id):
    data_id = []
    for d in data:
        data_id.append(_text2id(d, token2id))
    return data_id

def create_data_id_merged(data, token2id, n):
    data_id_merged = []
    for d in data:
        for i in range(len(d) - n):
            data_id_merged.append((d[i:i+n], d[i+n]))
    return data_id_merged
