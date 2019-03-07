import numpy as np
import pandas as pd
import pickle
import os
from torch.utils.data import Dataset
import unicodedata
import re

from collections import namedtuple

import torch

import global_variables

SOS_token = global_variables.SOS_token
EOS_token = global_variables.EOS_token
UNK_IDX = global_variables.UNK_IDX
PAD_IDX = global_variables.PAD_IDX


class Lang:
	def __init__(self, name, minimum_count = 5):
		self.name = name
		self.word2index = {}
		self.word2count = {}
		self.index2word = [None]*4
		self.index2word[SOS_token] = 'SOS'
		self.index2word[EOS_token] = 'EOS'
		self.index2word[UNK_IDX] = 'UNK'
		self.index2word[PAD_IDX] = 'PAD'
		self.n_words = 4  # Count SOS and EOS

		self.minimum_count = minimum_count;

	def addSentence(self, sentence):
		for word in sentence.split(' '):
			self.addWord( word.lower() )

	def addWord(self, word):
		if word not in self.word2count.keys():
			self.word2count[word] = 1
		else:
			self.word2count[word] += 1
			
		if self.word2count[word] >= self.minimum_count:
			if word not in self.index2word:
				self.word2index[word] = self.n_words
				self.index2word.append(word)
				self.n_words += 1

	def vec2txt(self, list_idx):
		word_list = []
		if type(list_idx) == list:
			for i in list_idx:
				if i not in set([EOS_token]):
					word_list.append(self.index2word[i])
		else:
			for i in list_idx:
				if i.item() not in set([EOS_token,SOS_token,PAD_IDX]):
					word_list.append(self.index2word[i.item()])
		return (' ').join(word_list)


# Turn a Unicode string to plain ASCII, thanks to
# http://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
	return ''.join(
		c for c in unicodedata.normalize('NFD', s)
		if unicodedata.category(c) != 'Mn'
	)

# Lowercase, trim, and remove non-letter characters


def normalizeString(s):
	s = unicodeToAscii(s.lower().strip())
	s = re.sub(r"([.!?])", r" \1", s)
	s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
	return s

def read_dataset(file):
	# Read the file and split into lines
	lines = open(file, encoding='utf-8').\
		read().strip().split('\n')

	# Split every line into pairs and normalize
	pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]
	return pd.DataFrame(pairs, columns = ['source_data', 'target_data'])


def token2index_dataset(df, source_lang_obj, target_lang_obj):
	for lan in ['source','target']:
		indices_data = []
		if lan=='source':
			lang_obj = source_lang_obj
		else:
			lang_obj = target_lang_obj
			
		for tokens in df[lan+'_tokenized']:
			
			index_list = [lang_obj.word2index[token] if token in lang_obj.word2index else UNK_IDX for token in tokens]
			index_list.append(EOS_token)
			indices_data.append(index_list)
			
		df[lan+'_indized'] = indices_data
		
	return df

def load_or_create_language_obj(source_name, source_lang_obj_path, source_data, minimum_count):
	
	if not os.path.exists(source_lang_obj_path):
		os.makedirs(source_lang_obj_path)
	
	full_file_path = os.path.join(source_lang_obj_path, source_name+'_lang_obj_'+'min_count_'+str(minimum_count)+'.p')
	
	if os.path.isfile(full_file_path):
		source_lang_obj = pickle.load( open( full_file_path, "rb" ) );
	else:
		source_lang_obj = Lang(source_name, minimum_count);
		for i, line in enumerate(source_data):
#           if i%10000 == 0:
#               print(i, len(source_data))
#               print(str(float(i/len(source_data))*100)+' done');
			source_lang_obj.addSentence(line);
		pickle.dump( source_lang_obj, open(full_file_path , "wb" ) )
		
	return source_lang_obj


def load_language_pairs(filepath, source_name = 'en', target_name = 'vi',
						lang_obj_path = '.',  minimum_count = 5):
	main_df = read_dataset(filepath);
	
	
	source_lang_obj = load_or_create_language_obj(source_name, lang_obj_path, main_df['source_data'], minimum_count);
	target_lang_obj = load_or_create_language_obj(target_name, lang_obj_path, main_df['target_data'], minimum_count);
	
	for x in ['source', 'target']:
		main_df[x+'_tokenized'] = main_df[x + "_data"].apply(lambda x:x.lower().split() );
		main_df[x+'_len'] = main_df[x+'_tokenized'].apply(lambda x: len(x)+1) #+1 for EOS
	
	main_df = token2index_dataset(main_df, source_lang_obj, target_lang_obj);
	
	# main_df = main_df[ np.logical_and( np.logical_and(main_df['source_len'] >=2, main_df['target_len'] >=2) , 
	#                               np.logical_and( main_df['source_len'] <= Max_Len, main_df['target_len'] <= Max_Len) ) ];

	main_df =  main_df[  np.logical_and(main_df['source_len'] >=2, main_df['target_len'] >=2 ) ]
	
	return main_df, source_lang_obj, target_lang_obj
	

class LanguagePair(Dataset):
	def __init__(self, source_name, target_name, filepath, 
					lang_obj_path, val = False, minimum_count = 5, max_num = None):
		
		self.source_name = source_name;
		self.target_name = target_name;
		self.val = val;
		self.minimum_count = minimum_count;

		self.main_df, self.source_lang_obj, self.target_lang_obj = load_language_pairs(filepath, 
																			  source_name, target_name, lang_obj_path, minimum_count);

		self.max_num = max_num;
		
	def __len__(self):
		return len( self.main_df ) if self.max_num is None else self.max_num
	
	def __getitem__(self, idx):
		
		return_list = [self.main_df.iloc[idx]['source_indized'], self.main_df.iloc[idx]['target_indized'], 
					self.main_df.iloc[idx]['source_len'], self.main_df.iloc[idx]['target_len'] ]

		if self.val:
			return_list.append(self.main_df.iloc[idx]['target_data'])
		
		return return_list 


def vocab_collate_func(batch, MAX_LEN):
	source_data = []
	target_data = []
	source_len = []
	target_len = []

	for datum in batch:
		source_len.append(datum[2])
		target_len.append(datum[3])

	MAX_LEN_Source = np.min([ np.max(source_len), MAX_LEN ]);
	MAX_LEN_Target = np.min([np.max(target_len), MAX_LEN]);

	source_len = np.clip(source_len, a_min = None, a_max = MAX_LEN_Source )
	target_len = np.clip(target_len, a_min = None, a_max = MAX_LEN_Target )
	# padding
	for datum in batch:
		if datum[2]>MAX_LEN_Source:
			padded_vec_s1 = np.array(datum[0])[:MAX_LEN_Source]
		else:
			padded_vec_s1 = np.pad(np.array(datum[0]),
								pad_width=((0,MAX_LEN_Source - datum[2])),
								mode="constant", constant_values=PAD_IDX)
		if datum[3]>MAX_LEN_Target:
			padded_vec_s2 = np.array(datum[1])[:MAX_LEN_Target]
		else:
			padded_vec_s2 = np.pad(np.array(datum[1]),
								pad_width=((0,MAX_LEN_Target - datum[3])),
								mode="constant", constant_values=PAD_IDX)
		source_data.append(padded_vec_s1)
		target_data.append(padded_vec_s2)
		
	
	named_returntuple = namedtuple('namedtuple', ['text_vec', 'text_lengths', 'label_vec', 'label_lengths'])
	return_tuple =named_returntuple( torch.from_numpy(np.array(source_data)), 
									 torch.from_numpy(np.array(source_len)),
									 torch.from_numpy(np.array(target_data)),
									 torch.from_numpy(np.array(target_len)) );

	return return_tuple
