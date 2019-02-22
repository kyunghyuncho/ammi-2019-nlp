import os
from io import open
import torch

UNK_TOKEN = '<unk>'

class Dictionary(object):
	def __init__(self):
		
		self.word2idx = {}
		self.idx2word = []
		
		self.add_word(UNK_TOKEN);

	def add_word(self, word):
		if word not in self.word2idx:
			self.idx2word.append(word)
			self.word2idx[word] = len(self.idx2word) - 1
		return self.word2idx[word]

	def __len__(self):
		return len(self.idx2word)


class Corpus(object):
	def __init__(self, path):
		self.dictionary = Dictionary()
		self.train = self.tokenize(os.path.join(path, 'train.txt'))
		self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
		self.test = self.tokenize(os.path.join(path, 'test.txt'))

	def tokenize(self, path):
		"""Tokenizes a text file."""
		assert os.path.exists(path)
		# Add words to the dictionary
		with open(path, 'r', encoding="utf8") as f:
			tokens = 0
			for line in f:
				words = line.split() + ['<eos>']
				tokens += len(words)
				for word in words:
					self.dictionary.add_word(word)

		# Tokenize file content
		with open(path, 'r', encoding="utf8") as f:
			ids = torch.LongTensor(tokens)
			token = 0
			for line in f:
				words = line.split() + ['<eos>']
				for word in words:
					ids[token] = self.dictionary.word2idx[word]
					token += 1

		return ids
	
	def tokenize_sentence(self, sentence):
	   
		# Tokenize string
		token = 0
		words = sentence.split() + ['<eos>']
		ids = torch.LongTensor(len(words))
		for word in words:
			if word in list(self.dictionary.word2idx.keys()):
				ids[token] = self.dictionary.word2idx[word]
			else:
				ids[token] = self.dictionary.word2idx[UNK_TOKEN]
			token += 1
			
		return ids