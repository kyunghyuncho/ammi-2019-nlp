import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PAD_IDX = 0
UNK_IDX = 1 
SOS_IDX = 2
EOS_IDX = 3

PAD_TOKEN = '<pad>'
UNK_TOKEN = '<unk>' 
SOS_TOKEN = '<sos>'
EOS_TOKEN = '<eos>'
