import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
# English to German datasets
from torchtext.datasets import Multi30k
# Preprocessing
from torchtext.data import Field, BucketIterator
# Tokenizer
import spacy
# Nice loss plots
from torch.utils.tensorboard import SummaryWriter

spacy_ger = spacy.load('de')
spacy_neg = spacy.load('en')

def tokenizer_ger(text):
    return [tok.text for tok in spacy_ger.tokenizer(text)]
def tokenizer_eng(text):
    return [tok.text for tok in spacy_eng.tokenizer(text)]

german = Field(tokenize=tokenizer_ger, lower=True, init_token = '<sos>', eos_token='<eos>')
german = Field(tokenize=tokenizer_eng, lower=True, init_token = '<sos>', eos_token='<eos>')

train_data, validation_data, test_data = Multi30k.splits(exts=('.de', '.en'), fields=(german, english)) # extensionsand fields

german.build_vocab(train_data, max_size=10000, min_freq = 2) # Once만 나온다면, 이를 vocabulary에 추가 안함 최소 2번은 나와야 추가
english.build_vocab(train_data, max_size=10000, min_freq = 2) # Once만 나온다면, 이를 vocabulary에 추가 안함 최소 2번은 나와야 추가

#TODO: bucket추가해야 함

class Encoder(nn.Module):
    # input_size 이경우 German vocab의 사이즈일 것이고, embedding_size d dimensional
    def __init__(self, input_size, embedding_size, hidden_size, num_layers, p):
        # super 클래스란, 자식 클래스에서 부모 클래스의 내용을 사용하고 싶을 경우 사용
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.dropout = nn.Dropout(p)
        # Embedding을 하는 부분
        self.embedding = nn.Embedding(input_size, embedding_size)
        # Embedding 된 벡터에 대해 LSTM을 수행할 것이다
        self.rnn = nn.LSTM(embeddign_size, hidden_size, num_layers, dropout=p)
    
    # sentence -> Tokenize된 뒤 -> vocab으로 map -> embedding -> LSTM
    def forward(self, k):
        # x shape: (seq_length, N)
        embedding = self.dropout(self.embedding(x))
        # embedding shape: (seq_length, N, embedding_size)
        
        self.rnn(embedding)
        # hidden과 cell 이 context
        outputs, (hidden, cell) = self.rnn(embedding)

        return hidden, cell


class Decoder(nn.Module):
    pass

# Combine the Encoder and Decoder
class Seq2Seq(nn.Module):
    pass


