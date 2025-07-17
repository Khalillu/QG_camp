import numpy
import torch
import math
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib
import torch.functional as F
from torchaudio.models.wav2vec2.components import FeedForward

example = "Hello?! This is an example of a lut of using torch to build Embedding, which is very crucial to Transformer."

# 去掉特殊字符，提取单词
def tokenize(sequence):
    for punc in ["!", ".", "?",","]:
        sequence = sequence.replace(punc, "")
    return[token.lower() for token in sequence.split(" ")]

# 为每一个单词分配一个数字
def build_vocab(data):
    vocab = list(set(tokenize(data)))
    vocab.sort()
    stoi = {word: i for i, word in enumerate(vocab)}
    return stoi

stoi = build_vocab(example)

#全部词的数量
vocab_size = len(stoi)
# embedding dimensions
d_model = 64

# 生成embedding层
   # matrix of size（17,3）
embeddings = torch.rand(vocab_size, d_model)

def position_embedding(tensor):                                                # torch.rand(*size, *, dtype=None, device=None)，生成一个在区间[0,1)内均匀分布的随机变量
    for pos in range(vocab_size):
        for i in range(d_model):
            if i % 2==0:
                position_embeddings = math.sin(pos / (10000**(2*i / d_model)))
                embeddings[pos, i] += position_embeddings
            else:
                position_embeddings = math.cos(pos / (10000**(2*i / d_model)))
                embeddings[pos, i] += position_embeddings

position_embedding(embeddings)

General_embeddings = embeddings*math.sqrt(d_model)

x = General_embeddings
x1 = x
x2 = x

def LayerNorm(feature:torch.Tensor,beta=0,gamma=1,eps=1e-5):
    # unbiased=False    pytorch中的LayerNorm默认使用这方式
    var_mean = torch.var_mean(feature,dim=1,keepdim=True,unbiased=False)
    mean = var_mean[1]
    std = var_mean[0]
    feature = (feature - mean[...,None]) / torch.sqrt(std[...,None] + eps)
    feature = feature * gamma + beta

    return feature

def Feed_Forward(tensor):
    linear_layer = nn.Linear(d_model, 4*d_model)
    x = linear_layer(tensor)
    x = F.relu(x , inplace=True)
    linear_layer = nn.Linear(4*d_model, d_model)
    x = linear_layer(x)

    return x

batch_size = 1
seq_len = 20
def Multi_Head_Attention(x, n_heads, d_model, h, dropout=0.1):
    query = nn.Linear(d_model, d_model)(x)
    key = nn.Linear(d_model, d_model)(x)
    value = nn.Linear(d_model, d_model)(x)

    assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
    head_dim = d_model // n_heads

    query = query.view(batch_size, seq_len, n_heads, head_dim).transpose(1, 2)
    key = key.view(batch_size, seq_len, n_heads, head_dim).transpose(1, 2)
    value = value.view(batch_size, seq_len, n_heads, head_dim).transpose(1, 2)
    for i in range(0, h):
        xi = Scale_dot_product_Attention(query, key, value, n_heads, dropout)
    for i in range(0, h):
        x = torch.cat(xi)
    x = nn.Linear(x, General_embeddings)
    return x

def Masked_Multi_Head_Attention(x, n_heads, d_model, h, dropout=0.1):
    mask = torch.triu(torch.ones(d_model, d_model), diagonal=1).bool()
    query = nn.Linear(d_model, d_model)(x)
    key = nn.Linear(d_model, d_model)(x)
    value = nn.Linear(d_model, d_model)(x)
    for i in range(0, h):
        QK_Ti = query @ key.T
        QK_Ti @= mask
        QK_Ti = torch.softmax(QK_Ti, dim=-1)
        Z = QK_Ti@value
        Zi = Scale_dot_product_Attention(query, key, value, n_heads, dropout)
    for i in range(0, h):
        x = torch.cat(Zi)
    x = nn.Linear(x, General_embeddings)


    assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
    head_dim = d_model // n_heads

    query = query.view(batch_size, seq_len, n_heads, head_dim).transpose(1, 2)
    key = key.view(batch_size, seq_len, n_heads, head_dim).transpose(1, 2)
    value = value.view(batch_size, seq_len, n_heads, head_dim).transpose(1, 2)

    return x



def Scale_dot_product_Attention(query, key, value,d_model, dropout=0.1):
    x = torch.softmax(torch.bmm(query/math.sqrt(d_model), key.T/math.sqrt(d_model)), dim=-1)*value
    return x

class Encoder(nn.Module):
    def __init__(self, d_model, n_heads, h, x1,dropout=0.1):
        super(Encoder, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.h = h
        self.x = x1
        x = Multi_Head_Attention(x1, n_heads, d_model, h, dropout=0.1)
        x = LayerNorm(x + Multi_Head_Attention(x, n_heads, d_model, h, dropout=0.1))
        x = LayerNorm(x + Feed_Forward(x))
        x = Feed_Forward(x)
        x = LayerNorm(x + Multi_Head_Attention(x, n_heads, d_model, h, dropout=0.1))
        x = LayerNorm(x + Feed_Forward(x))
        return x

class Decoder(nn.Module):
    def __init__(self, d_model, n_heads, h, x2,dropout=0.1):
        super(Decoder, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.h = h
        self.x = x2
        x = Masked_Multi_Head_Attention(x2, n_heads, d_model, h, dropout=0.1)
        x = LayerNorm(x + Multi_Head_Attention(x, n_heads, d_model, h, dropout=0.1))
        x = LayerNorm(x + Feed_Forward(x))
      #  x = Multi_Head_Attention(x, n_heads, d_model, h, dropout=0.1)
        x = LayerNorm(x + Multi_Head_Attention(x, n_heads, d_model, h, dropout=0.1))
        x = LayerNorm(x + Feed_Forward(x))
        x = Feed_Forward(x)
        x = LayerNorm(x + Multi_Head_Attention(x, n_heads, d_model, h, dropout=0.1))
        x = LayerNorm(x + Feed_Forward(x))
        x = nn.Linear()
        x = torch.softmax(x, dim=-1)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

n=input()
for _ in range(n):
    x = Encoder(x)
    x = Decoder(x)

