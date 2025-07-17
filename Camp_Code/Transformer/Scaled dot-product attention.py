import math
import torch
from 遮蔽Softmax import masked_softmax
from torch import nn
from d2l import torch as d2l

class DotProductAttention(nn.Module):
    def __init__(self, dropout=0, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)

    # queries的形状：(batch_size, 查询的个数, d)
    # keys的形状：(batch_size, ”键-值“对的个数, d)
    # values的形状：(batch_size, "键-值"对的个数, 值的维度)
    # valid_lens的形状：(batch_size, )或者(batch_size, 查询的个数)

    def forward(self, queries, keys, values, valid_lens=None):
        d = queries.shape[-1]
        # 设置transpose_b = True为了交换keys的最后两个维度
        scores = torch.bmm(queries, keys.transpose(1, 2))/math.sqrt(d)
        self.attention_weights = masked_softmax(scores, valid_lens)
        return torch.bmm(self.dropout(self.attention_weights), values)

if __name__ == '__main__':
    queries = torch.normal(0, 1,(2, 1, 2))
    keys = torch.ones((2, 10, 2))
    values = torch.arange(40, dtype=torch.float32).reshape(1, 10, 4).repeat(2, 1, 1)
    valid_lens = torch.tensor([2, 6])
    attention = DotProductAttention(dropout=0.5)
    attention.eval()
    x = attention(queries,keys,values,valid_lens)
    print(x)
    d2l.show_heatmaps(attention.attention_weights.reshape((1, 1, 2, 10)),
                      xlabel='Keys', ylabel='Queries')
    d2l.plt.show()