import math
import torch
from torch import nn
from d2l import torch as d2l
from 遮蔽Softmax import masked_softmax

class AdditiveAttention(nn.Module):
    def __init__(self, key_size, query_size, num_hiddens, dropout, **kwargs):
        super(AdditiveAttention, self).__init__(**kwargs)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=False)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=False)
        self.W_v = nn.Linear(num_hiddens, 1, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, valid_lens):
        queries = self.W_q(queries)
        keys = self.W_k(keys)
        # 在维度扩展后，queries的形状：(batch_size, 查询个数, 1, num_hiddens)
        # key的形状：(batch_size, 1, "键-值"对的个数, num_hiddens)
        # 使用广播方式进行求和
        features = queries.unsqueeze(2) + keys.unsqueeze(1)
        features = torch.tanh(features)
        # self.w_v仅有一个输出，因此从形状中移除最后那个维度
        # scores的形状：(batch_size, 查询的个数, "键-值"对的个数)
        scores = self.W_v(features).squeeze(-1)
        self.attention_weights = masked_softmax(scores, valid_lens)

        return torch.bmm(self.dropout(self.attention_weights), values)

queries, keys = torch.normal(0, 1, (2, 1, 20)), torch.ones((2, 10, 2))
values = torch.arange(40, dtype=torch.float32).reshape(1, 10, 4).repeat(2, 1, 1)
# 输入数据：
# queries: 形状(2,1,20)，两个批量，每个批量1个查询，每个查询20维
# keys: 形状(2,10,2)，两个批量，每个批量10个键，每个键2维
# values: 形状(2,10,4)，由arange(40)生成并重复，包含数值0-39
# valid_lens: [2,6]，表示第一个批量只看前2个键值对，第二个批量看前6个
valid_lens = torch.tensor([2, 6])

attention = AdditiveAttention(key_size=2, query_size=20, num_hiddens=8, dropout=0.1)
attention.eval()
x = attention(queries, keys, values, valid_lens)
print(x)

d2l.show_heatmaps(attention.attention_weights.reshape((1, 1, 2, 10)),
                  xlabel='Keys', ylabel='Queries')
d2l.plt.show()

