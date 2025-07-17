import math
import torch
from torch import nn
from d2l import torch as d2l

# softmax操作用于输出一个概率分布作为注意力权重。在某些情况下，
# 并非所有的值都应该被纳入到注意力汇聚中。
# 例如，某些文本序列被填充了没有意义的特殊词元。
# 为了仅将有意义的词元作为值来获取注意力汇聚，可以指定一个有效序列长度。
# 以便在计算softmax时过滤掉超出指定范围的位置。
# 下面的masked_softmax函数实现了这样的掩蔽softmax操作
# 其中超出有效长度的位置都被掩蔽并置为0。

def masked_softmax(X, valid_lens):
    # 通过最后一个轴上的遮蔽元素来进行softmax操作
    # X:3D张量，valied_lens：1D或2D张量
    if valid_lens is None:
        return nn.functional.softmax(X, dim=-1)
    else:
        shape = X.shape
        if valid_lens.dim() ==1:
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        else:
            valid_lens = valid_lens.reshape(-1)
        # 最后一轴上被掩蔽的元素使用一个非常大的负值替换，从而其softmax输出为0
        X = d2l.sequence_mask(X.reshape(-1,shape[-1]), valid_lens,value=-1e6)

        return nn.functional.softmax(X.reshape(shape), dim=-1)

# 考虑由两个2×4矩阵表示的样本，这两个样本的有效长度分别为2和3。
# 经过softmax操作，超出有效长度的值都被遮蔽为0。
x = masked_softmax(torch.rand(2, 2, 4), torch.tensor([2, 3]))
print(x)