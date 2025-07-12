## embedding：

input：向量x

将单词x，通过学习一个长为d的向量来表示

output：长为d的向量

## positional encoding（PE）：

input：

长为d的向量

向量加上PE的位置信息

公式:

 PE(pos,2i) = sin(pos / 10000**2i/d_model)

PE(pos,2i+1) = cos(pos / 10000**2i/d_model)

output：

分为三个维度
Value, Key, Query的向量

## Multi-Head Attention:

input：

输入一个向量Value，三个维度Value，Key，Query

将Query、Key、Value通过Linear层降维，投影到d_k, d_k, d_v维，进入h个scaled Dot-product Attention，得到h个output，之后concat在一起，再做一次Linear投影，还需要通过ResNet连接在一起

output：

权重矩阵

## scaled Dot-Product attention

input：values、queries与keys

通过一些函数来计算两个内积，内积越大，cos值越大，相似度越高。

output：

softmax(Q K.T/sqrt(d_k)) V

## Feed Forward

input:d维的向量x

通过W1将向量x投影到4d维，再通过W2将维度投影到512层

输出：

FFN(x) = max(0, xW1 + b1)W2 + b2

## Normalization

input：

样本

将一行（即一个样本）的各个数据做正态分布中的标准化，即使均值为0，方差为1

（每一个特征向量 - 均值 ）/ 标准差

output：

## Masked 

input：

向量x

将kt即kt以后的值变成非常大的负数，此时再softmax

output：

向量x，但第t维之后全部变成非常大的负数