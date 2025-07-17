## D2L注意力机制及Transformer流程

#### 生物学中的注意力提示：

受试者基于非自主性提示和自主性提示有选择地引导注意力的焦点。 

非自主性提示是基于环境中**物体的突出性和易见性**。

想象一下，假如我们面前有五个物品：一份报纸、一篇 研究论文、一杯咖啡、一本笔记本和一本书，就像 图10.1.1。所有纸制品都是黑白印刷的，但咖啡杯是红色 的。换句话说，这个咖啡杯在这种视觉环境中是**突出和显眼**的，不由自主地引起人们的注意。所以我们会把视力最敏锐的地方放到咖啡上，如 图10.1.1所示。

![image-20250717091030637](https://khalillu.oss-cn-guangzhou.aliyuncs.com/khalillu/20250717091030738.png)

喝咖啡后，我们会变得兴奋并想读书，所以转过头，重新聚焦眼睛，然后看看书，就像 图10.1.2中描述那样。 与 图10.1.1中由于**突出性**导致的选择不同，此时选择书是受到了**认知和意识**的控制，因此注意力在基于自主 性提示去辅助选择时将更为谨慎。受试者的主观意愿推动，选择的力量也就更强大。

![image-20250717091226978](https://khalillu.oss-cn-guangzhou.aliyuncs.com/khalillu/20250717091227028.png)

#### 查询、键和值

##### 只使用非自主性提示：

将选择偏向于感官输入，则可以简单地使用参数化的全连接层，甚至是非参数化的最大汇聚层（Max Pooling）或平均汇聚层（Average Pooling）。

##### 包含自主性提示：

**自主性提示**被称为**查询（query）**。给定任何查询，注意力机制通过**注意力汇聚**（Attention Pooling）将选择引导至**感官输入**（sensory inputs，例如中间特征表示）。在注意力机制中，这些感官输入被称为**值（value）**。更通俗的解释，每个值都与一个**键（Key）**配对，这可以想象为感官输入的**非自主提示**。图10.1.3，可以通过设计注意力汇聚的方式，便于给定的**查询**（自主性提示）与**键**（非自主性提示）进行**匹配**，这将引导得出**最匹配的值**（感官输入）。

**Query与Key匹配，得出最匹配的Value**

![image-20250717092132205](https://khalillu.oss-cn-guangzhou.aliyuncs.com/khalillu/20250717092132240.png)

平均汇聚层可以被视为输入的加权平均值，其中各输入的权重是一样的。实际上，注意力汇聚得到的是**加权平均的总和值**，其中**权重**是在**给定的查询和不同的键**之间计算得出的。

### 注意力汇聚：

#### Nadaraya-Watson核回归模型：

非参数注意力汇聚：

$f(x) = \sum_{i=1}^{n}\alpha(x,x_i)y_i$

其中，$x$查询，$(x_i,y_i)$为键值对。

注意力汇聚是$y_i$的加权平均。将查询$x$和键$x_i$之间的关系建模 为注意力权重（attention weight）$\alpha(x,x_i)$，这个权重将被分配给每一个对应值$y_i$。对于任何查询，模型在所有键值对注意力权重都是一个有效的概率分布：它们是非负的且总和为1。

如果一个键$x_i$越是**接近**给定的查询$x$，那么分配给这个键对应的值$y_i$​​的注意力**权重就会越大**。

#### 非参数注意力汇聚：

平均汇聚忽略了输入$x_i$。

于是NW提出一个更好的想法：**根据输入位置对输出$y_i$进行加权**

高斯核：

![image-20250717140908021](https://khalillu.oss-cn-guangzhou.aliyuncs.com/khalillu/20250717140908087.png)

其中K是核（kernel）。

####  带参数注意力汇聚

非参数的NW核回归具有一致性的优点：如果有足够的数据，此模型会收敛到最优结果。尽管如此，我们还是可以轻松地将可学习的参数集成到注意力汇聚中。

![image-20250717143451228](https://khalillu.oss-cn-guangzhou.aliyuncs.com/khalillu/20250717143451264.png)

##### 批量矩阵乘法：

为了更有效地计算小批量数据的注意力，我们可以利用深度学习开发框架中提供的批量矩阵乘法。

假设第一个小批量数据包含n个矩阵$x_1,...x_n$，形状为a×b，第二个小批量包含n个矩阵$y_1,...,y_n$，形状为b×c。他们的批量矩阵乘法得到n个矩阵$x_1y_1,...,x_n,y_n$，形状为a×c。因此，假设两个张量的形状别分是（n, a, b）和（n, b, c），它们的批量矩阵乘法输出的形状为（n, a, c）

eg:

```X = torch.ones((2, 1, 4))```

`Y = torch.ones((2, 4, 6))`

`torch.bmm(X, Y).shape`

输出：`torch.Size([2, 1, 6])`

在注意力机制的背景中，我们可以使用小批量矩阵乘法来计算小批量数据中的加权平均值。

`weights = torch.ones((2, 10))*0.1`

`values = torch.arange(20.0).reshape((2, 10))`

`torch.bmm(weights.unsqueeze(1), values.unsqueeze(-1))`

输出：`tensor([[[ 4.5000]],`

​			  ` [[14.5000]]])`

### 注意力评分函数：



高斯核指数部分可以视为注意力评分函数（attention scoring function），简称评分函数，然后把这个函数的输出结果输入到softmax函数中进行运算。通过上述步骤，将得到与键对应的值的概率分布（注意力权重）。最后注意力汇聚的输出就是基于这些注意力权重的值的加权和。

![image-20250717145511139](https://khalillu.oss-cn-guangzhou.aliyuncs.com/khalillu/20250717145511201.png)

#### 掩蔽softmax操作：

```python
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
```



#### 加性注意力：

![image-20250717151657055](https://khalillu.oss-cn-guangzhou.aliyuncs.com/khalillu/20250717151657091.png)

一般来说，当查询和键是不同长度的矢量时，可以使用加性注意力作为评分函数。给定查询$q \in R^q$和键$k \in R^k$，加性注意力（additive attention）的评分函数为：

$\alpha(q,k) = w_v^Ttanh(W_qq+W_kk) \in R$

其中$W_q \in R^{h×q}, W_k \in R^{h×k}, W_v \in R^h $

将查询和键连结起来后输入到一层多层感知机（MLP）中，感知机包含一个隐藏层，其隐藏单元数是一个超参数h。通过使用tanh作为激活函数，并且禁用偏执项

#### 缩放点积注意力（Scaled dot-product attention）

使用点积可以得到计算效率更高的评分函数，但是点积操作要求查询和键具有相同的长度d。假设查询和键的所有元素都是独立的随机变量，并且都满足零均值和单位方差，那么两个向量的点积的均值为0，方差为d。

为确保无论向量长度如何，点积的方差在不考虑向量长度的情况下仍然是1，我们再将点积除以$\sqrt{d}$，则缩放点积注意力评分函数为

​							$\alpha(q,k)=q^Tk/\sqrt{d}$

从实践中，从小批量的角度来考虑提高效率，例如基于n个查询，m个键-值对计算注意力，其中查询和键的长度为d，值的长度为v。查询$Q \in R^{n×d}$和值$V \in R^{m×v}$的缩放点积注意力是：

​						$softmax(QK^T/\sqrt{d})V \in R^{n×v}$

