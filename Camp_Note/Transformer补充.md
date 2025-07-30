## Transformer补充

### 输入与输出：

#### 输入：

$X$ 和 $X_{lens}$。其中 $X$ 为输入特征，尺寸为 $(B, T, d)$，$B$ 为batch-size（一个训练轮次中模型接受的样本数量），$T$ 为当前batch中各音频序列的最大长度，$d$ 为特征维度。$X_{lens}$ 为 $B$ 个样本各自的序列长度，尺寸为 $(B,)$

#### 输出：

$Y$ 和 $Y_{lens}$。其中$Y$为输出的文本后验概率分布，尺寸为 $(B, T', V)$，$B$ 为batch-size（一个训练轮次中模型接受的样本数量），$T'$ 为当前batch中各文本序列的最大长度，$V$为可能的候选字个数，即词表大小；$Y_{lens}$ 为各个文本序列的长度长度，尺寸为 $(B,)$​

“后验概率分布”通常指**给定已生成 token（或已观测 patch、帧）后，下一个 token（或 patch、帧）的条件概率分布**——也就是 **$p(yₜ |y_{<t} x)$**，其中

- x：源序列（encoder 可见的全部输入）；
- $y_{<t}$：截止到 t-1 时刻已生成的目标序列；
- yₜ ：t 时刻要预测的 token（或 patch/帧）

#### 损失函数计算中：

由于不同样本的序列长度不同，比如“早上好！”的音频长度只有2秒，对应大概 $T = 200$（100Hz），而"你今天真好看"的音频有4秒，对应大概  $T = 400$。

为了让模型批量处理不同长度的音频，我们将同一个批次中的输入填充 (padding) 到同样长度。比如 $T = 200 < 400$ ，那么 $[0, 200]$ 内就是“早上好！”对应的样本信息， $[200,400)$​​就是填充的无用信息。这些填充的部分后续计算损失函数时会被丢掉，因此具体填充为何值对运算结果无影响，通常填零，不过也可以填其他值。

### 注意力机制：

对于输出的某一部分，输入不同部分的重要形式不一样的。

其中线条的颜色的颜色深浅表示相连的输入输出字符之间的关联性，颜色越深表示关联性越大。这一“关联性”，其实就是注意力的体现。

![img](https://picx.zhimg.com/v2-7df72f94a32d67171412cba2f4da3ca3_r.jpg)



#### 实现方式：

Transformer中用到的注意力机制包括Query(Q), Key(K)和Value(V)三个组成部分。可以理解为，V是我们手头已经有的所有资料，即作为一个知识库；Q是我们待查询的东西，我们希望把V和Q有关的信息都找出来；而K是V这个知识库的钥匙，V中每个位置的信息对应于一个K。对于V中每个位置的信息而言（即K），若Q与该钥匙的匹配程度越高，就可以找出和Q相关更多的内容。

##### 举例：

四不像作为Q，以[鹿, 牛, 羊, 驴, 青蛙]同时作为V和K，然后发现四不像和鹿的相似度为1/3、和牛的相似度为1/6，和羊、驴的相似度均为1/4，和青蛙的相似度为0，那么最终的查询结果就是1/3鹿+1/6牛+1/4羊+1/4驴+0青蛙。

因此计算注意力的流畅可以分解为：

1. 计算Q和K的相似度
2. 根据计算得到的相似度，取出$V$每条信息中和$Q$有关的内容

##### 计算方法：

![img](https://pic3.zhimg.com/v2-c15933ad0345513dee7a459eed136a22_r.jpg)

**计算Q和K的相似度**：既然要计算相似度，我们应当首先确定两个向量之间的相似度的度量标准。简单起见，我们直接以内积作为衡量两个向量之间相似度的方式，比如$Q_1 = [0.3, 0.3], K_3 = [0.9, 0.9]$ （$Q_1$表示$Q$的第一个位置的元素，$K,V $同），那么$Q_1$和$K_3$ 之间的相似度为 $Q_1K_3^T = 0.3*0.9+0.3*0.9 = 0.54$​

以此类推可以计算出score矩阵

![image-20250728161828828](https://khalillu.oss-cn-guangzhou.aliyuncs.com/khalillu/20250728161835893.png)

之后用softmax函数逐行对score进行归一化。归一化之后，$\forall i\in{1,2,3},[Q_i^TK_1,Q_i^TK_2,Q_i^TK_3,Q_i^TK_4]$​都是一个离散的概率分布。将softmax函数逐行归一化之后结果记为attention

![image-20250728162156261](https://khalillu.oss-cn-guangzhou.aliyuncs.com/khalillu/20250728162156299.png)

取出V中每条信息中和Q有关的内容：

得到Q和K之间的相似度attention之后，我们就可以计算Q中每一个位置的元素和V中所有位置元素的注意力运算结果，记该运算结果为$O = [O_1;O_2;O_3]$，其中$O_i$为一个$1\times2$的行向量。以$Q_1$为例，$Q_1$和$[K_1;K_2;K_3;K_4]$的相似度的概率分布为[0.2199, 0.2633, 0.2969, 0.2199]，我们将该概率分布和V进行逐元素相乘，可以得到：

![image-20250728162820195](https://khalillu.oss-cn-guangzhou.aliyuncs.com/khalillu/20250728162820234.png)

将上述整个运算过程写成矩阵相乘的形式：

![image-20250728162845287](https://khalillu.oss-cn-guangzhou.aliyuncs.com/khalillu/20250728162845326.png)

注意力机制本质上可以被认为是一个求离散概率分布的数学期望。定义一个离散的随机变量$X$，随机变量$X$的所有可能取值为$[X_1,X_2,X_3,X_4], X\sim P$，离散分布$P = [p_1,p_2,p_3,p_4]$，于是注意力的计算结果就是随机变量$X$的数学期望$E(x) = \sum_{i = i}^{4}X_ip_i$​，而离散分布P就是上文通过Q和K计算得到的softmax归一化之后的attention。

交叉注意力机制和自注意力机制计算上没有区别，只是在Q、K和V的选取稍有不同。

Transformer对上述注意力机制进行了改进，使用了“多头注意力机制”(Multi-Head Attention)。多头注意力机制假设输入的特征空间可以分为互不相交的几个子空间，然后我们只需要在几个子空间内单独计算注意力，最后将多个子空间的计算结果拼接即可。

举例：

举个例子，假设$Q,K,V$的维度都是512，长度都是$L$ ，现将维度512的特征空间分成8个64维的子空间，每个子空间内单独计算注意力，于是每个子维度的计算结果的尺寸为$(L, 64)$ ，然后再将8个子维度的计算结果沿着特征维度拼起来，得到最终的计算结果，尺寸为$(L, 512)$ ，和输入的$Q$的尺寸保持一致。

多头注意力机制的伪代码如下：

```python
q_len, k_len, v_len = ...
batch_size, hdim = ...
head_dim = ...
assert hdim % head_dim = 0
assert k_len == v_len
num_head = hdim // head_dim
q = tensor(batch_size, q_len, hdim)
k = tensor(batch_size, k_len, hdim)
v = tensor(batch_size, v_len, hdim)


def multi_head_split(x):
  # x: (batch_size, len, hdim)
  b, l, hdim = x.size()
    # transpose主要是为了把"头"这个维度放到合适的位置，使得后续的矩阵乘法（Attention计算）能够按头并行计算。
  x = x.reshape(b, l, num_head, head_dim).transpose(1, 2)    # (b, num_head, l, dim)
  return x


def multi_head_merge(x, b):
  # x: (batch_ize, num_head, len, head_dim)
  b, num_head, l, head_dim = x.size()
  x = x.transpose(1, 2).reshape(b, l, num_head * head_dim)    #(batch_size, l, hdim)
  return x


q, k, v = map(multi_head_split, [q, k, v])
output = MultiHeadAttention(q, k, v)      # 该函数的具体实现后文给出
output = multi_head_merge(output, batch_size)
```

##### **multi_head_split 中的 reshape 和 transpose**：

- **为什么要 transpose(1, 2)？**
  - 为了把 `num_head` 放到维度 1，**seq_len** 放到维度 2。
  - 这样后面做 Attention 的时候，每个头都有自己的 `q, k, v`，可以 **并行计算**。
  - 最终 `q, k, v` 的形状变为：`(batch_size, num_head, seq_len, head_dim)`

##### **MultiHeadAttention 计算**：

- Attention 的公式是：

  Attention(*Q*,*K*,*V*)=softmax($\frac{QK^T}{\sqrt{d_{model}}}V$)

- 这里的 `Q, K, V` 已经是 `(batch_size, num_head, seq_len, head_dim)`。

- 计算 `QK^T` 时，最后两个维度是 `(seq_len, head_dim)` 和 `(head_dim, seq_len)`，正好可以矩阵乘法。

- **如果没有 transpose**，`QK^T` 的维度会是 `(l, num_head)` 和 `(num_head, l)`，**这不是我们想要的 Attention 计算方式**。

#### **multi_head_merge 中的 transpose**：

```python
x = x.transpose(1, 2)  # (b, l, num_head, head_dim)
x = x.reshape(b, l, num_head * head_dim)  # (b, l, hdim)
```

- 这里 `transpose(1, 2)` 是为了把 `num_head` 和 `seq_len` 换回来，使得 `reshape` 能把所有头合并回 `hdim`。

| 目的                       | 解释                                                         |
| -------------------------- | ------------------------------------------------------------ |
| **并行计算多头 Attention** | 把 `num_head` 放到维度 1，使得每个头可以独立计算 Attention   |
| **正确的矩阵乘法维度**     | 确保 `QK^T` 的维度是 `(seq_len, head_dim) @ (head_dim, seq_len)` |
| **最终合并回原始维度**     | 计算完后再把 `num_head` 合并回 `hdim`                        |

### 掩码机制

#### Encoder的self attention的长度mask：

###### 背景：

在上文我们提到，序列输入的长度可能是不一样的，我们当时的处理是将同一个batch中的所有样本都padding到最长长度，这样可以解决输入变长的问题，但是这样做的话，attention的计算会不会出问题？举个例子，当前batch的最大长度是10，而当前样本的长度为4，也就是说序列最后6个位置的数据都是padding填充的，应当舍弃，然而在计算自注意力的过程中，由于$Q、K$ 的长度都为10，所有最终计算出的attention长度也是10，其中包含了$Q$和$K$中padding部分的6个位置的attention计算结果。

因此，需要Masked掩码机制：

为了让attention的计算结果不受padding的影响，直接将padding对应位置的attention权重置为0即可。

我们仍然以"早上好！"为例，假设当前batch中最长的句子的长度为8，而 "早上好！" 的长度为4，因此我们需要padding 4个位置。那么在计算注意力时，记第$i$个字符和第$j$个字符之间的注意力权重为$p_{ij}$，那么 $p_{ij}$​应当满足：

$0 \leq p_{ij} \leq 1$	$1 \leq i$	$j \leq 4$	$p_{ij} = 0$

也就是说，除了当$Q$和$K$都为前4个token以外，其余情形的attention权重均为0，因为该情形下$Q$和$K$​总有一个为padding的部分。如下图所示，其中白色代表attention权重不为0，灰色代表attention权重为0，即被mask掉的部分。

![img](https://pic3.zhimg.com/v2-a0f302d4ac364f54c6114b19d5fc96aa_r.jpg)

子图(a)表示只要Q和K其一为padding，那么我们就将其attention权置为0；而子图(b)表示当K为padding时将对应的attention权重为0。实际模型训练过程中使用(b)而不是用(a)，使用(b)不会出错的是因为Q为padding的部分最终计算loss会被过滤掉，所以Q是否mask无影响。而使用(a)时，由于有些行的所有位置都被mask掉了，这部分计算attention时容易出现NaN。举个例子，我们可以将"早上好！"后面的4个位置的文本字符都用一个特殊的token "<ignore>"来填充，然后再计算交叉熵损失的过程中利用`torch.nn.functional.cross_entropy`的`ignore_idx`参数设成 "<ignore>" 将这部分的loss去掉，不纳入计算。

为了得到子图 (b) 所示的mask，我们只需要保留每个输入样本对应的样本长度即可，代码如下：

```python
def get_len_mask(b: int, max_len: int, feat_lens: torch.Tensor, device: torch.device) -> torch.Tensor:
    attn_mask = torch.ones((b, max_len, max_len), device=device)
    for i in range(b):
        attn_mask[i, :, :feat_lens[i]] = 0
    return attn_mask.to(torch.bool)


m = get_len_mask(2, 4, torch.tensor([2, 4]), "cpu")

# 为了打印方便，转为int
m = m.int()

# 输出
tensor([[[0, 0, 1, 1],
         [0, 0, 1, 1],
         [0, 0, 1, 1],
         [0, 0, 1, 1]],

        [[0, 0, 0, 0],
         [0, 0, 0, 0],
         [0, 0, 0, 0],
         [0, 0, 0, 0]]], dtype=torch.int32)
```

在上面的例子中，当前batch中有两个样本，长度分别为2和4，可以看到，长度为2的样本的后两个位置都被mask掉了。

“长度为 2 的样本”指的是 **当前批次（batch）里第一个序列样本的有效长度是 2**。

- 这个样本本来有 4 个时间步（`max_len=4`），但真正有用的信息只有 **前 2 个时间步**（`feat_lens[0]=2`）。
- 后 2 个时间步是 **padding（填充）**，为了防止模型去“注意”这些无意义的位置，所以用 mask 把它们 **屏蔽掉**。



```python
tensor([[[0, 0, 1, 1],
         [0, 0, 1, 1],
         [0, 0, 1, 1],
         [0, 0, 1, 1]], ...])
```

- 第 0 个样本（batch 中第一个样本）的 mask 是：
  - 第 0、1 列（前两个时间步）是 **0（不屏蔽）**；
  - 第 2、3 列（后两个时间步）是 **1（屏蔽）**。
- 这意味着模型在做 Attention 时，**不会让这个样本的第 2、3 个位置参与计算**。

得到mask之后，我们应该怎么用才能将padding部分的attention权重置为0呢？做法是直接将$Q$和$K$的内积计算结果中被mask掉的部分的值置为-inf，这样的话，过一层softmax之后，padding部分的attention权重就是0了，这一点可以使用PyTorch的masked_fill函数实现。

```python
scores = torch.matmul(Q, K.transpose(-1, -2)) / torch.sqrt(d_k)
if attn_mask is not None:
  scores.masked_fill_(attn_mask, -float("inf"))
```

#### Decoder的self attention的causal mask

Transformer中除了上述由于长度padding带来的mask以外，Transformer中还有一类常用的mask，即causal mask，中文直译为“因果掩码”。这一mask通常用在文本预测/生成相关任务中。

语言模型建模的基本形式为

$argmax\sum_{i = 1}^n[p(x_i|x_1,...,x_{i-1},\theta)]$，其中$\theta$为待优化的模型参数，$x$表示一个长度为n的输入样本，$p(x_i|x_1,...,x_{i-1})$表示条件概率。直观上看，语言模型的目标是尽可能最大化输入样本x的对数似然，根据链式法则，$p(x) = p(x_1)p(x_1|x_2)p(x_3|x_1,x_2)...p(x_n|x_1,x_2,...,x_{n-1})$。在用Transformer对上述概率进行建模时，我们可以让第i个位置接受到第1至第i-1位置上的信息输入，然后输出$p(x_i|x_1,...,x_{n-1})$的预测值。

在这一过程中，我们给模型的完整输入为 $(x_1,x_2,...x_n)$，但是预测 $p(x_i|x_1,...,x_{i-1})$时只用到了前$i-1$个输入的信息，也就是说后面的$n - (i - 1)$个位置的信息我们应当mask掉，只能根据历史信息来预测当前位置的输出。这很好理解，假如模型的输入为“早上好”，那么应该通过“早”来预测“上”，通过“早上”来预测“好”，而不是通过“早上好”来预测“上”或者“好”，因为这相当于直接把ground truth告诉模型了。

![img](https://pic3.zhimg.com/v2-cabd330eddb252ee017536b4cd4c20f0_r.jpg)

```python
def get_subsequent_mask(b: int, max_len: int, device: torch.device) -> torch.Tensor:
    """
    Args:
        b: batch-size.
        max_len: the length of the whole sequence.
        device: cuda or cpu.
    """
    return torch.triu(torch.ones((b, max_len, max_len), device=device), diagonal=1).to(torch.bool)     # or .to(torch.uint8)
```

> again，我们不用关心padding部分，因为padding部分最终计算loss时会被舍弃。

这是因为encoder中只是起到提取特征的作用，不需要像decoder那样计算自回归的交叉熵损失函数，所以不需要额外的causal约束。理论上来说，给encoder加上causal mask也可以，但是其意义和非causal存在一定差异。对于encoder中的第$i$个位置而言，不加入causal mask时，该位置可看到整个序列的所有信息；加入causal mask时，该位置只能看到第$1\sim(i - 1)$个位置的信息，这一约束没有必要。

下面的encoder和decoder之间的mask也是类似，在计算encoder和decoder的cross attention的mask的时候，由于decoder可以获取到encoder的所有信息，因此我们不需要针对encoder做额外的causal mask。

#### Encoder的decoder的cross-attention的mask

比如"早上好！"对应的音频特征长度为600，但是encoder当前batch中音频特征最长为800，因此在做cross-attention时，encoder特征后面的200个位置的特征应当被mask掉，如下图所示：

![img](https://pic3.zhimg.com/v2-15b9d97638e6e61279cf0e3afed71b46_r.jpg)

> again，不用管Q中后面4个padding的位置，计算loss时会筛掉。

得到上述mask的代码如下所示：

```python
def get_enc_dec_mask(
    b: int, max_feat_len: int, feat_lens: torch.Tensor, max_label_len: int, device: torch.device
    # b：表示批量大小（batch size），即一次处理的样本数量。
	# max_feat_len：特征序列的最大长度。例如，在语音识别任务中，这可能是经过特征提取后的音频特征序列的最大长度。
	# feat_lens：一个形状为 (b,) 的张量，表示每个样本的实际特征序列长度。由于每个样本的特征序列长度可能不同，feat_lens 用于记录每个样本的实际长度。
	# max_label_len：标签序列的最大长度。例如，在语音识别任务中，这可能是文本标签的最大长度。
	# device：指定张量的设备（例如 CPU 或 GPU），用于确定计算和存储的位置。
) -> torch.Tensor:
    attn_mask = torch.zeros((b, max_label_len, max_feat_len), device=device)       # (b, seq_q, seq_k)
    """ 
    对于每个样本，从特征序列的实际长度 feat_lens[i] 开始，将对应位置的掩码值设置为 1。
这里的 attn_mask[i, :, feat_lens[i]:] 表示第 i 个样本的标签序列的所有位置（:）与特征序列中从 feat_lens[i] 到末尾的部分之间不能进行注意力计算。
例如，如果某个样本的特征序列长度是 10，而 max_feat_len 是 20，那么从第 11 到第 20 的位置会被标记为 1，表示这些位置是无效的。
	"""
    for i in range(b):
        attn_mask[i, :, feat_lens[i]:] = 1
    return attn_mask.to(torch.bool)
	# 将 attn_mask 转换为布尔类型（torch.bool），因为注意力掩码通常只需要表示位置是否有效（True 或 False）。返回最终的注意力掩码。
```

计算注意力机制的完整代码：



```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_k, d_v, d_model, num_heads, p=0.):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.num_heads = num_heads
        self.dropout = nn.Dropout(p)
        
        # linear projections
        self.W_Q = nn.Linear(d_model, d_k * num_heads)
        self.W_K = nn.Linear(d_model, d_k * num_heads)
        self.W_V = nn.Linear(d_model, d_v * num_heads)
        self.W_out = nn.Linear(d_v * num_heads, d_model)

        # Normalization
        # References: <<Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification>>
        # 假设输入和输出维度分别为 d_in 和 d_out 。目标是让每一层的输出方差 ≈ 1，梯度方差也 ≈ 1，这样深层网络更容易训练。
        nn.init.normal_(self.W_Q.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.W_K.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.W_V.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))
        nn.init.normal_(self.W_out.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

    def forward(self, Q, K, V, attn_mask, **kwargs):
        # Q: (N, q_len, d_model)
	    # N = batch size（一次送进模型的样本个数）
	    # q_len = 查询序列的长度
	    # d_model = 模型的隐藏维度
        N = Q.size(0)
        q_len, k_len = Q.size(1), K.size(1)
        d_k, d_v = self.d_k, self.d_v
        num_heads = self.num_heads

        # multi_head split
		# self.W_Q(Q)    # 输出形状：(N, seq_len, d_k * num_heads)
	    # 线性层把每个 token 的 d_model 维向量映射成 d_k * num_heads 维向量。对 K、V 同理，只是 V 映射到 d_v * num_heads 
        # -1 自动推断为 seq_len。结果形状：(N, seq_len, num_heads, d_k) 含义：把每个 token 的 d_k * num_heads 维拆成 num_heads 个 d_k 维向量。
        # 把第 1、2 维交换，形状变为：(N, num_heads, seq_len, d_k) 含义：把“头”提到最前，方便后续所有头并行计算注意力。
        # Q	(N, num_heads, seq_len, d_k)	每个头的查询向量，准备做点积
        # K	(N, num_heads, seq_len, d_k)	每个头的键向量
        # V	(N, num_heads, seq_len, d_v)	每个头的值向量，准备加权求和
        Q = self.W_Q(Q).view(N, -1, num_heads, d_k).transpose(1, 2)
        K = self.W_K(K).view(N, -1, num_heads, d_k).transpose(1, 2)
        V = self.W_V(V).view(N, -1, num_heads, d_v).transpose(1, 2)
        # 先线性映射、再拆多头、最后转置维度，把 (batch, seq, d) 变成 (batch, heads, seq, d_k/d_v)，实现多头并行注意力
        
        # pre-process mask 
        if attn_mask is not None:
            # 掩码形状检查是否为(batch, q_len, k_len)
            assert attn_mask.size() == (N, q_len, k_len)
            # unsqueeze(1) 在第 1 维（heads 维）插入一个大小为 1 的维度。
            # repeat 把该维复制 num_heads 次，使得 每个头都拿到同一份掩码。
            attn_mask = attn_mask.unsqueeze(1).repeat(1, num_heads, 1, 1)    # broadcast	
            # 转成bool
            attn_mask = attn_mask.bool()

        # calculate attention weight
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)
        if attn_mask is not None:
            scores.masked_fill_(attn_mask, -1e4)
        attns = torch.softmax(scores, dim=-1)        # attention weights
        attns = self.dropout(attns)

        # calculate output
        output = torch.matmul(attns, V)

        # multi_head merge
        output = output.transpose(1, 2).contiguous().reshape(N, -1, d_v * num_heads)
        output = self.W_out(output)

        return output
```

### 位置编码

从attention的形式看，attention并不具备上述对位置进行编码的能力，这是因为attention对所有输入都一视同仁，即 $P(好|早上) = P(好|上早)$

为了保留原始输入的顺序关系，我们需要在Transformer的输入中加入表征顺序的特征，Transformer原文中位置编码的计算方式如下：

$PE(pos,2i) = sin(pos/10000^{2i/d_{model}})$ 

$PE(pos,2i+1) = cos(pos/10000^{2i/d_{model}})$

其中，pos为位置序号，$d_{model}$为特征的维度，i表示特征的第i维。

位置编码的代码如下：

```python
def pos_sinusoid_embedding(seq_len, d_model):
    embeddings = torch.zeros((seq_len, d_model))
    for i in range(d_model):
        f = torch.sin if i % 2 == 0 else torch.cos
        embeddings[:, i] = f(torch.arange(0, seq_len) / np.power(1e4, 2 * (i // 2) / d_model))
    return embeddings.float()
```

### 逐位置前馈网络

本质是一个两层的MLP

代码：

```python
class PoswiseFFN(nn.Module):
    def __init__(self, d_model, d_ff, p=0.):
        super(PoswiseFFN, self).__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.conv1 = nn.Conv1d(d_model, d_ff, 1, 1, 0)
        self.conv2 = nn.Conv1d(d_ff, d_model, 1, 1, 0)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=p)

    def forward(self, X):
        out = self.conv1(X.transpose(1, 2))     # (N, d_model, seq_len) -> (N, d_ff, seq_len)
        out = self.relu(out)
        out = self.conv2(out).transpose(1, 2)   # (N, d_ff, seq_len) -> (N, d_model, seq_len)
        out = self.dropout(out)
        return out
```

> MLP的实现方式，以上用的是一维卷积，也可以直接通过nn.Linear实现

### 编码器

编码器的主要用途是进行信息抽取，比如语音识别中，"Inputs"为原始音频提取的Fbank或者MFCC特征，尺寸为(batch_size, seq_len, hdim)，我们现在希望从这些人工特征中得到具有高层语义的特征，于是就将其输入Transformer的encoder，进行特征编码。

![img](https://pic2.zhimg.com/v2-96f8217758d3c102e81a67d00240745f_r.jpg)

编码器的主要结构包括三部分：特征编码、位置编码、N个encoder layer。其中，特征编码的用途是对原始的输入信息进行特征提取，得到**连续**向量，以作为后续输入；位置编码的用途是给特征编码加上位置信息；encoder layer的作用是实现高层语义特征的提取。每个encoder layer的模型结构完全相同，为一个多头注意力和一个MLP，再加上一些归一化和残差连接。

EncoderLayer层：

```python
class EncoderLayer(nn.Module):
    def __init__(self, dim, n, dff, dropout_posffn, dropout_attn):
        """
        Args:
            dim: input dimension
            n: number of attention heads
            dff: dimention of PosFFN (Positional FeedForward)
            dropout_posffn: dropout ratio of PosFFN
            dropout_attn: dropout ratio of attention module
        """
        assert dim % n == 0
        hdim = dim // n     # dimension of each attention head
        super(EncoderLayer, self).__init__()
        # LayerNorm
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        # MultiHeadAttention
        self.multi_head_attn = MultiHeadAttention(hdim, hdim, dim, n, dropout_attn)
        # Position-wise Feedforward Neural Network
        self.poswise_ffn = PoswiseFFN(dim, dff, p=dropout_posffn)

    def forward(self, enc_in, attn_mask):
        # reserve original input for later residual connections
        residual = enc_in
        # MultiHeadAttention forward
        context = self.multi_head_attn(enc_in, enc_in, enc_in, attn_mask)
        # residual connection and norm
        out = self.norm1(residual + context)
        residual = out
        # position-wise feedforward
        out = self.poswise_ffn(out)
        # residual connection and norm
        out = self.norm2(residual + out)

        return out
```

完整的Encoder：

```python
class Encoder(nn.Module):
    def __init__(
            self, dropout_emb, dropout_posffn, dropout_attn,
            num_layers, enc_dim, num_heads, dff, tgt_len,
    ):
        """
        Args:
            dropout_emb: dropout ratio of Position Embeddings.
            dropout_posffn: dropout ratio of PosFFN.
            dropout_attn: dropout ratio of attention module.
            num_layers: number of encoder layers
            enc_dim: input dimension of encoder
            num_heads: number of attention heads
            dff: dimensionf of PosFFN
            tgt_len: the maximum length of sequences
        """
        super(Encoder, self).__init__()
        # The maximum length of input sequence
        self.tgt_len = tgt_len
        self.pos_emb = nn.Embedding.from_pretrained(pos_sinusoid_embedding(tgt_len, enc_dim), freeze=True)
        self.emb_dropout = nn.Dropout(dropout_emb)
        self.layers = nn.ModuleList(
            [EncoderLayer(enc_dim, num_heads, dff, dropout_posffn, dropout_attn) for _ in range(num_layers)]
        )
    
    def forward(self, X, X_lens, mask=None):
        # add position embedding
        batch_size, seq_len, d_model = X.shape
        # torch.arange(seq_len)：生成 [0, 1, …, seq_len-1] 的位置索引。self.pos_emb：一个 可学习的或固定的 (seq_len, d_model) 位置编码表。逐元素相加，把位置信息注入到每个 token 的表示里。
        out = X + self.pos_emb(torch.arange(seq_len, device=X.device))  # (batch_size, seq_len, d_model)
        out = self.emb_dropout(out)
        # encoder layers
        # 把输入 x 依次流过所有层，逐步提取更深层特征。
        for layer in self.layers:
            out = layer(out, mask)
        return out
    	# 每层都会做：
        # 多头自注意力 → Add&Norm → 前馈网络 → Add&Norm
        # 如果提供了 mask（比如 Padding 掩码），会在注意力里使用，屏蔽无效位置。
        # 最终 out 的形状仍是 (batch_size, seq_len, d_model)，但已经融入了深层上下文信息，可直接接下游任务（CTC、Attention Decoder 等）。

```

| 部分                         | 含义                                                         |
| ---------------------------- | ------------------------------------------------------------ |
| `nn.ModuleList`              | PyTorch 提供的**容器**，会把所有子模块注册到网络中，让优化器、`.to(device)`、`state_dict()` 等操作都能递归处理它们。 |
| `EncoderLayer(...)`          | 每一个编码器子层，内部通常包含 **多头自注意力 + 前馈网络（PoswiseFFN）+ 残差 & LayerNorm**。 |
| `for _ in range(num_layers)` | 连续创建 `num_layers` 个这样的子层，实现“深度”编码器。       |

### 解码器

解码器的主要用途是根据编码器的信息推断出对应的文本是什么，比如我们现在通过encoder拿到了”早上好！“这句话对应的音频特征，我们希望让decoder具备推理能力，可以在接受这些音频特征作为输入的前提下去判断对应的文本内容是什么。

解码器的主要结构和编码器十分类似，也包括特征编码、位置编码和decoder layer三部分，只是decoder layer相比encoder layer而言多了一个和encoder输出之间的**交叉注意力**。

```python
class Decoder(nn.Module):
    def __init__(
            self, dropout_emb, dropout_posffn, dropout_attn,
            num_layers, dec_dim, num_heads, dff, tgt_len, tgt_vocab_size,
    ):
        """
        Args:
            dropout_emb: dropout ratio of Position Embeddings.
            dropout_posffn: dropout ratio of PosFFN.
            dropout_attn: dropout ratio of attention module.
            num_layers: number of encoder layers
            dec_dim: input dimension of decoder
            num_heads: number of attention heads
            dff: dimensionf of PosFFN
            tgt_len: the target length to be embedded.
            tgt_vocab_size: the target vocabulary size.
        """
        super(Decoder, self).__init__()

        # output embedding
        self.tgt_emb = nn.Embedding(tgt_vocab_size, dec_dim)
        self.dropout_emb = nn.Dropout(p=dropout_emb)                            # embedding dropout
        # position embedding
        self.pos_emb = nn.Embedding.from_pretrained(pos_sinusoid_embedding(tgt_len, dec_dim), freeze=True)
        # decoder layers
        self.layers = nn.ModuleList(
            [
                DecoderLayer(dec_dim, num_heads, dff, dropout_posffn, dropout_attn) for _ in
                range(num_layers)
            ]
        )

    def forward(self, labels, enc_out, dec_mask, dec_enc_mask, cache=None):
        # output embedding and position embedding
        tgt_emb = self.tgt_emb(labels)
        pos_emb = self.pos_emb(torch.arange(labels.size(1), device=labels.device))
        dec_out = self.dropout_emb(tgt_emb + pos_emb)
        # decoder layers
        # 2. 逐层解码
        for layer in self.layers:
                dec_out = layer(dec_out, enc_out, dec_mask, dec_enc_mask)
        return dec_out
```



| 变量名         | 含义                                        |
| -------------- | ------------------------------------------- |
| `labels`       | 解码器输入序列（通常是目标文本 token ID）   |
| `enc_out`      | 编码器输出 `(batch, src_len, d_model)`      |
| `dec_mask`     | 解码器自注意力掩码（防止看到未来 token）    |
| `dec_enc_mask` | 解码器-编码器交叉注意力掩码（屏蔽 padding） |
| `dec_out`      | 解码器最终输出 `(batch, tgt_len, d_model)`  |

### Transformer：

#### 整个模型结构：

```python
class Transformer(nn.Module):
    def __init__(
            self, frontend: nn.Module, encoder: nn.Module, decoder: nn.Module,
            dec_out_dim: int, vocab: int,
    ) -> None:
        super().__init__()
        self.frontend = frontend     # feature extractor 特征提取器
        self.encoder = encoder
        self.decoder = decoder
        self.linear = nn.Linear(dec_out_dim, vocab)

    def forward(self, X: torch.Tensor, X_lens: torch.Tensor, labels: torch.Tensor):
        X_lens, labels = X_lens.long(), labels.long()
        b = X.size(0)
        device = X.device
        # frontend
        out = self.frontend(X)
        max_feat_len = out.size(1)                            # compute after frontend because of optional subsampling
        max_label_len = labels.size(1)
        # encoder
        enc_mask = get_len_mask(b, max_feat_len, X_lens, device)
        enc_out = self.encoder(out, X_lens, enc_mask)
        # decoder
        dec_mask = get_subsequent_mask(b, max_label_len, device)
        dec_enc_mask = get_enc_dec_mask(b, max_feat_len, X_lens, max_label_len, device)
        dec_out = self.decoder(labels, enc_out, dec_mask, dec_enc_mask)
        logits = self.linear(dec_out)

        return logits
```

#### 模型验证：

```python
if __name__ == "__main__":
    # constants
    batch_size = 16                 # batch size
    max_feat_len = 100              # the maximum length of input sequence
    max_lable_len = 50              # the maximum length of output sequence
    fbank_dim = 80                  # the dimension of input feature
    hidden_dim = 512                # the dimension of hidden layer
    vocab_size = 26                 # the size of vocabulary

    # dummy data
    fbank_feature = torch.randn(batch_size, max_feat_len, fbank_dim)        # input sequence
    feat_lens = torch.randint(1, max_feat_len, (batch_size,))               # the length of each input sequence in the batch
    labels = torch.randint(0, vocab_size, (batch_size, max_lable_len))      # output sequence
    label_lens = torch.randint(1, max_label_len, (batch_size,))             # the length of each output sequence in the batch

    # model
    feature_extractor = nn.Linear(fbank_dim, hidden_dim)                    # alinear layer to simulate the audio feature extractor
    encoder = Encoder(
        dropout_emb=0.1, dropout_posffn=0.1, dropout_attn=0.,
        num_layers=6, enc_dim=hidden_dim, num_heads=8, dff=2048, tgt_len=2048
    )
    decoder = Decoder(
        dropout_emb=0.1, dropout_posffn=0.1, dropout_attn=0.,
        num_layers=6, dec_dim=hidden_dim, num_heads=8, dff=2048, tgt_len=2048, tgt_vocab_size=vocab_size
    )
    transformer = Transformer(feature_extractor, encoder, decoder, hidden_dim, vocab_size)

    # forward check
    logits = transformer(fbank_feature, feat_lens, labels)
    print(f"logits: {logits.shape}")     # (batch_size, max_label_len, vocab_size)

    # output msg
    # logits: torch.Size([16, 100, 26])
```