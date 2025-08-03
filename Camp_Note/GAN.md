## GAN

### GAN主要应用目标：

生成式任务（生成、重建、超分辨率、风格迁移、补全、上采样等）

**GAN的核心思想：**

生成器G和判别器D的一代代博弈

**生成器**：

生成网络，通过输入生成图像

**判别器**：

二分类网络，将生成器生成图像作为负样本，真实图像作为正样本

**learn 判别器D：**

给定G，通过G生成图像产生负样本，并结合真实图像作为正样本来训练D

**learn 生成器G：**

给定D，以**使得D对G生成图像的评分尽可能接近正样本**作为目标来训练G



G和D的训练过程交替进行，这个对抗的过程使得**G生成的图像越来越逼真**，D“打假”的能力也越来越强。

### 极大似然估计



> **补充：**
> **分布的表示：$P(x)$**
> 表示该分布中采样到样本x的概率，试想如果我们知道该分布中每个样本的采样概率，那么这个分布也就可以以这种形式表示出来了。
> **确定分布的表示：$P(x;\theta)$** 
> 其中 $\theta$表示该分布的参数，该分布的具体形式确定了（比如 $P(x;\theta)$可以是高斯分布，$\theta$ 就是高斯分布的均值 $\mu$和$\sigma$方差 

#### 要解决的问题：

* 给定一个数据分布$P_{data}(x)$
* 给定一个由参数 $\theta$ 定义的数据分布$P_G(x;\theta)$
* 我们希望求得参数$\theta$使得$P_G(x;\theta)$尽可能接近$P_{data}(x)$

可以理解成：

$P_G(x;\theta)$是某一具体的分布（如简单的高斯分布），而$P_{data}(x)$是未知的，我们希望通过极大似然估计来确定$\theta$，让$P_G(x;\theta)$能够大体表达$P_{data}(x)$。

#### 解决方案：

![image-20250731170846673](https://khalillu.oss-cn-guangzhou.aliyuncs.com/khalillu/20250731170846736.png)

#### 本质：

找到$\theta$使得$P_G(x;\theta)$与目标分布$P_{data}(x)$的KL散度尽可能低，也就是使得这两者的分布尽可能接近，实现用确定的分布$P_G(x;\theta)$极大似然估计$P_{data}(x)$

### 基本思想：

#### 生成器：

GAN的主要应用是集中在生成

本质就是在做一个极大似然估计的事情，希望可以用某一种具体的分布形式$P_G(x;\theta)$尽可能表达分布$P_{data}(x)$，这样我们就相当于得到了$P_{data}(x)$，并根据$P_{data}(x)$的分布$P_G(x;\theta)$采样做生成。

1. 确定具体分布的形式$P_G(x;\theta)$

2. 极大似然估计求得$\theta$。我们认为我们可以使用$P_G(x;\theta)$近似表达$P_{data}(x)$

3. 基于$P_G(x;\theta)$采样做生成

最直接的想法：

$P_G(x;\theta)$直接用高斯分布模型，但是高斯分布中的capacity太弱了，不能很有效地推广至去拟合各种差异很大地目标图像分布。

`capacity定义：模型能表示的分布族大小或最大互信息`

- `一个k维高斯𝒩(μ, Σ) 的自由度是k + k(k+1)/2 （均值 + 协方差独立参数）。`
- `有时把log|Σ|（对数行列式）当作“容量度量”，因为它正比于微分熵：`

`$h =\frac{1}{2}log[(2πe)^k |Σ|]。$`

`即代表该模型能装下多少信息量或多少数据复杂度。`

不妨设计一个神经网络G来得到更general的$P_G(x;\theta)$

![img](https://pic3.zhimg.com/v2-7f9c475f7ed92773caa7ff3a9f50a118_r.jpg)

##### 解释：

整体pipeline：

1. 先选取简单的先验分布$P_{prior}$，并从该先验分布重采样z作为输入，输入到神经网络G，得$G(z)=x$生成图像x。我们通过这种方式构建了生成分布$P_G(x;\theta)$。此时该分布主要有神经网络G决定，参数$\theta$由网络参数定义。我们可以通过输入z来在该分布上采样x。

2. 目标是$P_{data}(x)$，我们希望构建的$P_G(x;\theta)$与它尽可能接近。我们无法获得$P_{data}(x)$的具体表达形式，我们只能获得它的样本。
3. 类似极大似然估计，我们通过比较两个分布样本的差异设计loss来调节优化神经网络G的参数$\theta$，从而实现将分布$P_G$向$P_{data}$拉近，从而达到用$P_G$拟合表达$P_{data}$的效果。

$P_{prior}$表示一个先验分布，我们生成图像x需要输入的code z就是服从这个先验分布的。

这个先验分布可以是：高斯分布

$P_G(x;\theta) = \int_{z}P_{prior}(z)I_{[G(z;\theta)=x]}dz$

指示函数$I_{[G(z;\theta)=x]}$表示当[]内的条件为真时取值为1，为假时取值为0

也就是说分布$P_G$采样x的概率是所有能够是能$G(z;\theta)=x$成立的z出现的概率之和，而z在这里是符合先验分布$P_{prior}(z)$​​的。

##### 总结：

$P_G(a; θ)$ 就是“生成器把噪声 z 变成图片 a 的总概率”。
它等于“高维积分”，理论上需要把所有能生成 a 的 z 的概率加起来；
因为维度太高而算不动，我们才发明各种训练技巧去逼近它。

#### 判别器：

因为$P_G(x;\theta)$计算方式极其复杂，导致使用极大似然估计根本无从下手，因此引入了判别器D

对于生成器G：

1. $G$是一个函数，输入$z\sim P_{prior}$，输出$x\sim P_G$
2. 先验分布$P_{prior}$，$P_{prior}$和$G$共同决定的分布$P_G$

对于判别器D：

1. D是一个函数，输入$x \sim P_G$，输出一个scalar
2. D用于评估$P_{G}(x;\theta)$和$P_{data}(x)$之间的差异

GAN的最终目标：
$G^* = argmin_Gmax_DV(G,D)$

目标是得到使得式子$max_DV(G,D)$最小的生成器$G^*$

关于$V(G,D)$

$V(G,D) = E_{X\sim P_{data}}[logD(x)] + E_{x\sim P_G}[log(1-D(x))]$

> 给定G，$max_DV(G,D)$衡量的就是分布$P_G$和$P_{data}$的差异
>
> 因此，$argmin_Gmax_DV(G,D)$也就是我们需要的使得差异最小的G

1. **警察先练眼力**（固定小偷，先 max D）
   给定一批**真画**和**当前小偷画的假画**，警察拼命学会分辨：
   **max_D V(G, D)** → 把判别器 D 调到最灵敏，让真假差距最大。
2. **小偷再改进造假技术**（再 min G）
   警察已经最厉害了，小偷就研究：
   **min_G [ max_D V(G, D) ]** → 让自己（G）画的假画在这个最厉害的警察面前也分不出来，于是真假差距被压到最小。

- **V(G, D)** = 警察抓小偷的“得分”
- **max_D V(G, D)** = 把警察能力调到最好 → 得分最高
- **min_G [ … ]** = 小偷让警察的最高得分尽可能低 → 差距最小

经过一系列数学推导之后

此时最优的G：

$G^* = argmin_GV(G,D^*)=argmin_GJSD(P_{data}(x)||P_G(x))$

也就是使得$JSD(P_{data}(x)||P_G(x))$最小的G

$0<JSD(P_{data}(x)||P_G(x))<log2$

当$JSD(P_{data}(x)||P_G(x))=0$时，表示这两个分布完全相同

对于

$G^* = argmin_Gmax_DV(G,D)$

令$L(G)=max_DV(G,D)=V(G,D^*)$

优化$G^*$的方法：

$\theta_G \leftarrow \theta_G - \eta \frac{\partial L(G)}{\partial \theta_G}$

#### JS散度解析：

距离：

- 有两位师傅 A 和 B 分别做苹果派。
- 你想知道 **他们的口味到底差多远**。
- **不能直接尝遍所有派**（分布无限大），于是你拿“平均做法”当裁判。

1. 先把 A、B 各做 **一半混合**，得到一个“平均派” M。
2. 分别量：
   - A 的派 离 M 多远（KL₁）
   - B 的派 离 M 多远（KL₂）
3. 把两段距离 **加起来再除以 2** → 这就是 **Jensen–Shannon 散度**。

数学写成：

$JS(P∥Q)=\frac{1}{2}KL(P∥M)+\frac{1}{2}KL(Q∥M),M=\frac{(P+Q)}{2}$

- 哪些来自**真实数据**（概率 P_data）
- 哪些来自**生成器**（概率 P_G）

如果给每个样本 a 打分 D(a)=“这张图是真实的概率”，
那么最自然的**最大似然**目标就是：

$max_DE_a∼P_{data}[logD(a)]+E_a∼P_G[log(1−D(a))]$

- 第一项：真样本 → 希望 D(a) 越大越好 → log D(a) 越大越好
- 第二项：假样本 → 希望 D(a) 越小越好 → 1-D(a) 越大越好 → log(1-D(a)) 越大越好

#### KL散度：

$KL(P真实∥Q模型)=∑_xP(x)log\frac{Q(x)}{P(x)}$​

具体例子：

![image-20250802204127330](https://khalillu.oss-cn-guangzhou.aliyuncs.com/khalillu/20250802204127451.png)

求最强大的“警察”，揪出有什么不同

![image-20250802203618649](https://khalillu.oss-cn-guangzhou.aliyuncs.com/khalillu/20250802203625781.png)

求出最佳的$D^*$值表达式

带入原式

![image-20250802203707029](https://khalillu.oss-cn-guangzhou.aliyuncs.com/khalillu/20250802203707125.png)

因为

![image-20250802204923651](https://khalillu.oss-cn-guangzhou.aliyuncs.com/khalillu/20250802204923760.png)

又因为$\int_x P_{data}(a)da=1$（概率之和为1）

所以$\int_x P_{data}(a)log2da=1*log2=log2$

同理$\int_xP_G(a)log2da = log2$

即-2log2

剩下刚好凑出一个JS散度



GAN实战例子：



```python
import numpy as np

# config
n_sample = 512
lr       = 1e-3
n_epoch  = 5001
k_d      = 1
hidden   = 10            # 关键：扩到 10 个隐单元

# 网络
sigmoid = lambda x: 1/(1+np.exp(-np.clip(x,-250,250)))

def G(z, W1, b1, W2, b2):
    h = sigmoid(z @ W1 + b1)
    return sigmoid(h @ W2 + b2)

def D(x, W1, b1, W2, b2):
    h = sigmoid(x @ W1 + b1)
    return sigmoid(h @ W2 + b2)

# 真实样本与样本
real = lambda n: np.random.normal(4, 1, (n,1))
noise = lambda n: np.random.normal(0, 1, (n,1))

# 权重设置
W1_G = 0.01*np.random.randn(1,hidden);   b1_G = np.zeros((1,hidden))
W2_G = 0.01*np.random.randn(hidden,1);   b2_G = np.zeros((1,1))
W1_D = 0.01*np.random.randn(1,hidden);   b1_D = np.zeros((1,hidden))
W2_D = 0.01*np.random.randn(hidden,1);   b2_D = np.zeros((1,1))

# 训练
for epoch in range(n_epoch):
    z  = noise(n_sample)
    x_real = real(n_sample)
    x_fake = G(z, W1_G, b1_G, W2_G, b2_G)

    # 更新 D
    for _ in range(k_d):
        d_real = D(x_real, W1_D, b1_D, W2_D, b2_D)
        d_fake = D(x_fake, W1_D, b1_D, W2_D, b2_D)

        # 简单 SGD（可换成 Adam）
        err_real = d_real - 1
        err_fake = d_fake
        # 对 W2_D
        dW2_D = (sigmoid(x_real @ W1_D + b1_D).T @ err_real +
                 sigmoid(x_fake @ W1_D + b1_D).T @ err_fake) / n_sample
        db2_D = (err_real + err_fake).mean(axis=0, keepdims=True)
        # 对 W1_D（链式法则略写）
        # 这里用对称更新，略掉显式推导，保持简洁
        W2_D -= lr * dW2_D
        b2_D -= lr * db2_D

    # 更新 G（只用假样本）
    x_fake = G(z, W1_G, b1_G, W2_G, b2_G)
    d_fake = D(x_fake, W1_D, b1_D, W2_D, b2_D)
    # 损失 = log(1-D)，梯度反向即可
    err_G = -d_fake
    # 同样简化反向传播
    W2_G -= lr * (sigmoid(z @ W1_G + b1_G).T @ err_G) / n_sample
    b2_G -= lr * err_G.mean(axis=0, keepdims=True)

    if epoch % 500 == 0:
        gen = G(noise(10000), W1_G, b1_G, W2_G, b2_G)
        print(f'Epoch {epoch:4d} | 生成均值 = {gen.mean():.3f}')

print('训练结束，真实均值 ≈ 4.00，生成均值 ≈', G(noise(10000), W1_G, b1_G, W2_G, b2_G).mean())
```