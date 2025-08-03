import numpy as np

# config
n_sample = 512
lr       = 1e-3
n_epoch  = 5001
k_d      = 3
hidden   = 10            # 关键：扩到 10 个隐单元

# 网络
# np.clip: 将x值限制在[-250, 250]上
# np.exp: 输入x，返还e^x
sigmoid = lambda x: 1/(1+np.exp(-np.clip(x,-250,250)))

def G(z, W1, b1, W2, b2):
    h = sigmoid(z @ W1 + b1)    # 将一维噪声 -> 十维空间
    return sigmoid(h @ W2 + b2) # 将十维空间 -> 一维伪造样本
# G 的输出直接被当作 “伪造样本的数值” 使用，而不再额外解释成概率；
# 所以你可以把它看成 “一个被剪裁到 (0,1) 的连续伪造样本”。

def D(x, W1, b1, W2, b2):
    h = sigmoid(x @ W1 + b1)    # 引入非线性，使网络能拟合非高斯分布
    return sigmoid(h @ W2 + b2) # 把结果压在(0, 1) -> 天然符合概率定义
# 把结果压到 (0,1) 区间 → 天然符合「概率」含义

# 真实样本与样本
real = lambda n: np.random.normal(4, 1, (n,1))
noise = lambda n: np.random.normal(0, 1, (n,1))

# 权重设置
# W1_G	生成器 输入→隐藏 权重	(1, hidden)	高斯 N(0,0.01²) 小随机数
# b1_G	生成器 隐藏层偏置	(1, hidden)	全 0
# np.random.normal(0, 0.01, (3,2))  # 均 0、方差 0.01²，形状 3×2
# np.random.randn(3,2) * 0.01       # 等价写法 正态分布，默认均值为零，方差为1，
W1_G = 0.01*np.random.randn(1,hidden);   b1_G = np.zeros((1,hidden))
# W2_G	生成器 隐藏→输出 权重	(hidden, 1)	同上
# b2_G	生成器 输出层偏置	(1, 1)	全 0
W2_G = 0.01*np.random.randn(hidden,1);   b2_G = np.zeros((1,1))
# W1_D, b1_D, W2_D, b2_D	判别器对称结构	同上	同上
W1_D = 0.01*np.random.randn(1,hidden);   b1_D = np.zeros((1,hidden))
W2_D = 0.01*np.random.randn(hidden,1);   b2_D = np.zeros((1,1))
# hidden = 10 已在前面设定 → 隐藏层 10 个神经元。
# 小随机数 + 零偏置是深度学习的标准初始化，防止对称性破坏梯度流动。

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