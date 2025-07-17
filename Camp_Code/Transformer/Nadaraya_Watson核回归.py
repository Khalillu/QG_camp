import torch
from torch import nn
from d2l import torch as d2l
import matplotlib.pyplot as plt

n_train = 50
x_train, _ = torch.sort(torch.rand(n_train)*5)  # 排序后的训练样本

def f(x):
    return 2*torch.sin(x) + x**0.8

y_train = f(x_train) + torch.normal(0.0, 0.5, (n_train,)) # 训练样本的输出
x_test = torch.arange(0, 5, 0.1) # 测试样本
y_truth = f(x_test)
n_test = len(x_test)

def plot_kernel_reg(y_hat):
    d2l.plot(x_test, [y_truth, y_hat], 'x', 'y', legend=['Truth', 'Pred'],
                xlim=[0, 5], ylim=[-1, 5])
    d2l.plt.plot(x_train, y_train, 'o', alpha=0.5);

# 定义模型
class NWKernelRegression(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        # self.w可学习的标量参数，用于控制核函数的带宽
        # 1.通过反向传播自动优化
        # 2.作用类似于高斯核的带宽参数，决定注意力权重的集中程度。
        self.w = nn.Parameter(torch.rand((1,), requires_grad=True))

    def forward(self, queries, keys, values):
        # queries和attention_weights的形状为（查询个数，“键-值”对个数）
        # 每个查询复制keys.shape[1]次（即每个键值对的个数），使其与键的维度对齐，
        # 若queries形状为(n_queries,)，keys形状为(n_queries, n_kv_pairs)，则输出形状为(n_queries, n_kv_pairs)
        queries = queries.repeat_interleave(keys.shape[1]).reshape((-1, keys.shape[1]))
        # 套公式
        self.attention_weights = nn.functional.softmax(
            -((queries - keys)* self.w)**2 / 2, dim=1)
        # value的形状为（查询个数，“键-值”对个数）
        # self.attention_weights.unsqueeze(1)将权重变为形状(n_queries, 1, n_kv_paris)
        # values.unsqueeze(-1)将值变为形状(n_queries, n_kv_pairs, 1)
        # torch.bmm 批量矩阵乘法，对每个查询计算加权和
        # 对值（Values）按注意力权重加权求和，得到最终预测结果。
        return torch.bmm(self.attention_weights.unsqueeze(1),
                         values.unsqueeze(-1)).reshape(-1)

x_tile = x_train.repeat((n_train, 1))
y_tile = y_train.repeat((n_train, 1))

keys = x_tile[(1 - torch.eye(n_train)).type(torch.bool)].reshape((n_train, -1))
values = y_tile[(1 - torch.eye(n_train)).type(torch.bool)].reshape((n_train, -1))

net = NWKernelRegression()
loss = nn.MSELoss(reduction='none')
trainer = torch.optim.SGD(net.parameters(), lr=0.5)
animator = d2l.Animator(xlabel='epoch', ylabel='loss',xlim=[1, 5])

for epoch in range(5):
    trainer.zero_grad()
    l = loss(net(x_train, keys, values), y_train)
    l.sum().backward()
    trainer.step()
    print(f'epoch {epoch + 1}, loss {float(l.sum()):.6f}')
    animator.add(epoch + 1, float(l.sum()))


keys = x_train.repeat((n_test, 1))

values = y_train.repeat((n_test, 1))
y_hat = net(x_test, keys, values).unsqueeze(1).detach()
plot_kernel_reg(y_hat)

d2l.show_heatmaps(net.attention_weights.unsqueeze(0).unsqueeze(0),
                  xlabel = 'Sorted training inputs',
                  ylabel = 'Sorted testing inputs',)
d2l.plt.show()
d2l.plt.pause(2)