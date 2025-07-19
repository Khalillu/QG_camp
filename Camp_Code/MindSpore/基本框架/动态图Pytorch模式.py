import mindspore as ms
from mindspore import nn, ops

# 定义网络
net = nn.SequentialCell([
    nn.Dense(10, 1)
])

# 创建损失函数和优化器
loss_fn = nn.MSELoss()
optimizer = nn.SGD(net.trainable_params(), learning_rate=0.01)

# 封装训练网络
train_net = nn.TrainOneStepCell(
    nn.WithLossCell(net, loss_fn),
    optimizer
)

# 训练
x = ops.randn((32, 10))
y = ops.randn((32, 1))
loss = train_net(x, y)
print("Loss:", loss)