import torch
import torch.nn.functional as F
# 在Planetoid数据集中，掩码是预定义的。
# Planetoid数据集下载时需要IP America的神秘魔法
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv

# 载入数据
#  加载Cora数据集（一个学术论文引用网络数据集，节点为论文，边为引用关系）
dataset = Planetoid(root="~/tmp/Cora", name='Cora')
# 获取数据集的第一个图
data = dataset[0]

# 掩码的生成方式：
"""
import torch

num_nodes = 1000
train_ratio = 0.8

# 随机生成训练掩码
train_mask = torch.zeros(num_nodes, dtype=torch.bool)
train_indices = torch.randperm(num_nodes)[:int(train_ratio * num_nodes)]
train_mask[train_indices] = True

# 测试掩码是训练掩码的补集
~ 代表补集
test_mask = ~train_mask
"""

# 定义网络架构
# Net继承torch.nn.Module的自定义GNN模型
# conv1第一层GCN，输入维度为节点特征数（dataset.num_features），输出维度为16
# conv2第二层GCN，输入维度为16，输出维度为类别数（dataset.num_classes）
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # GCNConv事PyTorch Geometric实现的图卷积网络层，用于聚合节点及其邻居的特征。
        # conv1：将输入特征维度（dataset.num_features）映射到隐藏层维度（16）
        self.conv1 = GCNConv(dataset.num_features, 16)  # 输入=节点特征维度，16是中间隐藏神经元个数
        # conv2：将隐藏层维度（16）映射到输出类别数（dataset.num_classes）
        self.conv2 = GCNConv(16, dataset.num_classes)
        # 输入：节点特征矩阵x（形状为[num_nodes, num_features]）和边索引edge_index（形状为[2, num_edges]）
        # 输出：经过图卷积后的节点表示（conv1输出形状为[num_nodes, 16]），conv2输出为[num_nodes, num_classes]
    # x：节点特征矩阵
    # edge_index：图的边结构（稀疏邻接矩阵的COO格式）
    # F.relu：ReLU激活函数，引入非线性
    # F.log_softmax对输出取对数softmax，用于多分类任务，配合F.nll_loss使用
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)
# 检查是否有GPU，若有则使用cuda，否则用cpu
# model.to(device)将模型移动到指定设备（GPU/CPU）
# data.to(device) 将数据（节点特征、边、标签等）移动到同一设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net().to(device)
data = data.to(device)
# Adam：自适应优化器
# weight_decay=5e-4：L2正则化系数，防止过拟合
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
# 模型训练
# model.train()将模型设置为训练模式（启用Dropout等训练专用层）
model.train()
for _ in range(200):
    # optimizer.zero_grad()清空梯度缓存
    optimizer.zero_grad()
    # 模型对节点的预测效果（对数概率）
    out = model(data.x, data.edge_index)
    # F.nll_loss 负对数似然损失，计算训练集（data.train_mask）的损失
    # data.test_mask是一个布尔掩码（Boolean Mask），用于标识数据集中哪些节点属于测试集，
    # 它的具体特性如下：类型：torch.Tensor(布尔张量)
    # 形状：[num_nodes]与节点数量相同
    # 取值：True：对应位置的节点属于测试集；False：对应位置的节点不属于测试集
    # Mask掩码，核心作用是划分数据集
    # loss.backward()：反向传播计算梯度
    # 禁用train_mask对应的节点计算损失，忽略其他节点
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    optimizer.step()
# 测试
# model.eval()：将模型设置为评估模式（禁用Dropout等）
model.eval()
# test_predict：模型对测试集节点的预测结果
# test_predict类型为 torch.Tensor（浮点数张量），形状为：[num_test_nodes, num_classes]
# 内容：模型对测试机姐弟啊的呢预测结果，每个结点的输出是经过log_softmax后的对数概率
test_predict = model(data.x, data.edge_index)[data.test_mask]
# torch.argmax：获取概率最大的预测类别
# max_index 和 test_true 的长度相同（均为测试节点数）
# 类型torch.Tensor（整数张量），形状：[num_test_nodes]
# 每个测试节点的预测类别索引（通过argmax从test_predict获得）
# 例如：max_index = [1, 0, 2, ...]表示第一个测试节点预测为类别1，第二个为类别零
max_index = torch.argmax(test_predict, dim=1)
# test_true：测试集的真实标签
# 类型torch.Tensor（整数张量），形状：[num_test_nodes]
# 测试集节点的真实类别标签（来自data.y并通过data.test_mask筛选）
# 例如：test_true = [1, 2, 0, ...]
test_true = data.y[data.test_mask]
# 统计预测正确的样本数
correct = 0
# 遍历所有测试节点，统计预测正确的数量
for i in range(len(max_index)):
    if max_index[i] == test_true[i]:
        correct += 1
print('测试机准确率为:{}%'.format(100 * correct / len(max_index)))