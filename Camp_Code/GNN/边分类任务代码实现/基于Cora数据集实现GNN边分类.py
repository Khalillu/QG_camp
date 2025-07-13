import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv
from torch_geometric.utils import negative_sampling

# 便分类模型
# 负采样（Negative Sampling）
# 方法：随机选择图中不存在的边作为负样本，数量通常与正样本相同。
# 作用：平衡正负样本比例，防止模型偏向多数类。
# 联合训练
# 正样本标签为 1，负样本标签为 0。
# 例如：labels = [1, 1, ..., 0, 0]（前一半是正样本，后一半是负样本）。
# 为什么不直接对存在的边分类？
# (1) 缺乏负样本的监督
# 如果仅用存在的边训练，模型无法学习“边不存在”的特征模式。
#
# 后果：模型可能对所有节点对都预测为“存在”（过拟合正样本）。
#
# (2) 无法处理新节点对
# 直接分类只能预测已知边（训练集见过的边），但实际任务需要预测未知节点对是否存在边
# （如推荐系统中的潜在好友）。
#
# (3) 任务本质是“链接预测”
# 边分类的目标通常是预测未观察到的边（如未来可能形成的引用关系），而非仅分类已知边

# 同样是利用Cora数据集，只是这个时候我们关注的不再是节点特征，而是边特征，
# 因此，在这里我们需要手动创建边标签的正例与负例。这是一个二分类问题
class EdgeClassifier(torch.nn.Module):
    # in_channels：输入节点特征的维度
    # out_channels：GCN层输出的隐藏特征维度
    def __init__(self, in_channels, out_channels):
        super(EdgeClassifier, self).__init__()
        # 将in_channels映射到out_channels
        self.conv = GCNConv(in_channels, out_channels)
        # 一个全连接层，输入维度为2*outchannels（因为要拼接两个节点的特征），
        # 输出维度为2（二分类任务，表示“边存在”或“边不存在”）
        # 2*out_channels的原因是对于边(u,v)，我们需要将两个节点的特征拼接起来
        # 拼接后的特征维度是64+64=128
        # 第0维：表示边不存在的分数；第1维表示边存在的分数。
        # 水平concat，得到两行向量，一列表示边不存在的分数，一列代表边存在的分数
        self.classifier = torch.nn.Linear(2 *(out_channels), 2)

    def forward(self, x, edge_index):
        # x：节点特征矩阵，形状为[num_nodes, in_channels]
        # edge_index：边索引矩阵，形状为[2, num_edges]，
        # edge_index 输入的正样本边索引，与poe_edge_index相同
        # 表示图中所有边的源节点和目标节点
        # 通过GCNConv层聚合邻居信息，输出形状为[num_nodes, out_channels]
        x = F.relu(self.conv(x, edge_index))
        # pos_edge_index：原始边（正样本），形状[2, num_edges]，
        # 图中实际存在的边（正样本），每列是一个边的源节点和目标节点索引。
        pos_edge_index = edge_index
        # negative_sampling生成与正样本数量相同的负样本边（不存在的边）
        # 返回形状为[2, num_edges]的负样本边索引
        # torch.cat：将正样本和负样本边在dim=1（列方向）拼接，生成total_edge_index
        # 形状为[2, 2 * num_edges]
        # negative_sampling() 生成的负样本边（不存在的边），数量与正样本边相等
        # total_edge_index 拼接后的所有边（正样本加负样本）
        # 作用：生成与正样本数量相同的负样本边（不存在的边）。
        #
        # 原理：随机选择节点对，确保这些节点对不在原始边集合中。
        # edge_index：原始边索引，用于避免重复采样。
        # num_neg_samples：负样本数量，此处设为正样本数量（pos_edge_index.size(1)）。
        # 输出：形状为 [2, num_edges] 的负样本边索引。
        total_edge_index = torch.cat([pos_edge_index,
                                      negative_sampling(edge_index,
                                    num_neg_samples=pos_edge_index.size(1))],
                                     dim=1)
        # x 表示节点特征矩阵，经过GCN后隐性表示，
        # edge_features 每条边的特征（源节点和目标节点特征的拼接）
        # total_edge_index[0]：所有边的源节点索引（形状 [2 * num_edges]）。
        # total_edge_index[1]：所有边的目标节点索引（形状 [2 * num_edges]）。
        # x[total_edge_index[0]]：源节点特征（形状 [2 * num_edges, out_channels]）。
        # x[total_edge_index[1]]：目标节点特征（形状 [2 * num_edges, out_channels]）。
        # torch.cat(..., dim=1)：在特征维度拼接，
        # 生成 edge_features（形状 [2 * num_edges, 2 * out_channels]）。
        # dim=1，沿列方向拼接正负样本，即水平方向
        edge_features = torch.cat([x[total_edge_index[0]],
                                   x[total_edge_index[1]]], dim=1)
        return self.classifier(edge_features)
# 加载数据集
dataset = Planetoid(root='./data/Cora/raw', name='Cora')
data = dataset[0]

# 创建train_mask 和test_mask
edges = data.edge_index.t().cpu().numpy()
num_edges = edges.shape[0]
# 创建两个全零布尔张量，分别表示训练集和测试集的掩码
train_mask = torch.zeros(num_edges, dtype=torch.bool)
test_mask = torch.zeros(num_edges, dtype=torch.bool)
# 训练集的边数量
train_size = int(0.8*num_edges)
# 随机选择训练边索引
# torch.randperm(num_edges)：生成[0, num_edges-1]的随机排列（）相当于打乱边的索引。
# [:train_size]取前train_size个索引作为训练集
train_indices = torch.randperm(num_edges)[:train_size]
# 训练集中被选中的训练边索引设为True，其余为False
train_mask[train_indices] = True
# ~train对train_mask取逻辑非（True变False，反之亦然）
test_mask[~train_mask] = True

# 定义模型和优化器/训练/测试

model = EdgeClassifier(dataset.num_features, 64)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

def train():
    model.train()
    optimizer.zero_grad()
    # data.x 节点特征矩阵，形状为[num_nodes, num_features]
    # data.edge_index 边索引矩阵，形状为[2, num_edges]
    logits = model(data.x, data.edge_index)
    pos_edge_index = data.edge_index
    # 将pos_edge_index这一Tensor的值全部设为1，表示边存在 pos_labels形状为[num_edges]
    pos_labels = torch.ones(pos_edge_index.size(1), dtype=torch.long)
    # 将neg_edge_index这一Tensor的值全部设为0，表示边不存在
    neg_labels = torch.zeros(pos_edge_index.size(1), dtype=torch.long)
    labels = (torch.cat([pos_labels, neg_labels], dim=0).
              to(logits.device))
    # train_mask：原始训练掩码，标记哪些正样本边用于训练（形状 [num_edges]）。
    # new_train_mask：将 train_mask 在行方向复制一次，形状变为 [2*num_edges]
    # ，以匹配正负样本拼接后的标签和预测结果。
    # 原因：正样本和负样本数量相同（各 num_edges），因此需要扩展掩码。
    new_train_mask = torch.cat([train_mask, train_mask], dim=0)
    loss = F.cross_entropy(logits[new_train_mask], labels[new_train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

def test():
    model.eval
    with (torch.no_grad()):
        logits = model(data.x, data.edge_index)
        pos_edge_index = data.edge_index
        pos_labels = torch.ones(pos_edge_index.size(1), dtype=torch.long)
        neg_labels = torch.zeros(pos_edge_index.size(1), dtype=torch.long)
        labels = torch.cat([pos_labels, neg_labels],dim=0).to(logits.device)
        new_test_mask = torch.cat([test_mask, test_mask], dim=0)

        predictions = logits[new_test_mask].max(1)[1]
        correct = predictions.eq(labels[new_test_mask]).sum().item()
        return correct / len(predictions)

for epoch in range(1, 1001):
    loss = train()
    acc = test()
    print(f"Epoch: {epoch:03d}, Loss: {loss:.4f}, Acc: {acc:.4fs}")