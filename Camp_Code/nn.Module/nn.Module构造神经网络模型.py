import os
import sys
from torch.utils.data import DataLoader
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from dataloader import iris_dataloader

# 初始化神经网络模型

class NN(nn.Module):
    # 将输入层、隐藏层、输出层的维度输入进神经网络模型中
    def __init__(self,in_dim,hidden_dim1,hidden_dim2,out_dim) -> None:
        super().__init__()
        self.layer1 = nn.Linear(in_dim,hidden_dim1)
        self.layer2 = nn.Linear(hidden_dim1,hidden_dim2)
        self.layer3 = nn.Linear(hidden_dim2,out_dim)

    #定义前向传播
    def forward(self,x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x

    # 定义计算环境
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 训练集，验证集和测试集

custom_dataset = iris_dataloader("D:\\Pycharm\\PyCharm 2024.3.1.1\\Pytorch实战\\nn.Module\\dataloader.py")

    # 划分数据集
train_size = int (len(custom_dataset)*0.7)
val_size = int(len(custom_dataset)*0.2)
    # or test_size = int(len(custom_dataset)) - train_size - val_size
test_size = int(len(custom_dataset)*0.1)

    # random_split 按比例的随机切分，两个参数，一个是需要切分的数据集，一个是划分数据集的比例
train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(custom_dataset, [train_size, val_size, test_size])

    # shuffle=True作用：在batch抽取一定量数据集出来之后，将数据集进行打乱
train_loader = DataLoader(train_dataset,batch_size=16,shuffle=True)

val_loader = DataLoader(val_dataset,batch_size=1,shuffle=False)

test_loader = DataLoader(test_dataset,batch_size=1,shuffle=False)

print("训练集的大小",train_size,"验证集的大小",val_size,"测试集的大小",test_size)

    # 定义一个推理函数，来计算并返回准确率

def infer(model,dataset,device):
    model.eval()
    acc_num = 0
    # with torch.no_grad 仅验证当前模型的性能，并不改变模型的参数
    with torch.no_grad():
        for data in dataset:
            datas,label = data
            # 该模型的返回结果（三种鸢尾花的可能性）
            outputs = model(datas.to(device))
            #此时第零维为batch维度，即训练数据的数量，第一维才是结果维度，即鸢尾花的可能性
            # torch.max 该函数返回一个元组，其中包含两个元素，第一个元素是最大值（每个样本的最大分数），第二个元素是最大值所在的索引（每个样本预测的类别索引），因此[1]代表我们只取索引部分，即预测的类别标签
            predict_y = torch.max(outputs,dim=1)[1]
            # 比较当前模型的预测结果(predicct_y)与真实结果（label.to(device)）
            # 由于每一次都是一批量数据加入该函数中，因此我们需要sum().item()来获取这一批量数据中预测正确的个数，而+=是对每一批量加起来的全部数据的预测个数，而item()则是让我们取到该结果的数值，即数量
            acc_num += torch.eq(predict_y,label.to(device)).sum().item()

    acc = acc_num / len(dataset)
    return acc

def main(lr=0.005,epochs=20):
    model = NN(4, 12, 6, 3).to(device)
    loss_f = nn.CrossEntropyLoss()

    # model.parameters()将会调用模型中的所有参数，if语句判断是否为可迭代的参数 如果p.requires_grad为True，则进入该列表中
    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(pg, lr=lr)

    # 权重文件存储路径，getcwd()将会获取目前文件夹的路径
    save_path = os.path.join(os.getcwd(),"results/weights")
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    # 开始训练
    for epoch in range(epochs):
        model.train()
        acc_num = torch.zeros(1).to(device)
        sample_num = 0

        train_bar = tqdm(train_loader, file=sys.stdout,ncols=100)
        for datas in train_bar:
            data,label = datas
            # 移除标签张量中大小为1的最后一个维度，(-1)表示指定要挤压（移除）的维度位置
            label = label.squeeze(-1)
            sample_num += data.shape[0]

            # 防止以前的梯度，对当前产生一些影响
            optimizer.zero_grad()
            outputs = model(data.to(device))
            pred_class = torch.max(outputs,dim=1)[1] # torch.max 返回值是一个元组，第一个元素是max的值，第二元素是max值的索引
            acc_num = torch.eq(pred_class, label.to(device)).sum()

            loss = loss_f(outputs,label.to(device))
            loss.backward()
            optimizer.step()

            train_acc = acc_num / sample_num
            # .desc打印在进度条上显示
            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1, epochs, loss)

        val_acc = infer(model, val_loader, device)
        print("train epoch[{}/{}] loss:{:.3f} train_acc{:.3f}".format(epoch + 1,epochs,loss,train_acc))
        torch.save(model.state_dict(), os.path.join(save_path, "nn.pth"))

        # 每次数据集迭代之后，要对初始化的指标清零
        train_acc = 0.
        val_acc = 0.
    print("Finished Training!")

    test_acc = (
        infer(model, test_loader, device))
    print("test_acc",test_acc)

if __name__ == "__main__":
    main(lr=0.005,epochs=20)

