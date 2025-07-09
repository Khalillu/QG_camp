import torch.nn as nn
import torch.nn.functional as F


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        # 其中2d表示2 dimension，即二维
        # pytorch中的官方包，Conv2d表示卷积层，第一个in_channels表示输入的特征矩阵的深度，out_channels表示卷积核的个数，kernel_size代表卷积核的大小，stride默认为一，padding默认为零
        self.conv1 = nn.Conv2d(3, 16, 5)
        # MaxPool2d kernel为池化层的大小，stride为步长
        # 池化（即下采样，压缩参数）
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 5)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # relu激活函数，max(0,x)
        x = F.relu(self.conv1(x))    # input(3, 32, 32) output(16, 28, 28)
        x = self.pool1(x)            # output(16, 14, 14)
        x = F.relu(self.conv2(x))    # output(32, 10, 10)
        x = self.pool2(x)            # output(32, 5, 5)
        # 通过view函数将x转化为一维向量，-1代表第一个维度（即batch）
        x = x.view(-1, 32*5*5)       # output(32*5*5)
        x = F.relu(self.fc1(x))      # output(120)
        x = F.relu(self.fc2(x))      # output(84)
        x = self.fc3(x)              # output(10)
        return x


