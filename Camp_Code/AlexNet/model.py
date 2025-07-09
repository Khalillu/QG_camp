import torch.nn as nn
import torch


class AlexNet(nn.Module):
    def __init__(self, num_classes=1000, init_weights=False):
        super(AlexNet, self).__init__()
        # 使用nn.Sequential来精简代码（即封装网络层次比较多的模块）
        self.features = nn.Sequential(
            # 卷积或实例化过程中，若计算结果不为整数，则向下取整
            nn.Conv2d(3, 48, kernel_size=11, stride=4, padding=2),  # input[3, 224, 224]  output[48, 55, 55] 48为channel，即深度，而batch没有写出来
            # inplace = True pytorch通过某种方法，增加计算量但是能够降低内存使用容量的一种方法
            nn.ReLU(inplace=True),
            # 每经过一次卷积层，就需要进行一次ReLU激活函数
            nn.MaxPool2d(kernel_size=3, stride=2),                  # output[48, 27, 27]
            nn.Conv2d(48, 128, kernel_size=5, padding=2),           # output[128, 27, 27]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),                  # output[128, 13, 13]
            nn.Conv2d(128, 192, kernel_size=3, padding=1),          # output[192, 13, 13]
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=3, padding=1),          # output[192, 13, 13]
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 128, kernel_size=3, padding=1),          # output[128, 13, 13]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),                  # output[128, 6, 6]
        )
        # 分类器，包含三个全连接层
        self.classifier = nn.Sequential(
            # p代表随机失活的比例
            nn.Dropout(p=0.5),
            # Pytorch中常常将channel放在首位，因此第一位为128
            # 每经过一次全连接层，也需要进行一次ReLU函数激活
            nn.Linear(128 * 6 * 6, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            # 全连接层的输入即等于上一层的输出
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            # num_classes代表我们需要输出的类别个数
            nn.Linear(2048, num_classes),
        )
        # 初始化权重
        if init_weights:
            self._initialize_weights()

    # 正向传播
    def forward(self, x):
        # 调用以上模块封装
        x = self.features(x)
        # tensor通道排列顺序(batch, channel, height, width)
        # flatten展平，从index=1的维度
        x = torch.flatten(x, start_dim=1)
        # 将它输入到分类结构当中，即三个全连接层
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        # self.modules的继承自nn.module
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # 一般Pytorch直接
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            # 若传进来的实例为全连接层，则使用normal
            elif isinstance(m, nn.Linear):
                # 通过一个正态分布来给权重进行赋值，均值为0，方差为0.01
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
