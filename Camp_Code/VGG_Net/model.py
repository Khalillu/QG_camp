import torch.nn as nn
import torch

# official pretrain weights
model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth'
}


class VGG(nn.Module):
    # features代表make_futures生成的参数
    def __init__(self, features, num_classes=1000, init_weights=False):
        super(VGG, self).__init__()
        self.features = features
        # 分类网络结构
        self.classifier = nn.Sequential(
            # 512*7*7代表展平之后的元素个数
            nn.Linear(512*7*7, 4096),
            # Linear()展平为全连接层或者conv2()卷积之后都需要经过ReLU激活函数
            nn.ReLU(True),
            # dropout防止过拟合，以50%比例随机失活神经元
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes)
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        # N x 3 x 224 x 224
        # 通过提取特征网络结构
        x = self.features(x)
        # N x 512 x 7 x 7
        # 从第一个维度开始展平，第零维是batch，第一维是channel，第二维是height，第三维是width
        x = torch.flatten(x, start_dim=1)
        # N x 512*7*7
        x = self.classifier(x)
        return x

    # 遍历神经网络的每一层，若遍历的神经网络为卷积层，那就要使用xavier取初始化卷积核的权重
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.xavier_uniform_(m.weight)
                # 若使用了偏置，则将偏置初始化为零
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                    # 若为全连接层，也是用xavier来初始化权重，将偏置设为零
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                # nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    # 提取特征网络结构
def make_features(cfg: list):
    # 使用一个列表layers存储所需要的神经网络训练过程
    layers = []
    # RGB彩色图片，通道数为3
    in_channels = 3
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            # v不是M时，代表卷积核的个数
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            layers += [conv2d, nn.ReLU(True)]
            in_channels = v
            # 代表通过非关键字参数传入进去的
            # 层的顺序通过文件列表的顺序来生成
    return nn.Sequential(*layers)


cfgs = {
    # M代表最大池化
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

# **kwargs包含分类个数与初始化权重变量的布尔值
def vgg(model_name="vgg16", **kwargs):
    assert model_name in cfgs, "Warning: model number {} not in cfgs dict!".format(model_name)
    # 将VHH16的key值传入到字典中，就可以得到我们所对应的配置列表config
    cfg = cfgs[model_name]
    # 再通过实例化VGG，make_features(cfg)将列表cfg的特征提取出来，**kwargs中**表示可变长度字典变量
    model = VGG(make_features(cfg), **kwargs)
    return model
