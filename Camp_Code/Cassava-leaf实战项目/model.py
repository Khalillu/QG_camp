import torch.nn as nn
import torch


class BasicBlock(nn.Module):
    # expansion对应残差结构中主分支所采用的卷积核的个数是否发生变化
    expansion = 1

    # out_channel输出矩阵深度，对应主分支上卷积核的个数，该BasicBlock表示Residual结构模块，
    # 而downsample代表该捷径是虚线还是实线，即下采样的方法
    def __init__(self, in_channel, out_channel, stride=1, downsample=None, **kwargs):
        super(BasicBlock, self).__init__()
        # eg:若stride=1,output=(input - kernel_size + 2*padding)/stride + 1
        # 上面的input指的是上一模块的output，即输出矩阵的高和宽
        # 使用Batch.normalization时，不需要使用到bias，因此将bias=False
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        # 传入我们定义的下采样方法，默认为None
        self.downsample = downsample

    def forward(self, x):
        # 将x赋值给identity，即捷径分支上的输出值
        identity = x
        if self.downsample is not None:
            # 对应实现的残差结构
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # 此时没有进行ReLU激活函数，因为需要将输出加上那个捷径分支的输出，之后再通过ReLU激活函数
        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    """
    注意：原论文中，在虚线残差结构的主分支上，第一个1x1卷积层的步距是2，第二个3x3卷积层步距是1。
    但在pytorch官方实现过程中是第一个1x1卷积层的步距是1，第二个3x3卷积层步距是2，
    这么做的好处是能够在top1上提升大概0.5%的准确率。
    可参考Resnet v1.5 https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch
    """
    # 对应卷积核个数的变化，为18-Layer、34-Layer的四倍，因此expansion =4
    expansion = 4

    def __init__(self, in_channel, out_channel, stride=1, downsample=None,
                 groups=1, width_per_group=64):
        super(Bottleneck, self).__init__()

        width = int(out_channel * (width_per_group / 64.)) * groups

        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=width,
                               kernel_size=1, stride=1, bias=False)  # squeeze channels
        self.bn1 = nn.BatchNorm2d(width)
        # -----------------------------------------
        self.conv2 = nn.Conv2d(in_channels=width, out_channels=width, groups=groups,
                               kernel_size=3, stride=stride, bias=False, padding=1)
        self.bn2 = nn.BatchNorm2d(width)
        # -----------------------------------------
        # conv3即第三个卷积核的个数为out_channel的expandsion倍
        self.conv3 = nn.Conv2d(in_channels=width, out_channels=out_channel*self.expansion,
                               kernel_size=1, stride=1, bias=False)  # unsqueeze channels
        # 此时out_channel = out_channel*self.expansion，即卷积核夸大了expansion倍
        self.bn3 = nn.BatchNorm2d(out_channel*self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    # blocks_num代表一个残差结构的残差块数目，是一个列表参数，对于34-Layer[3,4,6,3]
    def __init__(self,
                 block,
                 blocks_num,
                 num_classes=1000,
                 include_top=True,
                 groups=1,
                 width_per_group=64):
        super(ResNet, self).__init__()
        self.include_top = include_top
        # 对应3x3maxpooling之后所得到的特征矩阵的深度
        self.in_channel = 64

        self.groups = groups
        self.width_per_group = width_per_group

        self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=7, stride=2,
                               padding=3, bias=False)
        # bn1的输入为conv1所输出的特征矩阵的深度，即self.in_channel，即out_channel
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # layer1对应的即为conv2所对应的一系列结构，即18-Layer conv2.18,conv2.34,conv2.50,conv2.101,conv2.152
        # 以下同理
        self.layer1 = self._make_layer(block, 64, blocks_num[0])
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)
        if self.include_top:
            # 无论输入矩阵的高和宽是多少，通过自适应平均池化下采样之后得到的特征矩阵的高和宽都是一和一
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # output size = (1, 1)
            # 所输入的节点个数是通过平均池化下采样层之后所得到特征矩阵展平后得到的节点个数
            # channel=512（深度），要乘以block.expansion，才是最后input的节点个数
            self.fc = nn.Linear(512 * block.expansion, num_classes)

        # 对卷积层进行初始化操作
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    # block指上文中的BasicBlock or Bottleneck, channel表示残差结构中，第一层卷积层所使用的卷积核个数，在Bottleneck结构中，第三层的卷积核个数往往是第一层的expansion倍
    # block_num指的是残差块的个数
    def _make_layer(self, block, channel, block_num, stride=1):
        downsample = None
        # 第一层通常stride=1
        # stride != 1时，会改变特征矩阵的高和宽
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion))

        # 将第一层残差结构传入
        layers = []
        layers.append(block(self.in_channel,
                            channel,
                            downsample=downsample,
                            stride=stride,
                            groups=self.groups,
                             width_per_group=self.width_per_group))
        # 当block为Bottleneck时，output(self.in_channel)的值需要变成原来的expansion倍
        self.in_channel = channel * block.expansion

        for _ in range(1, block_num):
            # self.in_channel代表输入矩阵的深度，channel表示主分支上第一层卷积的卷积核个数
            layers.append(block(self.in_channel,
                                channel,
                                groups=self.groups,
                                width_per_group=self.width_per_group))
        # 通过非关键字参数的形式*layers传入nn.sequential()结构中
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.include_top:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)

        return x


def resnet34(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnet34-333f7ec4.pth
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)


def resnet50(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnet50-19c8e357.pth
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)


def resnet101(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnet101-5d3b4d8f.pth
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, include_top=include_top)


def resnext50_32x4d(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth
    groups = 32
    width_per_group = 4
    return ResNet(Bottleneck, [3, 4, 6, 3],
                  num_classes=num_classes,
                  include_top=include_top,
                  groups=groups,
                  width_per_group=width_per_group)


def resnext101_32x8d(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth
    groups = 32
    width_per_group = 8
    return ResNet(Bottleneck, [3, 4, 23, 3],
                  num_classes=num_classes,
                  include_top=include_top,
                  groups=groups,
                  width_per_group=width_per_group)
