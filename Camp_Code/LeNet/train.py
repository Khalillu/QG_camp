import torch
import torchvision
import torch.nn as nn
from model import LeNet
import torch.optim as optim
import torchvision.transforms as transforms


def main():
    transform = transforms.Compose(
        # ToTensor将PIL Image转化为numpy.ndarray在[0,255]范围内的(H, W, C)，再转化为在[0,1]范围内的(C, H, W)
        [transforms.ToTensor(),
        # 将数据标准化
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # 50000张训练图片
    # 第一次使用时要将download设置为True才会自动去下载数据集
    train_set = torchvision.datasets.CIFAR10(root='./data', train=True,
                                             download=False, transform=transform)
    # 每次加载一批数据集，每一批拿出36张图片
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=36,
                                               shuffle=True, num_workers=0)

    # 10000张验证图片
    # 第一次使用时要将download设置为True才会自动去下载数据集
    val_set = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=False, transform=transform)
    # shuffle决定是否在每一次取一批数据之后打乱索引
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=5000,
                                             shuffle=False, num_workers=0)
    val_data_iter = iter(val_loader)
    val_image, val_label = next(val_data_iter)
    
    classes = ('plane', 'car', 'bird', 'cat',
                'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    net = LeNet()
    # nn.CrossEntropyLoss() 交叉熵损失函数
    loss_function = nn.CrossEntropyLoss()
    # 将net中的参数使用随机梯度算法Adam来进行更新
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    for epoch in range(5):  # loop over the dataset multiple times

        running_loss = 0.0
        for step, data in enumerate(train_loader, start=0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            # 将历史损失梯度清零，防止对计算的历史梯度进行累加
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = net(inputs)
            loss = loss_function(outputs, labels)
            # loss.backward() 进行反向传播
            loss.backward()
            # 而optimizer.step()用于参数的更新
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if step % 500 == 499:    # print every 500 mini-batches
                # 此处为检验正确率，不需要计算每个结点的损失梯度
                with torch.no_grad():
                    outputs = net(val_image)  # [batch, 10]
                    # 第零维代表batch，第一维输出的十个结点中寻找最大值
                    # [1]代表index索引
                    predict_y = torch.max(outputs, dim=1)[1]
                    # 由于计算出来的是tenser张量，因此要将他转化为.item()，获取其中的数据
                    accuracy = torch.eq(predict_y, val_label).sum().item() / val_label.size(0)

                    print('[%d, %5d] train_loss: %.3f  test_accuracy: %.3f' %
                          (epoch + 1, step + 1, running_loss / 500, accuracy))
                    running_loss = 0.0

    print('Finished Training')

    save_path = './Lenet.pth'
    torch.save(net.state_dict(), save_path)


if __name__ == '__main__':
    main()
