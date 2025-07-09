import os
import sys
import json

import torch
import torch.nn as nn
from torchvision import transforms, datasets, utils
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
from tqdm import tqdm

from model import AlexNet


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    data_transform = {
        # RandomResizedCrop随机裁剪，将其裁剪到224*224像素大小
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     # 随机反转
                                     transforms.RandomHorizontalFlip(),
                                     # 转化为tensor
                                     transforms.ToTensor(),
                                     # 进行标准化处理
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
        "val": transforms.Compose([transforms.Resize((224, 224)),  # cannot 224, must (224, 224)
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])}

    # "..表示返回上层目录 ..//..表示返回上上层目录"
    data_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))  # get data root path
    image_path = r"D:\Pycharm\PyCharm 2024.3.1.1\deep-learning-for-image-processing-master\data_set\flower_data"  # flower data set path
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"),
                                        # 以上预处理函数
                                         transform=data_transform["train"])
    train_num = len(train_dataset)

    # {'daisy':0, 'dandelion':1, 'roses':2, 'sunflower':3, 'tulips':4}
    flower_list = train_dataset.class_to_idx
    # 遍历刚才所获得的字典，将key和value值反过来，即将键和值相反
    # 以上处理之后，在预测完之后，返回给我们的索引就能直接通过字典得到它所对应的类别
    cla_dict = dict((val, key) for key, val in flower_list.items())
    # write dict into json file
    # 将cla_dict进行编码成json的方式
    json_str = json.dumps(cla_dict, indent=4)
    # 生成json文件
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    batch_size = 32
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               # 加载数据所使用的线程个数
                                               num_workers=nw)
    # 通过imagefolder载入测试集，同样传入测试集所对应的预处理参数，统计测试集的文件个数
    validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"),
                                            transform=data_transform["val"])
    val_num = len(validate_dataset)
    # 载入测试集
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=4, shuffle=False,
                                                  num_workers=nw)

    print("using {} images for training, {} images for validation.".format(train_num,
                                                                           val_num))
    # test_data_iter = iter(validate_loader)
    # test_image, test_label = test_data_iter.next()
    #
    # def imshow(img):
    #     img = img / 2 + 0.5  # unnormalize
    #     plt.imshow(np.transpose(npimg, (1, 2, 0)))
    #     npimg = img.numpy()
    #     plt.show()
    #
    # print(' '.join('%5s' % cla_dict[test_label[j].item()] for j in range(4)))
    # imshow(utils.make_grid(test_image))

    net = AlexNet(num_classes=5, init_weights=True)

    net.to(device)
    # 交叉熵损失函数
    loss_function = nn.CrossEntropyLoss()
    # pata = list(net.parameters())
    optimizer = optim.Adam(net.parameters(), lr=0.0002)

    epochs = 10
    save_path = './AlexNet.pth'
    # 为了能保存准确率最高的模型，因此有best_acc
    best_acc = 0.0
    train_steps = len(train_loader)
    for epoch in range(epochs):
        # train
        # 通过net.train与net.eval管理我们的dropout方法
        # net.train()打开dropout
        net.train()
        # 统计平均损失
        running_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)
        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()
            outputs = net(images.to(device))
            loss = loss_function(outputs, labels.to(device))
            loss.backward()
            # 通过以上获得的损失信息，来更新每一个结点的参数
            optimizer.step()

            # print statistics
            # 打印训练进度
            running_loss += loss.item()

            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                     epochs,
                                                                     loss)

        # validate
        # net.eval()关闭dropout
        net.eval()
        acc = 0.0  # accumulate accurate number / epoch
        # 禁止pytorch对参数进行跟踪
        with torch.no_grad():
            val_bar = tqdm(validate_loader, file=sys.stdout)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

        val_accurate = acc / val_num
        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
              (epoch + 1, running_loss / train_steps, val_accurate))

        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)

    print('Finished Training')


if __name__ == '__main__':
    main()
