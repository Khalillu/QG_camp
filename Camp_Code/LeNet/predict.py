import torch
import torchvision.transforms as transforms
from PIL import Image

from model import LeNet


def main():
    transform = transforms.Compose(
        # 与train函数中相同
        # Resize是将图片的像素转化为32*32
        [transforms.Resize((32, 32)),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    net = LeNet()
    net.load_state_dict(torch.load('Lenet.pth'))

    im = Image.open('2.jpg')
    # 将im中[H, W, C]的结构，转化为[C, H, W]的结构
    im = transform(im)  # [C, H, W]
    # 增加一个维度，N表示batch，即批量数
    im = torch.unsqueeze(im, dim=0)  # [N, C, H, W]

    with torch.no_grad():
        outputs = net(im)
        predict = torch.max(outputs, dim=1)[1].numpy()
        # predict = torch.softmax(outputs, dim=1)
        # 或者使用soft.max函数将输入进行处理
    # print(predict)
    print(classes[int(predict)])


if __name__ == '__main__':
    main()
