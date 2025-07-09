from torch.utils.data import Dataset
import os
import pandas as pd
import numpy as np
import torch

# 封装数据集加载方式
# (flower_data)继承Dataset这样一个父类
# python要求对三个父类方法进行重写：初始化initial、getitem、len
class iris_dataloader(Dataset):
    def __init__(self,data_path):
        self.data_path = data_path

        # 检验该文件路径是否存在
        assert os.path.exists(self.data_path),"dataset does not exist"

        # names=[]给数据集中每个列命名
        df = pd.read_csv(self.data_path, names=[0,1,2,3,4])

        # 使用字典，将鸢尾花中名字映射到相应的数值
        d = {"setosa":0, "versicolor":1, "virginica":2}
        # 将df[4]中的列使用map映射为相应的值
        df[4] = df[4].map(d)

        data = df.iloc[1:,1:5]
        label = df.iloc[1:,5:]

        # 由于pandas的数据类型dataframe不能与pytorch兼容，因此不能直接使用，需要转化为numpy数组之后，再转化为torch中的张量tenser类型

        # 将数据集中的数值映射到均值为零，方差为1的数据分布中
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        data = (data - mean) / std # Z值化

        self.data = torch.from_numpy(np.array(data,dtype='float32'))
        self.label = torch.from_numpy(np.array(label,dtype='int64'))

        self.data_num = len(self.data)
        print("当前数据集的大小：",self.data_num)

    # 需要返回数据集的大小的原因是：训练模型时，可能会把数据集拆成好几个批量，而pytorch需要知道数据有多少，才能按照批量大小进行封装
    def __len__(self):
        return self.data_num

    # index数据中的索引
    def __getitem__(self, index):
        # 将data,label转化为可以迭代的数据类型
        self.data = list(self.data)
        self.label = list(self.label)

        return self.data[index], self.label[index]
