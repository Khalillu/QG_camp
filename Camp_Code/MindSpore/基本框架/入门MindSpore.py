import mindspore
import numpy as np

# 张量
# 张量，即存储多维数组（n-dimensional array）的数据结构。
x = mindspore.Tensor(np.random.rand(2, 3), mindspore.float32)
print(x)
# 判断输入对象是否为mind.Tensor
print(mindspore.is_tensor(x))

# 将numpy数组转换为张量
y = np.random.randn(2, 3)
print(y)
print(mindspore.from_numpy(y))

# 数据类型
print(mindspore.dtype)
# print(mindspore.Tensor_to_np(y))

# 全局种子
mindspore.set_seed(2)
# 随机种子
mindspore.get_seed(2)