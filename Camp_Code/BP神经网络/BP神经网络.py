import numpy as np


def nonlin(x, deriv=False):
    if (deriv == True):
        return x * (1 - x)
    return 1 /  (1 + np.exp(-x))

X = np.array([[0.35],[0.9]])    # 输入层
y = np.array([[0.5]])

np.random.seed(1)

W0 = np.array([[0.1, 0.8],[0.4, 0.6]])
W1 = np.array([[0.3, 0.9]])

print("Original W0:{0},W1:{1}".format(W0, W1))
for j in range(100):
    # l0输入层，l1隐藏层、l2输出层
    l0 = X  # 相当于文章中x0
    l1 = nonlin(np.dot(W0, l0)) # 相当于文章中y1
    l2 = nonlin(np.dot(W1, l1)) # 相当于文章中y2
    l2_error = y - l2
    Error = 1/2.0*(y - l2)**2
    print("Error:{0}".format(Error))
    print()
    l2_delta = l2_error * nonlin(l2, deriv=True)    # nonlin(l2, deric=True)作用是计算sigmoid函数中对l2的导数
    print("l2_delta:{0}".format(l2_delta))
    print()
    l1_error = l2_delta * W1    # 反向传播
    l1_delta = l1_error * nonlin(l1, deriv=True)

    W1 += l2_delta * l1.T
    W0 += l0.T.dot(l1_delta)
    print("W0:{0}".format(W0))
    print("W1:{0}".format(W1))
    print()
    print()