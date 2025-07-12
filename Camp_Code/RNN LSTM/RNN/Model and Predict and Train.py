import math
import torch
from d2l.torch import try_gpu
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l

batch_size, num_steps = 32,35
train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)

# 通过one-hot encoding 独热编码来将词元表示为更具表现力的特征向量
# 简言之，将每个索引映射为相互不同的单位向量：假设此表中不同词元的数目为N（len(vocab)），词元索引的范围为0到N-1.如果词元的索引是整数i，那么我们将创建一个长度为N的全零向量，并将第i处的元素设置为1.此向量是原始词元的一个独热向量。索引为0和2的独热向量如下所示：

# F.one_hot(torch.tensor([0, 2]), len(vocab))

# 即表示为

# tensor([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
# 0, 0, 0, 0],
# [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
# 0, 0, 0, 0]])

# 我们每次采样的小批量数据形状是二维张量：（批量大小，时间步数）。one_hot函数将这样一个小批量数据转换成三维张量，张量的最后一个维度等于词表大小（len(vocab)）。
# 我们经常转换输入的维度，以便获得形状为（时间步数，批量大小，词表大小）的输出。这将使我们能够更方便地通过最外层的维度，一步一步地更新小批量数据的隐状态。
#
X = torch.arange(10).reshape((2, 5))
F.one_hot(X.T, 28).shape
# 时间步数放在首位时，每次访问后X.T都是连续的，因为此时批量数据与len(vocab)在后面两个维度表示X_t，每次访问时，X_t都是连续的，因为它挪到了后面两个维度
# [时间步数, 批量大小, 每一个样本的批量长度]
torch.Size([5, 2, 28])
#
# 初始化模型参数
def get_params(vocab_size, num_hiddens, device):
    # 输入为一个个词，通过one-hot变为num_inputs向量之后，那么就变成长为vocab_size的向量
    # 输出也应该为vocab_size，因为输出的下一个词可以为vocab中的任意一个词
    num_inputs = num_outputs = vocab_size

    def normal(shape):
        # torch.randn 是putorch的函数，作用是生成服从标准正态分布（均值为0，标准差为1）的随机数
        # size = shape 制定了输出张量的形状
        # 将生成的随机数*0.01，相当于将标准差从1缩放到0.01
        # 这种小标准差（0.01）的初始化常用于神经网络的权重初始化，目的是避免初始化值过大导致训练不稳定（如梯度爆炸）
        return torch.randn(size=shape, device=device)*0.01

    # 隐藏层参数
    W_xh = normal((num_inputs, num_hiddens))
    # 上一个隐藏变量状态到下一个隐藏变量状态的转换
    W_hh = normal((num_hiddens, num_hiddens))
    b_h = torch.zeros(num_hiddens, device=device)

    # 输出层参数
    W_hq = normal((num_hiddens, num_outputs))
    b_q = torch.zeros(num_outputs, device=device)

    # 附加梯度
    params = [W_xh, W_hh, b_h, W_hq, b_q]
    for param in params:
        # 将参数置于一个列表中，为每个参数启用梯度计算
        param.requires_grad_(True)
    return params

# 定义循环神经网路模型，我们首先需要一个init_rnn_state函数在初始化时返回隐状态。这个函数的返回是一个张量，张量全用0填充，形状为（批量大小，隐藏单元数）。在后面的应用中，我们可能会遇到隐状态包含多个变量的情况，而使用元组可以更容易地处理些。

def init_rnn_state(batch_size, num_hiddens, device):
    # 任意时刻隐藏状态的shape都是(batch_size, num_hiddens)
    # LSTM中将会有两个状态，因此为tuple，这里代表一个状态
    # batch_size 表示同时处理的样本数量，为了高校计算，我们将会会并行处理一个批次中的所有样本
    # vocab代表字母（等于28），而batch_size代表n个词，因此batch_size，代表每个词给出几个字母，然后并行预测下一个字母是什么
    # 因此隐藏状态需要为每个样本单独维护一个状态向量
    return (torch.zeros((batch_size, num_hiddens), device=device),)

# 下面的rnn函数定义了如何在一个时间步内计算隐状态和输出，循环神经网络模型通过inputs最外层的维度实现循环，以便逐时间步更新小批量数据的隐状态H。
# 此外，这里使用tanh函数作为激活函数。
# 当元素在实数上满足均匀分布时，tanh函数的平均值为0。
def rnn(inputs, state, params):
    # input的形状：（时间步数，批量大小，词表大小）
    W_xh, W_hh, b_h, W_hq, b_q = params
    # 此处H为一个tuple，因此state也是个tuple，即(batch_size, num_hiddens)
    H, = state
    outputs = []
    # X的形状：X_t（批量大小，词表大小）
    # 沿着第一个维度去遍历inputs
    # 每一步算一个特定的时间步
    for X in inputs:
        # 计算H_t
        # X（当前状态）通过W_xh权重矩阵线性变换
        # H_{t-1}（前一个状态）通过W_hh权重矩阵线性变换（实现循环连接）
        # +b_h加上隐藏层的偏执
        H = torch.tanh(torch.mm(X, W_xh) + torch.mm(H, W_hh) + b_h)
        # 当前隐藏层H通过权重W_hq和偏执b_q线性变换，得到输出Y，即outputs中的元素
        Y = torch.mm(H, W_hq) + b_q
        outputs.append(Y)
        # 对于所有的Y之前
        # cat=concat，即合并，拼接
        # 可以理解为n个矩阵按照竖直方向拼接起来，因此列数没有变化，仍然是vocab_size
        # 行数变成了批量大小乘以时间长度
        # T个时间步乘以批量大小，表示全部词在每一个时刻的输出
    return torch.cat(outputs, dim=0), (H,)

# 定义一个类来包装这些函数，并存储从零开始实现的循环神经网络模型的参数
class RNNModelScratch:  # @save
    """从零开始的循环神经网络"""
    def __init__(self, vocab_size, num_hiddens, device, get_params, init_state, forward_fn):
        self.vocab_size, self.num_hiddens = vocab_size, num_hiddens
        self.params = get_params(vocab_size, num_hiddens, device)
        self.init_state, self.forward_fn = init_state, forward_fn

    # X等于num_step(T) * 批量大小 * vocab_size的input
    def __call__(self, X, state):
        X = F.one_hot(X.T, self.vocab_size).type(torch.float32)
        return self.forward_fn(X, state, self.params)

    def begin_state(self, batch_size, device):
        return self.init_state(batch_size, self.num_hiddens, device)

    # 检查输出是否具有正确的形状，例如隐状态的维数是否改变

num_hiddens = 512
net = RNNModelScratch(len(vocab), num_hiddens, d2l.try_gpu(), get_params, init_rnn_state, rnn)
state = net.begin_state(X.shape[0], d2l.try_gpu())
Y, new_state = net(X.to(d2l.try_gpu()), state)
Y.shape, len(new_state), new_state[0].shape

# output (torch.Size([10,28])) , 1, torch.Size([2, 512])

# Predict

# prefix前缀，句子的开头
# num_preds 需要生成多少个词，因为此处vocab使用的是字符，所以词和字符就没有区别，因此就是一个char
# vocab用于从一个值map到一个字符
def predict_ch8(prefix, num_preds, net, vocab, device):
    """在prefix后面生成新字符"""
    # 生成初始状态
    # batch_size=1，对一个字符串进行预测
    state = net.begin_state(batch_size=1, device=device)
    # 第一个字符，放到vocab中，得到对应整形的下标，放到outputs中
    # 因此一开始是一个字
    outputs = [vocab[[prefix][0]]]
    # 将上一次预测的一个词，作为下一时刻的输入
    # batch_size=1,num_steps=1
    get_input = lambda: torch.tensor([outputs[-1]], device=device).reshape((1,1))
    # 更新已知序列的隐藏状态
    for y in prefix[1:]:
        _, state = net(get_input(), state)
        # 因为有真实值，因此不需要使用预测值，以防累计误差
        outputs.append(vocab[y])
    for _ in range(num_preds):
        # 将prefix（y） 作为input放进net中
        y, state = net(get_input(), state)
        # reshape成一个标量张量tensor([x]),再用int()提取数值(x)
        # 取dim=1概率最大的向量，即vocab_size维度，取概率最大的字符索引
        # 即将最大概率的字母转换为它的索引
        outputs.append(int(y.argmax(dim=1).reshape(1)))
    # 将index转成token
    return ''.join([vocab.idx_to_token[i] for i in outputs])


predict_ch8('time travaller', 10, net, vocab, d2l.try_gpu())

# 梯度裁剪
def grad_clipping(net, theta):  #@save
    if isinstance(net, nn.Module):
        params = [p for p in net.parameters() if p.requires_grad]
    else:
        # 将所有层的参数拿出来放在一起进行梯度裁剪
        params = net.params
    norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
    if norm > theta:
        for param in params:
                param.grad[:] *= theta / norm

# 模型训练
# use_random_iter的作用是将下一个批量的第i个样本，与上一个批量的第i个样本没有关系
# use_sequence_iter则是下一个批量的第i个样本与上一个批量的第i个样本是相邻的
# 会导致隐藏层的状态不一样
def train_epoch_ch8(net, train_iter, loss, updater, device,
                    use_random_iter):
    # timer计时器
    state, timer = None, d2l.Timer()
    # 记录loss
    # metric：是一个长度为2的累加器（d2l.Accumulator(2)），其内容为：
    # metric[0]：所有样本的损失总和（未取平均的原始损失累加值）。
    # metric[1]：所有样本的词元总数（即 batch_size * num_steps 的累加值）。
    # timer.stop()：返回从计时器启动到停止的时间间隔（秒）。
    metric = d2l.Accumulator(2)
    for X, Y in train_iter:
        if state is None or use_random_iter:
            # 在第一次迭代或使用随机抽样时初始化state
            # 因为随机抽样时以前的sequence信息和当前的sequence信息并不是连续的，上一个批量的state不应该用到这一批量的state来
            # 序列连续模式（默认）：当前批次的初始state=上一批次的最终state（时间步连续）。
            # 随机抽样模式（use_random_iter=True）：每个批次从随机位置采样，state必须重新初始化。

            state = net.begin_state(batch_size=X.shape[0], device=device)
        else:
            if isinstance(net, nn.Module) and not isinstance(state, tuple):
            # state对于nn.GRU是个张量
            # 不把里面的值改掉，在backward传播的时候，在做梯度运算的时候只关心现在之后的运算，做forward
            # 连续模式：从第50页读到第60页时，需要记住第49页的内容（state），但不需要记住第1~48页的细节（detach_）。
            #
            # 随机模式：随机翻到第100页时，必须清空记忆（重新初始化state）。
            # RNN的state会携带之前时间步的计算历史（通过计算图记录梯度）。
            # 如果直接复用state，梯度会从当前批次反向传播到所有之前的时间步，导致：
            # 计算冗余：梯度计算超出当前批次范围。
            # 内存爆炸：计算图无限增长。
            # .detach_()的作用
            # 切断计算图：将state从原有计算图中分离（转为“叶子节点”），但仍保留其数值。
            # 只需要记录上一时刻的状态，因为上一时刻包含了以前全部时刻的历史信息！！！
                state.detach_()
            else:
            # state对于nn.LSTM或对于我们从零开始实现的模型是个张量
                for s in state:
                    s.detach_()
        # 将时间步维度拉前一个维度
        y = Y.T.reshape(-1)
        X, y = X.to(device), y.to(device)
        y_hat, state = net(X, state)
        # 批量大小乘以时间步个样本
        # y_hat：模型的预测输出

        # 形状通常为 (num_steps * batch_size, vocab_size)，例如 (350, 28)（35时间步×10批次，28是字符表大小）。
        #将模型预测与模型实际的y作比较，因为此处loss使用的是交叉熵函数，CrossEntropy，传入的参数需要变成int64，即y.long()
        l = loss(y_hat, y.long()).mean()
        # 若使用torch.optim.Optimizer应该要将updater，即更新器的梯度清零
        if isinstance(updater, torch.optim.Optimizer):
            updater.zero_grad()
            l.backward()
            # 未更新梯度前，对梯度做clipping，即若梯度大于1时，就需要往下投影
            grad_clipping(net, 1)
            # 更新梯度
            updater.step()
        else:
            l.backward()
            grad_clipping(net, 1)
            # 因为已经调用了mean函数
            updater(batch_size=1)
        metric.add(l * y.numel(), y.numel())
        # metric[0]/metric[1]代表crossEntropy，而计算perplexity困惑度只需要加上指数，即exp
        # metric[1] / timer.stop() 代表每秒处理的词元数量
    return math.exp(metric[0] / metric[1]), metric[1] / timer.stop()

# 循环神经网络模型的训练函数也可以使用高级的API实现
###
def train_ch8(net, train_iter, vocab, lr, num_epochs, device, use_random_iter=False):
    # 虽然为语言模型，但实际是标准的多分类
    loss = nn.CrossEntropyLoss()
    animator = d2l.Animator(xlabel='epoch', ylabel='perplexity',
                            legend=['train'], xlim=[10, num_epochs])

    # 初始化
    # 若net为nn.Module的子类，则使用torch.optim.SGD
    if isinstance(net, nn.Module):
        updater = torch.optim.SGD(net.parameters(), lr)
    else:
        updater = lambda batch_size: d2l.sgd(net.params, lr, batch_size)
    predict = lambda prefix: predict_ch8(prefix, 50, net, vocab, device)
    # 训练和预测
    for epoch in range(num_epochs):
        ppl, speed = train_epoch_ch8(net, train_iter, loss, updater, device,
                                     use_random_iter)
        if (epoch + 1) % 10 == 0:
            print(predict('time traveller'))
            animator.add(epoch + 1, [ppl])
    print(f'困惑度 {ppl:.3f},{speed:.3f} 词元/秒 {str(device)}')
    print(predict('time traveller'))
    print(predict('traveller'))

num_epochs, lr = 500, 1
train_ch8(net, train_iter, vocab, lr, num_epochs, d2l.try_gpu())