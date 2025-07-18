## GNN代码实现笔记

#### 图数据的定义：

图数据是由**节点**和**边**组成的数据，最简单的方式是使用邻接矩阵来表示图形结构，从而捕捉图形中的节点和边的相关性。

图数据的信息包含三个层面，分别是**节点信息（V）、边信息（E）、图整体信息（U）**，它们通常使用向量来表示。而图神经网络就是通过**学习数据从而得到三个层面向量的最优表示。**

#### 图数据的任务：

* 图层面的任务：

分类、回归

eg：分子是天然的图，原子是节点，化学键是边。现在要做一个分类，有一个苯环的分子分一类，两个苯环的分子分一类。这是**图**分类任务。

* 边层面的任务

分类、回归

eg：通过语义分割把人和环境分离开来。每个人都是节点，现在做预测，预测的是人之间的关系，是合作关系？还是竞争关系？是互利共生关系？还是天敌关系？即，节点间的关系（边）。这是**边**分类任务。

* 节点层面的任务

分类、回归

eg：假设一个跆拳道俱乐部里有A、B两个教练，所有的会员都是节点。有一天A、B两个跆拳道教练决裂，那么各个学员是愿意和A在一个阵营还是愿意和B在一个阵营？这是**节点**分类任务。

#### 图神经网络工作流程：

GNN是对图上的所有属性进行的一个可以优化的变换，**输入是一个图，输出也是一个图。**它只对**属性向量（即V、E、U）进行变换**，但它**不会改变图的连接性**（即哪些点互相连接经过GNN后时不会变的）。在获取优化后的属性向量之后，再根据实际的任务，**后接全连接神经网络**，进行**分类和回归**。即可以将图神经网络看做是**一个图数据**的在**三个维度**（V、E、U）的特征提取器。

GNN对属性优化的方法叫做**消息传递机制**。比如最原始的GNN是SUM求和传递机制；到后面发展成图卷积网络（GCN）就考虑到了节点的度，**度越大，权重越小**，使用了**加权的SUM**；再到后面发展为图注意力网络GAT，在消息传递过程中，引入了注意力机制；

![img](https://pic2.zhimg.com/v2-21e09751fb2f96323501fc260afeed47_r.jpg)

**不同GNN的本质差别就在于它们如何进行节点之间的信息传递和计算，也就是它们的消息传递机制不同。**

#### 图神经网络代码实现：

##### 库函数

PyG（[PyTorch Geometric](https://zhida.zhihu.com/search?content_id=222240356&content_type=Article&match_order=1&q=PyTorch+Geometric&zhida_source=entity)），它是一个为图形数据的处理和学习提供支持的PyTorch扩展库，如图分类、图回归、图生成等。

以下为PyG常用的**内置数据集**：

1. **Cora, Citeseer, Pubmed**：这些数据集是文献引用网络数据集，用于节点分类任务。
2. **PPI**：蛋白质蛋白相互作用网络数据集，用于边分类任务。
3. **Reddit**：Reddit社交网络数据集，用于节点分类任务。
4. **Amazon-Computers，Amazon-Photo**：Amazon商品共同购买网络数据集，用于节点分类和图分类任务。
5. **ENZYMES**：蛋白质分子结构数据集，用于图分类任务。
6. **MUTAG**：分子化合物数据集，用于图分类任务。
7. **QM7b**：有机分子数据集，用于图回归任务。

