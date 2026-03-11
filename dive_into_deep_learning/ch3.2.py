# -*- coding: utf-8 -*-
"""
线性回归从零开始实现 - 《动手学深度学习》第3.2节

本节内容：
1. 生成数据集
2. 读取数据集
3. 初始化模型参数
4. 定义模型
5. 定义损失函数
6. 定义优化算法
7. 训练模型
"""

# ==================== 导入必要的库 ====================
# 注意：matplotlib.use('Agg') 必须在导入 pyplot 之前执行！
import matplotlib
matplotlib.use('Agg')  # 设置非交互式后端，避免IDE中GUI错误

import random  # Python标准库，用于生成随机数（后续小批量采样会用到）
import torch   # PyTorch深度学习框架，提供张量运算和自动求导
import matplotlib.pyplot as plt  # 导入matplotlib用于绘图
from d2l import torch as d2l  # 导入《动手学深度学习》配套工具库


# ==================== 1. 生成数据集 ====================
def synthetic_data(w, b, num_examples):  # @save 标记表示该函数保存到d2l包中
    """
    生成带噪声的线性回归数据集
    公式: y = Xw + b + ε, 其中ε是随机噪声
    
    参数:
        w: 权重向量（真实参数）
        b: 偏置项（真实参数）
        num_examples: 生成的样本数量
    
    返回:
        X: 特征矩阵，形状为 (num_examples, len(w))
        y: 标签向量，形状为 (num_examples, 1)
    """
    # 生成特征矩阵 X，每个元素从标准正态分布 N(0,1) 中采样
    # 形状: (样本数, 特征数) = (num_examples, len(w))
    X = torch.normal(0, 1, (num_examples, len(w)))
    
    # 计算线性部分: y = Xw + b
    # torch.matmul 是矩阵乘法，X形状(m,n), w形状(n,), 结果形状(m,)
    y = torch.matmul(X, w) + b
    
    # 添加随机噪声，模拟真实数据中的测量误差
    # 噪声服从 N(0, 0.01) 正态分布，标准差为0.01（较小的噪声）
    y += torch.normal(0, 0.01, y.shape)
    
    # 将y重塑为列向量，形状从 (num_examples,) 变为 (num_examples, 1)
    # -1 表示自动推断该维度大小
    return X, y.reshape((-1, 1))


# 定义真实参数（模型需要学习的目标值）
true_w = torch.tensor([2, -3.4])  # 真实权重：第1个特征权重为2，第2个特征权重为-3.4
true_b = 4.2                       # 真实偏置项

# 生成包含1000个样本的训练数据集
# features: 特征矩阵，形状 (1000, 2)
# labels: 标签向量，形状 (1000, 1)

#feature包含1000个自变量x1和x2的值，labels包含1000个因变量y的值
features, labels = synthetic_data(true_w, true_b, 500)

# 打印部分数据查看
print('features:', features[0], '\nlabel:', labels[0])

# ==================== 数据可视化 ====================
# # 设置图形大小
# plt.figure(figsize=(6, 4))

# # 绘制散点图：features[:, 1] 是第二个特征，labels 是对应的标签
# # detach().numpy() 将PyTorch张量转换为NumPy数组以便绘图
# # 参数1: 第二个特征的值，参数2: 对应的标签值，参数3: 点的大小
# plt.scatter(features[:, 1].detach().numpy(), labels.detach().numpy(), s=1)

# # 添加标题和坐标轴标签
# plt.title('Features[1] vs Labels')
# plt.xlabel('Features[1]')
# plt.ylabel('Labels')

# # 保存图片到本地文件
# plt.savefig('features_labels_scatter.png', dpi=300, bbox_inches='tight')
# print('散点图已保存到: features_labels_scatter.png')


# ==================== 2. 读取数据集 ====================
def data_iter(batch_size, features, labels):
    """小批量随机梯度下降的数据迭代器"""
    num_examples = len(features)                      # 获取样本总数
    indices = list(range(num_examples))               # 生成样本索引列表
    random.shuffle(indices)                          # 随机打乱索引顺序
    for i in range(0, num_examples, batch_size):     # 按批次大小遍历
        batch_indices = torch.tensor(                # 取出当前批次的索引
            indices[i: min(i + batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices]  # 返回特征和标签，相当于盗用函数后作用域未消失,再次调用继续执行

batch_size = 10

for X, y in data_iter(batch_size, features, labels):
    print(X, '\n', y)
    break
# ==================== 3. 初始化模型参数 ====================
# 生成随机权重，均值为0，标准差为0.01，形状为(2,1)，并启用梯度计算，这样梯度计算才会传播
w = torch.normal(0, 0.01, size=(2,1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

# ==================== 4. 定义模型 ====================
# 定义线性回归模型 y = Xw + b
def linreg(X, w, b):  #@save
    """线性回归模型"""
    return torch.matmul(X, w) + b  # 矩阵乘法计算预测值：X是特征矩阵，w是权重向量，b是偏置



# ==================== 5. 定义损失函数 ====================
# 定义平方损失函数（均方误差）
def squared_loss(y_hat, y):  #@save
    """均方损失"""
    # 计算预测值与真实值的平方误差
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2
# ==================== 6. 定义优化算法 ====================
#在神经网络中，这些参数就变成感知机的权重参数，而线性回归的参数就是线性回归模型的参数
def sgd(params, lr, batch_size):  #@save
    """小批量随机梯度下降"""
    for param in params:
        param.data = param - lr * param.grad / batch_size
        param.grad.zero_()  # 清零梯度，防止累积
# ==================== 7. 训练 ====================
# 编写训练循环，迭代更新参数
lr = 0.03  # 学习率
num_epochs = 3  # 训练轮数
net = linreg  # 模型
loss = squared_loss  # 损失函数

for epoch in range(num_epochs):  # 迭代训练轮数
    # 取出特征值和标签
    for X, y in data_iter(batch_size,features,labels):
        l =loss(net(X, w, b), y)  # 计算损失值
        l.sum().backward()  # 把batch的loss都加到一块，然后做计算图反向传播 反向传播计算梯度
        sgd([w, b], lr, batch_size)  # 使用小批量随机梯度下降更新参数
    with torch.no_grad():  # with代表该层for循环wan'bi
        train_l = loss(net(features, w, b), labels)  # 计算整个训练集的损失值
        print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')  # 打印当前轮数和平均损失值
# ==================== 8. 小结 ====================
print(f'w的估计误差: {true_w - w.reshape(true_w.shape)}')
print(f'b的估计误差: {true_b - b}')
