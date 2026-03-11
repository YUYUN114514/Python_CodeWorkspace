#3.3. 线性回归的简洁实现
import numpy as np
import torch
from torch.utils import data
from d2l import torch as d2l
from torch import nn
# 3.1. 生成数据集
true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = d2l.synthetic_data(true_w, true_b, 1000)

# 3.2. 读取数据集
# 该函数用于构造一个 PyTorch 数据迭代器（DataLoader），用于批量读取训练数据。(封装一下变量，就好比定义一个时钟结构体,把数据填进去，没有的就默认初始化)
def load_array(data_arrays, batch_size, is_train=True):  #@save
    """构造一个PyTorch数据迭代器"""
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)

batch_size = 10
data_iter = load_array((features, labels), batch_size)

# print(next(iter(data_iter)))

# nn.Linear(2, 1)	全连接层（线性层）：2个输入特征 → 1个输出
# nn.Sequential(...)	模型容器，将层按顺序串联，要是多层的话就是好多个全连接层连到一块
# net	最终的神经网络对象

# 3.3. 定义模型
# 我们将两个参数传递到nn.Linear中。 第一个指定输入特征形状，即2，第二个指定输出特征形状，输出特征形状为单个标量，因此为1。
net = nn.Sequential(nn.Linear(2, 1))

# 3.4初始化模型参数
# 自动注册这两个学习参数，并且可以通过 net[0].weight 和 net[0].bias 访问它们。
# 通过 net[0].weight.data.normal_(0, 0.01) 和 net[0].bias.data.fill_(0) 初始化权重和偏差。
# weight 和 bias 是 PyTorch 框架预定义的属性名。
net[0].weight.data.normal_(0, 0.01)
net[0].bias.data.fill_(0)

# 3.5. 定义损失函数
loss = nn.MSELoss()

# 3.6. 定义优化算法
trainer = torch.optim.SGD(net.parameters(), lr=0.03)

# 3.7. 训练
num_epochs = 3
for epoch in range(num_epochs):
    for X, y in data_iter:
        l = loss(net(X) ,y)
        trainer.zero_grad()
        l.backward()
        trainer.step()
    l = loss(net(features), labels)
    print(f'epoch {epoch + 1}, loss {l:f}')

    w = net[0].weight.data
print('w的估计误差：', true_w - w.reshape(true_w.shape))
b = net[0].bias.data
print('b的估计误差：', true_b - b)