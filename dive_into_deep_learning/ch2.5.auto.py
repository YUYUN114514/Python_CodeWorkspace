import torch
x = torch.arange(4.0)
x.requires_grad_(True)  # 需要计算梯度,因此分配内存计算梯度
print(x.grad)  # None, 因为还没有计算梯度
y = 2 * torch.dot(x, x) 
y.backward()  # 计算梯度，构造计算图，追踪所有相关变量的梯度  
print(x.grad)  # 输出梯度, dy/dx = 4x

x.grad.zero_()  # 将梯度清零，准备下一次计算
y = x.sum()  # y = x1 + x2 + x3 + x4
y.backward()  # 计算梯度，构造计算图，追踪所有相关变量的梯度
print(x.grad)  # 输出梯度, dy/dx = 1

x.grad.zero_()
y = x * x
# 等价于y.backward(torch.ones(len(x)))
# 当被求梯度变量不为标量时，需要指定梯度，否则会报错,就时是指定权重，把向量这几个数变成标量，然后求和
y.sum().backward()
print(x.grad)
