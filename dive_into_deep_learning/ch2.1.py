#基础知识，数据操作
import torch
#创建一个长度为12的一维张量
x = torch.arange(12)
#print(x.numel())

X = x.reshape(-1, 4)
#print(X)
Z = torch.zeros((2, 3, 4))
#print(Z)
Y = X * torch.randn(3,4)
x = torch.tensor([1.0, 2, 4, 8])
y = torch.tensor([2, 2, 2, 2])
#print(x+y,x-y,x*y,x/y,x**y)
#print(torch.exp(x))
#这个函数指的是创建一个张量，其中包含从0到11的整数，并且每个整数都被转换为浮点数，然后再reshape成3行4列的矩阵
# 函数连结：torch.arange(12, dtype=torch.float32).reshape((3,4))
X = torch.arange(12, dtype=torch.float32).reshape((3,4))
Y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
Z = torch.cat((X, Y), dim=0), 
A = torch.cat((X, Y), dim=1)
# print(Z)
# print(A)  
# print(X == Y)
# print(X.sum())
# 广播机制，大小不同的张量运算时，会自动进行适当的扩展，使得它们具有相同的形状
a = torch.arange(3).reshape((3, 1))
b = torch.arange(2).reshape((1, 2))
c = a + b
# print(c)

#切片索引
x = torch.arange(12).reshape((3, 4))
print(x[-1], x[0], x[1], x[2])
x[0:2, :] = 100
print(x)

#节约内存
#用x += y来代替x = x + y，可以节约内存
#用x[:] = x + y来代替x = x + y，也可以节约内存
#用x.copy_(y)来代替x = y，可以节约内存

#转换python对象
# a = torch.tensor([3.5])
# a, a.item(), float(a), int(a)