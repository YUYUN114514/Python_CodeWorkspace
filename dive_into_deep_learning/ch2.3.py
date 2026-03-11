import torch
x = torch.arange(12).reshape((3, 4))
y = x.sum(axis=0)  # 沿着行的方向求和，保持维度不变
print(x)
print(y)