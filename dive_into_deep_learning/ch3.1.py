import math
import time
import numpy as np
import torch
from d2l import torch as d2l
import matplotlib.pyplot as plt
# 线性模型
# 损失函数
# 解析解(对问题有限制)
# 随机梯度下降(优化与泛化能力)
# batch加速
# 正态分布
def normal(x, mu, sigma):
    p = 1 / math.sqrt(2 * math.pi * sigma**2)
    return p * np.exp(-0.5 / sigma**2 * (x - mu)**2)

x = np.arange(-7, 7, 0.01)

# 均值和标准差对
params = [(0, 1), (0, 2), (3, 1)]

# 使用 matplotlib 绘制并保存图片
fig, ax = plt.subplots(figsize=(4.5, 2.5))
for mu, sigma in params:
    ax.plot(x, normal(x, mu, sigma), label=f'mean {mu}, std {sigma}')
ax.set_xlabel('x')
ax.set_ylabel('p(x)')
ax.legend()
plt.savefig('normal_distribution.png', dpi=150)
print("图片已保存到 normal_distribution.png")