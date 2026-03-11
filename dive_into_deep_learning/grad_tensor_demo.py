import torch

# ==================== 问题：向量不能直接 backward() ====================
x = torch.arange(4.0, requires_grad=True)
y = x * x  # y = [0, 1, 4, 9] 是向量

# ❌ 错误：向量不能直接 backward()
# y.backward()  # 会报错

# ==================== 方法1：求和后反向传播 ====================
x.grad.zero_()
y = x * x
y.sum().backward()
print("方法1（求和）:")
print(f"x.grad = {x.grad}")  # dy/dx = 2x = [0, 2, 4, 6]
print(f"相当于计算: d(x1² + x2² + x3² + x4²)/dx")

# ==================== 方法2：传入 grad_tensor（权重向量） ====================
x.grad.zero_()
y = x * x

# grad_tensor 是权重向量，用于加权求和
# v = [0, 0, 0, 1] 表示只对第4个元素求导
v = torch.tensor([0., 0., 0., 1.])
y.backward(v)
print("\n方法2（grad_tensor）:")
print(f"grad_tensor v = {v}")
print(f"x.grad = {x.grad}")  # dy/dx = 2x * v = [0, 0, 0, 6]
print(f"相当于计算: d(0*y1 + 0*y2 + 0*y3 + 1*y4)/dx = d(y4)/dx")

# ==================== 数学原理 ====================
print("\n--- 数学原理 ---")
print("当输出是向量 y = [y1, y2, y3, y4] 时：")
print("  直接 backward() 无法计算，因为不知道要算哪个元素的梯度")
print("")
print("y.backward(v) 相当于计算：")
print("  ∂(v[0]*y1 + v[1]*y2 + v[2]*y3 + v[3]*y4) / ∂x")
print("")
print("如果 v = [1, 1, 1, 1]，则等价于 y.sum().backward()")
print("如果 v = [0, 0, 0, 1]，则等价于只计算 y4 对 x 的导数")

# ==================== 实际应用场景 ====================
print("\n--- 实际应用 ---")
print("场景1：只想计算某个输出的梯度")
print("  grad_tensor = [0, 1, 0, 0]  # 只计算第2个元素的梯度")
print("")
print("场景2：加权损失函数")
print("  grad_tensor = [0.1, 0.2, 0.3, 0.4]  # 不同样本有不同权重")
print("")
print("场景3：最常见：所有权重为1（等价于求和）")
print("  y.sum().backward()  # grad_tensor = [1, 1, 1, 1]")
