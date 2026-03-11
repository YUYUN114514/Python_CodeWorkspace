# 数据预处理
import os
import pandas as pd
import torch
# 读取数据集

# 创建目录：在上级目录中创建名为 'data' 的文件夹
# exist_ok=True 表示如果目录已存在，不会报错
os.makedirs(os.path.join('..', 'data'), exist_ok=True)

# 定义数据文件路径：在上级目录的 data 文件夹下创建 house_tiny.csv 文件
# os.path.join 会自动处理不同操作系统的路径分隔符
data_file = os.path.join('..', 'data', 'house_tiny.csv')

# 以写入模式 ('w') 打开文件
# with 语句会自动管理文件资源，退出时自动关闭文件
with open(data_file, 'w') as f:
    # 写入CSV文件的第一行：列名（字段名）
    # 包含3个字段：房间数量、小巷类型、房价
    f.write('NumRooms,Alley,Price\n')  # 列名

    # 写入第1条数据：房间数缺失(NA)，小巷铺砖(Pave)，房价127500
    f.write('NA,Pave,127500\n')  # 每行表示一个数据样本

    # 写入第2条数据：2个房间，小巷类型缺失(NA)，房价106000
    f.write('2,NA,106000\n')

    # 写入第3条数据：4个房间，小巷类型缺失(NA)，房价178100
    f.write('4,NA,178100\n')

    # 写入第4条数据：房间数缺失(NA)，小巷类型缺失(NA)，房价140000
    f.write('NA,NA,140000\n')

# 提示文件创建成功
print(f"数据文件已创建: {data_file}")

# 使用 pandas 读取 CSV 文件
data = pd.read_csv(data_file)
print(data)  # 打印读取的数据

# ==================== 处理缺失值 ====================

# data.iloc 位置索引器，用于按位置选取数据
# 将 data 分成 inputs（特征）和 outputs（标签）
# inputs: 前两列（NumRooms, Alley），outputs: 最后一列（Price）
inputs, outputs = data.iloc[:, 0:2], data.iloc[:, 2]

# 对于数值型缺失值，用该列的均值填充
# 注意：只对数值列（exclude=['object']）计算均值，避免字符串列导致错误
# fillna() 填充缺失值，mean() 计算均值：只选择number类型的列进行均值计算
inputs = inputs.fillna(inputs.select_dtypes(include='number').mean())
print(inputs)

# 对于类别型数据（Alley），使用独热编码（One-Hot Encoding）
# pd.get_dummies 将类别变量转换为虚拟变量（0/1）
# dummy_na=True 表示将缺失值也作为一个类别进行编码
inputs = pd.get_dummies(inputs, dummy_na=True)
print(inputs)
#    NumRooms  Alley_Pave  Alley_nan
# 0       3.0        True      False
# 1       2.0       False       True
# 2       4.0       False       True
# 3       3.0       False       True
X = torch.tensor(inputs.to_numpy(dtype=float))
y = torch.tensor(outputs.to_numpy(dtype=float))

print(X)
print(y)