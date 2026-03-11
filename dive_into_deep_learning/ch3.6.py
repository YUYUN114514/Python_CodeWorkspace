# ch3.6.py: 使用 Softmax 回归实现 Fashion-MNIST 图像分类
# 本章实现了一个完整的神经网络训练流程，包括数据加载、模型定义、损失函数、训练和预测
import torch
from IPython import display
from d2l import torch as d2l

# ========== 数据加载 ==========
# d2l 里有直接写好的数据迭代器，直接调用就行了，只用指派 batch_size 的大小就行了
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)  # 返回训练集和测试集的迭代器

# ========== 模型参数配置 ==========
num_inputs = 784  # 输入特征数：28x28 图像展平后为 784 个像素
num_outputs = 10  # 输出类别数：Fashion-MNIST 有 10 个类别（T恤、裤子、裙子等）

class Accumulator:  #@save
    """在 n 个变量上累加
    
    用于在训练过程中累计多个指标，例如：
    - [正确预测数, 预测总数]
    - [损失总和, 正确预测数, 样本总数]
    """
    def __init__(self, n):
        """初始化累加器，创建 n 个初始值为 0 的累加项"""
        self.data = [0.0] * n

    def add(self, *args):
        """向累加器中添加多个值，逐元素累加"""
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        """重置所有累加项为 0"""
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        """通过索引获取累加器中指定位置的值"""
        return self.data[idx]
    

class Animator:  #@save
    """在动画中绘制数据"""
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                 figsize=(3.5, 2.5)):
        # 增量地绘制多条线
        if legend is None:
            legend = []
        # d2l.use_svg_display()  # 在非 Jupyter 环境中会报错，已注释
        self.fig, self.axes = d2l.plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        # 使用lambda函数捕获参数
        self.config_axes = lambda: d2l.set_axes(
            self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts

    def add(self, x, y):
        # 向图表中添加多个数据点
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        # display.display(self.fig)  # Jupyter 特有功能，已注释
        # display.clear_output(wait=True)  # Jupyter 特有功能，已注释
        d2l.plt.pause(0.1)  # 使用 matplotlib 的 pause 来更新图形

# ========== 模型参数初始化 ==========
W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
# 权重矩阵 W: 784x10，使用正态分布初始化，mean=0, std=0.01，需要计算梯度
b = torch.zeros(num_outputs, requires_grad=True)
# 偏置向量 b: 10，初始化为 0，需要计算梯度

# ========== Softmax 激活函数 ==========
def softmax(X):
    """Softmax 函数：将线性输出转换为概率分布
    
    对输入的每一行进行 softmax 变换，输出和为 1 的概率分布
    
    Args:
        X: 输入张量，形状为 (batch_size, num_classes)
    
    Returns:
        概率分布张量，形状与 X 相同，每行和为 1
    """
    X_exp = torch.exp(X)  # 步骤1：对每个元素求指数 e^x
    partition = X_exp.sum(1, keepdim=True)  # 步骤2：对每一行求和，保持维度
    return X_exp / partition  # 步骤3：归一化，广播机制确保维度正确

# ========== 神经网络模型定义 ==========
# 按 w 的大小 784*10，重新定义图像尺寸，变成 batch_size x 784
# 这种就是按纯线性计算的角度去定义模型，3.3 节那个是全连接网络的简洁表达
def net(X):
    """Softmax 回归模型（单层神经网络）
    
    前向传播：输入 -> 线性变换 -> softmax -> 输出概率
    
    Args:
        X: 输入图像张量，形状为 (batch_size, 28, 28)
    
    Returns:
        预测概率，形状为 (batch_size, 10)
    """
    return softmax(torch.matmul(X.reshape((-1, W.shape[0])), W) + b)

# ========== 交叉熵损失函数示例 ==========
y = torch.tensor([0, 2])  # 真实标签：第一个样本是第0类，第二个是第2类
y_hat = torch.tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])  # 预测概率
y_hat[[0, 1], y]  # 高级索引：取 [y_hat[0,0], y_hat[1,2]] = [0.1, 0.5]
# 这一步是交叉熵损失计算的关键：提取真实标签对应的预测概率

# 交叉熵损失函数：高级索引，标签的绝对值都是 1，所以计算公式里省略了
def cross_entropy(y_hat, y):
    """交叉熵损失函数
    
    计算预测分布 y_hat 与真实标签 y 之间的交叉熵损失
    Loss = -log(y_hat[真实标签对应的概率])
    
    Args:
        y_hat: 预测概率，形状为 (batch_size, num_classes)
        y: 真实标签，形状为 (batch_size,)
    
    Returns:
        交叉熵损失张量，形状为 (batch_size,)
    """
    return - torch.log(y_hat[range(len(y_hat)), y])  # 提取真实标签对应的概率并取负对数

cross_entropy(y_hat, y)  # 计算示例损失

# ========== 精度计算 ==========
def accuracy(y_hat, y):  #@save
    """计算预测正确的数量
    
    Args:
        y_hat: 预测结果，可以是概率分布 (batch_size, num_classes) 或类别索引 (batch_size,)
        y: 真实标签，形状为 (batch_size,)
    
    Returns:
        预测正确的样本数（浮点数）
    """
    #检查维数和第二维的大小
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        # 执行 argmax(axis=1)
        # axis=0：按列找最大值（纵向）
        # axis=1：按行找最大值（横向）← 我们用的是这个
        y_hat = y_hat.argmax(axis=1)  # 如果是概率分布，取概率最大的类别作为预测
    cmp = y_hat.type(y.dtype) == y  # 比较预测与真实标签，得到布尔张量
    return float(cmp.type(y.dtype).sum())  # 统计 True 的数量，即正确预测数

def evaluate_accuracy(net, data_iter):  #@save
    """计算在指定数据集上模型的精度
    
    Args:
        net: 神经网络模型
        data_iter: 数据迭代器
    
    Returns:
        模型在该数据集上的准确率（0到1之间的浮点数）
    """
    if isinstance(net, torch.nn.Module):
        net.eval()  # 将模型设置为评估模式（不计算梯度，关闭 Dropout 等）
    metric = Accumulator(2)  # 累加器：[正确预测数, 预测总数]
    with torch.no_grad():  # 不计算梯度，节省内存
        for X, y in data_iter:
            metric.add(accuracy(net(X), y), y.numel())  # 累加正确数和总样本数
    return metric[0] / metric[1]  # 返回准确率 = 正确数 / 总数


def train_epoch_ch3(net, train_iter, loss, updater):  #@save
    """训练模型一个迭代周期（定义见第3章）"""
    # 将模型设置为训练模式
    if isinstance(net, torch.nn.Module):
        net.train()
    # 训练损失总和、训练准确度总和、样本数
    metric = Accumulator(3)
    for X, y in train_iter:
        # 计算梯度并更新参数
        y_hat = net(X)
        l = loss(y_hat, y)
        if isinstance(updater, torch.optim.Optimizer):
            # 使用PyTorch内置的优化器和损失函数
            updater.zero_grad()
            l.mean().backward()
            updater.step()
        else:
            # 使用定制的优化器和损失函数
            l.sum().backward()
            updater(X.shape[0])
        metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
    # 返回训练损失和训练精度
    return metric[0] / metric[2], metric[1] / metric[2]


def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):  #@save
    """训练模型（定义见第3章）"""
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
                        legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        train_metrics = train_epoch_ch3(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)
        animator.add(epoch + 1, train_metrics + (test_acc,))
    train_loss, train_acc = train_metrics
    assert train_loss < 0.5, train_loss
    assert train_acc <= 1 and train_acc > 0.7, train_acc
    assert test_acc <= 1 and test_acc > 0.7, test_acc
def predict_ch3(net, test_iter, n=6, save_path='predict_result.png'):  #@save
    """预测标签并保存结果图片

    从测试集中取一批数据，预测其类别，并与真实标签对比显示

    Args:
        net: 训练好的神经网络模型
        test_iter: 测试数据迭代器
        n: 要显示的图像数量
        save_path: 保存图片的路径
    """
    for X, y in test_iter:
        break  # 取第一个批次
    trues = d2l.get_fashion_mnist_labels(y)  # 获取真实标签的名称
    preds = d2l.get_fashion_mnist_labels(net(X).argmax(axis=1))  # 获取预测标签的名称
    titles = [true +'\n' + pred for true, pred in zip(trues, preds)]  # 组合标题（真实/预测）

    # 创建图形并保存图片
    fig, ax = d2l.plt.subplots(1, n, figsize=(n * 2, 2.5))  # 创建子图
    for i in range(n):
        ax[i].imshow(X[i].reshape(28, 28), cmap='gray')  # 显示灰度图
        ax[i].set_title(titles[i])  # 设置标题
        ax[i].axis('off')  # 隐藏坐标轴
    d2l.plt.tight_layout()  # 调整布局
    d2l.plt.savefig(save_path, dpi=150, bbox_inches='tight')  # 保存图片到文件
    d2l.plt.close()  # 关闭图形释放内存
    print(f"预测结果已保存到: {save_path}")


if __name__ == '__main__':  # Windows 多进程必须添加此保护
    evaluate_accuracy(net, test_iter)  # 在测试集上评估初始精度（应该接近 0.1）

    # ========== 启动训练 ==========
    lr = 0.1  # 学习率：控制参数更新的步长

    def updater(batch_size):
        """自定义的 SGD 参数更新器

        Args:
            batch_size: 批次大小，用于归一化梯度

        Returns:
            一个函数，调用该函数会执行一次 SGD 参数更新
        """
        return d2l.sgd([W, b], lr, batch_size)

    num_epochs = 10  # 训练轮数
    train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, updater)

    # ========== 预测与可视化 ==========
    predict_ch3(net, test_iter)  # 执行预测并显示结果
# ========== 预测与可视化函数 ==========



# ========== 主程序入口 ==========
