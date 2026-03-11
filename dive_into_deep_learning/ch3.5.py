#3.5. 图像分类数据集
import torch
import matplotlib.pyplot as plt
import torchvision
from torch.utils import data
from torchvision import transforms
from d2l import torch as d2l
# d2l.use_svg_display()
# 1. 读取数据集
# 通过ToTensor实例将图像数据从PIL类型变换成32位浮点数格式，
# 并除以255使得所有像素的数值均在0～1之间
trans = transforms.ToTensor()
mnist_train = torchvision.datasets.FashionMNIST(
    root="../data", train=True, transform=trans, download=True)
mnist_test = torchvision.datasets.FashionMNIST(
    root="../data", train=False, transform=trans, download=True)
# 下载数据集，并将其存储在“../data”目录下，会自动检查，不会重复下载
# print(len(mnist_train), len(mnist_test))
# print(mnist_train[0][0].shape, mnist_train[0][1])
# 数字标签索引及其文本名称之间进行转换：把数字标签转换为文本标签
def get_fashion_mnist_labels(labels):  #@save
    """返回Fashion-MNIST数据集的文本标签"""
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]

# 绘制图像，可视化
def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5, filename=None):  #@save
    """绘制图像列表"""
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        if torch.is_tensor(img):
            # 图片张量
            ax.imshow(img.numpy(), cmap='gray')
        else:
            # PIL图片
            ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"图片已保存到: {filename}")
        plt.close()
    return axes

# X, y = next(iter(data.DataLoader(mnist_train, batch_size=18)))
# show_images(X.reshape(18, 28, 28), 2, 9, titles=get_fashion_mnist_labels(y), filename='fashion_mnist_samples.png');
# 2. 读取小批量
batch_size = 256

def get_dataloader_workers():  #@save
    """使用4个进程来读取数据"""
    return 4

train_iter = data.DataLoader(mnist_train, batch_size, shuffle=True,
                             num_workers=get_dataloader_workers())
timer = d2l.Timer()
if __name__ == '__main__':
    for X, y in train_iter:
        continue
    print(f'{timer.stop():.2f} sec')
# 3. 整合所有组件
# 4. 小结# 2. 读取小批量






