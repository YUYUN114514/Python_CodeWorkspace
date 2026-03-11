# coding: utf-8
import sys, os
# 获取当前文件的上级目录（ch03的路径）
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取根目录（ch03的上层目录）
root_dir = os.path.dirname(current_dir)
# 将根目录添加到Python环境变量
sys.path.append(root_dir)
sys.path.append(os.pardir)  # 为了导入父目录的文件而进行的设定
import numpy as np
from dataset.mnist import load_mnist
from PIL import Image


def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()

(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)

img = x_train[0]
label = t_train[0]
print(label)  # 5

print(img.shape)  # (784,)
img = img.reshape(28, 28)  # 把图像的形状变为原来的尺寸
print(img.shape)  # (28, 28)

img_show(img)
