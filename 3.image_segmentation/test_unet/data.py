from typing import Any
from torch.utils.data import Dataset, DataLoader
import torch
import os
import torchvision
from PIL import Image
import cv2
from torchvision.transforms import InterpolationMode
from our1314.myutils.ext_transform import Resize1, PadSquare
import numpy as np
import torchvision.transforms.functional as F
from our1314.myutils.ext_transform import *


# 数据增强的种类：1.平移、翻转、旋转、尺寸、仿射变换 2.亮度、颜色、噪声，其中1部分需要同时对图像和标签进行操作，2部分只对图像有效部分进行操作
#input_size = (448-32, 448-32)#图像尺寸应该为16的倍数
input_size = (448,448)
transform1 = torchvision.transforms.Compose([
    ToTensors(),
    Resize1(input_size[0]),#等比例缩放
    PadSquare(),
    randomaffine_imgs(0.5, [-5,5], [-0.1,0.1], [-0.1,0.1], [0.9,1/0.9]),
    randomvflip_imgs(0.5),
    randomhflip_imgs(0.5)
])

transform2 = torchvision.transforms.Compose([
    torchvision.transforms.GaussianBlur(kernel_size=(1, 13)),  # 随机高斯模糊
    torchvision.transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.3)
])

transform_val = torchvision.transforms.Compose([
    ToTensors(),
    Resize1(input_size[0]),  # 按比例缩放
    PadSquare()  # 四周补零
])

class data_seg(Dataset):
    def __init__(self, data_path, transform1=None, transform2=None):
        self.transform1 = transform1
        self.transform2 = transform2

        self.Images = [os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith('.jpg')]
        self.Labels = [os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith('.png')]

    def __len__(self):
        return len(self.Images)

    def __getitem__(self, item):
        '''
        由于输出的图片的尺寸不同，我们需要转换为相同大小的图片。首先转换为正方形图片，然后缩放的同样尺度(256*256)。
        否则dataloader会报错。
        '''
        image = Image.open(self.Images[item])
        label = Image.open(self.Labels[item])

        if self.transform1 != None:
            image,label = self.transform1([image,label])

        if self.transform2 != None:
            image = self.transform2(image)
        return image, label


if __name__ == '__main__':
    
    data = data_seg('D:/desktop/choujianji/roi/mask/train', transform1=transform1, transform2=transform2)
    data_loader = DataLoader(data, batch_size=1, shuffle=True)
    for image, label in data_loader:
        F.to_pil_image((0.7*image+0.3*label)[0]).show()
