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

# 数据增强的种类：1.平移、翻转、旋转、透视变换 2.颜色、噪声，其中1部分需要同事对图像和标签进行操作，2部分只对图像有效部分进行操作
input_size = (448, 448)#图像尺寸应该为16的倍数

transform_basic = [
    Resize1(input_size),# 按比例缩放
    PadSquare(),# 填充为正方形
    torchvision.transforms.ToTensor(),
    torchvision.transforms.RandomVerticalFlip(0.5),
    torchvision.transforms.RandomHorizontalFlip(0.5),

    torchvision.transforms.RandomRotation(90, interpolation=InterpolationMode.NEAREST)
    # torchvision.transforms.RandomRotation(90, expand=False, interpolation=InterpolationMode.BILINEAR),
    # torchvision.transforms.CenterCrop(input_size),
]
torchvision.transforms.ToPILImage
transform_advan = [
    # torchvision.transforms.Pad(300, padding_mode='symmetric'),
    torchvision.transforms.GaussianBlur(kernel_size=(3, 7)),  # 随机高斯模糊
    torchvision.transforms.ColorJitter(brightness=(0.5, 1.5))
    # , contrast=(0.8, 1.2), saturation=(0.8, 1.2)),  # 亮度、对比度、饱和度
    # torchvision.transforms.ToTensor()
]

trans_train_mask = torchvision.transforms.Compose(transform_basic)
trans_train_image = torchvision.transforms.Compose(transform_basic + transform_advan)


transform_val = torchvision.transforms.Compose([
    Resize1(300),  # 按比例缩放
    PadSquare(),  # 四周补零
    torchvision.transforms.ToTensor()])


class data_seg(Dataset):
    def __init__(self, data_path, transform_image=None, transform_mask=None):
        self.transform_image = transform_image
        self.transform_mask = transform_mask

        self.Images = [os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith('.jpg')]
        self.Labels = [os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith('.png')]

    def __len__(self):
        return len(self.Images)

    def __getitem__(self, item):
        '''
        由于输出的图片的尺寸不同，我们需要转换为相同大小的图片。首先转换为正方形图片，然后缩放的同样尺度(256*256)。
        否则dataloader会报错。
        '''

        # 取出图片路径
        image_path = self.Images[item]
        label_path = self.Labels[item]

        image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED) # type:cv2.Mat
        label = cv2.imdecode(np.fromfile(label_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED) # type:cv2.Mat

        h, w, _ = image.shape
        scale = min(input_size[0]/w, input_size[1]/h)
    
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        label = cv2.resize(label, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)

        h, w, _ = image.shape
        left = (input_size[0] - w)//2
        right = input_size[0] - w - left
        top = (input_size[1] - h)//2
        bottom = input_size[1] - h - top 
    
        image = cv2.copyMakeBorder(image, top, bottom, left, right, borderType=cv2.BORDER_CONSTANT, value=0)
        label = cv2.copyMakeBorder(label, top, bottom, left, right, borderType=cv2.BORDER_CONSTANT, value=0)

        image = F.ToTensor(image)
        label = F.ToTensor(label)

        return image, label


if __name__ == '__main__':
    data = data_seg('D:/desktop/choujianji/roi/mask', transform_image=trans_train_image, transform_mask=trans_train_mask)

    data_loader = DataLoader(data, batch_size=1, shuffle=True)
    for image, label in data_loader:
        torchvision.transforms.ToPILImage()(image[0]*label[0]).show()
