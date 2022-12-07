import os

import cv2
import torch
import torchvision
from PIL.Image import Image
from torch.utils.data import Dataset


# 将label转换为7x7x30的张量.
from object_detection.手写yolov1.utils import basic


class data(Dataset):
    def __init__(self, data_path, image_size=448, grid_size=7, num_bboxes=2, num_classes=20):
        self.image_size = image_size
        self.grid_size = grid_size
        self.num_bboxes = num_bboxes
        self.num_classes = num_classes

        self.Images = []
        self.Labels = []

        images_path = os.path.join(data_path, "Images")
        labels_path = os.path.join(data_path, "Labels")

        self.Images = [f for f in os.listdir(images_path) if f.endswith('.jpg') or f.endswith('.png')]
        self.Labels = [f for f in os.listdir(labels_path) if f.endswith('.txt')]

    def __len__(self):
        return len(self.Images)

    def __getitem__(self, item):
        image_path = self.Images[item]
        label_path = self.Labels[item]

        # region 1、读取图像
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        max_len = max(img.shape)
        padding_lr = (img.shape[1] - max_len) / 2
        padding_up = (img.shape[0] - max_len) / 2
        img = cv2.copyMakeBorder(padding_up, padding_up, padding_lr, padding_lr, cv2.BORDER_CONSTANT, value=0)
        img = cv2.resize(img, dsize=(self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        # endregion

        # region 2、读取标签
        labels = []
        # label = torch.zeros(self.grid_size, self.grid_size, (self.num_bboxes * (4 + 1) + self.num_classes))
        with open(label_path, 'r') as f:
            lines = f.readlines()
            lines = [l.split(' ') for l in lines]

            for data_list in lines:
                data_list = [float(f.strip()) for f in data_list]
                cls = data_list[0]
                x = int(data_list[1] * 7)  # 向下取整
                y = int(data_list[2] * 7)  # 向下取整
                w = data_list[3]
                h = data_list[4]

                # label = torch.tensor([x, y, w, h])
                labels.append(torch.tensor([cls, x, y, w, h]))
        labels = basic.encode(labels)
        # endregion

        return img, labels


if __name__ == '__main__':
    Images = [f for f in os.listdir('D:/下载/Rotated-RetinaNet-master') if f.endswith('.py')]

    a = torch.rand(3, 4)
    print(a)
    mask = torch.randint(2, (3, 4))
    print(mask)
    mask = mask == 1
    print(mask)

    print(a[mask])

    # expand(),expand_as()函数只能将size = 1的维度扩展到更大的尺寸，如果扩展其他size（）的维度会报错。
