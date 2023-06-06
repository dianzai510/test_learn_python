import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import torch
import torchvision
from torch.utils.data import Dataset

import myutils.myutils
from myutils.myutils import yolostr2data
from object_detection.A单目标坐标回归测试1.data.data_xray_毛刺 import data_xray_毛刺


class data1(Dataset):
    def __init__(self, data_path):
        self.Images = [os.path.join(data_path, f) for f in os.listdir(data_path) if
                       f.endswith(".png") or f.endswith(".jpg") or f.endswith(".bmp")]  # 列表解析
        # self.transform = transform

    def __len__(self):
        return len(self.Images)

    def __getitem__(self, item):
        image_path = self.Images[item]

        # 读取图像
        img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), -1)  # type:cv2.Mat
        img = cv2.resize(img, (256, 256))
        img_tensor = myutils.myutils.mat2tensor(img)
        return img_tensor, img_tensor


if __name__ == '__main__':
    transf = torchvision.transforms.ToTensor()
    mydata = data_xray_毛刺('D:\desktop\XRay毛刺检测\TO252样品图片\TO252编带好品\ROI\out1/train', transf)
    # im = Image.open(data[0])
    # img.show(img)

    for img, pos in mydata:
        dis = myutils.myutils.tensor2mat(img)
        H, W, CH = dis.shape
        x0, y0, w, h = pos
        x0 *= W
        y0 *= H
        w *= W
        h *= H

        pt1 = (int(x0 - w / 2), int(y0 - h / 2))
        pt2 = (int(x0 + w / 2), int(y0 + h / 2))
        cv2.rectangle(dis, pt1, pt2, (0, 0, 255), 1)
        cv2.imshow("dis", dis)
        cv2.waitKey(500)
