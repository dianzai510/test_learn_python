"""
1、损失函数
2、优化器
3、训练过程
    训练
    验证
"""
import argparse
import os
import pathlib
import time
from datetime import datetime

import PIL
import cv2
import torch
import torchvision
from PIL.Image import Image
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from torch import nn

from data.MyData import data_ic
from models.net_resnet18 import net_resnet18
import numpy
import numpy as np


def test(opt):
    # 定义设备
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    loss_fn = nn.MSELoss()
    mydata = data_ic('C:/work/files/deeplearn_dataset/坐标回归测试/test')
    datasets_val = DataLoader(mydata, batch_size=1, shuffle=True)

    net = net_resnet18()
    path = './run/train/weights/epoch=189-train_acc=0.9998342990875244.pth'
    checkpoint = torch.load(path)
    net.load_state_dict(checkpoint['net'])
    net.to(device)

    for epoch in os.listdir():

        # 验证
        net.eval()
        total_val_loss = 0
        total_val_accuracy = 0
        with torch.no_grad():
            for imgs, labels in datasets_val:
                imgs = imgs.to(device)

                labels = labels.to(device)
                out = net(imgs)

                loss = loss_fn(out, labels)
                total_val_loss += loss

                acc = (out.argmax(1) == labels).sum()
                total_val_accuracy += acc

                img = imgs[0]
                shape = img.shape
                ww = shape[2]
                hh = shape[1]

                out = out.squeeze(dim=0)
                x0 = (out[0] * ww).item()
                y0 = (out[1] * hh).item()
                w = (out[2] * ww).item()
                h = (out[3] * hh).item()

                pt1 = (int(x0 - w / 2), int(y0 - h / 2))
                pt2 = (int(x0 + w / 2), int(y0 + h / 2))

                img = imgs[0].numpy() * 255  # type:numpy.ndarray
                img = img.swapaxes(0, 2)
                img = img.swapaxes(0, 1)
                img = img.astype(np.uint8)

                # image = np.zeros((512, 512, 3), dtype=np.uint8)
                # print(type(img), type(image))
                # cv2.rectangle(image, pt1, pt2, (0, 0, 255), 1, cv2.LINE_8, 3)  # 红
                img = img.copy()
                cv2.rectangle(img, pt1, pt2, (0, 0, 255), 2)
                cv2.imshow("dis", img)
                cv2.waitKey(1500)
                # img = torchvision.transforms.functional.to_pil_image(imgs[0])
                # Image.show(img)


        val_acc = total_val_accuracy / len(datasets_val)
        val_loss = total_val_loss

        print(f"epoch:{epoch}, val_acc={val_acc}, val_loss={val_loss}")

        pathlib.Path(f'{opt.model_save_path}/weights').mkdir(parents=True,
                                                             exist_ok=True)  # https://zhuanlan.zhihu.com/p/317254621


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
    parser.add_argument('--save_period', type=int, default=-1, help='Log model after every "save_period" epoch')
    parser.add_argument('--model_save_path', default='run/train', help='save to project/name')

    opt = parser.parse_args()
    test(opt)
