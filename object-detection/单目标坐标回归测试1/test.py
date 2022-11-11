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
import torch
import torchvision
from PIL.Image import Image
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from torch import nn

from data.MyData import data_ic
from models.net_resnet18 import net_resnet18
import numpy as np


def test(opt):
    # 定义设备
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    loss_fn = nn.MSELoss()
    mydata = data_ic('D:/work/files/data/test_yolo/ic')
    datasets_val = DataLoader(mydata, batch_size=1, shuffle=True)

    net = net_resnet18()
    path = 'run/train/weights/epoch=266-train_acc=1.0.pth'
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
