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


def train(opt):
    # 定义设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")

    # 读取数据
    mydata = data_ic('d:/work/files/deeplearn_datasets/test_datasets/单目标回归测试/train')
    datasets_train = DataLoader(mydata, batch_size=5, shuffle=True)
    datasets_val = DataLoader(mydata, batch_size=5, shuffle=True)

    # 训练轮数
    epoch_count = 500
    net = net_resnet18()
    net.to(device)
    loss_fn = nn.MSELoss()

    # optimizer = torch.optim.SGD(net.parameters(), lr=0.1)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    writer = SummaryWriter(f"{opt.model_save_path}/logs")

    start_epoch = 0
    print(f"训练集的数量:{len(datasets_train)}")
    print(f"验证集的数量:{len(datasets_val)}")

    for epoch in range(start_epoch, epoch_count):
        print(f"----第{epoch}轮训练开始----")

        # 训练
        net.train()
        total_train_accuracy = 0
        total_train_loss = 0

        for imgs, labels in datasets_train:
            imgs = imgs.to(device)
            labels = labels.to(device)

            out = net(imgs)
            loss = loss_fn(out, labels)
            # 优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            acc = loss
            total_train_accuracy += acc
            total_train_loss += loss

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

                acc = loss
                total_val_accuracy += acc

        train_acc = 1 - total_train_accuracy / len(datasets_train)
        train_loss = total_train_loss
        val_acc = total_val_accuracy / len(datasets_val)
        val_loss = total_val_loss

        print(f"epoch:{epoch}, train_acc={train_acc}, train_loss={train_loss}, val_acc={val_acc}, val_loss={val_loss}")

        writer.add_scalar("train_acc", train_acc, epoch)
        writer.add_scalar("train_loss", train_loss, epoch)
        writer.add_scalar("val_acc", val_acc, epoch)
        writer.add_scalar("val_loss", val_loss, epoch)

        # 保存训练模型
        state_dict = {'net': net.state_dict(),
                      # 'optimizer': optimizer.state_dict(),# 不保存优化器权重文件体积非常小，可以上传至github
                      'epoch': epoch}

        pathlib.Path(f'{opt.model_save_path}/weights').mkdir(parents=True,
                                                             exist_ok=True)  # https://zhuanlan.zhihu.com/p/317254621
        f = f'{opt.model_save_path}/weights/epoch={epoch}-train_acc={str(train_acc.item())}.pth'
        torch.save(state_dict, f)
        print(f"第{epoch}轮模型参数已保存")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', nargs='?', default='run/train/weights/best.pth', help='')
    parser.add_argument('--resume', nargs='?', const=True, default=True, help='resume most recent training')
    parser.add_argument('--save_period', type=int, default=-1, help='Log model after every "save_period" epoch')
    parser.add_argument('--model_save_path', default='run/train', help='save to project/name')

    opt = parser.parse_args()
    train(opt)
