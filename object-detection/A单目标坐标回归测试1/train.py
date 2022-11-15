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

    mydata = data_ic('C:/work/files/deeplearn_dataset/坐标回归测试/train')
    datasets_train = DataLoader(mydata, batch_size=5, shuffle=True)
    datasets_val = DataLoader(mydata, batch_size=5, shuffle=True)

    # 训练轮数
    epoch_count = 500
    net = net_resnet18()  # classify_net1()
    # if opt.model != '':
    #     checkpoint = torch.load(opt.model)
    #     net.load_state_dict(checkpoint['net'])

    net.to(device)
    # loss_fn = nn.CrossEntropyLoss()
    loss_fn = nn.MSELoss()

    # optimizer = torch.optim.SGD(net.parameters(), lr=0.1)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    writer = SummaryWriter(f"{opt.model_save_path}/logs")
    x = torch.tensor(np.random.randn(1, 3, 256, 306), dtype=torch.float)  # type:torch.Tensor
    x = x.to(device)
    writer.add_graph(net, x)
    start_epoch = 0
    if opt.resume:
        '''
        断点继续参考：
        https://www.zhihu.com/question/482169025/answer/2081124014
        '''
        lists = os.listdir(f"{opt.model_save_path}/weights")  # 获取模型路径下的模型文件
        if len(lists) > 0:
            lists.sort(key=lambda fn: os.path.getmtime(f"{opt.model_save_path}/weights" + "\\" + fn))  # 按时间排序
            last_pt_path = os.path.join(f"{opt.model_save_path}/weights", lists[len(lists) - 1])
            #checkpoint = torch.load(last_pt_path)
            checkpoint = torch.load(opt.model)
            start_epoch = checkpoint['epoch']
            net.load_state_dict(checkpoint['net'])
            optimizer.load_state_dict(checkpoint['optimizer'])

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

            # region 显示训练图像
            # img1 = imgs[0, :, :, :]
            # img2 = imgs[1, :, :, :]
            # img1 = torchvision.transforms.ToPILImage()(img1)
            # img1.show()
            # img2 = torchvision.transforms.ToPILImage()(img2)
            # img2.show()
            # endregion

            # if epoch < 2:  # 保存两轮的训练图像
            #     for i in range(imgs.shape[0]):
            #         img1 = imgs[i, :, :, :]
            #         img1 = torchvision.transforms.ToPILImage()(img1)
            #         img1.save(f'{opt.model_save_path}/{datetime.now().strftime("%Y.%m.%d_%H.%M.%S.%f")}.png', 'png')

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

                # region 显示训练图像
                # img1 = imgs[0, :, :, :]
                # img2 = imgs[1, :, :, :]
                # img1 = torchvision.transforms.ToPILImage()(img1)
                # img1.show()
                # img2 = torchvision.transforms.ToPILImage()(img2)
                # img2.show()
                # endregion

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
                      'optimizer': optimizer.state_dict(),
                      'epoch': epoch}

        pathlib.Path(f'{opt.model_save_path}/weights').mkdir(parents=True,
                                                             exist_ok=True)  # https://zhuanlan.zhihu.com/p/317254621
        f = f'{opt.model_save_path}/weights/epoch={epoch}-train_acc={str(train_acc.item())}.pth'
        torch.save(state_dict, f)
        print(f"第{epoch}轮模型参数已保存")

        # 保存TensorBoard日志

        # 保存
        # net.state_dict()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', nargs='?', default='run/train/weights/best.pth', help='')
    parser.add_argument('--resume', nargs='?', const=True, default=True, help='resume most recent training')
    parser.add_argument('--save_period', type=int, default=-1, help='Log model after every "save_period" epoch')
    parser.add_argument('--model_save_path', default='run/train', help='save to project/name')

    opt = parser.parse_args()
    train(opt)
