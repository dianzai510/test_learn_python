import argparse
import os.path
import pathlib
from datetime import datetime

import numpy as np
import torch
import torchvision.transforms
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from img_cls.FCN图像分类.data import data_xray
from img_cls.FCN图像分类.models.net_xray import net_xray
from img_cls.FCN图像分类.utils import utils


def train(opt):
    # 定义设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 训练轮数
    epoch_count = opt.epoch
    # 网络
    net = net_xray(True)  # 加载官方预训练权重

    # 初始化网络权重
    if opt.weights != "":
        checkpoint = torch.load(opt.weights)
        net.load_state_dict(checkpoint['net'])  # 加载checkpoint的网络权重

    net.to(device)
    loss_fn = nn.CrossEntropyLoss()  # 定义损失函数
    optimizer = torch.optim.SGD(net.parameters(), lr=opt.lr)  # 定义优化器
    # optimizer = torch.optim.Adam(net.parameters(), lr=opt.lr)

    start_epoch = 0
    if opt.resume:
        start_epoch = checkpoint['epoch']  # 加载checkpoint的优化器epoch
        optimizer.load_state_dict(checkpoint['optimizer'])  # 加载checkpoint的优化器

    # 初始化TensorBoard
    writer = SummaryWriter(f"{opt.out_path}/logs")

    # 绘制网络图
    if opt.add_graph:
        x = torch.tensor(np.random.randn(1, 3, 110, 310), dtype=torch.float)
        x = x.to(device)
        writer.add_graph(net, x)

    # 加载数据集
    dataloader_train = DataLoader(data_xray.datasets_train, 10, shuffle=True)
    dataloader_val = DataLoader(data_xray.datasets_val, 4, shuffle=True)

    print(f"训练集的数量：{len(data_xray.datasets_train)}")
    print(f"验证集的数量：{len(data_xray.datasets_val)}")
    cnt = 0
    for epoch in range(start_epoch, epoch_count):
        print(f"----第{epoch}轮训练----")

        # 训练
        net.train()
        acc_train = 0
        loss_train = 0

        for imgs, labels in dataloader_train:
            imgs = imgs.to(device)
            labels = labels.to(device)

            out = net(imgs)
            loss = loss_fn(out, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            acc = (out.argmax(1) == labels).sum()
            acc_train += acc
            loss_train += loss

            # region 保存指定数量的训练图像
            path = f'{opt.out_path}/img'
            img_count = len(os.listdir(path))
            if img_count < opt.train_img:
                cnt += 1
                path = f'{opt.out_path}/img'
                if os.path.exists(path) is not True:
                    os.makedirs(path)

                for i in range(imgs.shape[0]):
                    img = imgs[i, :, :, :]
                    img = torchvision.transforms.ToPILImage()(img)
                    img.save(f'{opt.out_path}/img/{datetime.now().strftime("%Y.%m.%d_%H.%M.%S.%f")}.png', 'png')
            # endregion

        # 验证
        net.eval()
        acc_val = 0
        loss_val = 0
        with torch.no_grad():
            for imgs, labels in dataloader_val:
                imgs = imgs.to(device)
                labels = labels.to(device)

                out = net(imgs)
                loss = loss_fn(out, labels)
                acc = (out.argmax(1) == labels).sum()
                acc_val += acc
                loss_val += loss
                # region 保存验证失败的图像
                # endregion

        # 打印一轮的训练结果
        mean_acc_train = acc_train / len(data_xray.datasets_train)
        mean_loss_train = loss_train
        mean_acc_val = acc_val / len(data_xray.datasets_val)
        mean_loss_val = loss_val
        print(f"epoch:{epoch}, "
              f"acc_train:{mean_acc_train}({acc_train}/{len(data_xray.datasets_train)})",
              f"loss_train:{mean_loss_train}",
              f"acc_val:{mean_acc_val}({acc_val}/{len(data_xray.datasets_val)})",
              f"loss_val:{mean_loss_val}")

        writer.add_scalar("acc_train", mean_acc_train, epoch)
        writer.add_scalar("loss_train", mean_loss_train, epoch)
        writer.add_scalar("acc_val", mean_acc_val, epoch)
        writer.add_scalar("loss_val", mean_loss_val, epoch)

        # region 保存模型
        # 保存best
        best_path = f'{opt.out_path}/weights/best.pth'
        f = best_path if os.path.exists(best_path) else utils.getlastfile(opt.out_path + '/' + 'weights')
        if f is not None:
            checkpoint = torch.load(f)
            acc_last = checkpoint['acc']
            loss_last = checkpoint['loss']
            if mean_acc_train >= acc_last and mean_loss_train < loss_last:
                # 保存训练模型
                checkpoint = {'net': net.state_dict(),
                              'optimizer': optimizer.state_dict(),
                              'epoch': epoch,
                              'acc': mean_acc_train,
                              'loss': mean_loss_train}
                torch.save(checkpoint, f'{opt.out_path}/weights/best.pth')
                print(
                    f"epoch{epoch}已保存为best.pth,acc:{mean_acc_train}({acc_train}/{len(data_xray.datasets_train)}),loss:{mean_loss_train}")

        # 按周期保存模型
        if epoch % opt.save_period == 1:
            # 创建目录
            pathlib.Path(f'{opt.out_path}/weights').mkdir(parents=True, exist_ok=True)
            # 保存训练模型
            checkpoint = {'net': net.state_dict(),
                          'optimizer': optimizer.state_dict(),
                          'epoch': epoch,
                          'acc': mean_acc_train,
                          'loss': mean_loss_train}
            torch.save(checkpoint, f'{opt.out_path}/weights/epoch={epoch}.pth')
            print(f"第{epoch}轮模型参数已保存")
        # endregion


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', default='run/train/exp/weights/best.pth',
                        help='指定权重文件，未指定则使用官方权重！')
    parser.add_argument('--resume', default=False, type=bool,
                        help='True表示从--weights参数指定的epoch开始训练,False从0开始')

    parser.add_argument('--epoch', default='300', type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--out_path', default='run/train/exp', type=str)
    parser.add_argument('--add_graph', default=False, type=bool)
    parser.add_argument('--save_period', default=5, type=int, help='多少轮保存一次，')
    parser.add_argument('--train_img', default=100, type=int, help='保存指定数量的训练图像')

    opt = parser.parse_args()
    train(opt)
