import argparse
import os
import sys

import torch
from torch import nn
from torch.utils.data import DataLoader
from image_segmentation.test_unet2.data import data_seg, SEGData
from image_segmentation.test_unet2.model import UNet


def train(opt):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    datasets_train = data_seg('D:/work/files/deeplearn_datasets/test_datasets/xray_real')
    datasets_val = data_seg('D:/work/files/deeplearn_datasets/test_datasets/xray_real')

    dataloader_train = DataLoader(datasets_train, batch_size=4, shuffle=True, num_workers=1, drop_last=True)
    dataloader_val = DataLoader(datasets_val, batch_size=4, shuffle=True, num_workers=1, drop_last=True)

    net = UNet()
    net.to(device)

    loss_fn = nn.BCELoss()
    # optimizer = torch.optim.SGD(net.parameters(), lr=0.001)  # 定义优化器 momentum=0.99
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01)  # 定义优化器 momentum=0.99

    # 加载预训练模型
    if os.path.exists(opt.weights):
        checkpoint = torch.load(opt.weights)
        net.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])

    cnt = 0
    for epoch in range(0, 1000):
        print(f"----第{epoch}轮训练----")

        # 训练
        net.train()
        loss_train = 0

        for images, labels in dataloader_train:
            images = images.to(device)
            labels = labels.to(device)

            out = net(images)
            loss = loss_fn(input=out, target=labels)  # 损失函数参数要分input和labels，反了计算值可能时nan 2023.2.24
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_train += loss

        # 打印一轮的训练结果
        mean_loss_train = loss_train

        result_epoch_str = f"epoch:{epoch}, " \
                           f"loss_train:{mean_loss_train}"

        print(f"{result_epoch_str}\n")

        # if epoch % 10 == 0:
        #     checkpoint = {'net': net.state_dict(),
        #                   'optimizer': optimizer.state_dict(),
        #                   'epoch': epoch,
        #                   'loss': mean_loss_train}
        #     torch.save(checkpoint, f'./epoch={epoch}.pth')

        path_best = './run/best.pth'
        if os.path.exists(path_best):
            checkpoint = torch.load(path_best)
            if mean_loss_train < checkpoint['loss']:
                checkpoint = {'net': net.state_dict(),
                              'optimizer': optimizer.state_dict(),
                              'epoch': epoch,
                              'loss': mean_loss_train}
                torch.save(checkpoint, path_best)
        else:
            checkpoint = {'net': net.state_dict(),
                          'optimizer': optimizer.state_dict(),
                          'epoch': epoch,
                          'loss': mean_loss_train}
            torch.save(checkpoint, path_best)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', default='./run/best.pth',
                        help='指定权重文件，未指定则使用官方权重！')
    parser.add_argument('--resume', default=False, type=bool,
                        help='True表示从--weights参数指定的epoch开始训练,False从0开始')
    parser.add_argument('--data', default='D:/work/files/deeplearn_datasets/test_datasets/xray_real')  # 修改
    parser.add_argument('--epoch', default=400, type=int)
    parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--batch_size', default=60, type=int)
    parser.add_argument('--out_path', default='run/train/exp_xray_sot23e', type=str)  # 修改
    parser.add_argument('--add_graph', default=False, type=bool)
    parser.add_argument('--save_period', default=20, type=int, help='多少轮保存一次，')
    parser.add_argument('--train_img', default=200, type=int, help='保存指定数量的训练图像')

    opt = parser.parse_args()
    train(opt)
