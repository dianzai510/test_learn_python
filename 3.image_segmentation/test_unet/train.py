import argparse
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from data import data_seg, transform1, transform2, transform_val
from model import UNet
import torch.nn.functional as F
import datetime 


def train(opt):
    os.makedirs(opt.out_path, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    datasets_train = data_seg(opt.data_path_train, transform1, transform2)
    datasets_val = data_seg(opt.data_path_val, transform_val)

    dataloader_train = DataLoader(datasets_train, batch_size=opt.batch_size, shuffle=True, num_workers=1, drop_last=True)
    dataloader_val = DataLoader(datasets_val, batch_size=opt.batch_size, shuffle=True, num_workers=1, drop_last=True)

    net = UNet()
    net.to(device)

    loss_fn = nn.BCELoss()
    # loss_fn = dice_loss()

    optimizer = torch.optim.SGD(net.parameters(), lr=opt.lr, momentum=0.99)  # 定义优化器 momentum=0.99
    #optimizer = torch.optim.Adam(net.parameters(), opt.lr)  # 定义优化器 momentum=0.99

    # 学习率更新策略
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)

    #lf = lambda x: ((1 + math.cos(x * math.pi / opt.epoch)) / 2) * (1 - 0.2) + 0.2
    #scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    # 加载预训练模型
    loss_best = 9999
    path_best = os.path.join(opt.out_path,opt.weights)
    if os.path.exists(path_best):
        checkpoint = torch.load(path_best)
        net.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        time,epoch,loss = checkpoint['time'],checkpoint['epoch'],checkpoint['loss']
        loss_best = checkpoint['loss']
        print(f"{time}: best.pth, epoch: {epoch}, loss: {loss}")
    
    for epoch in range(1, opt.epoch):
        # 训练
        net.train()
        loss_train = 0
        for images, labels in dataloader_train:
            images = images.to(device)
            labels = labels.to(device)

            out = net(images)
            out = F.sigmoid(out) #bceloss，标签必须要在0到1之间,因此输出需要加个sigmoid
            loss = loss_fn(input=out, target=labels) #损失函数参数要分input和labels，反了计算值可能是nan 2023.2.24

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #scheduler.step()
            loss_train += loss

        # 验证
        net.eval()
        loss_val = 0
        with torch.no_grad():
            for images, labels in dataloader_val:
                images = images.to(device)
                labels = labels.to(device)
                out = net(images)
                out = F.sigmoid(out) #bceloss，标签必须要在0到1之间,因此输出需要加个sigmoid
                loss = loss_fn(input=out, target=labels) #损失函数参数要分input和labels，反了计算值可能是nan 2023.2.24
                loss_val += loss

        # 打印一轮的训练结果
        loss_train = loss_train / len(dataloader_train.dataset)
        loss_val = loss_val / len(dataloader_val.dataset)
        print(f"epoch:{epoch}, loss_train:{loss_train}, loss_val:{loss_val}, lr:{optimizer.param_groups[0]['lr']}")

        # 保存best.pth
        if loss_val < loss_best:
            loss_best = loss_val
            checkpoint = {'net': net.state_dict(),
                          'optimizer': optimizer.state_dict(),
                          'epoch': epoch,
                          'loss': loss_val.item(),
                          'time': datetime.date.today()}
            torch.save(checkpoint, path_best)
            print(f'已保存:{path_best}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', default='best.pth', help='指定权重文件，未指定则使用官方权重！')
    parser.add_argument('--out_path', default='./run/train', type=str)  # 修改
    parser.add_argument('--resume', default=False, type=bool, help='True表示从--weights参数指定的epoch开始训练,False从0开始')
    parser.add_argument('--data_path_train', default='D:/desktop/choujianji/roi/mask/train')  # 修改
    parser.add_argument('--data_path_val', default='D:/desktop/choujianji/roi/mask/val')  # 修改
    parser.add_argument('--epoch', default=1000, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--batch_size', default=10, type=int)

    opt = parser.parse_args()

    train(opt)
