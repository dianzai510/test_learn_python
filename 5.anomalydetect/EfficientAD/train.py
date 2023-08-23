import argparse
import os
from data import DatasetSplit,IMAGENET_MEAN,IMAGENET_STD
import torch
from torch import nn
from torch.utils.data import DataLoader
from data import CJJDataset
from model import Teacher, Student, AutoEncoder
import datetime 
import random
import numpy as np
import cv2
import itertools
from tqdm import tqdm


def set_seeds(seed, with_torch=True, with_cuda=True):
    """Fixed available seeds for reproducibility.
    Args:
        seed: [int] Seed value.
        with_torch: Flag. If true, torch-related seeds are fixed.
        with_cuda: Flag. If true, torch+cuda-related seeds are fixed
    """
    random.seed(seed)
    np.random.seed(seed)
    if with_torch:
        torch.manual_seed(seed)
    if with_cuda:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

@torch.no_grad()
def teacher_normalization(teacher, train_loader, device):
    mean_outputs = []
    for train_image, _ in tqdm(train_loader, desc='Computing mean of features'):
        train_image = train_image.to(device)

        teacher_output = teacher(train_image)
        mean_output = torch.mean(teacher_output, dim=[0, 2, 3])
        mean_outputs.append(mean_output)
    channel_mean = torch.mean(torch.stack(mean_outputs), dim=0)
    channel_mean = channel_mean[None, :, None, None]

    mean_distances = []
    for train_image, _ in tqdm(train_loader, desc='Computing std of features'):
        train_image = train_image.to(device)

        teacher_output = teacher(train_image)
        distance = (teacher_output - channel_mean) ** 2
        mean_distance = torch.mean(distance, dim=[0, 2, 3])
        mean_distances.append(mean_distance)
    channel_var = torch.mean(torch.stack(mean_distances), dim=0)
    channel_var = channel_var[None, :, None, None]
    channel_std = torch.sqrt(channel_var)

    return channel_mean, channel_std


def train(opt):
    os.makedirs(opt.out_path, exist_ok=True)
    set_seeds(0)
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    datasets_train = CJJDataset(opt.data_path) #MVTecDataset(opt.data_path, "pill")
    #datasets_val = MVTecDataset(opt.data_path_val, "pill")

    dataloader_train = DataLoader(datasets_train, batch_size=opt.batch_size, shuffle=True, num_workers=8, drop_last=True)
    #dataloader_val = DataLoader(datasets_val, batch_size=opt.batch_size, shuffle=True, num_workers=8, drop_last=True)

    teacher = Teacher()
    student = Student()
    autoencoder = AutoEncoder()

    teacher.to(device)
    student.to(device)
    autoencoder.to(device)

    optimizer = torch.optim.Adam(itertools.chain(student.parameters(), autoencoder.parameters()), lr=opt.lr, weight_decay=1e-5)  # 定义优化器 momentum=0.99
    #optimizer = torch.optim.Adam(net.parameters(), opt.lr)  # 定义优化器 momentum=0.99

    # 学习率更新策略
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=40, eta_min=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(0.95 * 70000), gamma=0.1)


    # 加载预训练模型
    loss_best = 9999
    if os.path.exists(opt.pretrain):
        checkpoint = torch.load(opt.pretrain)
        teacher.load_state_dict(checkpoint['teacher'])
        student.load_state_dict(checkpoint['net_student'])
        autoencoder.load_state_dict(checkpoint['net_autoencoder'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        time,epoch,loss = checkpoint['time'],checkpoint['epoch'],checkpoint['loss']
        #loss_best = checkpoint['loss']
        print(f"加载权重: {opt.pretrain}, {time}: epoch: {epoch}, loss: {loss}")
    
    teacher_mean, teacher_std = teacher_normalization(teacher, dataloader_train, device=device)

    for epoch in range(1, opt.epoch):
        # 训练
        teacher.eval()
        student.train()
        autoencoder.train()

        loss_train = 0
        for image_st, image_ae in dataloader_train:
            image_st = image_st.to(device)
            image_ae = image_ae.to(device)

            #student
            with torch.no_grad():
                teacher_output_st = teacher(image_st)
                teacher_output_st = (teacher_output_st - teacher_mean) / teacher_std

            student_output_st = student(image_st)[:, :384]
            distance_st = (teacher_output_st - student_output_st) ** 2
            d_hard = torch.quantile(distance_st, q=0.999)
            loss_hard = torch.mean(distance_st[distance_st >= d_hard])
            loss_st = loss_hard

            #autoencoder
            ae_output = autoencoder(image_ae)
            with torch.no_grad():
                teacher_output_ae = teacher(image_ae)
                teacher_output_ae = (teacher_output_ae - teacher_mean) / teacher_std

            student_output_ae = student(image_ae)[:, 384:]
            distance_ae = (teacher_output_ae - ae_output)**2
            distance_stae = (ae_output - student_output_ae)**2
            loss_ae = torch.mean(distance_ae)
            loss_stae = torch.mean(distance_stae)

            loss_train = loss_st + loss_ae + loss_stae

            optimizer.zero_grad()
            loss_train.backward()
            optimizer.step()
            scheduler.step()


        # 验证
        # net.eval()


        # 打印一轮的训练结果
        #loss_train = loss_train / len(dataloader_train)
        #loss_val = loss_val / len(dataloader_val)
        print(f"epoch:{epoch}, loss_train:{round(loss_train.item(), 6)}, lr:{optimizer.param_groups[0]['lr']}")


        # 保存权重
        if loss_train < loss_best:
            loss_best = loss_train

            teacher.eval()
            student.eval()
            autoencoder.eval()

            checkpoint = {'net_teacher': teacher.state_dict(),
                          'net_student': student.state_dict(),
                          'net_autoencoder': autoencoder.state_dict(),
                          'optimizer': optimizer.state_dict(),
                          'teacher_mean':teacher_mean,
                          'teacher_std':teacher_std,
                          'epoch': epoch,
                          'loss': loss_train,
                          'time': datetime.date.today()}
            torch.save(checkpoint, os.path.join(opt.out_path,opt.weights))
            print(f'已保存:{opt.weights}')
    

def predict(opt):
    student = Student()
    autoencoder = AutoEncoder()
    checkpoint = torch.load(opt.pretrain)
    student.load_state_dict(checkpoint['net_student'])
    autoencoder.load_state_dict(checkpoint['net_autoencoder'])
    teacher_mean = checkpoint['teacher_mean']
    teacher_std = checkpoint['teacher_std']

    datasets_test = CJJDataset(opt.data_path, split=DatasetSplit.TEST)
    i=0
    for data in datasets_test:
        images = data['image']#type:torch.Tensor
        images = images.unsqueeze(0)
        masks = net.predict(images)

        max_value = np.round(np.max(masks[0]),2)
        min_value = np.round(np.min(masks[0]),2)
        print("\nmax=",max_value,"min=",min_value)
            
        img = images[0]
        img = img.cpu().numpy()
        img = img.transpose([1,2,0])
        img = img*IMAGENET_STD + IMAGENET_MEAN
        img = img.astype('float32')
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        temp = masks[0]
        _,thr = cv2.threshold(temp, 1.5, 255, cv2.THRESH_BINARY)
        thr = thr.astype("uint8")
        temp = 1/(1+np.exp(-temp))#sigmoid

        #region 将mask转换为热力图
        temp = np.uint8(temp*255)
        img = np.uint8(img*255)
        temp = cv2.applyColorMap(temp,cv2.COLORMAP_JET)
        #endregion

        contours,_ = cv2.findContours(thr, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

        h,w,c = img.shape
        dis = cv2.hconcat([img,temp])
        cv2.drawContours(dis, contours, -1, (0,0,255),3)
        cv2.putText(dis, data['filename'], (0,18), cv2.FONT_ITALIC, 0.7, (0,0,255), 1)
        cv2.putText(dis, f"max={str(max_value)},min={str(min_value)}", (0,h-2), cv2.FONT_ITALIC, 0.8, (0,0,255), 1)
        cv2.imshow("dis", dis)
        cv2.waitKey()
        
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrain', default='./run/train/best.pth', help='指定权重文件，未指定则使用官方权重！')  # 修改
    parser.add_argument('--out_path', default='./run/train', type=str)  # 修改
    parser.add_argument('--weights', default='best.pth', help='指定权重文件，未指定则使用官方权重！')

    parser.add_argument('--resume', default=False, type=bool, help='True表示从--weights参数指定的epoch开始训练,False从0开始')
    parser.add_argument('--data_path', default='D:/work/files/deeplearn_datasets/choujianji/roi-mynetseg/test')
    parser.add_argument('--data_path_val', default='')
    parser.add_argument('--epoch', default=300, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--batch_size', default=16, type=int)

    opt = parser.parse_args()

    train(opt)
    #predict(opt)
