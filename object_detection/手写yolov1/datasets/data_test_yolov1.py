import os
import numpy as np
import cv2
import torch
import torchvision
from PIL.Image import Image
from torch.utils.data import Dataset, DataLoader
from utils import utils
from object_detection.手写yolov1.utils import basic


class data_test_yolov1(Dataset):
    def __init__(self, data_path, image_size=448, grid_size=7, num_bboxes=2, num_classes=20):
        self.image_size = image_size
        self.grid_size = grid_size
        self.num_bboxes = num_bboxes
        self.num_classes = num_classes

        self.Images = []
        self.Labels = []

        images_path = os.path.join(data_path, "images")
        labels_path = os.path.join(data_path, "labels")

        self.Images = [images_path + '/' + f for f in os.listdir(images_path) if
                       f.endswith('.jpg') or f.endswith('.png') or f.endswith('.bmp')]
        self.Labels = [labels_path + '/' + f for f in os.listdir(labels_path) if f.endswith('.txt')]

    def __len__(self):
        return len(self.Images)

    def __getitem__(self, item):
        image_path = self.Images[item]
        label_path = self.Labels[item]

        # region 1、读取图像,并按比例缩放到448,不足的地方补零
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        max_len = max(img.shape)
        padding_lr = int((max_len - img.shape[1]) / 2)
        padding_up = int((max_len - img.shape[0]) / 2)
        img = cv2.copyMakeBorder(img, padding_up, padding_up, padding_lr, padding_lr, borderType=cv2.BORDER_CONSTANT,
                                 value=0)
        img = cv2.resize(img, dsize=(self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        img = torchvision.transforms.ToTensor()(img)
        # endregion

        # region 2、读取标签,并根据图像的缩放方式调整坐标
        labels = []
        with open(label_path, 'r') as f:
            lines = f.readlines()
            lines = [l.strip().split(' ') for l in lines]

            d1 = len(lines)
            d2 = len(lines[0])
            labels = torch.empty((d1, d2))

            dis = utils.tensor2mat(img)  # type: np.ndarray
            dis = utils.drawgrid(dis, (7, 7))
            for i, data_list in enumerate(lines):
                data_list = [float(f.strip()) for f in data_list]
                cls, x, y, w, h = data_list

                # pt = np.array([[x], [y], [1]])
                # H = np.array([[448, 0, 0],
                #               [0, 252, (448 - 252) / 2],
                #               [0, 0, 1]
                #               ])
                #
                # pt = H.dot(pt)
                # pt = pt.flatten().tolist()
                # cp = (int(pt[0]), int(pt[1]))
                # cv2.circle(dis, cp, 5, (0, 0, 255), -1)
                # labels.append(torch.tensor([cls, x, y, w, h]))

                # 根据图像的处理方式，重新计算目标坐标
                y = (y * 252.0 + (448.0 - 252.0) / 2.0) / 448.0
                h = h * 252 / 448  # h重新归一化，因为使用时使用448进行
                labels[i, :] = torch.tensor([cls, x, y, w, h])

            # cv2.imshow('dis', dis)
            # cv2.waitKey()
        labels = basic.encode(labels, self.grid_size, self.num_bboxes, self.num_classes, dis)

        # cv2.imshow('dis', dis)
        # cv2.waitKey()
        # endregion

        return img, labels


if __name__ == '__main__':
    datasets_train = data_test_yolov1('D:/work/files/data/DeepLearningDataSets/x-ray/datasets-xray-sot23/train')
    dataloader_train = DataLoader(datasets_train, 1, shuffle=True)
    for imgs, labels in dataloader_train:
        img1 = imgs[0, ...]
        dis = utils.tensor2mat(img1)
        dis = utils.drawgrid(dis, (7, 7))

        label = labels[0]
        # obj_mask = label[:, :, 4] > 0
        # objs = label[obj_mask]
        for j in range(label.shape[0]):
            for i in range(label.shape[1]):
                c = label[j, i][4]
                if c > 0:
                    data = label[j, i]
                    d = 448 / 7
                    ij = torch.tensor([i, j]) * d
                    xy = ij + data[:2] * d
                    wh = data[2:4] * 448
                    utils.rectangle(dis, xy.numpy(), wh.numpy(), (255, 0, 0), 2)
        cv2.imshow('dis', dis)
        cv2.waitKey()

        # region 显示label

        # endregion

        #  label
        # grid_lu = np.array([i / 7.0 * 448.0, j / 7.0 * 448.0])
        # cv2.circle(dis, grid_lu.astype(np.int), 5, (255, 0, 0), -1)
        #
        # target_center = grid_lu + delta_xy.numpy() * 448 / 7
        # cv2.circle(dis, target_center.astype(np.int), 5, (0, 0, 255), -1)
        # target_wh = np.array([448 * wh[0], 448 * wh[1]])
        # utils.rectangle(dis, target_center.astype(np.int), target_wh.astype(np.int), (0, 0, 255), 2)
