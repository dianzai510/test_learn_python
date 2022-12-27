import argparse
import torch
from torch.utils.data import DataLoader
import numpy as np
import utils.utils
from object_detection.手写yolov1.model.yolov1 import yolov1
from object_detection.手写yolov1.datasets.data_test_yolov1 import data_test_yolov1


def detect(opt):
    # 1、加载网络
    checkpoint = torch.load(opt.weights)
    net = yolov1()
    net.load_state_dict(checkpoint['net'])

    # 2、加载数据
    datasets = data_test_yolov1("D:/work/files/deeplearn_datasets/test_datasets/test_yolo_xray/train")
    data_loader = DataLoader(datasets, 1)
    for images, labels in data_loader:
        pred = net(images)
        for index in range(images.size(0)):
            for i in range(7):
                for j in range(7):
                    box_pred = pred[index, i, j]
                    x0 = 0.0
                    y0 = 0.0
                    w = 0.0
                    h = 0.0
                    if box_pred[4] > 0.3 or box_pred[9] > 0.3:
                        box = []
                        if box_pred[4] > box_pred[9]:
                            box = box_pred[0:4]
                        else:
                            box = box_pred[4:9]
                        x0 = i + box[0]
                        y0 = j + box[1]
                        w = box[2]
                        h = box[3]
                        x0 *= 416 / 7
                        y0 *= 416 / 7
                        w *= 416
                        h *= 416
                        pass
                        img = images[index]
                        img = utils.utils.tensor2mat(img)
                        img = utils.utils.rectangle(img, np.array([x0.item(), y0.item()]), np.array([w.item(), h.item()]), (0, 0, 255), 2)


        box = box.contiguous().view(-1, 5)
        for i in range(0, box.size(0), 2):
            two_box = box[i:i + 2]
            two_box = two_box.contiguous().view(-1, 5)  # type:torch.Tensor
            # if two_box[4]
            conf = two_box[:, 4]
            if conf[0] > 0.3:
                box = two_box[:4]
                x0, y0, w, h = box

                pass
            elif conf[1] > 0.3:
                box = two_box[5:9]
                x0, y0, w, h = box
                pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', default='run/train/exp/weights/best.pth')

    opt = parser.parse_args()
    detect(opt)
