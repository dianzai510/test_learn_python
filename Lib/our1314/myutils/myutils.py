import os
import cv2
import numpy as np
from PIL import Image
from math import *
import torchvision.transforms as F

def pil2mat(image):
    mat = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    return mat

def mat2pil(mat):
    image = Image.fromarray(cv2.cvtColor(mat, cv2.COLOR_BGR2RGB))
    return image

def tensor2mat(data, dtype=np.uint8):
    """
    将给定的张量转换为Mat类型图像，并自动*255，并交换通道RGB→BGR
    :param data:张量,三个维度，[c,h,w]
    :param dtype:模板数据类型，默认np.uint8
    :return:OpenCV Mat，三个维度，[h,w,c]
    """
    assert len(data.shape)==3 , "张量维度不为3！"

    img = data.detach().numpy()  # type:np.ndarray
    img = img.copy()  # 没有这句会报错：Layout of the output array img is incompatible with cv::Mat
    img *= 255
    img = img.astype(np.uint8)
    img = np.transpose(img, (1, 2, 0))  # c,h,w → h,w,c
    img = img.copy()
    return img

def mat2tensor(img:np.array, dtype=np.uint8):
    """
    输入图像数据为0-255，某人为BGR通道，自动交换为RGB通道，归一化至0-1
    """
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    tensor = F.ToTensor(img)
    return tensor


def drawgrid(img, size, color=(0, 0, 255), linewidth=1):
    """
    在图像上绘制指定格式的网络线
    :param img:
    :param size:
    :param color:
    :param linewidth:
    :return:
    """
    dis = img.copy()
    x = np.arange(size[0]) * dis.shape[1] / size[0]
    y1 = np.zeros_like(x)
    y2 = dis.shape[0] * np.ones_like(x)
    p1 = np.vstack((x, y1)).T
    p2 = np.vstack((x, y2)).T

    for i in range(p1.shape[0]):
        _p1, _p2 = p1[i], p2[i]  # type:np.ndarray
        _p1 = _p1.astype(np.int)
        _p2 = _p2.astype(np.int)
        cv2.line(dis, _p1, _p2, color)

    y = np.arange(size[0]) * dis.shape[1] / size[0]
    x1 = np.zeros_like(x)
    x2 = dis.shape[0] * np.ones_like(x)
    p1 = np.vstack((x1, y)).T
    p2 = np.vstack((x2, y)).T

    for i in range(p1.shape[0]):
        _p1, _p2 = p1[i], p2[i]  # type:np.ndarray
        _p1 = _p1.astype(np.int)
        _p2 = _p2.astype(np.int)
        cv2.line(dis, _p1, _p2, color)

    return dis

def rectangle(img, center, wh, color, thickness):
    """
    给定中心和宽高绘制矩阵
    :param img:
    :param center:
    :param wh:
    :param color:
    :param thickness:
    :return:
    """
    pt1 = center - wh / 2.0  # type: np.ndarray
    pt2 = center + wh / 2.0  # type: np.ndarray
    pt1 = pt1.astype(np.int)
    pt2 = pt2.astype(np.int)
    cv2.rectangle(img, pt1, pt2, color, thickness)
    return img

# 获取按时间排序的最后一个文件
def getlastfile(path, ext='.pth'):
    if os.path.exists(path) is not True: return None
    list_file = [path + '/' + f for f in os.listdir(path) if f.endswith(ext)]  # 列表解析
    if len(list_file) > 0:
        list_file.sort(key=lambda fn: os.path.getmtime(fn))
        return list_file[-1]
    else:
        return None

def yolostr2data(yolostr: str):
    data = []
    yolostr = yolostr.strip()
    arr = yolostr.split('\n')
    arr = [f.strip() for f in arr]
    arr = [f for f in arr if f != ""]

    for s in arr:
        a = s.split(' ')
        a = [f.strip() for f in a]
        a = [f for f in a if f != ""]
        data.append((int(a[0]), float(a[1]), float(a[2]), float(a[3]), float(a[4])))
    return np.array(data)

def sigmoid(x):
    return 1. / (1 + np.exp(-x))


# if __name__ == '__main__':
#     x = torch.rand((3,300,300),dtype=torch.float)
#     y = tensor2mat(x)
#     y[y>0]=0
#     y=drawgrid(y, [10,10])
#     cv2.imshow('dis', y)
#     cv2.waitKey()
#     pass
