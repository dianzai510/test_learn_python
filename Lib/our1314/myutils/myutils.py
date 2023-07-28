import copy
import os
import cv2
import numpy as np
import torch
import torchvision
from PIL import Image
from torch.nn import Module
from math import *



def pil2mat(image):
    mat = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    return mat

def mat2pil(mat):
    image = Image.fromarray(cv2.cvtColor(mat, cv2.COLOR_BGR2RGB))
    return image

def tensor2mat(data, dtype=np.uint8):
    """
    将给定的张量转换为Mat
    :param data:张量,三个维度，[c,h,w]
    :param dtype:模板数据类型，默认np.uint8
    :return:OpenCV Mat，三个维度，[h,w,c]
    """

    size = data.size()
    if len(size) != 3:
        assert "张量维度不为3！"
        return None
    img = data.detach().numpy()  # type:np.ndarray
    img = img.copy()  # 没有这句会报错：Layout of the output array img is incompatible with cv::Mat
    img *= 255
    img = img.astype(np.uint8)
    img = np.transpose(img, (1, 2, 0))  # c,h,w → h,w,c
    img = img.copy()
    return img

def mat2tensor(mat, dtype=np.uint8):
    tensor = torchvision.transforms.ToTensor()(mat)
    return tensor

def drawgrid(img, size, color=(0, 0, 255), linewidth=2):
    """
    在图像上绘制指定格式的网络线
    :param img:
    :param size:
    :param color:
    :param linewidth:
    :return:
    """
    img = img.copy()
    x = np.arange(size[0]) * img.shape[1] / size[0]
    y1 = np.zeros_like(x)
    y2 = img.shape[0] * np.ones_like(x)
    p1 = np.vstack((x, y1)).T
    p2 = np.vstack((x, y2)).T

    for i in range(p1.shape[0]):
        _p1, _p2 = p1[i], p2[i]  # type:np.ndarray
        _p1 = _p1.astype(np.int)
        _p2 = _p2.astype(np.int)
        cv2.line(img, _p1, _p2, color)

    y = np.arange(size[0]) * img.shape[1] / size[0]
    x1 = np.zeros_like(x)
    x2 = img.shape[0] * np.ones_like(x)
    p1 = np.vstack((x1, y)).T
    p2 = np.vstack((x2, y)).T

    for i in range(p1.shape[0]):
        _p1, _p2 = p1[i], p2[i]  # type:np.ndarray
        _p1 = _p1.astype(np.int)
        _p2 = _p2.astype(np.int)
        cv2.line(img, _p1, _p2, color)

    return img

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
def getlastfile(path, ext):
    if os.path.exists(path) is not True: return None
    list_file = [path + '/' + f for f in os.listdir(path) if f.endswith(".pth")]  # 列表解析
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
    return data

def SE2(x,y,rad):
    r = np.array([
        [cos(rad),-sin(rad),x],
        [sin(rad),cos(rad),y],
        [0,0,1]
        ])
    return r

def _SE2(H):
    x,y = H[0,2],H[1,2]
    cos_theta = H[0,0]
    sin_theta = H[1,0]
    theta = atan2(cos_theta, sin_theta)
    return x,y,theta

#绕3D坐标系X轴旋转的旋转矩阵
def RX(rad):
    r = np.array([
        [1,0,0],
        [0,cos(rad),-sin(rad)],
        [0,sin(rad),cos(rad)]
        ])
    return r
#绕3D坐标系X轴旋转的旋转矩阵
def RY(rad):
    r = np.array([
        [cos(rad),0,sin(rad)],
        [0,1,0],
        [-sin(rad),0,cos(rad)]
        ])
    return r
#绕3D坐标系X轴旋转的旋转矩阵
def RZ(rad):
    r = np.array([
        [cos(rad),-sin(rad),0],
        [sin(rad),cos(rad),0],
        [0,0,1]
        ])
    return r

def Pxyz(x,y,z):
    H = np.eye(4)
    H[0,3],H[1,3],H[2,3]=x,y,z
    return H

def Homogeneous(m):
    h,w = m.shape
    m = np.column_stack([m,np.zeros(h,1)])
    m = np.row_stack([m,np.zeros(1,w+1)])
    m[-1,-1]=1
    return m

def SE3(px,py,pz,rx,ry,rz):
    Rx = Homogeneous(RX(rx))
    Ry = Homogeneous(RY(ry))
    Rz = Homogeneous(RZ(rz))
    P = Pxyz(px,py,pz)
    H = P@Rz@Ry@Rx
    return H

def _SE3(H):
    pass


def deg(rad):
    return rad*180/pi

def rad(deg):
    return deg*pi/180



def sigmoid(x):
    return 1. / (1 + np.exp(-x))


if __name__ == '__main__':
    pass
