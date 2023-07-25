import copy
import os
import cv2
import numpy as np
import torch
import torchvision
from PIL import Image
from torch.nn import Module
from math import *
import halcon
from halcon.numpy_interop import himage_from_numpy_array, himage_as_numpy_array


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
# 按比例将长边缩放至目标尺寸
class Resize3:
    def __init__(self, width):
        # self.resize = torchvision.transforms.Resize()
        self.width = width

    def __call__(self, x):
        _, h, w = x.shape
        scale = self.width / max(w, h)
        W = round(w * scale)
        H = round(h * scale)
        x = torchvision.transforms.Resize((H, W))(x)
        return x


# class Resize2():
#     def __init__(self, width):
#         self.width = width
#
#     def __call__(self, img):
#         h, w, c = img.shape
#         scale = self.width / max(w, h)
#         W, H = round(scale * w), round(scale * h)
#         dst = cv2.resize(img, (W, H), interpolation=cv2.INTER_LINEAR)
#         return dst


class PadSquare:
    def __call__(self, x):
        _, h, w = x.shape
        width = max(w, h)
        pad_left = round((width - w) / 2.0)
        pad_right = width - w - pad_left
        pad_up = round((width - h) / 2.0)
        pad_down = width - h - pad_up
        x = torchvision.transforms.Pad((pad_left, pad_up, pad_right, pad_down))(x)
        return x

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

def ndarray2hobject(mat):
    return himage_from_numpy_array(mat)

def hobject2ndarray(hobj):
    return himage_as_numpy_array(hobj)

def deg(rad):
    return rad*180/pi

def rad(deg):
    return deg*180/pi


if __name__ == '__main__':
    # a = getlastfile('D:/work/proj/xray/test_learn_python/image_classification/cnn_imgcls/run/train/oqa_agl/weights', '.pth')
    # x = torch.rand((1, 3, 110, 310))
    # x = Resize3(330)(x)
    # x = PadSquare()(x)
    
    # resize = torchvision.transforms.Resize((100, 100))
    # x = resize(x)
    # pass

    img = cv2.imread('d:/desktop/tmp.png', cv2.IMREAD_COLOR)
    hobj = ndarray2hobject(img)
    w, h = halcon.get_image_size(hobj)
    halcon.dev_open_window(0, 0, w, h, 'black', WindowHandle)
    halcon.dev_display (Image)
