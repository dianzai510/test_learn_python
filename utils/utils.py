import torch
import numpy as np
import cv2


def tensor2mat(data, dtype=np.uint8):
    img = data.numpy()  # type:np.ndarray
    img = img.copy()  # Layout of the output array img is incompatible with cv::Mat
    img = img * 255
    img = img.astype(np.uint8)
    img = np.transpose(img, (1, 2, 0))  # bgr转rgb

    img = img.copy()
    return img


def drawgrid(img, size, color=(0, 0, 255), linewidth=2):
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
    pt1 = center - wh / 2.0  # type: np.ndarray
    pt2 = center + wh / 2.0  # type: np.ndarray
    pt1 = pt1.astype(np.int)
    pt2 = pt2.astype(np.int)
    cv2.rectangle(img, pt1, pt2, color, thickness)
