import torchvision
import torch
import numpy as np
import cv2
import torchvision.transforms.functional as F

# 按比例将长边缩放至目标尺寸
class Resize1:
    def __init__(self, width):
        self.width = width

    def __call__(self, x):
        if isinstance(torch.Tensor):
            _, h, w = x.shape
            scale = self.width / max(w, h)
            W, H = round(w * scale), round(h * scale)
            x = F.resize(x, [H,W])
            return x
        elif isinstance(np.ndarray):
            h, w, c = x.shape
            scale = self.width / max(w, h)
            W, H = round(scale * w), round(scale * h)
            x = cv2.resize(x, (W, H), interpolation=cv2.INTER_LINEAR)
            return x


class PadSquare:
    def __call__(self, x):
        if isinstance(torch.Tensor):
            _, h, w = x.shape
            width = max(w, h)
            pad_left = round((width - w) / 2.0)
            pad_right = width - w - pad_left
            pad_up = round((width - h) / 2.0)
            pad_down = width - h - pad_up

            x = F.pad(x, [pad_left, pad_up, pad_right, pad_down])
            return x

        elif isinstance(np.ndarray):
            h, w, _ = x.shape
            width = max(w, h)
            pad_left = round((width - w) / 2.0)
            pad_right = width - w - pad_left
            pad_up = round((width - h) / 2.0)
            pad_down = width - h - pad_up

            x = cv2.copyMakeBorder(x, pad_up, pad_down, pad_left, pad_right, cv2.BORDER_CONSTANT, value=0)
            return x
