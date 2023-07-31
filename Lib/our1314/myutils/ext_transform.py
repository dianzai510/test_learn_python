import torchvision
import torch
import numpy as np
import cv2
import torchvision.transforms.functional as F
import random
from PIL import Image
from math import *
from our1314.myutils.mathexp import *
import typing

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
        else:
            assert '数据类型应该是张量或者ndarray'

class randomaffine_imgs:
    def __init__(self, rotate:list[float], transx:list[float], transy:list[float], scale:list[float]):
        self.rot_deg = random.uniform(rotate[0], rotate[1])
        self.transx = random.uniform(transx[0], transx[1])
        self.transy = random.uniform(transy[0], transy[1])
        self.scale = random.uniform(min(scale), max(scale))

    def __call__(self, imgs:list):
        assert type(imgs) == list,'类型不为list'
        assert len(imgs) !=0, '数量为0'

        result = []
        if isinstance(imgs[0], torch.Tensor):
            for x in imgs:
                _,h,w = x.shape
                img_trans = F.affine(x, self.rot_deg, [int(self.transx*w),int(self.transy*h)], self.scale, 1, interpolation=F.InterpolationMode.NEAREST)
                result.append(img_trans)
        
        elif isinstance(imgs[0], np.ndarray):
            for x in imgs:
                h,w = x.shape[0],x.shape[1]
                size_new = (w,h)
                angle_rad = rad(self.rot_deg)
                H1 = np.array([
                    [1,0,-w/2],
                    [0,1,-h/2],
                    [0,0,1]
                ])
                H2 = np.array([
                    [cos(angle_rad),-sin(angle_rad),0],
                    [sin(angle_rad),cos(angle_rad),0],
                    [0,0,1]
                ])
                H = np.linalg.inv(H1)@H2@H1
                img_trans = cv2.warpAffine(x, H[0:2,0:3], size_new)
                result.append(img_trans)
        else:
            assert '数据类型应该是张量或者ndarray'
        return result

class randomvflip_imgs:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, imgs:list):
        assert type(imgs) == list,'类型不为list'

        value = random.uniform(0,1)
        if value < self.p:
            if isinstance(imgs[0], torch.Tensor):
                imgs = [F.vflip(x) for x in imgs]
            elif isinstance(imgs[0], np.ndarray):
                imgs = [cv2.flip(x,0) for x in imgs]
            else:
                assert '数据类型应该是张量或者ndarray'
        return imgs
    
class randomhflip_imgs:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, imgs:list):
        assert type(imgs) == list,'类型不为list'

        value = random.uniform(0,1)
        if value < self.p:
            if isinstance(imgs[0], torch.Tensor):
                imgs = [F.hflip(x) for x in imgs]
            elif isinstance(imgs[0], np.ndarray):
                imgs = [cv2.flip(x,1) for x in imgs]
            else:
                assert '数据类型应该是张量或者ndarray'
        return imgs

if __name__ == "__main__":
    #a = randomaffine_imgs([-10,10],[-0.1,0.1],[-0.1,0.1],[0.9,1/0.9])
    image = cv2.imdecode(np.fromfile('D:/desktop/choujianji/roi/mask/LA22089071-0152_2( 4, 17 ).jpg', dtype=np.uint8), cv2.IMREAD_UNCHANGED) # type:cv2.Mat
    label = cv2.imdecode(np.fromfile('D:/desktop/choujianji/roi/mask/LA22089071-0152_2( 4, 17 ).png', dtype=np.uint8), cv2.IMREAD_UNCHANGED) # type:cv2.Mat
    r = randomaffine_imgs([-10,10],[-0.1,0.1],[-0.1,0.1],[0.9,1/0.9])#randomhflip_imgs(1)
    b1,b2 = r([F.to_tensor(image), F.to_tensor(label)])

    F.to_pil_image(b1).show()
    F.to_pil_image(b2).show()
