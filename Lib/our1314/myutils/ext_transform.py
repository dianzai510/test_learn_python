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
import os

# 按比例将长边缩放至目标尺寸
class Resize1:
    def __init__(self, width):
        self.width = width

    def __call__(self, imgs):
        assert type(imgs) == list,'类型不为list'
        assert len(imgs) !=0, '数量为0'

        result = []
        for x in imgs:
            if isinstance(x, torch.Tensor):
                h, w = x.shape[1],x.shape[2]
                scale = self.width / max(w, h)
                W, H = round(w * scale), round(h * scale)
                result.append(F.resize(x, [H,W]))

            elif isinstance(x, np.ndarray):
                h, w = x.shape[0],x.shape[1]
                scale = self.width / max(w, h)
                W, H = round(scale * w), round(scale * h)
                result.append(cv2.resize(x, (W, H), interpolation=cv2.INTER_LINEAR))

            else:
                assert '数据类型应该是张量或者ndarray'

        return result


class PadSquare:
    def __call__(self, imgs):
        assert type(imgs) == list,'类型不为list'
        assert len(imgs) !=0, '数量为0'

        result = []
        for x in imgs:
            if isinstance(x, torch.Tensor):
                h, w = x.shape[1],x.shape[2]
                width = max(w, h)
                pad_left = round((width - w) / 2.0)
                pad_right = width - w - pad_left
                pad_up = round((width - h) / 2.0)
                pad_down = width - h - pad_up

                result.append(F.pad(x, [pad_left, pad_up, pad_right, pad_down]))

            elif isinstance(x, np.ndarray):
                h, w = x.shape[0],x.shape[1]
                width = max(w, h)
                pad_left = round((width - w) / 2.0)
                pad_right = width - w - pad_left
                pad_up = round((width - h) / 2.0)
                pad_down = width - h - pad_up

                result.append(cv2.copyMakeBorder(x, pad_up, pad_down, pad_left, pad_right, cv2.BORDER_CONSTANT, value=0))

            else:
                assert '数据类型应该是张量或者ndarray'

        return result

class randomaffine_imgs:
    def __init__(self, rotate:list[float], transx:list[float], transy:list[float], scale:list[float]):
        self.rotate = rotate
        self.transx = transx
        self.transy = transy
        self.scale = scale

    def __call__(self, imgs:list):
        assert type(imgs) == list,'类型不为list'
        assert len(imgs) !=0, '数量为0'

        rot_deg = random.uniform(self.rotate[0], self.rotate[1])
        transx = random.uniform(self.transx[0], self.transx[1])
        transy = random.uniform(self.transy[0], self.transy[1])
        scale = random.uniform(min(self.scale), max(self.scale))

        result = []
        for x in imgs:
            if isinstance(x, torch.Tensor):
                h, w = x.shape[1],x.shape[2]
                img_trans = F.affine(x, rot_deg, [int(transx*w),int(transy*h)], scale, 1, interpolation=F.InterpolationMode.NEAREST)
                result.append(img_trans)
            
            elif isinstance(x, np.ndarray):
                h,w = x.shape[0],x.shape[1]
                angle_rad = rad(rot_deg)
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
                img_trans = cv2.warpAffine(x, H[0:2,0:3], (w,h))
                result.append(img_trans)
            else:
                assert '数据类型应该是张量或者ndarray'
        return result

class randomvflip_imgs:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, imgs:list):
        assert type(imgs) == list,'类型不为list'
        assert len(imgs) !=0, '数量为0'

        result = imgs.copy()
        value = random.uniform(0,1)
        if value < self.p:
            for i,x in enumerate(imgs):
                if isinstance(x, torch.Tensor):
                    result[i]=(F.vflip(x))
                elif isinstance(x, np.ndarray):
                    result[i]=cv2.flip(x,0)
                else:
                    assert '数据类型应该是张量或者ndarray'
        return result
    
class randomhflip_imgs:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, imgs:list):
        assert type(imgs) == list,'类型不为list'
        assert len(imgs) !=0, '数量为0'

        result = imgs.copy()
        value = random.uniform(0,1)
        if value < self.p:
            for i,x in enumerate(imgs):
                if isinstance(x, torch.Tensor):
                    result[i]=F.hflip(x)
                elif isinstance(x, np.ndarray):
                    result[i]=cv2.flip(x,1)
                else:
                    assert '数据类型应该是张量或者ndarray'
        return result


if __name__ == "__main__":
    transform1 = torchvision.transforms.Compose([
            Resize1(448),#等比例缩放
            # PadSquare(),
            # randomaffine_imgs([-10,10], [-0.1,0.1], [-0.1,0.1], [0.7,1/0.7]),
            # randomvflip_imgs(0.5),
            # randomhflip_imgs(0.5)
        ])
    
    data_path = 'D:/desktop/choujianji/roi/mask'
    Images = [os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith('.jpg')]
    Labels = [os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith('.png')]
    for i in range(len(Images)):
        image = cv2.imdecode(np.fromfile(Images[i], dtype=np.uint8), cv2.IMREAD_UNCHANGED) # type:cv2.Mat
        label = cv2.imdecode(np.fromfile(Labels[i], dtype=np.uint8), cv2.IMREAD_UNCHANGED) # type:cv2.Mat

        #r = randomaffine_imgs([-10,10],[-0.1,0.1],[-0.1,0.1],[0.9,1/0.9]) # randomhflip_imgs(1)
        #b1,b2 = r([F.to_tensor(image), F.to_tensor(label)])
        
        b1,b2 = transform1([F.to_tensor(image),F.to_tensor(label)])
        if isinstance(b1,np.ndarray):
            b2 = cv2.cvtColor(b2, cv2.COLOR_GRAY2BGR)
            bb = cv2.hconcat([b1,b2])
        elif isinstance(b1, torch.Tensor):
            b2 = b2.numpy()
            b2 = cv2.cvtColor(b2, cv2.COLOR_GRAY2BGR)
            b2 = F.to_tensor(b2)
            bb = torch.cat([b1,b2],dim=2)
        F.to_pil_image(bb).show()
