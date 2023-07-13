"""
操作步骤：
1、实例化网络模型
2、加载训练好的权重文件
3、输出某一层的特征图
4、截图特征图的通道转换为图像
"""

import torch
import torchvision
from torch.nn import Module
from torch import nn
import cv2
from torchvision import transforms

import sys
sys.path.append("D:/work/program/Python/DeepLearning/test_learn_python")
from myutils.myutils import *
    
if __name__ == '__main__':

    preprocess = transforms.Compose([   transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    
    fea = []
    def hookfn(module, fea_in, fea_out):
        fea.append(fea_out)

    net = torchvision.models.resnet18(pretrained=True)
    net.conv1.register_forward_hook(hook=hookfn)

    src = Image.open('image/2.jpg').convert('RGB')
    x = preprocess(src)
    x = x.unsqueeze(0)    
    x = net(x)

    f = fea[0].squeeze(0)
    for i,ch in enumerate(f):
        ch = ch.unsqueeze(0)
        img = tensor2mat(ch)
        img = cv2.resize(img, None, fx=1, fy=1, interpolation=cv2.INTER_NEAREST)
        cv2.imshow(f'dis{i}', img)
        rowcnt = 1920//120
        x = i%rowcnt
        y = i//rowcnt
        cv2.moveWindow(f'dis{i}',x*120,y*120)
        cv2.waitKey(1)
    cv2.waitKey()
    

    