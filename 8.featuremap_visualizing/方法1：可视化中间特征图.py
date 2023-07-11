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

import sys
sys.path.append("D:/work/program/Python/DeepLearning/test_learn_python")
from myutils.myutils import *
    
if __name__ == '__main__':
    def hookfn(module, fea_in, fea_out):
        fea_out.squeeze_(0)
        i=0
        for ch in fea_out:
            a = ch.unsqueeze(0)
            img = tensor2mat(a)
            #img = cv2.resize(img, None, fx=16, fy=16, interpolation=cv2.INTER_NEAREST)
            img = cv2.resize(img, None, fx=16, fy=16, interpolation=cv2.INTER_LINEAR)
            cv2.imshow(f'dis', img)
            cv2.waitKey()
            i+=1
            print(a.shape)
            pass
        cv2.waitKey()


    net = torchvision.models.resnet18(pretrained=True)
    print(net)
    print('\n')
    x = cv2.imread('D:/work/files/deeplearn_datasets/coco128/images/train2017/000000000625.jpg')
    x = mat2tensor(x)
    x = x.unsqueeze(0)
    # x = torch.rand((1,3,256,256))
    for name, m in net.named_children():
        print(m)
        print('#'*100)
        
        # if isinstance(m, nn.Conv2d):
        #     m.register_forward_hook(hook=hookfn)
        #     break
        if name =='layer4':
            m.register_forward_hook(hook=hookfn)
            break
        #x = m(x)
    
    x = net(x)
    

    