'''
Grad-CAM (Gradient-weighted Class Activation Mapping) 是一种可视化深度神经网络中哪些部分对于预测结果贡献最大的技术。
https://baijiahao.baidu.com/s?id=1763572174465139226&wfr=spider&for=pc

https://www.jianshu.com/p/fd2f09dc3cc9
论文《Learning Deep Features for Discriminative Localization》发现了CNN分类模型的一个有趣的现象：
CNN的最后一层卷积输出的特征图，对其通道进行加权叠加后，其激活值（ReLU激活后的非零值）所在的区域，即为图像中
的物体所在区域。而将这一叠加后的单通道特征图覆盖到输入图像上，即可高亮图像中物体所在位置区域。

作者：西北小生_
链接：https://www.jianshu.com/p/fd2f09dc3cc9
来源：简书
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。
'''

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
        pass

    net = torchvision.models.resnet18(pretrained=True)

    net.register_full_backward_hook

    