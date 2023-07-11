"""
操作步骤：
1、实例化网络模型
2、加载训练好的权重文件
3、输出某一层的特征图
4、截图特征图的通道转换为图像
"""

import torch
import torchvision
import torchviz
from torch.nn import Module

class net_xray(Module):
    def __init__(self, pretrained, cls_num=2):
        super(net_xray, self).__init__()
        self.resnet = resnet18(pretrained=pretrained)
        self.resnet.fc = Linear(512, cls_num, bias=True)

    def forward(self, x):
        x = self.resnet(x)
        # x = self.softmax(x)
        return x
    
if __name__ == '__main__':
    net = torchvision.models.resnet18(pretrained=True)
    
    pass