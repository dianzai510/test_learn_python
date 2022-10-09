#coding=gb2312

""" 2022.10.9
���ʣ�����³����ʦ����������������ľ���ǰ��������Լ���ģ�����ͨ�����һ����
��𣺱�д������֤���ֱ��ð����Ͳ����������Լ����������ͬһ���ݽ������㡣
���ۣ�pytorch�ľ�����ڲ������������Լ��ԭ�򣺡�
����ReLU����������Ϊ0��
û��ReLU����������Ϊ������
��֤��������������ľ����numpy����д������һ����
VGG16������ṹ��ʾ��ÿ�������󶼸���һ��relu�㣬����ҳ��չʾ��VGG�ṹ������������relu��
resnetһ���־�����batchnorm+relu����һ��������ԭ������Ӻ��ٽ�relu,��˶���relu�����ο���https://www.modb.pro/db/488020��
��Ϊ���Ҳ���������㣬���û�з����Լ�������˻���
"""
import torch.nn
import torchvision
from torch import nn
from torch.nn import Module
import cv2
import numpy as np
from torchvision.models import resnet18

#����Relu������
class MyNet_Relu(Module):
    def __init__(self):
        super(MyNet_Relu, self).__init__()
        self.Conv = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0)),#�����
            nn.ReLU()#�����Լ�����ReLU�������ĸ��������0
        )
        pass

    def forward(self, x):
        x = self.Conv(x)
        return x

#������Relu������
class MyNet(Module):
    def __init__(self):
        super(MyNet, self).__init__()
        self.Conv = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0)),#�����
        )
        pass

    def forward(self, x):
        x = self.Conv(x)
        return x

#����
d1 = -1 * torch.rand((1, 3, 3)) #type:torch.Tensor
net_relu = MyNet_Relu()
net = MyNet()
#��ʼ��Ȩ��
for m in net_relu.modules():
    if isinstance(m, nn.Conv2d):
        m.weight.data = torch.ones((1, 1, 3, 3))/9.0
        m.bias.data = torch.zeros(1)
        print(m.weight)

for m in net.modules():
    if isinstance(m, nn.Conv2d):
        m.weight.data = torch.ones((1, 1, 3, 3))/9.0
        m.bias.data = torch.zeros(1)
        print(m.weight)

print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
out_relu = net_relu(d1)
out = net(d1)
print(f"out_relu={out_relu},out={out}")

#numpy������
src = d1.numpy()#type:np.ndarray
src = src.reshape((-1))
kernel = torch.ones((3,3),dtype=float)/9.0#type:np.ndarray
kernel = kernel.reshape((-1))
dst = np.convolve(src, kernel, mode='valid')
pass

#��д�������
src = src[::-1]
src = src.reshape((1,9))
kernel = kernel.reshape((9,1))
s = src.dot(kernel)
pass

net = torchvision.models.vgg.vgg16()
print(net)
net = torchvision.models.resnet.resnet18()
print(net)
pass