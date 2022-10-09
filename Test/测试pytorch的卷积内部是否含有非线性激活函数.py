#coding=gb2312

"""
���ۣ�pytorch�ľ���㲻���������Լ��ԭ�򣺡�
�������ReLU�������ɸ��������0
"""
import torch.nn
from torch import nn
from torch.nn import Module
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
d1 = -1 * torch.rand((1, 3, 3))
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
        m.weight.data = torch.ones((1, 1, 3, 3)) / 9.0
        m.bias.data = torch.zeros(1)
        print(m.weight)

print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
out_relu = net_relu(d1)
out = net(d1)
print(f"out_relu={out_relu},out={out}")
pass
