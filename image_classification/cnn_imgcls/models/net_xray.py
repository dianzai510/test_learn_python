import torch
from torch.nn import Linear, Module
from torchvision.models.resnet import resnet18, resnet101
from image_classification.cnn_imgcls.data import data_xray_sot23


class net_xray(Module):
    def __init__(self, pretrained):
        super(net_xray, self).__init__()
        self.resnet = resnet18(pretrained=pretrained)
        self.resnet.fc = Linear(512, 2, bias=True)
        # self.resnet = resnet101(pretrained=pretrained)
        #self.resnet.fc = Linear(2048, 2, bias=True)

    def forward(self, x):
        x = self.resnet(x)
        return x


if __name__ == '__main__':
    input_shape = (1, 3) + data_xray_sot23.input_size
    img = torch.randn(size=input_shape)
    print(img.shape)
    net = net_xray(True)
    print(net)
    out = net(img)
    print(f'out shape: {out.shape}')
