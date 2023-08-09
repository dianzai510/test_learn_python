import cv2
import  torch
import torchvision
from torchvision.datasets import ImageFolder
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from  torchvision.models.detection.ssd import ssd300_vgg16
import  torch.nn as nn
from torchvision.io.image import read_image
from torchvision.models.segmentation import deeplabv3_resnet50,deeplabv3
from torchvision.transforms.functional import to_pil_image
from torch.nn import Module
import collections
from PIL import Image
import matplotlib.pyplot as pil
class deeplabv3Test(Module):
    def __init__(self):
        super(deeplabv3Test, self).__init__()
        self.resnet = deeplabv3_resnet50(pretrained=True,progress=False,num_classes=21
                                         ,aux_loss=True,pretrained_backbone=True)

        self.resnet2 =nn.Conv2d(21,2,1)
        self.softmax = torch.nn.Sigmoid()
        total_params = sum(p.numel() for p in self.resnet.parameters())
        print(f'总参数数量：{total_params}')
        total_trainable_params = sum(p.numel() for p in self.resnet.parameters() if p.requires_grad == True)
        print(f'可训练参数数量：{total_trainable_params}')

    # 训练时候不加，检测代码加，训练加，检测就不加20230213 x = self.softmax(x)
    def forward(self, x):
        x = self.resnet(x)
        # print(x)
        x =self.resnet2(x['out'])
        x = self.softmax(x)
        return x


if __name__ == '__main__':
    # img = torch.randn(size=(2, 3, 260, 260))
    img1 = Image.open("D:\\抽检机\\8050\\LA22089434-0825_3( 1, 1 ).jpg")
    # img1 = img1.numpy()
    tensot = torchvision.transforms.ToTensor()
    resize = torchvision.transforms.Resize((300,300))
    toPIL = torchvision.transforms.ToPILImage()
    img1 = tensot(img1)
    img1 = resize(img1)
    tuple1 = (img1,img1)
    img1 = torch.stack(tuple1)
    # img1 = torch.unsqueeze(img1,0)

    net = deeplabv3Test()
    out = net(img1)

    pil.figure()
    # pil.imshow(img1)
    print(out.shape)
    # img1 = toPIL(out[1][1])
    # img1.show("123")
    # print(f'aaa{out.shape}')
    # print("xxxxxxxxxxxx")

    pass





