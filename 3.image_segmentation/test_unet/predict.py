import os
import torch
from model import UNet
from data import input_size, transform_val
import cv2
import numpy as np
import torchvision
from our1314.myutils.myutils import tensor2mat

if __name__ == "__main__":
    path_best = f"./run/train/best.pth"
    checkpoint = torch.load(path_best)
    net = UNet()
    net.load_state_dict(checkpoint['net'])
    net.eval()

    dir = 'D:/desktop/choujianji/roi'
    files = [os.path.join(dir,f) for f in os.listdir(dir) if f.endswith('.jpg')]
    for image_path in files:
        #image_path = 'D:/desktop/choujianji/roi/mask/LA22089071-0152_2( 4, 17 ).jpg'
        src = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)   # type:cv2.Mat
        img = src.copy()
        h, w, _ = src.shape
        scale = min(input_size[0]/w, input_size[1]/h)
        img = cv2.resize(src, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        h, w, _ = img.shape
        left = (input_size[0] - w)//2
        right = input_size[0] - w - left
        top = (input_size[1] - h)//2
        bottom = input_size[1] - h - top 
        img = cv2.copyMakeBorder(img, top, bottom, left, right, borderType=cv2.BORDER_CONSTANT, value=0)
        img = torchvision.transforms.ToTensor()(img)
        img = torch.unsqueeze(img, dim=0)
        x = net(img)
        x = torch.sigmoid(x)
        x = torch.squeeze(x, dim=0)
        y = tensor2mat(x)
        _,y = cv2.threshold(y, 200, 255, cv2.THRESH_BINARY)
        src = tensor2mat(img.squeeze_())
        dis = cv2.copyTo(src, y)


        cv2.imshow("dis", dis)
        cv2.waitKey(1)
    cv2.destroyAllWindows()