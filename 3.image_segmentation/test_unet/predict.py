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

    image_path = 'D:/desktop/choujianji/roi/mask/LA22089071-0152_2( 4, 17 ).jpg'
    image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)   # type:cv2.Mat
    h, w, _ = image.shape
    scale = min(input_size[0]/w, input_size[1]/h)
    image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    h, w, _ = image.shape
    left = (input_size[0] - w)//2
    right = input_size[0] - w - left
    top = (input_size[1] - h)//2
    bottom = input_size[1] - h - top 
    image = cv2.copyMakeBorder(image, top, bottom, left, right, borderType=cv2.BORDER_CONSTANT, value=0)
    image = torchvision.transforms.ToTensor()(image)
    image = torch.unsqueeze(image, dim=0)
    x = net(image)
    x = torch.sigmoid(x)
    x = torch.squeeze(x, dim=0)
    y = tensor2mat(x)
    cv2.imshow("dis", y)
    cv2.waitKey()
    cv2.destroyAllWindows()