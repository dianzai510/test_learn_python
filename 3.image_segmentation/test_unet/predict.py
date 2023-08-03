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

    with torch.no_grad():
        dir = 'D:/work/files/deeplearn_datasets/choujianji/roi-seg/val'
        files = [os.path.join(dir,f) for f in os.listdir(dir) if f.endswith('.jpg')]
        for image_path in files:
            src = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)   # type:cv2.Mat
            
            img, = transform_val([src])
            x = net(img.unsqueeze(0))#type:torch.Tensor
            x = x.squeeze_(dim=0)
            t = 0.7
            x[x>t]=1.0
            x[x<=t]=0.0
            
            # dis = img.clone()
            # dis[0] = 0.7*img[0]+0.5*x
            # dis = tensor2mat(dis)

            mask = tensor2mat(x)
            img = tensor2mat(img)

            dis = cv2.copyTo(img,mask)
            cv2.imshow("dis", dis)
            cv2.waitKey(1)
        
    cv2.destroyAllWindows()