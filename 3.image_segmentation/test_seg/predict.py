import argparse
import os
import torch
from model import UNet
from model import deeplabv3,UNet
from data import transform_val
import cv2
import numpy as np
import torchvision
from our1314.myutils.myutils import tensor2mat

def predict(opt):
    path_weight = os.path.join(opt.out_path,opt.weights)
    checkpoint = torch.load(path_weight)
    net = deeplabv3()
    net.load_state_dict(checkpoint['net'])
    net.eval()

    with torch.no_grad():
        files = [os.path.join(opt.data_path_test,f) for f in os.listdir(opt.data_path_test) if f.endswith('.jpg')]
        for image_path in files:
            src = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)   # type:cv2.Mat
            
            img, = transform_val([src])
            x = net(img.unsqueeze(0))#type:torch.Tensor
            x = x.squeeze_(dim=0)

            t = opt.conf
            x[x>t]=1.0
            x[x<=t]=0.0
            
            # dis = img.clone()
            # dis[0] = 0.7*img[0]+0.5*x
            # dis = tensor2mat(dis)

            mask = tensor2mat(x)
            img = tensor2mat(img)

            mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            dis = cv2.addWeighted(img, 0.7, mask, 0.3, 0)
            #dis = cv2.copyTo(img,mask)
            cv2.imshow("dis", dis)
            cv2.waitKey()
        
    cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', default='best_out.pth', help='指定权重文件，未指定则使用官方权重！')
    parser.add_argument('--out_path', default='./run/train', type=str)  # 修改
    parser.add_argument('--data_path_test', default='D:/work/files/deeplearn_datasets/choujianji/roi-mynetseg/train')  # 修改
    parser.add_argument('--conf', default=0.95, type=float)

    opt = parser.parse_args()

    predict(opt)
