import sys
sys.path.append("D:/work/program/python/DeepLearning/test_learn_python")
import os
from random import shuffle
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor
from model1 import Model1
from model2 import Model2
import torch
from torchvision import transforms
from _5_anomalydetect.PatchCore.patchcore import PatchCore
import torchvision.transforms.functional as F
from PIL import Image
from math import *


net = Model2()
net.eval()

test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((100,100)),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

transform_img = [
        transforms.Resize(300),
        # transforms.RandomRotation(rotate_degrees, transforms.InterpolationMode.BILINEAR),
        #transforms.ColorJitter(brightness_factor, contrast_factor, saturation_factor),
        #transforms.RandomHorizontalFlip(h_flip_p),
        #transforms.RandomVerticalFlip(v_flip_p),
        #transforms.RandomGrayscale(gray_p),
        # transforms.RandomAffine(rotate_degrees, 
        #                         translate=(translate, translate),
        #                         scale=(1.0-scale, 1.0+scale),
        #                         interpolation=transforms.InterpolationMode.BILINEAR),
        
        #transforms.GaussianBlur(kernel_size=(7,7),sigma=(0.1,2.0)),#随机高斯模糊          

        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
transform = transforms.Compose(transform_img)

transform_img1 = [
        transforms.Resize(300),
        transforms.CenterCrop(224),
        ]
transform1 = transforms.Compose(transform_img1)


path = 'D:/work/files/deeplearn_datasets/choujianji/roi-mynetseg/test/test/ng/1.png'#input('输入图像路径：')
src = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)#type:np.ndarray
dir_image = "D:/work/files/deeplearn_datasets/choujianji/roi-mynetseg/test/train/good"
files_all = os.listdir(dir_image)
images_path = [os.path.join(dir_image, f) for f in files_all if f.endswith('.png') or f.endswith('.jpg') or f.endswith('.bmp')]
images_path = images_path[:100]

src = Image.open(path).convert("RGB")
d = transform1(src)
src = F.to_tensor(d).numpy()
src = src.transpose([1,2,0])
src = cv2.cvtColor(src,cv2.COLOR_RGB2BGR)

# images_path.append(path)
#shuffle(images_path)#随机排序
images_path.insert(0, path)

imgs = [cv2.imdecode(np.fromfile(f, dtype=np.uint8), cv2.IMREAD_COLOR) for f in images_path]
imgs = [transform(Image.open(f).convert('RGB')) for f in images_path]
imgs = torch.stack(imgs,dim=0)

#region patchcore提取特征
patchcore = PatchCore(torch.device("cuda:1"))

with torch.no_grad():
    input_image = imgs.to(torch.device("cuda:1"))
    feas = patchcore._embed(input_image)
    feas = np.array(feas)
    s = int(sqrt(len(feas)/101))
    feas = feas.reshape(-1,s,s,1024)
#endregion

#region 遍历图像，计算异常分
dd = []
for y in range(s):
    for x in range(s):
        X = feas[:,y,x,:]
        d = [np.linalg.norm(X[0]-p) for p in X[1:]]
        dd.append(np.mean(d))
dd = np.array(dd)
dd = dd.reshape(s,s) - 1
#dd = (dd - np.min(dd)) / (np.max(dd) - np.min(dd))
dd = cv2.resize(dd, (224,224), cv2.INTER_LINEAR)
cv2.imshow("dis", dd)
cv2.waitKey()
pass

#endregion
num = 224//s
#clf = LocalOutlierFactor(n_neighbors=40, contamination=0.01)#异常检测器
clf = LocalOutlierFactor(n_neighbors=10, contamination=1e-6, novelty=False)

def onmouse(*p):
    event, x, y, flags, param=p
    if event == cv2.EVENT_LBUTTONDOWN:
        dis = src.copy()

        #1、绘制鼠标位置
        #cv2.circle(dis, (x,y), 5, (0,0,255), 2)
        cv2.line(dis,(x,y),(x,y),(0,0,255),1)
        cv2.imshow("dis", dis)

        #3、异常检测
        # 1、收集异常点所在的图像索引
        # 2、所在图像索引上的异常坐标的聚类
        #X = imgs[:,y,x,:]
        y = int(y/num)
        x = int(x/num)
        X = feas[:,y,x,:]
        pred = clf.fit_predict(X)
        #pred = clf.score_samples(X)
        ng_index = [i for i,p in enumerate(pred) if p<0]
        p1 = [np.sqrt(np.sum(np.power(X[0]-p,2))) for p in X]
        p1 = [np.round(d,2) for d in p1]
        
        print(p1)
        print(np.max(p1[1:]),np.min(p1[1:]),np.mean(p1),np.std(p1))
        #print(ng_index)
        
        plt.clf()#清屏
        plt.ion()#不会阻塞线程
        plt.axis([0,50,0,2])
        plt.plot(p1, c='blue')
        plt.show()

cv2.namedWindow('dis')
cv2.setMouseCallback("dis",onmouse)
cv2.imshow('dis', src)
cv2.waitKey()

"""
总结：总体来说实现了思路，目前存在的问题如下
1、大量异常通常出现在图像未准确对齐的图像上
2、检测时间较长
解决方案：
1、正常测试时，图像想过应该比现在好很多，至少能保证不会出现过于模糊的图像。
2、图像对齐初步依靠模板匹配解决。
3、检测时间长的问题尝试使用深度学习训练的特征提取器提取特征进行解决。
第1、2在改造的机器上比较容易解决，现在优先测试第三个问题。
"""
