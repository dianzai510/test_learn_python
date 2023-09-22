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
import faiss
from our1314.work.Utils import GetAllFiles
from queue import Queue
import gc


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

cnt = 0
#dir_image = "D:/work/files/deeplearn_datasets/choujianji/roi-mynetseg/test/test"
dir_image = "D:/work/proj/抽检机/program/抽检机/bin/net7.0-windows/data/roi"
files_all = GetAllFiles(dir_image)
files_all = [f for f in files_all if f.endswith('.png') or f.endswith('.jpg') or f.endswith('.bmp')]

cnt_queue = 100
queuq_image = Queue(cnt_queue)
faiss_index = faiss.IndexFlatL2(1024)
patchcore = PatchCore(torch.device("cuda:1"))

for i,path in enumerate(files_all):
    images_queue = files_all[i:i+cnt_queue]

    imgs = [transform(Image.open(f).convert('RGB')) for f in images_queue]#读取队列内的所有图像
    imgs = torch.stack(imgs,dim=0)#合并为张量

    #region patchcore提取特征
    with torch.no_grad():
        input_image = imgs.to(torch.device("cuda:1"))
        feas = patchcore._embed(input_image)
        feas = np.array(feas)
        s = int(sqrt(len(feas)/cnt_queue))
        feas = feas.reshape(-1,s,s,1024)
        feas_train = feas.reshape(-1,1024)
        faiss_index.add(feas_train)
    #endregion

    #region 遍历图像，计算异常分
    """
    如果以当前特征与所有特征距离的平均值作为异常分可能有问题。
    尝试以最近的几个(10个)近邻的均值作为异常分。
    """
    dd = []
    for y in range(s):
        for x in range(s):
            X = feas[:,y,x,:]
            d = [np.linalg.norm(X[0]-p) for p in X[1:]]#计算当前特征与所有特征的距离
            d = np.sort(d)[:10]#取前10个

            # faiss_index.reset()
            # faiss_index.add(X[1:])
            # d = faiss_index.search(X[0:1],10)[0]

            dd.append(np.mean(d))#求平均值
            
    dd = np.array(dd)
    dd = dd.reshape(s,s)-0.5
    #dd = (dd - np.min(dd)) / (np.max(dd) - np.min(dd))
    dd = dd/2
    dd = dd + 0.29 - np.mean(dd)

    dd = cv2.resize(dd, (224,224), cv2.INTER_LINEAR)
    cv2.imshow("dis", dd)
    cv2.waitKey(1)
    #endregion
    dd = dd*255
    dd = dd.astype("int32")
    cv2.imwrite(f"D:/desktop/eee/{os.path.basename(path)}", dd)
    gc.collect()



#鼠标点选计算异常
src = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)#type:np.ndarray
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
