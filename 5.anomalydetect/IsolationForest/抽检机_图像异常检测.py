import os
from random import shuffle
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor

path = 'D:/desktop/tmp2.png'#input('输入图像路径：')
src = cv2.imdecode(np.fromfile(path, dtype=np.uint8), -1)#type:np.ndarray

dir_image = "D:\work\proj\抽检机\program\ChouJianJi\data\ic"
files_all = os.listdir(dir_image)
images_path = [os.path.join(dir_image, f) for f in files_all if f.endswith('.png') or f.endswith('.jpg') or f.endswith('.bmp')]
images_path.append(path)
shuffle(images_path)

imgs = [cv2.imdecode(np.fromfile(f, dtype=np.uint8), cv2.IMREAD_COLOR) for f in images_path]
imgs = np.array(imgs)

clf = LocalOutlierFactor(n_neighbors=80, contamination=0.01)

def onmouse(*p):
    event, x, y, flags, param=p
    if event == cv2.EVENT_LBUTTONDOWN:
        dis = src.copy()

        #1、绘制鼠标位置
        #cv2.circle(dis, (x,y), 5, (0,0,255), 2)
        cv2.line(dis,(x,y),(x,y),(0,0,255),1)
        cv2.imshow("dis", dis)

        #2、绘制曲线图
        b = imgs[:,y,x,0]
        g = imgs[:,y,x,1]
        r = imgs[:,y,x,2]
        #x = np.arange(0,imgs.shape[0])

        plt.clf()#清屏
        
        plt.subplot(311)
        plt.plot(b, c='blue')
        plt.subplot(312)
        plt.plot(g, c='green')
        plt.subplot(313)
        plt.plot(r, c='red')
        plt.legend([],[])
        plt.show()

        #3、异常检测
        # 1、收集异常点所在的图像索引
        # 2、所在图像索引上的异常坐标的聚类
        X = imgs[:,y,x,:]
        pred = clf.fit_predict(X)
        #pred = clf.score_samples(X)
        ng_index = [i for i,p in enumerate(pred) if p<0]
        print(ng_index)
        

cv2.namedWindow('dis')
cv2.setMouseCallback("dis",onmouse)
cv2.imshow('dis', src)
cv2.waitKey()


