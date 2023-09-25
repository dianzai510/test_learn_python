import cv2
import scipy
from scipy import interpolate
import torch
import numpy as np
import matplotlib.pyplot as plt
from our1314.work.Utils import *

"""
生成切割道训练图像思路：
1、生成矩形四个角点坐标
2、固定步长生成随机Y点坐标
3、样条插值连线
"""
def gen_rect1(len,off,num=20):
    #1
    pts1 = gen_line(len,off)
    pts1 = np.vstack([pts1, np.ones([1,pts1.shape[1]])])
    pts1  = SE2(-len//2,-len//2,0).dot(pts1)
    #4
    pts2 = gen_line(len,off)
    pts2 = np.vstack([pts2, np.ones([1,pts2.shape[1]])])
    pts2  = SE2(len//2,-len//2,pi/2).dot(pts2)
    pts = np.hstack([pts1,pts2])
    #2
    pts3 = gen_line(len,off)
    pts3 = np.vstack([pts3, np.ones([1,pts3.shape[1]])])
    pts3  = SE2(-len//2,len//2,0).dot(pts3)
    pts3 = np.fliplr(pts3)

    #3
    pts4 = gen_line(len,off)
    pts4 = np.vstack([pts4, np.ones([1,pts4.shape[1]])])
    pts4  = SE2(-len//2,-len//2,pi/2).dot(pts4)
    pts4 = np.fliplr(pts4)
    
    pts = np.hstack([pts1,pts2,pts3,pts4])
    return pts

def gen_line(len,off,num=20):
    _min,_max = off
    x = np.linspace(0,len,num)
    y = np.random.randint(_min,_max,size=num)
    f = interpolate.interp1d(x,y,kind='linear')
    
    x = np.arange(len)
    y = f(x)

    pts = np.stack([x,y],axis=0)
    return pts

for i in range(10**8):
    pts = gen_rect1(len=200,off=(-3,3),num=20)
    img = np.zeros([300,300,3], np.int32)
    h,w,c = img.shape
    theta = np.random.rand(1)*pi/12 - pi/24
    pts = SE2(h/2,w/2,theta).dot(pts)
    pts = pts[0:2,:].T.astype(np.int32)

    color = (random.randint(0,255),random.randint(0,255),random.randint(0,255))
    if np.linalg.norm(np.array(list(color)))<20:
        continue
       
    cv2.fillPoly(img,[pts],color)

    pts = gen_rect1(len=170,off=(-1.5,1.5),num=100)
    pts = SE2(h/2,w/2,theta).dot(pts)
    pts = pts[0:2,:].T.astype(np.int32)
    color2 = (random.randint(0,255),random.randint(0,255),random.randint(0,255))
    cv2.fillPoly(img,[pts],color2)

    color = (random.randint(0,80),random.randint(0,80),random.randint(0,80))
    cv2.polylines(img, [pts], True, color, random.randint(1,3))
    
    noise = np.random.randn(h,w,c) * random.randint(5,10)
    img[img>0] = img[img>0] + noise[img>0]

    img[img>255]=255
    img[img<=0]=0
    img = img.astype("uint8")

    cv2.imshow("dis",img)
    cv2.waitKey()