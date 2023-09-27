"""
思路：
0、转换为梯度图像
1、从切割道边缘开始滑窗+分类。
2、非极大值抑制
"""

import cv2
import numpy as np
from math import *
import matplotlib.pyplot as plt
from our1314.work.Utils import *

def 切割道图像转正(img):

    pass

def ransac_line(pts):
    choice = np.random.choice(len(pts),2)
    p1,p2 = pts[choice]
    pass




if __name__ == "__main__":
    # path = "D:/work/files/deeplearn_datasets/其它数据集/切割道检测/芯片尺寸/1C603AA20221223/85.jpg"
    # img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    # tmp = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # a,tmp = cv2.threshold(tmp, 20, 255, cv2.THRESH_BINARY)
    # tmp = cv2.bitwise_not(tmp)
    # cv2.imshow("dis",tmp)
    # cv2.waitKey()

    #1、生成直线，并随机抽取坐标点
    t = np.linspace(0,100,50)
    #t = np.random.choice(t,50)
    theta = pi/3
    x0,y0 = 0,0
    x = x0 + t*cos(theta)
    y = y0 + t*sin(theta)
    x = np.expand_dims(x,axis=0)
    y = np.expand_dims(y,axis=0)
    pts = np.vstack([x,y])

    #2、沿着直线垂直方向添加高斯噪声
    vec_line = np.array([[cos(theta),sin(theta)]])
    vec_line = SO2(pi/2).dot(vec_line.T)
    
    noise = np.random.randn(pts.shape[1])*vec_line
    pts = pts + noise

    #3、随机添加外点噪声
    outlier = np.random.randint(0,100,[2,10])
    pts = np.hstack([pts,outlier])

    plt.plot(pts[0,:],pts[1,:],'.r')
    plt.show()
    pass

