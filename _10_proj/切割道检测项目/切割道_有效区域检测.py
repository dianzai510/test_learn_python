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
    path = "D:/work/files/deeplearn_datasets/其它数据集/切割道检测/芯片尺寸/1C603AA20221223/85.jpg"
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    tmp = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    a,tmp = cv2.threshold(tmp, 20, 255, cv2.THRESH_BINARY)
    tmp = cv2.bitwise_not(tmp)
    size = tmp.shape[:2]
    cv2.imshow("dis",tmp)
    cv2.waitKey()
    pass

