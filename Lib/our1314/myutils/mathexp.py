import numpy as np
from math import *

def SE2(x,y,rad):
    r = np.array([
        [cos(rad),-sin(rad),x],
        [sin(rad),cos(rad),y],
        [0,0,1]
        ])
    return r

def _SE2(H):
    x,y = H[0,2],H[1,2]
    cos_theta = H[0,0]
    sin_theta = H[1,0]
    theta = atan2(cos_theta, sin_theta)
    return x,y,theta

#绕3D坐标系X轴旋转的旋转矩阵
def RX(rad):
    r = np.array([
        [1,0,0],
        [0,cos(rad),-sin(rad)],
        [0,sin(rad),cos(rad)]
        ])
    return r
#绕3D坐标系X轴旋转的旋转矩阵
def RY(rad):
    r = np.array([
        [cos(rad),0,sin(rad)],
        [0,1,0],
        [-sin(rad),0,cos(rad)]
        ])
    return r
#绕3D坐标系X轴旋转的旋转矩阵
def RZ(rad):
    r = np.array([
        [cos(rad),-sin(rad),0],
        [sin(rad),cos(rad),0],
        [0,0,1]
        ])
    return r

def Pxyz(x,y,z):
    H = np.eye(4)
    H[0,3],H[1,3],H[2,3]=x,y,z
    return H

def Homogeneous(m):
    h,w = m.shape
    m = np.column_stack([m,np.zeros(h,1)])
    m = np.row_stack([m,np.zeros(1,w+1)])
    m[-1,-1]=1
    return m

def SE3(px,py,pz,rx,ry,rz):
    Rx = Homogeneous(RX(rx))
    Ry = Homogeneous(RY(ry))
    Rz = Homogeneous(RZ(rz))
    P = Pxyz(px,py,pz)
    H = P@Rz@Ry@Rx
    return H

def _SE3(H):
    pass


def deg(rad):
    return rad*180/pi

def rad(deg):
    return deg*pi/180