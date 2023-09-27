import cv2
import torch
import numpy as np
import torchvision.transforms.functional as F

"""
结论：
卷积神经网络的操作实际与图像卷积是一样的。
"""

#region opencv
src = np.ones([400,400,1],dtype=np.uint8)*128


cv2.circle(src, (200,300),10,(255,255,255),-1)
cv2.rectangle(src, (300,200),(300+5,200+5),(255,255,255),-1)
img = src.copy()
cv2.imshow("dis",img)
cv2.waitKey()

kernel = np.zeros([11,11],dtype=float)
kernel = cv2.circle(kernel,(5,5),5,1)
kernel = kernel/np.sum(kernel)
a = cv2.filter2D(img, 3, kernel)
cv2.imshow("dis",a/255)
cv2.waitKey()

kernel = np.array([[-0.5,0,0.5]])
a = cv2.filter2D(img, 3, kernel)
cv2.imshow("dis",a/255)
cv2.waitKey()

kernel = np.array([[-0.5],[0],[0.5]])
a = cv2.filter2D(img, 3, kernel)
cv2.imshow("dis",a/255)
cv2.waitKey()
#endregion

#region pytorch
img = F.to_tensor(src)

conv1 = torch.nn.Conv2d(1,1,kernel_size=11,stride=1,padding=0,bias=False)

kernel = np.zeros([11,11],dtype=float)
kernel = cv2.circle(kernel,(5,5),5,1)
kernel = kernel/np.sum(kernel)
kernel = np.expand_dims(kernel,axis=0)
kernel = np.expand_dims(kernel,axis=0)
kernel = kernel.astype("float")
kernel = torch.from_numpy(kernel)
kernel = kernel.type(torch.float)
w1 = conv1.weight
conv1.weight = torch.nn.Parameter(kernel,requires_grad=False)
w2 = conv1.weight
a = conv1(img)
a = torch.permute(a,[1,2,0])
a = a.detach().numpy()
cv2.imshow("dis",a)
cv2.waitKey()

conv1 = torch.nn.Conv2d(in_channels=1,out_channels=1,kernel_size=(1,3),stride=1,padding=0,bias=False,)

kernel = np.array([[-0.5,0,0.5]],dtype=float)
kernel = np.expand_dims(kernel,axis=0)
kernel = np.expand_dims(kernel,axis=0)
kernel = kernel.astype("float")
kernel = torch.from_numpy(kernel)
kernel = kernel.type(torch.float)
conv1.weight = torch.nn.Parameter(kernel,requires_grad=False)
a = conv1(img)
a = torch.permute(a,[1,2,0])
a = a.detach().numpy()
cv2.imshow("dis",a)
cv2.waitKey()

conv1 = torch.nn.Conv2d(in_channels=1,out_channels=1,kernel_size=(1,3),stride=1,padding=0,bias=False,)

kernel = np.array([[-0.5],[0],[0.5]],dtype=float)
kernel = np.expand_dims(kernel,axis=0)
kernel = np.expand_dims(kernel,axis=0)
kernel = kernel.astype("float")
kernel = torch.from_numpy(kernel)
kernel = kernel.type(torch.float)
conv1.weight = torch.nn.Parameter(kernel,requires_grad=False)
a = conv1(img)
a = torch.permute(a,[1,2,0])
a = a.detach().numpy()
cv2.imshow("dis",a)
cv2.waitKey()
pass
#endregion