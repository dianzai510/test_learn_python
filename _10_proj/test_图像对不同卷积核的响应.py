import cv2
import numpy as np


img = np.ones([400,400,3],dtype=np.uint8)*128
cv2.circle(img, (200,300),10,(255,255,255),-1)
cv2.rectangle(img, (300,200),(300+5,200+5),(255,255,255),-1)
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