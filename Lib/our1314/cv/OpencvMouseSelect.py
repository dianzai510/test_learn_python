import cv2
import numpy as np

class mouseSelect_simple():
    def __init__(self, src, windowName='dis'):
        self.src = src
        self.windowName = windowName
        self.down = False
        
        cv2.namedWindow(windowName)
        cv2.setMouseCallback(windowName, self.onmouse)
        cv2.imshow(windowName, src)
        cv2.waitKey()

    def onmouse(self, *p):
        event, x, y, flags, param = p   
        if event == cv2.EVENT_LBUTTONDOWN:
            self.down = True
            self.pt1 = np.array([x,y])

        if event == cv2.EVENT_MOUSEMOVE and self.down==True:
            self.pt2 = np.array([x,y])
            dis = self.src.copy()
            cv2.rectangle(dis, self.pt1, self.pt2, (0,0,255), 2)
            cv2.imshow(self.windowName , dis)

        if event == cv2.EVENT_LBUTTONUP:
            self.down = False
            self.pt2 = np.array([x,y])
            return self.pt1, self.pt2
        
        if event == cv2.EVENT_RBUTTONDOWN:
            cv2.waitKey(200)
            cv2.destroyWindow(self.windowName)


# if __name__ == '__main__':
#     path = input('输入图像路径：')
#     src = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)#type:np.ndarray
#     a = mouseSelect(src)
#     print(f'{a.pt1},{a.pt2}')
    