import cv2
import numpy as np

class mouseSelect():
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
            cv2.waitKey(1)

        if event == cv2.EVENT_LBUTTONUP:
            self.down = False
            self.pt2 = np.array([x,y])
            cv2.waitKey(200)
            cv2.destroyWindow(self.windowName)
            return self.pt1, self.pt2


if __name__ == '__main__':
    path = 'd:/desktop/1.jpg'
    src = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)#type:np.ndarray
    a = mouseSelect(src)
    