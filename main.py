import numpy as np
import cv2

img = np.zeros((512,512,3),np.uint8)
cv2.line(img,(0,0),(511,511),(245,0,0),5)
winname = "example"
cv2.namedWindow(winname)
cv2.imshow(winname,img)
cv2.waitKey(0)
cv2.destroyWindow(winname)