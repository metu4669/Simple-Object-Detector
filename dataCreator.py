import cv2
import tensorflow as tf
import numpy as np
from PIL import Image
import os

drawing = False


def draw(event, x, y, flags, param):
    global drawing
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            cv2.circle(img1, (x, y), 10, (255, 0, 0), -1)


img1 = np.zeros((512, 512, 1), np.uint8)
i = 0
while 1:
    cv2.setMouseCallback("Image", draw)
    cv2.imshow("Image", img1)

    k = cv2.waitKey(1) & 0XFF
    if k == ord('s'):
        break
    elif k == ord('k'):
        img2 = img1/255.0
        img2 = cv2.resize(img2, (28, 28), cv2.INTER_AREA)
        img2 = img2.reshape(1, 28, 28)

        height, width, depth = img1.shape
        imgScale = 28 / width
        newX, newY = img1.shape[1] * imgScale, img1.shape[0] * imgScale
        new = cv2.resize(img1, (int(newX), int(newY)))

        path = os.getcwd()+"/DATA/person/"
        cv2.imwrite(str(path) + str(i) + ".jpg", new)
        i += 1
    elif k == ord('c'):
        img1 = np.zeros((512, 512, 1), np.uint8)
        img3 = np.zeros((512, 512, 1), np.uint8)
        pr = ''
cv2.destroyAllWindows()
