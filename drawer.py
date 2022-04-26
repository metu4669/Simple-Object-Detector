import cv2
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import  Image
import os

img3 = np.zeros((512, 512, 3), np.uint8)
font = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (5, 50)
fontScale = 1
fontColor = (255, 255, 255)
lineType = 2
'''
mnist_data = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist_data.load_data()
x_train, x_test = x_train/255, x_test/255

createdModel = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(512, activation=tf.nn.relu),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

createdModel.compile(optimizer='adam',
                     loss='sparse_categorical_crossentropy',
                     metrics=['accuracy'])
createdModel.fit(x_train, y_train)
createdModel.evaluate(x_test, y_test)
'''
createdModel = tf.keras.models.load_model("final.model")
drawing = False


def draw(event, x, y, flags, param):
    global drawing
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        print("LEFT Clicked")
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            cv2.circle(img1, (x, y), 10, (255, 0, 0), -1)


img1 = np.zeros((512, 512, 1), np.uint8)

cv2.namedWindow("Drawing Image")
cv2.setMouseCallback("Image", draw)
pr = ""
while 1:
    cv2.setMouseCallback("Image", draw)
    cv2.imshow("Image", img1)
    # Display the image
    cv2.putText(img3, 'Predicted Result: ' + str(pr),
                bottomLeftCornerOfText,
                font,
                fontScale,
                fontColor,
                lineType)
    cv2.imshow("img", img3)

    k = cv2.waitKey(1) & 0XFF
    if k == ord('s'):
        break
    elif k == ord('p'):
        img2 = img1/255.0
        img2 = cv2.resize(img2, (28, 28), cv2.INTER_AREA)
        img2 = img2.reshape(1, 28, 28)
        pr = createdModel.predict_classes(img2)
    elif k == ord('c'):
        img1 = np.zeros((512, 512, 1), np.uint8)
        img3 = np.zeros((512, 512, 1), np.uint8)
        pr = ''
cv2.destroyAllWindows()
