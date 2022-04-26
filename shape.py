import cv2
import numpy as np
from keras.models import model_from_json
from keras import backend as k
import tensorflow as tf


def get_img_contour_thresh(img):
    x, y, w, h = 0, 50, 300, 300
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    ret, thresh1 = cv2.threshold(blur, 175, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    thresh1 = thresh1[y:y + h, x:x + w]
    contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    return img, contours, thresh1


def get_img_gradient(img):
    edges = cv2.Canny(img, 100, 200)
    return edges


def show_web_cam(mirror=False):
    # load json and create model
    model = tf.keras.models.load_model('rakam_final.model')
    print("Loaded model from disk")

    cap = cv2.VideoCapture('http://192.168.104.101:4747/video')
    while True:
        ret, img = cap.read()
        # print(ret)
        img, contours, thresh = get_img_contour_thresh(img)
        edge = get_img_gradient(img)
        ans = ''

        centres = []
        if len(contours) > 0:
            contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(contour) > 500:
                x, y, w, h = cv2.boundingRect(contour)
                new_image = thresh[y:y + h, x:x + w]
                new_image = cv2.resize(new_image, (28, 28))
                new_image = np.array(new_image)
                new_image = new_image.astype('float32')
                new_image /= 255
                
                if k.image_data_format() == 'channels_first':
                    new_image = new_image.reshape(1, 28, 28)
                else:
                    new_image = new_image.reshape(28, 28)

                new_image = np.expand_dims(new_image, axis=0)
                ans = model.predict(new_image).argmax()
        x, y, w, h = 0, 50, 300, 300
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 3)

        cv2.putText(img, "Recognized Value : " + str(ans), (0, 320),
                    cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(img, "Created By", (0, 370),
                    cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 255), 1)
        cv2.putText(img, "Omer CELEBI", (0, 390),
                    cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 255), 1)
        cv2.putText(img, "Furkan COBAN", (0, 410),
                    cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 255), 1)

        cv2.imshow("Frame", img)
        cv2.imshow("Contours", thresh)
        #cv2.imshow("Edge", edge)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


def main():
    show_web_cam(mirror=False)


if __name__ == '__main__':
    main()


'''
            for c in contours:
                M = cv2.moments(c)

                # calculate x,y coordinate of center
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                else:
                    cX, cY = 0, 0
                cv2.circle(thresh, (cX, cY), 5, (255, 255, 255), -1)
                # cv2.putText(img, "centroid", (cX - 25, cY - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                cv2.rectangle(thresh, (cX - 25, cY - 25), (cX + 20, cY + 20), (0, 0, 255), 3)
                x, y, w, h = 0, 50, 300, 300
                cv2.rectangle(thresh, (x, y), (x + w, y + h), (0, 0, 255), 3)

                # display the image
                # cv2.imshow("Image", thresh)
'''