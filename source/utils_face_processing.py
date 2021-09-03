from typing import List
import cv2
import dlib
import numpy as np


def read_img(path):
    img = cv2.imread(path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = np.zeros_like(img_gray)
    return img, img_gray, mask


def images_show(images: List):
    order = 1
    for img in images:
        cv2.imshow(f'{order}', img)
        order += 1
    cv2.waitKey()
    cv2.destroyAllWindows


def face_detector(img, img_gray):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
    faces = detector(img_gray)
    for face in faces:
        landmarks = predictor(img_gray, face)
        landmarks_points = []
        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            landmarks_points.append((x, y))
            cv2.circle(img, (x, y), 3, (0, 0, 255), -1)

    images_show([img])


img, img_gray, mask = read_img('./image_data/face1.jpg')
# images_show([img, img_gray, mask])
face_detector(img, img_gray)
