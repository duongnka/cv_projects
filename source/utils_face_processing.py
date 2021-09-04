from typing import List
import cv2
import dlib
import numpy as np
from numpy.lib.twodim_base import tri


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
    return landmarks_points


def detect_convex_hull(img, mask, landmarks_points):
    points = np.array(landmarks_points, np.int32)
    convexhull = cv2.convexHull(points)
    cv2.polylines(img, [convexhull], True, (255, 0, 0), 3)
    images_show([img])
    cv2.fillConvexPoly(mask, convexhull, 255)
    face = cv2.bitwise_and(img, img, mask=mask)
    return face, convexhull


def delaunay_triangle(convexhull, img):
    rect = cv2.boundingRect(convexhull)
    subdiv = cv2.Subdiv2D(rect)
    subdiv.insert(landmarks_points)
    triangles = subdiv.getTriangleList()
    triangles = np.array(triangles, dtype=np.int32)

    for t in triangles:
        draw_triangle(t, img)

    images_show([img])


def draw_triangle(triangle, img):
    pt1 = (triangle[0], triangle[1])
    pt2 = (triangle[2], triangle[3])
    pt3 = (triangle[4], triangle[5])

    cv2.line(img, pt1, pt2, (0, 0, 255), 2)
    cv2.line(img, pt2, pt3, (0, 0, 255), 2)
    cv2.line(img, pt1, pt3, (0, 0, 255), 2)


img, img_gray, mask = read_img('./image_data/face2.jpg')
# images_show([img, img_gray, mask])
landmarks_points = face_detector(img, img_gray)
face, convexhull = detect_convex_hull(img, mask, landmarks_points)
delaunay_triangle(convexhull, img)
# images_show([face])
