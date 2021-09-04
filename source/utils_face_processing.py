from typing import List
import cv2
import dlib
import numpy as np
from numpy.lib.index_tricks import IndexExpression
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


def face_detector(img, img_gray, isDisplay=False):
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
            if isDisplay:
                cv2.circle(img, (x, y), 3, (0, 0, 255), -1)
    images_show([img])
    return landmarks_points


def detect_convex_hull(img, mask, landmarks_points, isDisplay=False):
    points = np.array(landmarks_points, np.int32)
    convexhull = cv2.convexHull(points)
    if isDisplay:
        cv2.polylines(img, [convexhull], True, (255, 0, 0), 3)
        images_show([img])
    cv2.fillConvexPoly(mask, convexhull, 255)
    face = cv2.bitwise_and(img, img, mask=mask)
    return face, convexhull, points


def delaunay_triangle_face2nd(img2, indexes_triangles, landmarks_points, isDisplay=False):
    indexes_triangles2 = []
    for triangle_index in indexes_triangles:
        pt1 = landmarks_points[triangle_index[0]]
        pt2 = landmarks_points[triangle_index[1]]
        pt3 = landmarks_points[triangle_index[2]]
        indexes_triangles2.append([pt1, pt2, pt3])
        if isDisplay:
            cv2.line(img2, pt1, pt2, (0, 0, 255), 2)
            cv2.line(img2, pt3, pt2, (0, 0, 255), 2)
            cv2.line(img2, pt1, pt3, (0, 0, 255), 2)
    if isDisplay:
        images_show([img2])
    return indexes_triangles2


def delaunay_triangle(convexhull, img, landmarks_points, points, isDisplay=False):
    rect = cv2.boundingRect(convexhull)
    subdiv = cv2.Subdiv2D(rect)
    subdiv.insert(landmarks_points)
    triangles = subdiv.getTriangleList()
    triangles = np.array(triangles, dtype=np.int32)
    indexes_triangeles = []
    for t in triangles:
        pt1, pt2, pt3 = draw_triangle(t, img, isDisplay)
        index_pt1 = extract_index_nparray(
            np.where((points == pt1).all(axis=1)))
        index_pt2 = extract_index_nparray(
            np.where((points == pt2).all(axis=1)))
        index_pt3 = extract_index_nparray(
            np.where((points == pt3).all(axis=1)))

        if index_pt1 is not None and index_pt2 is not None and index_pt3 is not None:
            triangle = [index_pt1, index_pt2, index_pt3]
            indexes_triangeles.append(triangle)
    if isDisplay:
        images_show([img])
    return indexes_triangeles


def draw_triangle(triangle, img, isDisplay):
    pt1 = (triangle[0], triangle[1])
    pt2 = (triangle[2], triangle[3])
    pt3 = (triangle[4], triangle[5])
    if isDisplay:
        cv2.line(img, pt1, pt2, (0, 0, 255), 2)
        cv2.line(img, pt2, pt3, (0, 0, 255), 2)
        cv2.line(img, pt1, pt3, (0, 0, 255), 2)
    return pt1, pt2, pt3


def extract_index_nparray(nparray):
    index = None
    for num in nparray[0]:
        index = num
        break
    return index


def wrap_faces(img, img2, indexes_triangles, landmarks_points_img, landmarks_points_img2):
    for triangle_index in indexes_triangles:
        points, cropped_triangle, rect1 = get_triangle(
            img, landmarks_points_img, triangle_index)
        points2, cropped_triangle2, rect2 = get_triangle(
            img2, landmarks_points_img2, triangle_index)
        # Warp triangles
        (x, y, w, h) = rect2
        points = np.float32(points)
        points2 = np.float32(points2)
        M = cv2.getAffineTransform(points, points2)
        warped_triangle = cv2.warpAffine(cropped_triangle, M, (w, h))
        # images_show([cropped_triangle, cropped_triangle2, warped_triangle])
        # break


def get_triangle(img, landmarks_points, triangle_index):
    pt1 = landmarks_points[triangle_index[0]]
    pt2 = landmarks_points[triangle_index[1]]
    pt3 = landmarks_points[triangle_index[2]]
    triangle = np.array([pt1, pt2, pt3], np.int32)
    rect = cv2.boundingRect(triangle)
    (x, y, w, h) = rect
    cropped_triangle = img[y: y + h, x: x + w]
    cropped_mask = np.zeros((h, w), np.uint8)
    points = np.array([[pt1[0] - x, pt1[1] - y],
                      [pt2[0] - x, pt2[1] - y],
                      [pt3[0] - x, pt3[1] - y]], np.int32)
    cv2.fillConvexPoly(cropped_mask, points, 255)
    cropped_triangle = cv2.bitwise_and(cropped_triangle, cropped_triangle,
                                       mask=cropped_mask)
    return points, cropped_triangle, rect


img, img_gray, mask = read_img('./image_data/face1.jpg')
img2, img2_gray, _ = read_img('./image_data/face2.jpg')
# images_show([img, img_gray, mask])
landmarks_points_img = face_detector(img, img_gray)
landmarks_points_img2 = face_detector(img2, img2_gray)

face, convexhull, points = detect_convex_hull(img, mask, landmarks_points_img)
indexes_triangles = delaunay_triangle(
    convexhull, img, landmarks_points_img, points)
# indexes_triangles2 = delaunay_triangle_face2nd(
#     img2, indexes_triangles, landmarks_points_img2)

wrap_faces(img, img2, indexes_triangles,
           landmarks_points_img, landmarks_points_img2)

# images_show([face])
