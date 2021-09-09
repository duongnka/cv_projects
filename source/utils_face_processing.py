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

def extract_index_nparray(nparray):
    index = None
    for num in nparray[0]:
        index = num
        break
    return index

def images_show(images: List):
    order = 1
    for img in images:
        cv2.imshow(f'{order}', img)
        order += 1
    cv2.waitKey()
    cv2.destroyAllWindows

def get_biggest_face_rect(img_gray):
    detector = dlib.get_frontal_face_detector()
    face = max(detector(img_gray), key= lambda r: r.area())
    return face

def get_landmarks_points(img_gray):
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
    face = get_biggest_face_rect(img_gray)
    landmarks = predictor(img_gray, face)
    landmarks_points = []
    for n in range(0, 68):
        x = landmarks.part(n).x
        y = landmarks.part(n).y
        landmarks_points.append((x, y))
    
    return landmarks_points

def get_triangles_indees(landmarks_points):
    points = np.array(landmarks_points, np.int32)
    convexhull = cv2.convexHull(points)
    rect = cv2.boundingRect(convexhull)
    subdiv = cv2.Subdiv2D(rect)
    subdiv.insert(landmarks_points)
    triangles = subdiv.getTriangleList()
    triangles = np.array(triangles, dtype=np.int32)

    indexes_triangles = []
    for t in triangles:
        pts = [(t[0], t[1]), (t[2], t[3]), (t[4], t[5])]
        triangle = []
        for pt in pts:
            index = np.where((points == pt).all(axis=1))
            index = extract_index_nparray(index)
            if index is None:
                break
            triangle.append(index)
        if len(triangle) == 3:
            indexes_triangles.append(triangle)
    return indexes_triangles

def get_triangle(landmarks_points, index):
    return [ landmarks_points[index[0]], 
             landmarks_points[index[1]], 
             landmarks_points[index[2]] ]
             
def swap_face(img, img_gray, img2, img2_gray):

    landmarks_points = get_landmarks_points(img_gray)
    landmarks_points2 = get_landmarks_points(img2_gray)

    indexes_triangles = get_triangles_indees(landmarks_points)

    lines_space_mask = np.zeros_like(img_gray)
    lines_space_new_face = np.zeros_like(img2)
    height, width, channels = img2.shape
    img2_new_face = np.zeros((height, width, channels), np.uint8)

    # Triangulation of both faces
    for triangle_index in indexes_triangles:
        # Triangulation of the first face
        tr1_pt = get_triangle(landmarks_points, triangle_index)
        triangle1 = np.array([tr1_pt[0], tr1_pt[1], tr1_pt[2]], np.int32)

        rect1 = cv2.boundingRect(triangle1)
        (x, y, w, h) = rect1
        cropped_triangle = img[y: y + h, x: x + w]
        cropped_tr1_mask = np.zeros((h, w), np.uint8)

        points = np.array([[tr1_pt[0][0] - x, tr1_pt[0][1] - y],
                        [tr1_pt[1][0] - x, tr1_pt[1][1] - y],
                        [tr1_pt[2][0] - x, tr1_pt[2][1] - y]], np.int32)

        cv2.fillConvexPoly(cropped_tr1_mask, points, 255)

        # Lines space
        cv2.line(lines_space_mask, tr1_pt[0], tr1_pt[1], 255)
        cv2.line(lines_space_mask, tr1_pt[1], tr1_pt[2], 255)
        cv2.line(lines_space_mask, tr1_pt[0], tr1_pt[2], 255)
        lines_space = cv2.bitwise_and(img, img, mask=lines_space_mask)

        # Triangulation of second face
        tr2_pt1 = landmarks_points2[triangle_index[0]]
        tr2_pt2 = landmarks_points2[triangle_index[1]]
        tr2_pt3 = landmarks_points2[triangle_index[2]]
        triangle2 = np.array([tr2_pt1, tr2_pt2, tr2_pt3], np.int32)

        rect2 = cv2.boundingRect(triangle2)
        (x, y, w, h) = rect2

        cropped_tr2_mask = np.zeros((h, w), np.uint8)

        points2 = np.array([[tr2_pt1[0] - x, tr2_pt1[1] - y],
                            [tr2_pt2[0] - x, tr2_pt2[1] - y],
                            [tr2_pt3[0] - x, tr2_pt3[1] - y]], np.int32)

        cv2.fillConvexPoly(cropped_tr2_mask, points2, 255)

        # Warp triangles
        points = np.float32(points)
        points2 = np.float32(points2)
        M = cv2.getAffineTransform(points, points2)
        warped_triangle = cv2.warpAffine(cropped_triangle, M, (w, h))
        warped_triangle = cv2.bitwise_and(
            warped_triangle, warped_triangle, mask=cropped_tr2_mask)

        # Reconstructing destination face
        img2_new_face_rect_area = img2_new_face[y: y + h, x: x + w]
        img2_new_face_rect_area_gray = cv2.cvtColor(
            img2_new_face_rect_area, cv2.COLOR_BGR2GRAY)
        _, mask_triangles_designed = cv2.threshold(
            img2_new_face_rect_area_gray, 1, 255, cv2.THRESH_BINARY_INV)
        warped_triangle = cv2.bitwise_and(
            warped_triangle, warped_triangle, mask=mask_triangles_designed)

        img2_new_face_rect_area = cv2.add(img2_new_face_rect_area, warped_triangle)
        img2_new_face[y: y + h, x: x + w] = img2_new_face_rect_area

    points2 = np.array(landmarks_points2, np.int32)
    convexhull2 = cv2.convexHull(points2)
    # Face swapped (putting 1st face into 2nd face)
    img2_face_mask = np.zeros_like(img2_gray)
    img2_head_mask = cv2.fillConvexPoly(img2_face_mask, convexhull2, 255)
    img2_face_mask = cv2.bitwise_not(img2_head_mask)


    img2_head_noface = cv2.bitwise_and(img2, img2, mask=img2_face_mask)
    result = cv2.add(img2_head_noface, img2_new_face)

    bb = cv2.boundingRect(convexhull2)
    center_face2 = (bb[0] + int(bb[2] / 2), bb[1] + int(bb[3] / 2))

    seamlessclone = cv2.seamlessClone(
        result, img2, img2_head_mask, center_face2, cv2.NORMAL_CLONE)
    return seamlessclone

img, img_gray, mask = read_img('./image_data/face1.jpg')
img2, img2_gray, mask2 = read_img('./image_data/face2.jpg')
swapped_face = swap_face(img, img_gray, img2, img2_gray)
images_show([swapped_face])