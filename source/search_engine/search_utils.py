from typing import List
import imutils
import cv2
from matplotlib import pyplot as plt
import glob
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle
import os

def save_obj(obj, name ):
    with open('obj/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open('./search_engine/obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)


def chi2_distance(histA, histB, eps=1e-10):
    d = 0.5 * np.sum([((a - b) ** 2) / (a + b + eps)
                      for (a, b) in zip(histA, histB)])
    return d


def get_list_images_paths(folder_path = "./data_set/17flowers/*.jpg"):
    return glob.glob(folder_path)


def create_inverted_index(images_histograms, isSave = False):

    inverted_histogram_index = {}

    for i, feature in enumerate(images_histograms.values()):
        for fi, value in enumerate(feature):
            value_item = []
            feature_item = {}

            if fi in inverted_histogram_index:
                feature_item = inverted_histogram_index[fi]

            if value in feature_item:
                value_item = feature_item[value]

            value_item.append(i)
            feature_item[value] = value_item
            inverted_histogram_index[fi] = feature_item
    
    if isSave:
        save_obj(inverted_histogram_index, 'inverted_histogram_index')

    return inverted_histogram_index


def get_combined_list(list_1, list_2):

    set_1 = set(list_1)
    set_2 = set(list_2)
    list_2_items_not_in_list_1 = list(set_2 - set_1)
    combined_list = list_1 + list_2_items_not_in_list_1

    return combined_list


class RGBHistogram:
    def __init__(self, bins : List[int]):
        self.bins = bins

    def describe(self, image):
        hist = cv2.calcHist([image], [0, 1, 2],
            None, self.bins, [0, 256, 0, 256, 0, 256])
        if imutils.is_cv2():
            hist = cv2.normalize(hist)
        else:
            hist = cv2.normalize(hist,hist)

        return hist.flatten()

    def describe_images(self, list_images, isSave = False):
        images_histograms = {}
        
        for image_path in list_images:
            k = image_path[image_path.rfind("/") + 1:]
            img = cv2.imread(image_path)
            features = self.describe(img)
            images_histograms[k] = [round(digit) for digit in features * 10e2]
        
        if isSave:
            save_obj(images_histograms, 'images_histograms')
            save_obj(list_images, 'list_images')

        return images_histograms
    
def init_color_search():
    list_images = get_list_images_paths()
    print(list_images)
    describer = RGBHistogram([4,4,4])
    images_histograms = describer.describe_images(list_images, isSave=True)
    create_inverted_index(images_histograms, isSave=True)
