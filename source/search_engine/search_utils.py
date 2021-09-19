from __future__ import print_function 
from __future__ import division
from typing import List
import imutils
import cv2
import glob
import numpy as np
import pickle
import os
import torch
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt
import glob
from PIL import Image

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


def create_inverted_index(images_histograms, index_type, isSave = False):

    inverted_index = {}

    for i, feature in enumerate(images_histograms.values()):
        for fi, value in enumerate(feature):
            value_item = []
            feature_item = {}

            if fi in inverted_index:
                feature_item = inverted_index[fi]

            if value in feature_item:
                value_item = feature_item[value]

            value_item.append(i)
            feature_item[value] = value_item
            inverted_index[fi] = feature_item
    
    if isSave:
        save_obj(inverted_index, f'inverted_{index_type}_index')

    return inverted_index


def get_combined_list(list_1, list_2):

    set_1 = set(list_1)
    set_2 = set(list_2)
    list_2_items_not_in_list_1 = list(set_2 - set_1)
    combined_list = list_1 + list_2_items_not_in_list_1

    return combined_list



INPUT_SIZE = 244
DATA_TRANSFORM = transforms.Compose([
        transforms.Resize(INPUT_SIZE),
        transforms.CenterCrop(INPUT_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
DATA_RESULTS_TRANSFORM = transforms.Compose([
        transforms.Resize(INPUT_SIZE),
        transforms.CenterCrop(INPUT_SIZE),
        transforms.ToTensor()])
        #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# DEVICE = torch.device('cpu')

def read_img_PIL(path):
  f = open(path,'rb')
  img = Image.open(f)
  return img

class Resnet18SearchEngine:
    def __init__(self) -> None:
        self.load_model()

    def load_model(self):
        modelPath = './obj/feature_extractor_resnet18.pth'
        self.model = torch.load(modelPath, map_location= DEVICE)

    def feature_extractor(self, img):
        img = DATA_TRANSFORM(img)
        img = img.to(DEVICE)
        feature = self.describe(img)
        feature = [round(digit,3) for digit in feature]
        return feature

    def describe(self, img):
        xb = img.unsqueeze(0)
        yb = self.model(xb)
        return yb.cpu().data.numpy().flatten()

    def describe_images(self, list_images, isSave = False):
        images_features = {}

        for image_path in list_images:
            k = image_path[image_path.rfind("/") + 1:]
            img = read_img_PIL(image_path)
            feature = self.feature_extractor(img)
            images_features[k] = feature
        
        if isSave:
            save_obj(list_images, 'list_images')
            save_obj(images_features, 'image_features_resnet18')

        return images_features
        

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
    describer = RGBHistogram([4,4,4])
    images_histograms = describer.describe_images(list_images, isSave=True)
    create_inverted_index(images_histograms, 'histogram', isSave=True)

def init_features_search():
    list_images = get_list_images_paths()
    describer = Resnet18SearchEngine()
    images_features = describer.describe_images(list_images, True)
    create_inverted_index(images_features, 'features', isSave=True)

init_features_search()