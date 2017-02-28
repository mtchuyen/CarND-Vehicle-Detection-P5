#! /usr/bin/env python
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import time
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from lesson_functions import *
# NOTE: the next import is only valid for scikit-learn version <= 0.17
# from sklearn.cross_validation import train_test_split
# for scikit-learn >= 0.18 use:
from sklearn.model_selection import train_test_split
'''
Processing of an image is as follows:

1. Perform a Histogram of Oriented Gradients (HOG) Feature extraction 
    on a labeled training set of images
2. Train a Linear SVM classifier
    Note: do not forget to normalize features and randomize a selection for training and testing
3. Optionally, apply a color transform and append binned color features, 
    as well as histograms of color to your HOG feature vector
4. Implement a sliding window technique and use your trained classifier to search 
    for vehicles in images

Videos should do the above, then:
    - create a heat map of recurring detections frame by frame
    - reject outliers
    - follow detected vehicles
    - estimate a bounding box for vehicles detected
'''
class Image():
    def __init__(self, image):
        self.image = image
        
    def process_image(self):
        pass
    

def main():
    img = mpimg.imread("test_images/test1.jpg")
    plt.imshow(img)
    plt.show()
if __name__ == "__main__":
    main()