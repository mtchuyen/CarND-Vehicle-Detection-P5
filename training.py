import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import time
from skimage.feature import hog
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScalar
# note this is only valid for scikit-learn >= 0.18
from sklearn.model_select import train_test_split
from lesson_functions import *



