import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import time
from skimage.feature import hog
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
# note this is only valid for scikit-learn >= 0.18
from sklearn.model_selection import train_test_split
from lesson_functions import *

def main():
   f = open("cars.txt",'r')
   cars = f.readlines()
   f = open("notcars.txt",'r')
   notcars = f.readlines()

   # define feature parameters
   color_space = 'HSV' # can be RGB, HSV, LUV, HLS, YUV, YCrCb
   orient = 6
   pix_per_cell = 8
   cell_per_block = 2
   hog_channel = "ALL" # can be 0, 1, 2, or "ALL"
   spatial_size = (16, 16) # spatial binning dimensions
   hist_bins = 16 # number of histogram bins
   spatial_feat = True # spatial_features on or off
   hist_feat = True # Histogram features on or off
   hog_feat = True # HOG features on or off
   
   t = time.time()
   n_samples = 2000
   # safer to generate two sets of random numbers as the lengths of the
   # sample sets may be different
   random_car_idxs = np.random.randint(0, len(cars), n_samples)
   random_notcar_idxs = np.random.randint(0, len(notcars), n_samples)
   test_cars = np.array(cars)[random_car_idxs]
   test_notcars = np.array(notcars)[random_notcar_idxs]
   car_features = extract_features(test_cars, color_space=color_space, 
                        spatial_size=spatial_size, hist_bins=hist_bins, 
                        orient=orient, pix_per_cell=pix_per_cell, 
                        cell_per_block=cell_per_block, 
                        hog_channel=hog_channel,
                        spatial_feat=spatial_feat, hist_feat=hist_feat, 
                        hog_feat=hog_feat)

   notcar_features = extract_features(test_notcars, color_space=color_space, 
                        spatial_size=spatial_size, hist_bins=hist_bins, 
                        orient=orient, pix_per_cell=pix_per_cell, 
                        cell_per_block=cell_per_block, 
                        hog_channel=hog_channel,
                        spatial_feat=spatial_feat, hist_feat=hist_feat, 
                        hog_feat=hog_feat)
   
   print(time.time()-t, ' Seconds to compute features...')
   
   # define the features vector by stacking car and notcar features
   # convert to float values so we can process it
   X = np.vstack((car_features, notcar_features)).astype(np.float64)   

   # fit a per-column scaler
   X_scaler = StandardScaler().fit(X)
   
   # apply the scaler to X
   scaled_X = X_scaler.transform(X)
   
   # define the labels vector - car is one, not car is 0
   y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

   # Split up data into randomized training and "test" (validation) sets 
   rand_state = np.random.randint(0, 100)
   X_train, X_test, y_train, y_test = train_test_split(
        scaled_X, y, test_size=0.1, random_state=rand_state)
   
   print('Using ', orient, ' orientations,', pix_per_cell,
         ' pixels per cell, ', cell_per_block, ' cells per block',
         hist_bins, ' histogram bins, and ', spatial_size,
         ' spatial samplng')
   print('Feature vector length: ', len(X_train[0]))
   
   #Use a linear SVC
   svc = LinearSVC()
   
   # check the training time for the SVC
   t = time.time()
   svc.fit(X_train, y_train)
   print(round(time.time()-t,2), 'Seconds to train SVC...')
   
   # Check the accuracy score of the SVC
   print('Test accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
if __name__ == "__main__":
    main()