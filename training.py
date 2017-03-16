import cv2
import glob
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pickle
from scipy.ndimage.measurements import label
from skimage.feature import hog
# note this is only valid for scikit-learn >= 0.18
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC,SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import time
from lesson_functions import *
# scan image once then subsample the array to extract features for each window
def test_images2(svc, X_scaler, spatial_size,
                hist_bins, orient, pix_per_cell, cell_per_block,
                hog_channel, spatial_feat, hist_feat, hog_feat, 
                xy_window, rescale):
    out_images = []
    out_maps = []
    out_titles = []
    out_boxes = []
    # consider a narrower swath in y
    ystart = 380
    ystop = 636
    # rather than take each window and downsize it, instead
    # scale the entire image, apply HOG to it, and subsample that
    # array. The effect is to resample at different window sizes
    scale = rescale
    # iterate over test images
    # extract test images
    searchpath = "test_images/*"
    example_images = glob.glob(searchpath)
    for img_src in example_images:
        img_boxes = []
        t = time.time()
        count = 0
        img = mpimg.imread(img_src)
        draw_img = np.copy(img)

        # make a heatmap, initially zeros
        # add detections to this heatmap, then 
        # threshold it to remove potential false positives
        heatmap = np.zeros_like(img[:,:,0])
        # images need to be normalized as the system was trained on pngs
        # but tested on jpgs        
        img = img.astype(np.float32)/255
        
        # restrict image to search area by cropping it
        img_tosearch = img[ystart:ystop,:,:]
        
        # transform to different color space for better contrast
        #ctrans_tosearch = convert_color(img_tosearch, conv='RGB2YCrCb')
        ctrans_tosearch = convert_color(img_tosearch, conv='RGB2HSV')
        
        # convert image to different size if scaling
        # instead of changing window size, change size of image
        # with constant window size
        if scale != 1:
            imshape = ctrans_tosearch.shape
            ctrans_tosearch = cv2.resize(ctrans_tosearch,
                                (np.int(imshape[1]/scale),
                                np.int(imshape[0]/scale)))
        
        # break out the three color channels
        ch1 = ctrans_tosearch[:,:,0]
        ch2 = ctrans_tosearch[:,:,1]
        ch3 = ctrans_tosearch[:,:,2]
        
        # define blocks and steps as above for first color channel
        # nxblocks and nyblocks hold the number of HOG cells across
        # each dimension for the particular image
        nxblocks = (ch1.shape[1] // pix_per_cell) - 1
        nyblocks = (ch1.shape[0] // pix_per_cell) - 1
        
        # number of features per block are we going to be extracting
        nfeat_per_block = orient * cell_per_block**2
        
        # size of the original window
        window = 64
        
        # total number of blocks per window
        nblocks_per_window = (window // pix_per_cell) - 1
        cells_per_step = 2 # instead of overlap, define how many cells to step
        nxsteps = (nxblocks - nblocks_per_window)//cells_per_step
        nysteps = (nyblocks - nblocks_per_window)//cells_per_step
        
        # computer individual channel HOG features for the entire image
        # with feature_vec as False we are getting a multi-dimensional 
        # array rather than a flattened vector, so that we can see
        # the features as a function of position on the image
        hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block,
                                feature_vec=False)
        hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block,
                                feature_vec=False)
        hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block,
                                feature_vec=False)
        
        for xb in range(nxsteps):
            for yb in range(nysteps):
                count += 1
                # calculate the new position
                ypos = yb * cells_per_step
                xpos = xb * cells_per_step
                
                # extrace HOG for the current patch of the image
                hog_feat1 = hog1[ypos:ypos + nblocks_per_window, 
                                 xpos:xpos + nblocks_per_window].ravel()
                hog_feat2 = hog2[ypos:ypos + nblocks_per_window, 
                                 xpos:xpos + nblocks_per_window].ravel()
                hog_feat3 = hog3[ypos:ypos + nblocks_per_window, 
                                 xpos:xpos + nblocks_per_window].ravel()
                hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))
                
                # get the boundaries of where we are on the image
                xleft = xpos * pix_per_cell
                ytop  = ypos * pix_per_cell
                
                # Extract the image patch
                subimg = cv2.resize(ctrans_tosearch[ytop:ytop + window,
                                                    xleft: xleft + window],
                                                    xy_window)
                                                    
                # Get color features
                spatial_features = bin_spatial(subimg, size=spatial_size)
                hist_features = color_hist(subimg, nbins=hist_bins)
                
                # Scale features and make a prediction
                test_features = X_scaler.transform(np.hstack((
                                                spatial_features, 
                                                hist_features,
                                                hog_features))).reshape(1, -1)
                test_prediction = svc.predict(test_features)
                
                if test_prediction == 1:
                    xbox_left = np.int(xleft*scale)
                    ytop_draw = np.int(ytop*scale)
                    win_draw = np.int(window*scale)
                    '''
                    cv2.rectangle(draw_img, (xbox_left, ytop_draw + ystart),
                                  (xbox_left + win_draw, ytop_draw + win_draw + ystart),
                                  (0, 0, 255))
                    '''
                    img_boxes.append(((xbox_left, ytop_draw + ystart),
                                      (xbox_left + win_draw, 
                                      ytop_draw + win_draw+ ystart)))
                    # add to heatmap where we think we found it
                    heatmap[ytop_draw+ystart:ytop_draw+win_draw + ystart,
                            xbox_left:xbox_left + win_draw] += 1
        print(time.time() - t, ' seconds to run, total windows= ', count)        
        out_titles.append(img_src[-12:])
        out_titles.append(img_src[-12:])
        # heatmap = 255*heatmap/np.max(heatmap)
        out_images.append(heatmap)
        out_maps.append(heatmap)
        out_boxes.append(img_boxes)
    
    fig = plt.figure(figsize=(12, 24))
    visualize(fig, 8, 2, out_images, out_titles)
        
def test_images(svc, X_scaler, spatial_size,
                hist_bins, orient, pix_per_cell, cell_per_block,
                hog_channel, spatial_feat, hist_feat, hog_feat,
                xy_window, threshold=2):
    # extract test images
    searchpath = "test_images/*"
    example_images = glob.glob(searchpath)
    images = []
    titles = []
    img = mpimg.imread(example_images[0])
    # make a heatmap of zeros
    heatmap = np.zeros_like(img[:,:,0])
    # use these values to restrict coordinates to search over
    # so you're not searching over sky, trees, etc
    y_start_stop = [400, 656] # Min max to search in slide window
    x_start_stop = [None, None] # Min max to search in slide window
    overlap = 0.5
    # for each image filename:
    #    read in the image, 
    #    calculate the set of windows to test over
    #    search the windows and find the hot spots
    #    draw boxes where the cars may be
    #    collect the images and titles 
    
    for img_src in example_images:
        t1 = time.time()
        img = mpimg.imread(img_src)
        draw_img = np.copy(img)
        # images need to be normalized as the system was trained on pngs
        # but tested on jpgs
        img = img.astype(np.float32)/255
        print(np.min(img), np.max(img))
        # calculate the list of boxes
        # xy_window of 128x128 found a few
        # 64x64 found a few but appeared to be too small
        # found a few at 96x96
        # search windows should be integer multiple of your 
        # cell_size for HOG because 
        for window_size in [(64,64), (96, 96), (128, 128)]:
            windows = slide_window(img, x_start_stop=x_start_stop,
                                  y_start_stop=y_start_stop, 
                                  xy_window = window_size[0], 
                                  xy_overlap = (overlap, overlap))
            # find the hot spots
            hot_windows = search_windows(img, windows, svc, X_scaler,
                            spatial_size=spatial_size, hist_bins=hist_bins,
                            orient=orient, pix_per_cell=pix_per_cell, 
                            cell_per_block=cell_per_block, 
                            hog_channel=hog_channel,
                            spatial_feat=spatial_feat, hist_feat=hist_feat, 
                            hog_feat=hog_feat)
            # add to heatmap where we think we found it
            for hw in hot_windows:
                heatmap[hw[0][1]:hw[1][1],hw[0][0]:hw[1][0]] += 1
        max = np.amax(heatmap)
        heatmap = apply_threshold(heatmap, threshold)
        labels = label(heatmap)
        window_img = draw_labeled_bboxes(img, labels)
          
        images.append(window_img)
        titles.append("")
        print(time.time()-t1, ' seconds to process one image searching ',
              len(windows), ' windows')
    fig = plt.figure(figsize=(12, 18), dpi=300)
    visualize(fig, 6, 2, images, titles)

# Define a single function that can extract features using hog sub-sampling and make predictions
def find_cars(img, ystart, ystop, xstart, xstop, scale, svc, X_scaler, orient, pix_per_cell, 
              cell_per_block, hog_channel, spatial_size, hist_bins, xy_window, 
              color_space, spatial_feat, hist_feat, hog_feat):
    
    draw_img = np.copy(img)
    
    # make a heatmap of zeros
    heatmap = np.zeros_like(img[:,:,0])
    
    
    # limit image to y area to search (ignore sky and hood of car)
    img_tosearch = img[ystart:ystop,xstart:xstop,:]
    
    # color convert to better color space
    #ctrans_tosearch = convert_color(img_tosearch, conv='RGB2YCrCb')
    ctrans_tosearch = convert_color(img_tosearch, input='RGB', conv=color_space)
    # normalize image and convert to float
    ctrans_tosearch = ctrans_tosearch.astype(np.float32)/255

    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale),
                                                        np.int(imshape[0]/scale)))
        
    ch1 = ctrans_tosearch[:,:,0]
    ch2 = ctrans_tosearch[:,:,1]
    ch3 = ctrans_tosearch[:,:,2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell)-1
    nyblocks = (ch1.shape[0] // pix_per_cell)-1 
    nfeat_per_block = orient*cell_per_block**2
    # 64 was the original sampling rate, with 8 cells and 8 pix per cell
    window = 64
    
    nblocks_per_window = (window // pix_per_cell)-1 
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step
    
    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)
    
    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb*cells_per_step
            xpos = xb*cells_per_step
            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            if hog_channel == "ALL":
                hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))
            elif hog_channel == 0:
                hog_features = hog_feat1
            elif hog_channel == 1:
                hog_features = hog_feat2
            elif hog_channel == 2:
                hog_features = hog_feat3
            xleft = xpos*pix_per_cell
            ytop = ypos*pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], xy_window)
          
            # Get color features
            spatial_features = bin_spatial(subimg, size=spatial_size)
            hist_features = color_hist(subimg, nbins=hist_bins)
            features = []
            if spatial_feat:
                features.append(spatial_features)
            if hist_feat:
                features.append(hist_features)
            if hog_feat:
                features.append(hog_features)
            feature_vec = np.concatenate(features)
            # Scale features and make a prediction
            #test_features = X_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))    
            test_features = X_scaler.transform(feature_vec.reshape(1, -1))
            #test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))    
            test_prediction = svc.predict(test_features)
            
            if test_prediction == 1:
                xbox_left = np.int(xleft*scale)
                ytop_draw = np.int(ytop*scale)
                win_draw = np.int(window*scale)
                cv2.rectangle(draw_img,(xbox_left+xstart, ytop_draw+ystart),(xbox_left+win_draw+xstart,ytop_draw+win_draw+ystart),(0,0,255),6) 
                # add to heatmap where we think we found it
                heatmap[ytop_draw+ystart:ytop_draw+win_draw + ystart,
                        xbox_left+xstart:xbox_left + win_draw + xstart] += 1                
    return draw_img, heatmap

def process_frame(img, ystart, ystop, xstart, xstop, scale, svc, X_scaler, orient, pix_per_cell,
                  cell_per_block, hog_channel, spatial_size, hist_bins, threshold, xy_window, 
                  color_space, spatial_feat, hist_feat, hog_feat):
    heatmap = np.zeros_like(img[:,:,0])
    for xy_win in [(32,32), (64, 64), (96,96), (128 ,128)]:
        draw_img, next_heatmap = find_cars(img, ystart, ystop, xstart, xstop, scale, svc, X_scaler, orient,
                                      pix_per_cell, cell_per_block, hog_channel, spatial_size, 
                                      hist_bins, xy_win, color_space, spatial_feat, 
                                      hist_feat, hog_feat)
        heatmap += next_heatmap
    max_val = np.amax(heatmap)
    heatmap = apply_threshold(heatmap, threshold)
    labels = label(heatmap)
    draw_img = draw_labeled_bboxes(img, labels)
    return draw_img, heatmap

def process_images(ystart, ystop, xstart, xstop, scale, svc, X_scaler, orient, pix_per_cell,
                  cell_per_block, hog_channel, spatial_size, hist_bins, xy_window, threshold,
                  color_space, spatial_feat, hist_feat, hog_feat):
    out_images = []
    out_titles = []
    searchpath = "test_images/*"
    example_images = glob.glob(searchpath)
    for img_src in example_images:
        img = mpimg.imread(img_src)
        draw_img, heatmap = process_frame(img, ystart, ystop, xstart, xstop, scale, svc, X_scaler, orient, pix_per_cell,
                  cell_per_block, hog_channel, spatial_size, hist_bins, threshold, 
                  xy_window, color_space, spatial_feat, hist_feat, hog_feat)
        out_images.append(draw_img)
        out_images.append(heatmap)
        out_titles.append(img_src[-12:])
        out_titles.append(img_src[-12:])
    fig = plt.figure(figsize=(12,24))
    visualize(fig, 7, 2, out_images, out_titles)

def train(color_space, orient, pix_per_cell, cell_per_block, hog_channel,
          spatial_size, hist_bins, spatial_feat, hist_feat, hog_feat, n_samples):
   f = open("cars.txt",'r')
   cars = f.readlines()
   f = open("notcars.txt",'r')
   notcars = f.readlines()
   
   t = time.time()
   test_cars = []
   test_notcars = []
   if n_samples == 0:
       # the car images are too close, so decimate them
       total = len(cars)
       # KITTI seems OK, others seem repetitive
       counter = 0
       for idx in range(total):
           if cars[idx].find("KITTI") >= 0:
               test_cars.append(cars[idx])
               test_notcars.append(notcars[idx])
           else:
               if 8*(counter//8) == counter:
                   test_cars.append(cars[idx])
                   test_notcars.append(notcars[idx])
               counter += 1
   else:
       # safer to generate two sets of random numbers as the lengths of the
       # sample sets may be different
       random_car_idxs = np.random.randint(0, len(cars), n_samples)
       random_notcar_idxs = np.random.randint(0, len(notcars), n_samples)
       test_cars =  np.array(cars)[random_car_idxs]
       test_notcars =  np.array(notcars)[random_notcar_idxs]
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
        scaled_X, y, test_size=0.15, random_state=rand_state)
   
   print('Using ', len(test_cars), ' samples with ', orient, ' orientations,', pix_per_cell,
         ' pixels per cell, ', cell_per_block, ' cells per block',
         hist_bins, ' histogram bins, and ', spatial_size,
         ' spatial samplng')
   print('Feature vector length: ', len(X_train[0]))
   
   #Use a linear SVC
   svc = LinearSVC()
   #svc = SVC(kernel='poly',degree=2)
   # check the training time for the SVC
   t = time.time()
   svc.fit(X_train, y_train)
   print(round(time.time()-t,2), 'Seconds to train SVC...')
   
   # Check the accuracy score of the SVC
   print('Test accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
   # get false positives and negatives from confusion matrix
   # see http://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html#sklearn.metrics.confusion_matrix
   # where
   # TP = cm[0][0]
   # FP = cm[0][1]
   # FN = cm[1][0]
   # TN = cm[1][1]
   cm = confusion_matrix( y_test, svc.predict(X_test))
   print("false positives=", round(100.0*float(cm[0][1])/len(y_test),4),"%")
   print("false negatives=", round(100.0*float(cm[1][0])/len(y_test),4),"%")
   return (X_scaler, svc)
def main():
   # define feature parameters
   color_space = 'YCrCb' # can be RGB, HSV, LUV, HLS, YUV, YCrCb
   orient = 9
   pix_per_cell = 8
   cell_per_block = 2
   scale = 1
   xstart = 300
   xstop = 1280
   ystart = 380
   ystop = 636
   hog_channel = "ALL" # can be 0, 1, 2, or "ALL"
   spatial_size = (32, 32) # spatial binning dimensions
   hist_bins = 32 # number of histogram bins
   spatial_feat = True # spatial_features on or off
   hist_feat = True # Histogram features on or off
   hog_feat = True # HOG features on or off
   xy_window = (64, 64)
   threshold = 2
   (X_scaler, svc) = train(color_space, orient, pix_per_cell, cell_per_block, hog_channel,
          spatial_size, hist_bins, spatial_feat, hist_feat, hog_feat, 0)   
   # run on test data
   '''    
   test_images(svc, X_scaler, spatial_size,
                hist_bins, orient, pix_per_cell, cell_per_block,
                hog_channel, spatial_feat, hist_feat, hog_feat,
                xy_window)
   test_images2(svc, X_scaler, spatial_size,
                hist_bins, orient, pix_per_cell, cell_per_block,
                hog_channel, spatial_feat, hist_feat, hog_feat, xy_window,
                rescale=1)
   '''
   process_images(ystart, ystop, xstart, xstop, scale, svc, X_scaler, orient, pix_per_cell,
                  cell_per_block, hog_channel, spatial_size, hist_bins, xy_window,
                  threshold, color_space,spatial_feat, hist_feat, hog_feat)    
if __name__ == "__main__":
    main()