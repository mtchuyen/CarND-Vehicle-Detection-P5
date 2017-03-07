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
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
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
    ystart = 400
    ystop = 656
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
                xy_window):
    # extract test images
    '''
    np.save does not seem to work at the moment
    svc = pickle.load(open("svc.p", "rb"))
    X_scaler = np.load('x_scaler.npy')
    '''
    searchpath = "test_images/*"
    example_images = glob.glob(searchpath)
    images = []
    titles = []
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
        windows = slide_window(img, x_start_stop=x_start_stop,
                              y_start_stop=y_start_stop, 
                              xy_window = xy_window, 
                              xy_overlap = (overlap, overlap))
        # find the hot spots
        hot_windows = search_windows(img, windows, svc, X_scaler,
                        spatial_size=spatial_size, hist_bins=hist_bins,
                        orient=orient, pix_per_cell=pix_per_cell, 
                        cell_per_block=cell_per_block, 
                        hog_channel=hog_channel,
                        spatial_feat=spatial_feat, hist_feat=hist_feat, 
                        hog_feat=hog_feat)
        # draw boxes
        window_img = draw_boxes(draw_img, hot_windows, color=(0,0,255),
                                thick=6)
        images.append(window_img)
        titles.append("")
        print(time.time()-t1, ' seconds to process one image searching ',
              len(windows), ' windows')
    fig = plt.figure(figsize=(12, 18), dpi=300)
    visualize(fig, 5, 2, images, titles)

# Define a single function that can extract features using hog sub-sampling and make predictions
def find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins):
    
    draw_img = np.copy(img)
    
    # make a heatmap of zeros
    heatmap = np.zeros_like(img[:,:,0])
    
    # normalize image and convert to float
    img = img.astype(np.float32)/255
    
    # limit image to y area to search (ignore sky and hood of car)
    img_tosearch = img[ystart:ystop,:,:]
    
    # color convert to better color space
    #ctrans_tosearch = convert_color(img_tosearch, conv='RGB2YCrCb')
    ctrans_tosearch = convert_color(img_tosearch, conv='RGB2HSV')
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
        
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
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

            xleft = xpos*pix_per_cell
            ytop = ypos*pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))
          
            # Get color features
            spatial_features = bin_spatial(subimg, size=spatial_size)
            hist_features = color_hist(subimg, nbins=hist_bins)

            # Scale features and make a prediction
            test_features = X_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))    
            #test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))    
            test_prediction = svc.predict(test_features)
            
            if test_prediction == 1:
                xbox_left = np.int(xleft*scale)
                ytop_draw = np.int(ytop*scale)
                win_draw = np.int(window*scale)
                cv2.rectangle(draw_img,(xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart),(0,0,255),6) 
                '''
                img_boxes.append(((xbox_left, ytop_draw + ystart),
                                  (xbox_left + win_draw, 
                                  ytop_draw + win_draw+ ystart)))
                '''
                # add to heatmap where we think we found it
                heatmap[ytop_draw+ystart:ytop_draw+win_draw + ystart,
                        xbox_left:xbox_left + win_draw] += 1                
    return draw_img, heatmap

def process_frame(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell,
                  cell_per_block, spatial_size, hist_bins, threshold):
    draw_img, heatmap = find_cars(img, ystart, ystop, scale, svc, X_scaler, orient,
                                  pix_per_cell, cell_per_block, spatial_size, hist_bins)
    heatmap = apply_threshold(heatmap, threshold)
    labels = label(heatmap)
    draw_img = draw_labeled_bboxes(img, labels)
    return draw_img, heatmap

def process_images(ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell,
                  cell_per_block, spatial_size, hist_bins):
    out_images = []
    out_titles = []
    threshold = 3
    searchpath = "test_images/*"
    example_images = glob.glob(searchpath)
    for img_src in example_images:
        img = mpimg.imread(img_src)
        draw_img, heatmap = process_frame(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell,
                  cell_per_block, spatial_size, hist_bins, threshold)
        out_images.append(draw_img)
        out_images.append(heatmap)
        out_titles.append(img_src[-12:])
        out_titles.append(img_src[-12:])
    fig = plt.figure(figsize=(12,24))
    visualize(fig, 6, 2, out_images, out_titles)

def train(color_space, orient, pix_per_cell, cell_per_block, hog_channel,
          spatial_size, hist_bins, spatial_feat, hist_feat, hog_feat):
   f = open("cars.txt",'r')
   cars = f.readlines()
   f = open("notcars.txt",'r')
   notcars = f.readlines()
   
   t = time.time()
   n_samples = 1000
   # safer to generate two sets of random numbers as the lengths of the
   # sample sets may be different
   random_car_idxs = np.random.randint(0, len(cars), n_samples)
   random_notcar_idxs = np.random.randint(0, len(notcars), n_samples)
   test_cars =  cars #np.array(cars)[random_car_idxs]
   test_notcars =  notcars #np.array(notcars)[random_notcar_idxs]
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
   '''
   np.save does not seem to work right now
   pickle.dump(svc, open("svc.p", "wb"))
   np.save('x_scaler.npy',X_scaler)
   '''
   return (X_scaler, svc)
def main():
   # define feature parameters
   color_space = 'HSV' # can be RGB, HSV, LUV, HLS, YUV, YCrCb
   orient = 9
   pix_per_cell = 8
   cell_per_block = 2
   scale = 1
   ystart = 400
   ystop = 656
   hog_channel = "ALL" # can be 0, 1, 2, or "ALL"
   spatial_size = (16, 16) # spatial binning dimensions
   hist_bins = 16 # number of histogram bins
   spatial_feat = True # spatial_features on or off
   hist_feat = True # Histogram features on or off
   hog_feat = True # HOG features on or off
   xy_window = (64, 64)
   (X_scaler, svc) = train(color_space, orient, pix_per_cell, cell_per_block, hog_channel,
          spatial_size, hist_bins, spatial_feat, hist_feat, hog_feat)   
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
   process_images(ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell,
                  cell_per_block, spatial_size, hist_bins)    
if __name__ == "__main__":
    main()