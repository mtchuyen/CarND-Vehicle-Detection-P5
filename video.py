#! /usr/bin/env python
import imageio
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage.measurements import label
from lesson_functions import *
from training import train,find_cars
class Video():
    def __init__(self, X_scaler, svc, queue_max, file_name, output_name, color_space, 
                 orient, pix_per_cell,
                 cell_per_block = 2, scale = 1, ystart = 400, 
                 ystop = 656, hog_channel = "ALL",
                 spatial_size = (16, 16),
                 hist_bins = 16,
                 spatial_feat = True,
                 hist_feat = True,
                 hog_feat = True,
                 xy_window = (96, 96),
                 threshold = 3):
        self.X_scaler = X_scaler
        self.svc = svc
        self.file_name = file_name
        self.output_name = output_name
        self.color_space = color_space
        self.orient = orient
        self.pix_per_cell = pix_per_cell
        self.cell_per_block = cell_per_block
        self.scale = scale
        self.ystart = ystart
        self.ystop = ystop
        self.hog_channel = hog_channel
        self.spatial_size = spatial_size
        self.hist_bins = hist_bins
        self.spatial_feat = spatial_feat
        self.hist_feat = hist_feat
        self.hog_feat = hog_feat
        self.xy_window = xy_window
        self.threshold = threshold
        self.queue_max = queue_max
        self.heatmap_list = []
        
    def get_video_reader(self, file_name):
        return imageio.get_reader(file_name, 'ffmpeg')
        
    def get_image_from_mp4(self, reader , num):
        image = None
        try:
            image = reader.get_data(num)
        except:
            pass
        return image
    
    def run_video(self):
        reader = self.get_video_reader(self.file_name)
        writer = imageio.get_writer(self.output_name, fps=30)
        num = 0
        next_image = self.get_image_from_mp4(reader, num)
        while next_image != None:
            draw_img, heatmap = find_cars(next_image, self.ystart, self.ystop, self.scale, 
                                  self.svc, self.X_scaler, self.orient,
                                  self.pix_per_cell, self.cell_per_block, self.spatial_size, 
                                  self.hist_bins, self.xy_window, self.color_space)
            self.heatmap_list.append(heatmap)
            if len(self.heatmap_list) == self.queue_max:
                heatmap = np.sum(self.heatmap_list,axis=0)
                heatmap = apply_threshold(heatmap, self.threshold)
                labels = label(heatmap)
                draw_img = draw_labeled_bboxes(next_image, labels)
                #plt.imshow(heatmap)
                #plt.show()
                #plt.imshow(draw_img)
                #plt.show()
                del(self.heatmap_list[self.queue_max-1])
                writer.append_data(draw_img[:,:,:])
            num = num + 1
            next_image = self.get_image_from_mp4(reader, num)
        writer.close()


def main():

   color_space = 'HSV' # can be RGB, HSV, LUV, HLS, YUV, YCrCb
   orient = 9
   queue_max = 3
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
   threshold = 4
   file_name = "test_video.mp4"
   output_name = "output_test.mp4"
   (X_scaler, svc) = train(color_space, orient, pix_per_cell, cell_per_block, hog_channel,
      spatial_size, hist_bins, spatial_feat, hist_feat, hog_feat)   
   video = Video(X_scaler, svc, queue_max, file_name, output_name, color_space, 
                 orient, pix_per_cell,
                 cell_per_block, scale, 
                 ystart, ystop, 
                 hog_channel,
                 spatial_size,
                 hist_bins,
                 spatial_feat,
                 hist_feat,
                 hog_feat,
                 xy_window,
                 threshold)
   video.run_video()
if __name__ == "__main__":
    main()
