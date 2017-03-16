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
                 cell_per_block = 2, scale = 1, 
                 xstart=0, xstop=1280, 
                 ystart = 400, 
                 ystop = 656, hog_channel = "ALL",
                 spatial_size = (16, 16),
                 hist_bins = 16,
                 spatial_feat = True,
                 hist_feat = True,
                 hog_feat = True,
                 xy_window = (96, 96),
                 threshold = 3,
                 debug = False,
                 min_display=300,
                 skip=0):
        self.X_scaler = X_scaler
        self.svc = svc
        self.file_name = file_name
        self.output_name = output_name
        self.color_space = color_space
        self.orient = orient
        self.pix_per_cell = pix_per_cell
        self.cell_per_block = cell_per_block
        self.scale = scale
        self.xstart = xstart
        self.xstop = xstop
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
        self.debug = debug
        self.min_display = min_display
        self.skip = skip
        self.fps = 30
        
    def get_video_reader(self, file_name):
        return imageio.get_reader(file_name, 'ffmpeg')
        
    def get_image_from_mp4(self, reader , num):
        image = None
        try:
            image = reader.get_data(num)
        except:
            pass
        return image
    
    def getFrame(self, frameNumber,outputFileName):
        reader = self.get_video_reader(self.file_name)
        next_image = None
        for i in range(frameNumber):
            next_image = self.get_image_from_mp4(reader, i)
        imageio.imwrite(outputFileName, next_image)

        
            
            
    def run_video(self):
        reader = self.get_video_reader(self.file_name)
        writer = imageio.get_writer(self.output_name, fps=30)
        num = self.skip
        next_image = self.get_image_from_mp4(reader, num)
        while next_image is not None:
            windows = [(32,32), (64, 64), (96,96), (128 ,128)]
            for xy_window in windows:
                draw_img, heatmap = find_cars(next_image, self.ystart, self.ystop, self.xstart, self.xstop,
                                    self.scale, 
                                      self.svc, self.X_scaler, self.orient,
                                      self.pix_per_cell, self.cell_per_block, self.hog_channel,
                                      self.spatial_size, 
                                      self.hist_bins, self.xy_window, self.color_space,
                                      self.spatial_feat, self.hist_feat, self.hog_feat)
                self.heatmap_list.append(heatmap)
            if len(self.heatmap_list) == self.queue_max*len(windows):
                raw_heatmap = np.sum(self.heatmap_list,axis=0)
                max = np.amax(raw_heatmap) # for tuning/debugging
                heatmap = np.copy(raw_heatmap) # copy because otherwise it gets changed by reference
                heatmap = apply_threshold(heatmap, self.threshold)
                labels = label(heatmap)
                draw_img = draw_labeled_bboxes(next_image, labels)
                if self.debug and self.fps*(num//self.fps) == num and num >= self.min_display:
                    plt.imshow(draw_img)
                    plt.show()
                    plt.imshow(raw_heatmap)
                    plt.show()
                for idx in range(len(windows)):
                    del(self.heatmap_list[self.queue_max*len(windows)-(idx+1)])
                writer.append_data(draw_img[:,:,:])
            num = num + 1
            next_image = self.get_image_from_mp4(reader, num)
        writer.close()


def main():
   color_space = 'YCrCb' # can be RGB, HSV, LUV, HLS, YUV, YCrCb
   orient = 9
   pix_per_cell = 8
   queue_max = 6
   cell_per_block = 2
   scale = 1
   xstart = 600
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
   threshold = 6
   file_name = "project_video.mp4"
   output_name = "output_project.mp4"
   n_samples = 0
   debug = False # diagnotics... if debug on, display current frame and raw heatmap every second
   min_display=300 # initial frame to start from
   skip =0 # fast forward to this frame, default = 0
      
   (X_scaler, svc) = train(color_space, orient, pix_per_cell, cell_per_block, hog_channel,
      spatial_size, hist_bins, spatial_feat, hist_feat, hog_feat,n_samples)   
   video = Video(X_scaler, svc, queue_max, file_name, output_name, color_space, 
                 orient, pix_per_cell,
                 cell_per_block, scale, 
                 xstart, xstop,
                 ystart, ystop, 
                 hog_channel,
                 spatial_size,
                 hist_bins,
                 spatial_feat,
                 hist_feat,
                 hog_feat,
                 xy_window,
                 threshold, debug, 
                 min_display,
                 skip)
   '''
   video.getFrame(600, "test7.jpg")
   '''
   video.run_video()
if __name__ == "__main__":
    main()
