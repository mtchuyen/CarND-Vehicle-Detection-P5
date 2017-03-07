#! /usr/bin/env python
from image import Image
import imageio
import cv2
import pickle
import sys
class Video():
    def __init__(self, ksize=3, debug=False):
        self.image = Image(ksize, debug)
        
    def get_video_reader(self, file_name):
        return imageio.get_reader(file_name, 'ffmpeg')
        
    def get_image_from_mp4(self, reader , num):
        image = None
        try:
            image = reader.get_data(num)
        except:
            pass
        return image
    
    def run_video(self, video_name, output):
        ksize=3
        reader = self.get_video_reader(video_name)
        writer = imageio.get_writer(output, fps=30)
        num = 0
        next_image = self.get_image_from_mp4(reader, num)
        while next_image != None:
            img = Image(next_image)
            out_image = img.process_image()
            heatmap = img.heatmap(out_image)
            bboxes = img.get_bboxes(heatmap)
            writer.append_data(result[:,:,:])
            num = num + 1
            next_image = self.get_image_from_mp4(reader, num)
        writer.close()


def main():
    video_proc = Video(3, debug=False)
    video_proc.run_video('project_video.mp4', 'output_images/challenge_output_corrected.mp4')
if __name__ == "__main__":
    main()
