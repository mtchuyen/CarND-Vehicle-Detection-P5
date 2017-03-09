import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from lesson_functions import convert_color
img = mpimg.imread("test_images/test4.jpg")
conv = "HSV" # conv can be RGB, HSV, LUV, HLS, YUV, YCrCb
draw_img = convert_color(img, "BGR", conv)
f, (a,b,c,d)=plt.subplots(4)
a.imshow(draw_img[:,:,0])
a.set_title('chan 0')
b.imshow(draw_img[:,:,1])
b.set_title('chan 1')
c.imshow(draw_img[:,:,2])
c.set_title('chan 2')
plt.show()
