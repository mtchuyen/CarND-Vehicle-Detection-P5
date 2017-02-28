import matplotlib.image as mpimg
import numpy as np
import matplotlib.pyplot as plt
# Read in the image
image = mpimg.imread('test_images/cutouts/cutout1.jpg')

# Take histograms in R, G, and B
rhist = np.histogram(image[:,:,0], bins=32, range=(0, 256))
ghist = np.histogram(image[:,:,1], bins=32, range=(0, 256))
bhist = np.histogram(image[:,:,2], bins=32, range=(0, 256))
'''
With np.histogram(), you don't actually have to specify the number of bins or the 
range, but here I've arbitrarily chosen 32 bins and specified range=(0, 256) 
in order to get orderly bin sizes. np.histogram() returns a tuple of two arrays. 
In this case, for example, rhist[0] contains the counts in each of the bins and rhist[1] 
contains the bin edges (so it is one element longer than rhist[0]).

To look at a plot of these results, we can compute the bin centers from the bin edges. 
Each of the histograms in this case have the same bins, so I'll just use the rhist bin edges:
'''
# Generating bin centers
bin_edges = rhist[1]
bin_centers = (bin_edges[1:]  + bin_edges[0:len(bin_edges)-1])/2

#Plot a figure with all three bar charts
fig = plt.figure(figsize=(12,3))
plt.subplot(131)
plt.bar(bin_centers, rhist[0])
plt.xlim(0, 256)
plt.title('R Histogram')
plt.subplot(132)
plt.bar(bin_centers, ghist[0])
plt.xlim(0, 256)
plt.title('G Histogram')
plt.subplot(133)
plt.bar(bin_centers, bhist[0])
plt.xlim(0, 256)
plt.title('B Histogram')
plt.show()