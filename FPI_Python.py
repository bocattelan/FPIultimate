from skimage import data, io, filters
from skimage.color import rgb2gray, gray2rgb
from skimage.util import img_as_ubyte
from skimage.transform import resize

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage

from skimage.morphology import watershed
from skimage.feature import peak_local_max


foreground = io.imread('C:\Users\Bruno\Workspace\FPIultimate\\narwhal.jpg')
background = io.imread('C:\Users\Bruno\Workspace\FPIultimate\\banaSplit.jpg')

foreground = resize(foreground, (640, 640))

#into grayscale
foreground_gray = rgb2gray(foreground)
#background_gray = rgb2gray(background) 

#quantization
foreground_quantized = img_as_ubyte(foreground_gray) 
foreground_quantized = foreground_quantized // 64


#background_quantized = img_as_ubyte(background_gray) 
#background_quantized = background_quantized // 8

#foreground_edges = filters.sobel(foreground_quantized)
#foreground_edges = 255 - foreground_edges
#(100, 100)
mask = foreground_quantized < 3
foreground_quantized[mask] = 128 

foreground_RGB = gray2rgb(foreground_quantized)
merged = foreground_RGB + background
print(foreground_quantized)
io.imshow(merged)
#io.imshow(foreground_quantized, cmap='gray')
#io.show()


# Now we want to separate the two objects in image
# Generate the markers as local maxima of the distance to the background

#foreground_edges = 255 - foreground_edges
distance = ndimage.distance_transform_edt(foreground_quantized)
local_maxi = peak_local_max(distance, indices=False, footprint=np.ones((3, 3)),
                            labels=foreground_quantized)
markers = ndimage.label(local_maxi)[0]
labels = watershed(-distance, markers, mask=foreground_quantized)

fig, axes = plt.subplots(ncols=3, figsize=(8, 2.7))
ax0, ax1, ax2 = axes

ax0.imshow(foreground_quantized, cmap=plt.cm.gray, interpolation='nearest')
ax0.set_title('Overlapping objects')
ax1.imshow(-distance, cmap=plt.cm.jet, interpolation='nearest')
ax1.set_title('Distances')
ax2.imshow(labels, cmap=plt.cm.spectral, interpolation='nearest')
ax2.set_title('Separated objects')

for ax in axes:
    ax.axis('off')

fig.subplots_adjust(hspace=0.01, wspace=0.01, top=1, bottom=0, left=0,
                    right=1)
plt.show()