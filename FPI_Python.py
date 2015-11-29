from skimage import data, io, filters
from skimage.color import rgb2gray, gray2rgb
from skimage.util import img_as_ubyte, img_as_float
from skimage.transform import resize

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage

from skimage.morphology import watershed
from skimage.feature import peak_local_max
from skimage.morphology import erosion, disk

from skimage.color import convert_colorspace






def _gaussian(sigma=0.5, size=None):
    """Gaussian kernel numpy array with given sigma and shape.
    """
    sigma = max(abs(sigma), 1e-10)

    x = np.arange(-(size[0] - 1) / 2.0, (size[0] - 1) / 2.0 + 1e-8)
    y = np.arange(-(size[1] - 1) / 2.0, (size[1] - 1) / 2.0 + 1e-8)

    Kx = np.exp(-x ** 2 / (2 * sigma ** 2))
    Ky = np.exp(-y ** 2 / (2 * sigma ** 2))
    ans = np.outer(Kx, Ky) / (2.0 * np.pi * sigma ** 2)
    return ans / sum(sum(ans))

def growImage(input_image, synth_mask, window):
    """This function performs constrained synthesis. It grows the texture
    of surrounding region into the unknown pixels.

    Parameters
    ---------
    input_image : (M, N) array, np.uint8
+        Input image whose texture is to be calculated
+    synth_mask : (M, N) array, bool
+        Texture for `True` values are to be synthesised.
+    window : int
+        Size of the neighborhood window
+
+    Returns
+    -------
+    image : array, float
+        Texture synthesised input_image.
+
+    """

    MAX_THRESH = 0.3
    ERR_THRESH = 0.1

    # Padding
    pad_size = tuple(np.array(input_image.shape) + np.array(window) - 1)
    image = np.mean(input_image) * np.ones(pad_size, dtype=np.float32)
    mask = np.zeros(pad_size, bool)
    h, w = input_image.shape
    i0, j0 = window, window
    i0 /= 2
    j0 /= 2
    image[i0:i0 + h, j0:j0 + w] = img_as_float(input_image)
    mask[i0:i0 + h, j0:j0 + w] = synth_mask

    sigma = window / 6.4
    gauss_mask = _gaussian(sigma, (window, window))
    ssd = np.zeros(input_image.shape, np.float)

    while mask.any():
        progress = 0

        # Generate the boundary of ROI (region to be synthesised)
        boundary = mask - erosion(mask, disk(1))
        if not boundary.any():  # If the remaining region is 1-pixel thick
            boundary = mask

        bound_list = np.transpose(np.where(boundary == 1))

        for i_b, j_b in bound_list:
            template = image[(i_b - window / 2):(i_b + window / 2 + 1),
                             (j_b - window / 2):(j_b + window / 2 + 1)]
            mask_template = mask[(i_b - window / 2):(i_b + window / 2 + 1),
                                 (j_b - window / 2):(j_b + window / 2 + 1)]
            valid_mask = gauss_mask * (1 - mask_template)

            # best_matches = find_matches(template, valid_mask, image, window)
            total_weight = np.sum(valid_mask)
            for i in xrange(input_image.shape[0]):
                for j in xrange(input_image.shape[1]):
                    sample = image[i:i + window, j:j + window]
                    dist = (template - sample) ** 2
                    ssd[i, j] = np.sum(dist * valid_mask) / total_weight

            # Remove the case where sample == template
            ssd[i_b - window / 2, j_b - window / 2] = 1.

            best_matches = np.transpose(np.where(ssd <= ssd.min() * (
                                        1 + ERR_THRESH)))

            rand = np.random.randint(best_matches.shape[0])
            matched_index = best_matches[rand, :]

            if ssd[tuple(matched_index)] < MAX_THRESH:
                image[i_b, j_b] = image[tuple(matched_index + [window / 2,
                                                               window / 2])]
                mask[i_b, j_b] = False
                progress = 1

        if progress == 0:
            MAX_THRESH = 1.1 * MAX_THRESH

    return image[i0:-i0, j0:-j0]


















foreground = io.imread('C:\Users\Bruno\Workspace\FPIultimate\\narwhalP.jpg')
background = io.imread('C:\Users\Bruno\Workspace\FPIultimate\\mountainP.jpg')
foreground_gray = rgb2gray(foreground)
background_gray = rgb2gray(background)


trueMap = [[False for x in range(foreground_gray.shape[0])] for x in range(foreground_gray.shape[1])] 
print(foreground.max())
foreground_quantized = img_as_ubyte(foreground_gray)
foreground_quantized = foreground_quantized //16


for i in range(foreground_gray.shape[0]):
    for j in range(foreground_gray.shape[1]):
        if foreground_gray[i,j] < 0.7:
            trueMap[i][j] = True
            #print('oi')

foreground_texture = foreground[:,:,:]

for i in range(0,foreground_gray.shape[0]):
    for j in range(0,foreground_gray.shape[1]):
        if trueMap[i][j] == True:
            for m in range(4):
                for n in range(4):
                    print('olar')
                    #foreground_texture[i+m-2,j+n-2,0] = background[i+m-2,j+n-2,0]
                    #foreground_texture[i+m-2,j+n-2,1] = background[i+m-2,j+n-2,1]
                    #foreground_texture[i+m-2,j+n-2,2] = background[i+m-2,j+n-2,2]
'''         
for i in range(0,foreground_gray.shape[0]):
    for j in range(0,foreground_gray.shape[1]):
            if trueMap[i][j] == False:
                foreground_xyz[i,j,0] = background_xyz[i,j,0]
                foreground_xyz[i,j,1] = background_xyz[i,j,1]
                foreground_xyz[i,j,2] = background_xyz[i,j,2]
'''      
background_gray[:,:] = background_gray[:,:] - foreground_gray[:,:]
mask = background_gray < 0
background_gray[mask] = 0
foreground_texture= growImage(background_gray,trueMap,3)

io.imshow(foreground_texture,cmap='gray')
io.show()
print('terminei')
#print(trueMap)

















'''



foreground = io.imread('C:\Users\Bruno\Workspace\FPIultimate\\narwhal.png')
background = io.imread('C:\Users\Bruno\Workspace\FPIultimate\\banaSplit.jpg')

foreground = resize(foreground, (640, 640))

#into grayscale
foreground_gray = rgb2gray(foreground)
#background_gray = rgb2gray(background) 
print(foreground_gray.max())
#quantization
foreground_quantized = img_as_ubyte(foreground_gray) 
print(foreground_quantized.max())
foreground_quantized = foreground_quantized // 3 #diz quantos tons tu quer de grayscale
print(foreground_quantized.max())

#background_quantized = img_as_ubyte(background_gray) 
#background_quantized = background_quantized // 8

#foreground_edges = filters.sobel(foreground_quantized)
#foreground_edges = 255 - foreground_edges
#(100, 100)
#mask = foreground_quantized > 4
#foreground_quantized[mask] = 0 
foreground_RGB = gray2rgb(foreground_quantized)
merged = foreground_RGB
#print(foreground_quantized)
io.imshow(merged, cmap='gray')
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
'''