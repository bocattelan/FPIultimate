from skimage import data, io, filters
from skimage.color import rgb2gray
from skimage.util import img_as_ubyte




img = io.imread('C:\Users\Bruno\Workspace\CamouflageImages\\eu.png')
img_gray = rgb2gray(img)
quantized = img_as_ubyte(img_gray) 
quantized = quantized // 32
io.imshow(quantized, cmap='gray')
io.show()