from skimage import data, io, filters
from skimage.color import rgb2gray
from skimage.util import img_as_ubyte
from skimage.transform import resize



foreground = io.imread('C:\Users\Bruno\Workspace\CamouflageImages\\eu.png')
background = io.imread('C:\Users\Bruno\Workspace\CamouflageImages\\banaSplit.jpg')

#foreground = resize(foreground, (640, 640))

#into grayscale
foreground_gray = rgb2gray(foreground)
background_gray = rgb2gray(background) 

#quantization
foreground_quantized = img_as_ubyte(foreground_gray) 
foreground_quantized = foreground_quantized // 8

background_quantized = img_as_ubyte(background_gray) 
background_quantized = background_quantized // 8



#(100, 100)
merged = foreground_quantized * background_quantized
io.imshow(merged, cmap='gray')
io.show()