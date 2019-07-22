import sys

from PIL import Image

from edge.otsu_thresholding import otsu

try:
    img = Image.open(sys.argv[1])
    img.load()
    img.show()
    bw = img.convert('L')
    otsu.otsu(bw)
except IOError:
    print("Unable to open file. Please try another format or check spelling.")
