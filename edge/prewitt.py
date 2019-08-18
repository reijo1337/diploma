"""
Written in Python 2.7!
This module takes an image and converts it to grayscale, then applies a
Prewitt operator.
"""
import time

__author__ = "Kevin Gay"

from PIL import Image
import math


class Prewitt(object):
    def __init__(self, im_path):
        start_time = time.time()
        im = Image.open(im_path).convert('L')
        self.width, self.height = im.size
        mat = im.load()

        prewittx = [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]
        prewitty = [[1, 1, 1], [0, 0, 0], [-1, -1, -1]]

        self.prewittIm = Image.new('L', (self.width, self.height))
        pixels = self.prewittIm.load()

        lin_scale = .3
        # For each pixel in the image
        for row in range(self.width-len(prewittx)):
            for col in range(self.height-len(prewittx)):
                Gx = 0
                Gy = 0
                for i in range(len(prewittx)):
                    for j in range(len(prewitty)):
                        val = mat[row+i, col+j] * lin_scale
                        Gx += prewittx[i][j] * val
                        Gy += prewitty[i][j] * val

                pixels[row+1,col+1] = int(math.sqrt(Gx*Gx + Gy*Gy))
        self.prewitt_time = time.time() - start_time

    def save_im(self, name):
        self.prewittIm.save(name)


def test():
    im = 'jaguar'
    inName = im + '.jpg'
    outName = im + '-prewitt.jpg'
    prewitt = Prewitt(inName)
    prewitt.save_im(outName)