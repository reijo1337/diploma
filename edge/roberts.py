"""
Written in Python 2.7!
This module takes an image and converts it to grayscale, then applies a
Roberts Cross operator.
"""
import time

__author__ = "Kevin Gay"

from PIL import Image
import math


class Roberts(object):
    def __init__(self, im_path):
        start_time = time.time()
        im = Image.open(im_path).convert('L')
        self.width, self.height = im.size
        mat = im.load()

        robertsx = [[1,0],[0,-1]]
        robertsy = [[0,1],[-1,0]]

        self.sobelIm = Image.new('L', (self.width, self.height))
        pixels = self.sobelIm.load()

        lin_scale = .7
        for row in range(self.width-len(robertsx)):
            for col in range(self.height-len(robertsy)):
                Gx = 0
                Gy = 0
                for i in range(len(robertsx)):
                    for j in range(len(robertsy)):
                        val = mat[row+i, col+j] * lin_scale
                        Gx += robertsx[i][j] * val
                        Gy += robertsy[i][j] * val

                pixels[row+1,col+1] = int(math.sqrt(Gx*Gx + Gy*Gy))
        self.roberts_time = time.time() - start_time

    def save_im(self, name):
        self.sobelIm.save(name)


def test():
    im = 'jaguar'
    inName = im + '.jpg'
    outName = im + '-roberts.jpg'
    roberts = Roberts(inName)
    roberts.save_im(outName)