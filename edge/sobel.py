"""
Written in Python 2.7!
This module takes an image and converts it to grayscale, then applies a
Sobel operator.
"""

__author__ = "Kevin Gay"

from PIL import Image
import math


class Sobel(object):

    def __init__(self, im_path):

        self.im = Image.open(im_path).convert('L')
        self.width, self.height = self.im.size
        mat = self.im.load()

        sobelx = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
        sobely = [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]

        self.sobelIm = Image.new('L', (self.width, self.height))
        pixels = self.sobelIm.load()

        lin_scale = .25

        # For each pixel in the image
        for row in range(self.width-len(sobelx)):
            for col in range(self.height-len(sobelx)):
                gx = 0
                gy = 0
                for i in range(len(sobelx)):
                    for j in range(len(sobely)):
                        val = mat[row+i, col+j] * lin_scale
                        gx += sobelx[i][j] * val
                        gy += sobely[i][j] * val

                pixels[row+1, col+1] = int(math.sqrt(gx*gx + gy*gy))

    def save_gray(self, name):
        self.im.save(name)

    def save_im(self, name):
        self.sobelIm.save(name)


def test():
    im = 'jaguar'
    inName = im + '.jpg'
    out_name = im + '-sobelNew.jpg'
    sobel = Sobel(inName)
    sobel.save_im(out_name)
    sobel.save_gray('jaguar-gray.jpg')
