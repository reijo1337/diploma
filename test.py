import os

import cv2

from PIL import Image

from edge.canny_filter.utils.utils import rgb2gray
from preprocessing.detector import skin_detector

from scipy.ndimage.filters import convolve
import edge.canny_filter.canny_edge_detector as ced
import matplotlib.image as mpimg

if __name__ == "__main__":
    shape_path = "/home/tantsevov/diploma/data/asl-alphabet/shape"
    for root, dirs, files in os.walk("/home/tantsevov/diploma/data/asl-alphabet/asl_alphabet_train/asl_alphabet_train"):
        for name in files:
            label = root.split("/")[-1]
            directory = shape_path + "/" + label
            if not os.path.exists(directory):
                os.makedirs(directory)
            img = cv2.imread(os.path.join(root, name))
            img = skin_detector(img)
            cv2.imwrite(directory+"/"+name, img)
