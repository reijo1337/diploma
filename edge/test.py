import os
import time
import unittest

import cv2
import matplotlib
import pandas as pd
import matplotlib.image as mpimg
from matplotlib.pyplot import gray, figure, subplot, imshow, savefig, close
import numpy as np
import edge.canny_filter.canny_edge_detector as ced

from edge.canny_filter.utils.utils import rgb2gray
from edge.opencv_filter.opencv import opencv, skel, shape
from edge.skelet_dnn.main import skeleton
from edge.sobel import Sobel
from edge.prewitt import Prewitt
from edge.roberts import Roberts
from PIL import Image
from scipy.ndimage.filters import convolve

from preprocessing.detector import skin_detector

asl_dataset = "../data/dataset/asl-alphabet/asl_alphabet_test"
datamix_dataset = "../data/dataset/data_mix_300/test"
colombian_dataset = "../data/dataset/dataset/test"
asl2_dataset = "../data/dataset/dataset5/test"


def canny_func(frame):
    start_time = time.time()
    img = rgb2gray(frame)
    detector = ced.cannyEdgeDetector([img], sigma=1.0, kernel_size=5, lowthreshold=0.01, highthreshold=0.07,
                                     weak_pixel=100)
    img_final = detector.detect()[0]
    if img_final.shape[0] == 3:
        img_final = img_final.transpose(1, 2, 0)
    gray()
    return img_final, time.time() - start_time


class MyTestCase(unittest.TestCase):
    @staticmethod
    def test_convert():
        for image in os.listdir(asl2_dataset):
            image_jpg = image.split(".")[0] + ".jpg"
            im = Image.open(os.path.join(asl2_dataset, image))
            os.remove(os.path.join(asl2_dataset, image))
            im.save(os.path.join(asl2_dataset, image_jpg), quality=95)

    @staticmethod
    def test_canny():
        path = "../data/temp2"
        original = mpimg.imread(os.path.join(path, "bmstu.png"))
        original_gray = rgb2gray(original)
        Image.fromarray(original_gray).convert('RGB').save(os.path.join(path, "bmstu_gray.png"))
        detector = ced.cannyEdgeDetector([original_gray], sigma=1.0, kernel_size=5, lowthreshold=0.03,
                                         highthreshold=0.07, weak_pixel=100)
        smoothed1 = convolve(original_gray, detector.gaussian_kernel(5, 1.0))
        Image.fromarray(smoothed1).convert('RGB').save(os.path.join(path, "bmstu_smoothed.png"))

        gradient_mat2, theta_mat2 = detector.sobel_filters(smoothed1)
        Image.fromarray(gradient_mat2).convert('RGB').save(os.path.join(path, "bmstu_gradient.png"))

        non_max3 = detector.non_max_suppression(gradient_mat2, theta_mat2)
        Image.fromarray(non_max3).convert('RGB').save(os.path.join(path, "bmstu_non_max.png"))

        threshold4 = detector.threshold(non_max3)
        Image.fromarray(threshold4).convert('RGB').save(os.path.join(path, "bmstu_threshold.png"))

        final5 = detector.hysteresis(threshold4)
        Image.fromarray(final5).convert('RGB').save(os.path.join(path, "bmstu_final.png"))

    @staticmethod
    def test_compare():
        start_time = time.time()
        matplotlib.rcParams.update({'font.size': 30})
        print("Compare")
        dataset_path = asl2_dataset
        output_path = "compare_asl2"
        canny_times, roberts_times, prewitt_times, sobel_times, \
            openc_times, skel_opencv_times, skel_dnn_times = [], [], [], [], [], [], []
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        for image in os.listdir(dataset_path):
            print(f'Processing {image}')
            orig = mpimg.imread(os.path.join(dataset_path, image))
            canny, canny_time = canny_func(orig)
            roberts = Roberts(os.path.join(dataset_path, image))
            prewitt = Prewitt(os.path.join(dataset_path, image))
            sobel = Sobel(os.path.join(dataset_path, image))
            openc, openc_time = opencv(orig)
            skek_opencv, skel_opencv_time = skel(openc.copy())
            skel_dnn, skel_dnn_time = skeleton(orig.copy())

            canny_times.append(canny_time)
            roberts_times.append(roberts.roberts_time)
            prewitt_times.append(prewitt.prewitt_time)
            sobel_times.append(sobel.sobel_time)
            openc_times.append(openc_time)
            skel_opencv_times.append(skel_opencv_time + openc_time)
            skel_dnn_times.append(skel_dnn_time)

            figure(figsize=(20, 20))
            subplot(3, 3, 1, title="а", xticks=[], yticks=[])
            imshow(canny, "gray")
            subplot(3, 3, 2, title="б", xticks=[], yticks=[])
            imshow(roberts.sobelIm, "gray")
            subplot(3, 3, 3, title="в", xticks=[], yticks=[])
            imshow(prewitt.prewittIm, "gray")
            subplot(3, 3, 4, title="г", xticks=[], yticks=[])
            imshow(sobel.sobelIm, "gray")
            subplot(3, 3, 5, title="д", xticks=[], yticks=[])
            imshow(openc, "gray")
            subplot(3, 3, 6, title="е", xticks=[], yticks=[])
            imshow(skek_opencv)
            subplot(3, 3, 7, title="ж", xticks=[], yticks=[])
            imshow(skel_dnn)
            subplot(3, 3, 8, title="з", xticks=[], yticks=[])
            imshow(orig)
            savefig(os.path.join(output_path, image))
            close()
        data = {"min": [], "max": [], "avg": []}
        data["min"].append(min(canny_times))
        data["max"].append(max(canny_times))
        data["avg"].append(np.average(canny_times))

        data["min"].append(min(roberts_times))
        data["max"].append(max(roberts_times))
        data["avg"].append(np.average(roberts_times))

        data["min"].append(min(prewitt_times))
        data["max"].append(max(prewitt_times))
        data["avg"].append(np.average(prewitt_times))

        data["min"].append(min(sobel_times))
        data["max"].append(max(sobel_times))
        data["avg"].append(np.average(sobel_times))

        data["min"].append(min(openc_times))
        data["max"].append(max(openc_times))
        data["avg"].append(np.average(openc_times))

        data["min"].append(min(skel_opencv_times))
        data["max"].append(max(skel_opencv_times))
        data["avg"].append(np.average(skel_opencv_times))

        data["min"].append(min(skel_dnn_times))
        data["max"].append(max(skel_dnn_times))
        data["avg"].append(np.average(skel_dnn_times))

        result = pd.DataFrame(data=data,
                              index=["canny", "roberts", "prewitt", "sobel", "opencv", "skel_opencv", "skel_dnn"])
        print(result)
        result.to_csv(os.path.join(output_path, "result.csv"))
        print(f'total time: {time.time() - start_time}')

    @staticmethod
    def test_shape():
        shape_path = "../data/asl-alphabet/shape"
        for root, dirs, files in os.walk("../data/asl-alphabet/asl_alphabet_train/asl_alphabet_train"):
            for name in files:
                label = root.split("/")[-1]
                directory = shape_path + "/" + label
                if not os.path.exists(directory):
                    os.makedirs(directory)
                img = cv2.imread(os.path.join(root, name))
                img = cv2.resize(img, dsize=(img.shape[0] // 3, img.shape[1] // 3), interpolation=cv2.INTER_CUBIC)
                img = skin_detector(img)
                cv2.imwrite(directory+"/"+name, img)

if __name__ == '__main__':
    shape_path = "../data/asl-alphabet/shape"
    for root, dirs, files in os.walk("../data/asl-alphabet/asl_alphabet_train/asl_alphabet_train"):
        for name in files:
            label = root.split("/")[-1]
            directory = shape_path + "/" + label
            if not os.path.exists(directory):
                os.makedirs(directory)
            img = cv2.imread(os.path.join(root, name))
            img = shape(img)
            cv2.imwrite(directory+"/"+name, img)
