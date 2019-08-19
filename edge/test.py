import os
import time
import unittest
import pandas as pd
import matplotlib.image as mpimg
from matplotlib.pyplot import imsave, gray, figure, subplot, imshow, savefig
import numpy as np
import edge.canny_filter.canny_edge_detector as ced

from edge.canny_filter.utils.utils import rgb2gray
from edge.opencv_filter.opencv import opencv, paln_point, skel
from edge.skelet_dnn.main import skeleton
from edge.sobel import Sobel
from edge.prewitt import Prewitt
from edge.roberts import Roberts

nothing_image_path = "../data/dataset/asl-alphabet/asl_alphabet_test/nothing_test.jpg"


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
    # def test_sobel(self):
    #     print(f'SOBEL')
    #     dataset_path = "../data/dataset/asl-alphabet/asl_alphabet_test"
    #     output_path = "sobel"
    #     if not os.path.exists(output_path):
    #         os.makedirs(output_path)
    #     for image in os.listdir(dataset_path):
    #         print(f'Processing {image}')
    #         out_name = image
    #         sobel = Sobel(os.path.join(dataset_path, image))
    #         sobel.save_im(os.path.join(output_path, out_name))

    # def test_prewitt(self):
    #     print(f'PREWITT')
    #     dataset_path = "../data/dataset/asl-alphabet/asl_alphabet_test"
    #     output_path = "prewitt"
    #     if not os.path.exists(output_path):
    #         os.makedirs(output_path)
    #     for image in os.listdir(dataset_path):
    #         print(f'Processing {image}')
    #         out_name = image
    #         sobel = Prewitt(os.path.join(dataset_path, image))
    #         sobel.save_im(os.path.join(output_path, out_name))

    # def test_roberts(self):
    #     print(f'ROBERTS')
    #     dataset_path = "../data/dataset/asl-alphabet/asl_alphabet_test"
    #     output_path = "roberts"
    #     if not os.path.exists(output_path):
    #         os.makedirs(output_path)
    #     for image in os.listdir(dataset_path):
    #         print(f'Processing {image}')
    #         out_name = image
    #         sobel = Roberts(os.path.join(dataset_path, image))
    #         sobel.save_im(os.path.join(output_path, out_name))

    # def test_canny(self):
    #     print("CANNY")
    #     dataset_path = "../data/dataset/asl-alphabet/asl_alphabet_test"
    #     output_path = "canny"
    #     if not os.path.exists(output_path):
    #         os.makedirs(output_path)
    #     for image in os.listdir(dataset_path):
    #         print(f'Processing {image}')
    #         img = mpimg.imread(os.path.join(dataset_path, image))
    #         img = rgb2gray(img)
    #         detector = ced.cannyEdgeDetector([img], sigma=1.0, kernel_size=5, lowthreshold=0.01, highthreshold=0.07,
    #                                          weak_pixel=100)
    #         img_final = detector.detect()[0]
    #         if img_final.shape[0] == 3:
    #             img_final = img_final.transpose(1, 2, 0)
    #         gray()
    #         imsave(os.path.join(output_path, image), img_final)

    def test_opencv(self):
        print("OPENCV")
        dataset_path = "../data/dataset/asl-alphabet/asl_alphabet_test"
        output_path = "opencv"
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        if not os.path.exists(os.path.join(output_path, 'skel')):
            os.makedirs(os.path.join(output_path, 'skel'))
        for image in os.listdir(dataset_path):
            print(f'Processing {image}')
            img = mpimg.imread(os.path.join(dataset_path, image))
            img = opencv(img)
            # img = paln_point(img)
            sk = skel(img)
            img = img + sk
            imsave(os.path.join(output_path, image), img)
            imsave(os.path.join(output_path, 'skel', image), sk)

    def test_compare(self):
        print("Compare")
        dataset_path = "../data/dataset/asl-alphabet/asl_alphabet_test"
        output_path = "compare"
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
            skel_opencv_times.append(skel_opencv_time)
            skel_dnn_times.append(skel_dnn_time)

            figure(figsize=(20, 20))
            subplot(3, 3, 1, title="canny", xticks=[], yticks=[])
            imshow(canny, "gray")
            subplot(3, 3, 2, title="roberts", xticks=[], yticks=[])
            imshow(roberts.sobelIm, "gray")
            subplot(3, 3, 3, title="prewitt", xticks=[], yticks=[])
            imshow(prewitt.prewittIm, "gray")
            subplot(3, 3, 4, title="sobel", xticks=[], yticks=[])
            imshow(sobel.sobelIm, "gray")
            subplot(3, 3, 5, title="opencv", xticks=[], yticks=[])
            imshow(openc, "gray")
            subplot(3, 3, 6, title="skel_openv", xticks=[], yticks=[])
            imshow(skek_opencv)
            subplot(3, 3, 7, title="skel_dnn", xticks=[], yticks=[])
            imshow(skel_dnn)
            subplot(3, 3, 8, title="original", xticks=[], yticks=[])
            imshow(orig)
            savefig(os.path.join(output_path, image))
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

        result = pd.DataFrame(data=data, index=["canny", "roberts", "prewitt", "sobel", "opencv", "skel_opencv", "skel_dnn"])
        print(result)
        result.to_csv("result.csv")


if __name__ == '__main__':
    unittest.main()
