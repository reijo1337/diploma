import os
import unittest

import matplotlib.image as mpimg
from matplotlib.pyplot import imsave, gray, figure, subplot, imshow, savefig

import edge.canny_filter.canny_edge_detector as ced

from edge.canny_filter.utils.utils import rgb2gray
from edge.opencv_filter.opencv import opencv, paln_point, skel
from edge.sobel import Sobel
from edge.prewitt import Prewitt
from edge.roberts import Roberts

nothing_image_path = "../data/dataset/asl-alphabet/asl_alphabet_test/nothing_test.jpg"


class MyTestCase(unittest.TestCase):
    def test_sobel(self):
        print(f'SOBEL')
        dataset_path = "../data/dataset/asl-alphabet/asl_alphabet_test"
        output_path = "sobel"
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        for image in os.listdir(dataset_path):
            print(f'Processing {image}')
            out_name = image
            sobel = Sobel(os.path.join(dataset_path, image))
            sobel.save_im(os.path.join(output_path, out_name))

    def test_prewitt(self):
        print(f'PREWITT')
        dataset_path = "../data/dataset/asl-alphabet/asl_alphabet_test"
        output_path = "prewitt"
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        for image in os.listdir(dataset_path):
            print(f'Processing {image}')
            out_name = image
            sobel = Prewitt(os.path.join(dataset_path, image))
            sobel.save_im(os.path.join(output_path, out_name))

    def test_roberts(self):
        print(f'ROBERTS')
        dataset_path = "../data/dataset/asl-alphabet/asl_alphabet_test"
        output_path = "roberts"
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        for image in os.listdir(dataset_path):
            print(f'Processing {image}')
            out_name = image
            sobel = Roberts(os.path.join(dataset_path, image))
            sobel.save_im(os.path.join(output_path, out_name))

    def test_canny(self):
        print("CANNY")
        dataset_path = "../data/dataset/asl-alphabet/asl_alphabet_test"
        output_path = "canny"
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        for image in os.listdir(dataset_path):
            print(f'Processing {image}')
            img = mpimg.imread(os.path.join(dataset_path, image))
            img = rgb2gray(img)
            detector = ced.cannyEdgeDetector([img], sigma=1.0, kernel_size=5, lowthreshold=0.01, highthreshold=0.07,
                                             weak_pixel=100)
            img_final = detector.detect()[0]
            if img_final.shape[0] == 3:
                img_final = img_final.transpose(1, 2, 0)
            gray()
            imsave(os.path.join(output_path, image), img_final)

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
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        for image in os.listdir(dataset_path):
            print(f'Processing {image}')
            orig = mpimg.imread(os.path.join(dataset_path, image))
            canny = mpimg.imread(os.path.join("canny", image))
            roberts = mpimg.imread(os.path.join("roberts", image))
            prewitt = mpimg.imread(os.path.join("prewitt", image))
            sobel = mpimg.imread(os.path.join("sobel", image))
            openc = mpimg.imread(os.path.join("opencv", image))
            figure(figsize=(20, 20))
            subplot(3, 2, 1, title="canny", xticks=[], yticks=[])
            imshow(canny, "gray")
            subplot(3, 2, 2, title="roberts", xticks=[], yticks=[])
            imshow(roberts, "gray")
            subplot(3, 2, 3, title="prewitt", xticks=[], yticks=[])
            imshow(prewitt, "gray")
            subplot(3, 2, 4, title="sobel", xticks=[], yticks=[])
            imshow(sobel, "gray")
            subplot(3, 2, 5, title="opencv", xticks=[], yticks=[])
            imshow(openc, "gray")
            subplot(3, 2, 6, title="original", xticks=[], yticks=[])
            imshow(orig)
            savefig(os.path.join(output_path, image))


if __name__ == '__main__':
    unittest.main()
