import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

from preprocessing.detector import skin_detector


def rsl_loader():
    images, labels = [], []
    for root, dirs, files in os.walk("data/rsl/rsl-alphabet-dataset-master"):
        for name in files:
            img = cv2.imread(os.path.join(root, name))
            label = name.split("_")[1]
            img = cv2.resize(img, dsize=(40, 40), interpolation=cv2.INTER_CUBIC)
            img = skin_detector(img)
            images.append(img)
            labels.append(label)
    unique_val = np.unique(np.array(labels))

    images = np.array(images)
    images = images / 255

    label_binrizer = LabelBinarizer()
    labels = label_binrizer.fit_transform(labels)

    x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.1, stratify=labels, random_state=7)

    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)

    return x_train, y_train, x_test, y_test, unique_val
