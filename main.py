import os
from datetime import datetime

from skimage.transform import resize
import matplotlib.image as mpimg

from analitics import fit_history_plot, evaluation_results, plot_prediction, save_model_summary
from classifier.capsnet import CapsNet
from config import Config
from edge.opencv_filter.opencv import shape
from sklearn.preprocessing import LabelBinarizer
import numpy as np
import tensorflow as tf

from test import create_model, train
from utils import get_distribution_strategy, get_strategy_scope


def load_images(root_dir):
    x_train = []
    y_train = []
    x_test = []
    y_test = []

    for root, dirs, files in os.walk("data/data_mix_300/data_mix_300/train"):
        for name in files:
            img = mpimg.imread(os.path.join(root, name))
            label = root.split("/")[-1]
            img = shape(img)
            img = resize(img, (img.shape[0] // 3, img.shape[1] // 3),
                         anti_aliasing=True)
            x_train.append(img)
            y_train.append(label)

    for root, dirs, files in os.walk("data/data_mix_300/data_mix_300/valid"):
        for name in files:
            img = mpimg.imread(os.path.join(root, name))
            label = root.split("/")[-1]
            img = shape(img)
            img = resize(img, (img.shape[0] // 3, img.shape[1] // 3),
                         anti_aliasing=True)
            x_test.append(img)
            y_test.append(label)
    label_binarizer = LabelBinarizer()
    unique_val = np.unique(np.array(y_test))

    y_train = label_binarizer.fit_transform(y_train)
    y_test = label_binarizer.fit_transform(y_test)
    x_test = np.array(x_test)
    x_train = np.array(x_train)

    h, w = x_test.shape[1], x_test.shape[2]

    x_train = x_train.reshape(x_train.shape[0], h, w, 1)
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)
    x_train = x_train / 255
    x_test = x_test / 255
    return x_train, y_train, x_test, y_test, unique_val


def store_config(name, config, path):
    data = str(name) + '\n' + str(config)
    with open("data/data_mix_300/plots/" + datetime.now().strftime('%Y-%m-%d_%H-%M-%S_') + str('config') + '.txt', 'w') as f:
        f.write(data)


if __name__ == "__main__":
    config = Config()
    store_config('capsule network', config, config.PLOTS_DIR)

    strategy = get_distribution_strategy(
        distribution_strategy=config.DISTRIBUTION_STRATEGY,
        num_gpus=config.NUM_GPUS,
        tpu_address=config.TPU)

    strategy_scope = get_strategy_scope(strategy)

    x_train, y_train, x_test, y_test, classes = load_images(config.DATA_DIR)

    config.CLASS_NAMES = classes

    with strategy_scope:
        model = create_model(x_test.shape[1:], len(config.CLASS_NAMES), config.NUM_ROUTING)
        model.compile(optimizer='adam',
                      loss=tf.keras.losses.categorical_crossentropy,  # margin_loss
                      metrics={'output_1': ['accuracy']})
        model.summary()
        train(model=model, data=((x_train, y_train), (x_test, y_test)))

    y_pred, x_recon = model.predict([x_test, y_test], batch_size=1)
    print('-'*50)
    print('Test acc:', np.sum(np.argmax(y_pred, 1) == np.argmax(y_test, 1))/y_test.shape[0])
