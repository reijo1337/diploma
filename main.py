import os
from datetime import datetime

from skimage.transform import rescale, resize, downscale_local_mean
import matplotlib.image as mpimg

from analitics import fit_history_plot, evaluation_results, plot_prediction, save_model_summary
from classifier.capsnet import CapsNet
from config import Config
from edge.opencv_filter.opencv import shape
from sklearn.preprocessing import LabelBinarizer
import numpy as np
import tensorflow as tf

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

    with strategy_scope:
        model = CapsNet(config)
        model.compile(optimizer='adam',
                      loss=tf.keras.losses.categorical_crossentropy,  # margin_loss
                      metrics={'output_1': ['accuracy']})

    imgs, labels, imgs_test, labels_test, classes = load_images(config.DATA_DIR)

    validation_split = 0.2
    imgs_train = imgs[:int((1 - validation_split) * len(imgs))]
    labels_train = labels[:int((1 - validation_split) * len(labels))]
    imgs_validation = imgs[int(-validation_split * len(imgs)):]
    labels_validation = labels[int(-validation_split * len(labels)):]

    config.CLASS_NAMES = classes

    # callbacks
    tensorboard = tf.keras.callbacks.TensorBoard(log_dir=config.LOGS_DIR)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(config.CHECKPOINT_FILE,
                                                    save_best_only=True,
                                                    save_weights_only=True,
                                                    verbose=1)
    # training
    history = model.fit(x=imgs_train, y=labels_train,
                        batch_size=config.BATCH_SIZE, epochs=config.NUM_EPOCH,
                        validation_data=(imgs_validation, labels_validation),
                        shuffle=True, callbacks=[tensorboard, checkpoint])
    fit_history_plot(history, config.PLOTS_DIR)

    # evaluate
    [loss_, accuracy_] = model.evaluate(x=imgs_test, y=labels_test, batch_size=config.BATCH_SIZE, )
    loss__ = 'loss:     {:.5}'.format(loss_)
    accuracy__ = 'accuracy: {:.3%}'.format(accuracy_)
    evaluation_results([loss__, accuracy__], config.PLOTS_DIR)

    # prediction
    labels_test_pred = model.predict(imgs_test)
    pred = np.zeros_like(labels_test_pred)
    for index, value in enumerate(np.argmax(labels_test_pred, axis=-1)):
        pred[index][value] = 1.0
    labels_test_str = config.CLASS_NAMES[np.where(labels_test.numpy() == 1.0)[1]]
    labels_test_pred_str = config.CLASS_NAMES[np.where(pred == 1.0)[1]]
    plot_prediction(imgs_test, labels_test_str, labels_test_pred_str, config.PLOTS_DIR)

    # model summary
    save_model_summary(model, config.PLOTS_DIR)
    model.summary()
