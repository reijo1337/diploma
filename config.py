import argparse
import os
import pathlib
from datetime import datetime
import numpy as np


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


class Config:

    def __init__(self):
        # input data
        args = self.parse_input
        self.DATA_DIR = args.data_dir
        self.LOGS_DIR = os.path.join(args.data_dir, args.logs_dir)
        self.CHECKPOINT_FILE = os.path.join(args.data_dir, args.checkpoint_file)
        self.MODEL_PATH = os.path.join(args.data_dir, args.model_path)
        self.PLOTS_DIR = os.path.join(args.data_dir, args.plots_dir)
        self.SHOW_MODEL_SUMMARY = bool(args.show_model_summary)
        self.RECONSTRUCTION_ON = bool(args.reconstruction_on)
        self.NUM_EPOCH = int(args.num_epoch)
        self.LEARNING_RATE = int(args.learning_rate)
        self.BATCH_SIZE = int(args.batch_size)
        self.IMG_HEIGHT = int(args.img_height)
        self.IMG_WIDTH = int(args.img_width)
        self.CHANNELS = int(args.channels)
        self.ROUTING_ALGO = str(args.routing_algo)
        self.DEBUG_MODE_ON = bool(args.debug_mode_on)
        self.NUM_OF_CAPSULE_LAYERS = int(args.number_of_caps_layers)

        image_count = 1115

        self.STEPS_PER_EPOCH = int(np.ceil(image_count / self.BATCH_SIZE))
        self.CLASS_NAMES = np.sort(np.array([0,1,2,3,4,5,6,7,8,9]))

        self.NUM_ROUTING = 3

        self.DISTRIBUTION_STRATEGY = args.distribution_strategy
        self.NUM_GPUS = bool(args.num_gpus)
        self.TPU = args.tpu

    def __str__(self):
        result = []
        for x in dir(self):
            if x.isupper():
                result += [x + ': ' + str(self.__dict__[x])]
        return '\n'.join(result)

    @property
    def parse_input(self):
        # parse the arguments
        parser = argparse.ArgumentParser()
        parser.add_argument('--data_dir', help='directory where the data is stored with the folder names = labels',
                            default='/home/tantsevov/diploma/data/sign-language-between-0-9/')
        parser.add_argument('--logs_dir', help='directory where the logs are being stored',
                            default='logs/' + datetime.now().strftime('%Y%m%d-%H%M%S'))
        parser.add_argument('--model_dir', help='directory where the logs are being stored',
                            default='model/')
        parser.add_argument('--checkpoint_file', help='define the folder where the checkpoints of the models are being saved',
                            default='ckpt/weights.ckpt')  # https://github.com/keras-team/keras/issues/10652 (no '{}' are allowed in the filename path
        parser.add_argument("--show_model_summary", type=str2bool, nargs='?',
                            const=True, default=True,
                            help='show the summary of the keras model to be trained')
        parser.add_argument('--model_path', help='The folder where the model shall be saved. (Do not enter the whole path to the folder.)',
                            default='model/model.h5')
        parser.add_argument("--reconstruction_on", type=str2bool, nargs='?',
                            const=True, default=False,
                            help='show the summary of the keras model to be trained')
        parser.add_argument('--num_epoch', help='The number of epochs used for training.',
                            default=500)
        parser.add_argument('--batch_size', help='The batch size used for training.',
                            default=32)
        parser.add_argument('--img_height', help='The height of the image in pixel.',
                            default=300)
        parser.add_argument('--img_width', help='The width of the image in pixel.',
                            default=300)
        parser.add_argument('--channels', help='The number of channels used in the picture. (if black and white => 1, if 3 colours => 3)',
                            default=1)
        parser.add_argument('--routing_algo',
                            help='The routing algorithm used in the capsule layer. (standard: \'scalar_product\', min-max: \'min_max\')',
                            default='scalar_product')
        parser.add_argument('--plots_dir', help='The folder where all the plots are being saved',
                            default='plots/')
        parser.add_argument('--learning_rate', help='Specify the learning rate for the training. (integer)',
                            default=0.001, type=int)
        parser.add_argument("--debug_mode_on", type=str2bool, nargs='?',
                            const=True, default=False,
                            help='turn the debug mode settings on (small dataset)')
        parser.add_argument('--number_of_caps_layers', help='Specify how many capsule layers shall be included. (Must be greater or equal to one.)',
                            type=lambda x: (x > 0) and x.is_integer(), default=1)
        parser.add_argument('--distribution_strategy', help='Specify which type of distribution use',
                            default="off")
        parser.add_argument('--num_gpus', help='Specify count of gpu',
                            default=0, type=int)
        parser.add_argument('--tpu', help='Specify tpu address',
                            default="")
        args = parser.parse_args()
        return args

