import keras.backend as K
from keras import layers
class Length(layers.Layer):
    def call(self, inputs, **kwargs):
        return K.sqrt(K.sum(K.square(inputs), -1))
    def compute_output_shape(self, input_shape):
        return input_shape[:-1]