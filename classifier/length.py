import tensorflow as tf


class Length(tf.keras.layers.Layer):
    """
    Compute the length of vectors. This is used to compute a Tensor that has the same shape with y_true in margin_loss
    inputs: shape=[dim_1, ..., dim_{n-1}, dim_n]
    output: shape=[dim_1, ..., dim_{n-1}]
    """
    def call(self, inputs, **kwargs):

        temp = tf.keras.backend.sum(tf.keras.backend.square(inputs), axis=-1, keepdims=False) + tf.keras.backend.epsilon()
        y_prob = tf.keras.backend.sqrt(temp)  # safe with K.epsilon()
        y_pred = tf.reshape(tf.one_hot(tf.math.top_k(y_prob)[1],
                                       depth=y_prob.get_shape()[-1]),
                            [-1, y_prob.get_shape()[-1]])
        return y_prob

    def compute_output_shape(self, input_shape):
        return input_shape[:-1]
