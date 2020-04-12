import tensorflow as tf


def save_squash(vectors, axis=-1, keepdims=True):
    """
    The non-linear activation used in Capsule. It drives the length of a large vector to near 1 and small vector to 0
    :param vectors: some vectors to be squashed, N-dim tensor
    :param axis: the axis to squash
    :return: a Tensor with same shape as input vectors
    """
    s_squared_norm = tf.keras.backend.sum(tf.keras.backend.square(vectors), axis, keepdims=keepdims) + \
                     tf.keras.backend.epsilon()
    scale = s_squared_norm / (1 + s_squared_norm) / tf.keras.backend.sqrt(s_squared_norm)
    return scale * vectors
