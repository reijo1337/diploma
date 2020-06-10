import tensorflow as tf
import keras.backend as K
class CapsuleLayer(layers.Layer):
  def call(self, inputs):
    inputs_expand = K.expand_dims(K.expand_dims(inputs, 2), 2)
    inputs_tiled = K.tile(inputs_expand, [1, 1, self.num_capsule, 1, 1])
    inputs_hat = tf.scan(lambda ac, x: K.batch_dot(x, self.W, [3, 2]), elems=inputs_tiled, initializer=K.zeros([self.input_num_capsule, self.num_capsule, 1, self.dim_vector]))
    assert self.num_routing>0, 'The num_routing should be >0.'
    for i in range(self.num_routing):
      c = tf.nn.softmax(self.bias, dim=2) 
      outputs = squash(K.sum(c * inputs_hat, 1, keepdims=True))
      if i != self.num_routing - 1:
        self.bias += K.sum(inputs_hat * outputs, -1, keepdims=True)
    return K.reshape(outputs, [-1, self.num_capsule, self.dim_vector])

