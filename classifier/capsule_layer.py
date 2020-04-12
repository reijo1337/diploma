import tensorflow as tf

from classifier.save_squash import save_squash


class CapsuleLayer(tf.keras.layers.Layer):
    """
    The capsule layer. It is similar to Dense layer. Dense layer has `in_num` inputs, each is a scalar, the output of the
    neuron from the former layer, and it has `out_num` output neurons. CapsuleLayer just expand the output of the neuron
    from scalar to vector. So its input shape = [None, input_num_capsule, input_dim_vector] and output shape = \
    [None, num_capsule, dim_vector]. For Dense Layer, input_dim_vector = dim_vector = 1.

    :param num_capsule: number of capsules in this layer
    :param dim_vector: dimension of the output vectors of the capsules in this layer
    :param num_routings: number of iterations for the routing algorithm
    """

    def __init__(self, num_capsule,
                 dim_vector,
                 num_routing=3,
                 routing_algo='scalar_product',
                 **kwargs):
        super(CapsuleLayer, self).__init__(**kwargs)
        self.num_capsule = num_capsule
        self.dim_vector = dim_vector
        assert num_routing > 0, 'The num_routing should be > 0.'

        # routing algorithm details
        possible_routing_algos = ['scalar_product', 'min_max']
        assert routing_algo in possible_routing_algos, 'please select one of the possible routing algorithms: ' + str(possible_routing_algos)
        self.routing_algo = routing_algo
        self.num_routing = num_routing

    def build(self, input_shape):
        assert len(input_shape) >= 3, "The input Tensor should have shape=[None, input_num_capsule, input_dim_vector]"
        self.input_num_capsule = input_shape[1]
        self.input_dim_vector = input_shape[2]

        # Transform matrix
        self.W = self.add_weight(shape=[self.input_num_capsule, self.num_capsule, self.dim_vector, self.input_dim_vector],
                                 initializer='glorot_uniform', name='W', trainable=True)  # tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.5)

        # Coupling coefficient. The redundant dimensions are just to facilitate subsequent matrix calculation.
        self.bias = self.add_weight(shape=[self.input_num_capsule, self.num_capsule, 1, 1],
                                    initializer='zeros',
                                    name='bias',
                                    trainable=False)
        self.built = True

    def call(self, inputs):
        # inputs.shape=[None, input_num_capsule, input_dim_vector]
        # Expand dims to [batch_size, input_num_capsule, 1, 1, input_dim_vector]
        inputs_expand = tf.keras.backend.expand_dims(tf.keras.backend.expand_dims(inputs, -1), 2)

        # Replicate num_capsule dimension to prepare being multiplied by W
        # Now it has shape = [None, input_num_capsule, num_capsule, 1, input_dim_vector]
        inputs_tiled = tf.tile(inputs_expand, [1, 1, self.num_capsule, 1, 1])

        """  
        # Compute `inputs * W` by expanding the first dim of W. More time-consuming and need batch_size.
        # Now W has shape  = [batch_size, input_num_capsule, num_capsule, input_dim_vector, dim_vector]
        w_tiled = K.tile(K.expand_dims(self.W, 0), [self.batch_size, 1, 1, 1, 1])

        # Transformed vectors, inputs_hat.shape = [None, input_num_capsule, num_capsule, 1, dim_vector]
        inputs_hat = K.batch_dot(inputs_tiled, w_tiled, [4, 3])
        """

        # self.W needs an additional dimension to cope with the batch size.
        # matmul does not care if the batch_size does not match. It just uses the same self.W multiple times
        # inputs_hat.shape = [batch_size, input_num_capsule, num_capsule, 1, 16]
        inputs_hat = tf.matmul(tf.keras.backend.expand_dims(self.W, 0), inputs_tiled,
                               name='u_hat_transformation')

        if self.routing_algo is 'scalar_product':
            # b = tf.keras.backend.expand_dims(tf.keras.backend.zeros(shape=[self.input_num_capsule, self.num_capsule, 1, 1],
            #                                                        name='b'), 0)
            b = tf.keras.backend.expand_dims(self.bias, 0)
            for i in range(self.num_routing - 1):
                c = tf.nn.softmax(b, axis=2, name='routing_weights')
                v = save_squash(tf.keras.backend.sum(c * inputs_hat, axis=1, keepdims=True), axis=-2)
                v_tiled = tf.tile(v, [1, self.input_num_capsule, 1, 1, 1], name="v_tiled")
                agreement = tf.matmul(inputs_hat, v_tiled, transpose_a=True, name="agreement")
                b = agreement + b + tf.keras.backend.expand_dims(self.bias, 0)
            c_last = tf.nn.softmax(b, axis=2, name="routing_weights")
            outputs = save_squash(tf.keras.backend.sum(c_last * inputs_hat, 1, keepdims=True), axis=-2)
        elif self.routing_algo is 'min_max':
            p = tf.constant(0.0)
            q = tf.constant(1.0)
            b = tf.keras.backend.expand_dims(self.bias, 0)
            c = tf.keras.backend.ones_like(b)
            for i in range(self.num_routing - 1):
                v = save_squash(tf.keras.backend.sum(c * inputs_hat, axis=1, keepdims=True), axis=-2)
                v_tiled = tf.tile(v, [1, self.input_num_capsule, 1, 1, 1], name="v_tiled")
                agreement = tf.matmul(inputs_hat, v_tiled, transpose_a=True, name="agreement")
                b = agreement + b
                b_min = tf.keras.backend.min(b, axis=-4, keepdims=True)
                b_max = tf.keras.backend.max(b, axis=-4, keepdims=True)
                c = p + tf.math.divide(b - b_min, b_max - b_min, name='min_max_division') * (q - p)
            outputs = save_squash(tf.keras.backend.sum(c * inputs_hat, axis=1, keepdims=True), axis=-2)

        return tf.keras.backend.reshape(outputs, [-1, self.num_capsule, self.dim_vector])

    def compute_output_shape(self, input_shape):
        return tuple([None, self.num_capsule, self.dim_vector])
