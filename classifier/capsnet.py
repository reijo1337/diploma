import tensorflow as tf

# own classes
from classifier.capsule_layer import CapsuleLayer
from classifier.length import Length
from classifier.save_squash import save_squash


class CapsNet(tf.keras.Model):

    def __init__(self, config, **kwargs):
        super(CapsNet, self).__init__(**kwargs)

        self.config = config
        conv1_strides = 1
        conv1_kernel_size = 9
        conv1_filters = 256

        conv1_output_width = int(((config.IMG_WIDTH - conv1_kernel_size) / conv1_strides) + 1)
        conv1_output_height = int(((config.IMG_HEIGHT - conv1_kernel_size) / conv1_strides) + 1)

        conv2_strides = 2
        conv2_kernel_size = 9

        conv2_output_width = int(((conv1_output_width - conv2_kernel_size) / conv2_strides) + 1)
        conv2_output_height = int(((conv1_output_height - conv2_kernel_size) / conv2_strides) + 1)

        primary_capsule_input_dim = 8
        primary_capsule_map = 32
        capsule_output_dim = 16


        # Layer 1: Just a conventional Conv2D layer
        self.conv1 = tf.keras.layers.Conv2D(filters=conv1_filters,
                                            kernel_size=conv1_kernel_size,
                                            strides=conv1_strides,
                                            padding='valid',
                                            activation='relu',
                                            name='conv1')

        # Layer 2: Conv2D layer with `squash` activation, then reshape to [None, num_capsule, dim_vector]
        self.conv2 = tf.keras.layers.Conv2D(filters=primary_capsule_input_dim * primary_capsule_map,
                                            kernel_size=conv2_kernel_size,
                                            strides=conv2_strides,
                                            padding='valid')
        self.reshape1 = tf.keras.layers.Reshape(target_shape=[conv2_output_width * conv2_output_height * primary_capsule_map, primary_capsule_input_dim, ],
                                                name='primarycaps')
        self.squash = tf.keras.layers.Lambda(save_squash, name='squash')

        # Layer 3: Capsule layer. Routing algorithm works here.
        if self.config.NUM_OF_CAPSULE_LAYERS > 1:
            self.caps_1 = CapsuleLayer(num_capsule=len(config.CLASS_NAMES),
                                       dim_vector=capsule_output_dim,
                                       num_routing=config.NUM_ROUTING,
                                       routing_algo=config.ROUTING_ALGO,
                                       name='caps_1')
        self.caps_out = CapsuleLayer(num_capsule=len(config.CLASS_NAMES),
                                     dim_vector=capsule_output_dim,
                                     num_routing=config.NUM_ROUTING,
                                     routing_algo=config.ROUTING_ALGO,
                                     name='caps_out')

        # Layer 4: This is an auxiliary layer to replace each capsule with its length. Just to match the true label's shape.
        # If using tensorflow, this will not be necessary. :)
        self.y_prob = Length(name='y_prob')

    def call(self, inputs, *args, **kwargs):

        # validate input
        if not self.config.RECONSTRUCTION_ON:
            img = inputs
        elif len(inputs) == 2:
            assert self.config.RECONSTRUCTION_ON, 'The reconstruction layer is on and therefore we require two inputs'
            img = inputs['input_1']
            y = inputs['input_2']

        # route date through network
        conv1 = self.conv1(img)
        conv2 = self.conv2(conv1)
        reshape1 = self.reshape1(conv2)
        squash = self.squash(reshape1)

        if self.config.NUM_OF_CAPSULE_LAYERS > 1:
            caps_1 = self.caps_1(squash)
            caps_out = self.caps_out(caps_1)
        else:
            caps_out = self.caps_out(squash)

        y_prob = self.y_prob(caps_out)

        # don't build the net any further if we don't need the reconstruction
        if not self.config.RECONSTRUCTION_ON:
            return y_prob

        masked = self.masked([caps_out, y])
        secondarycaps = self.secondarycaps(masked)
        x_recon1 = self.x_recon1(secondarycaps)
        x_recon2 = self.x_recon2(x_recon1)
        x_recon3 = self.x_recon3(x_recon2)
        reshape2 = self.reshape2(x_recon3)

        return ([y_prob, reshape2])
