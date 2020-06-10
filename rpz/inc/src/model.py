from keras import layers, models
def CapsNet(input_shape, n_class, num_routing):
    x = layers.Input(shape=input_shape, name="image")
    conv5 = layers.Conv2D(filters=256, kernel_size=9, strides=1, padding='valid', activation='relu', name='conv')(x)
    primarycaps = PrimaryCap(conv5, dim_vector=8, n_channels=32, kernel_size=9, strides=2, padding='valid')
    digitcaps = CapsuleLayer(num_capsule=n_class, dim_vector=16, num_routing=num_routing, name='digitcaps')(primarycaps)
    out_caps = Length(name='out')(digitcaps)

    return models.Model(x, out_caps)
