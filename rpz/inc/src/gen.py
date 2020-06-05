# Creating of N=layerCount RNN layers
rnnLayer = MultiRNNCell([GRUCell(self.hiddenSize) for _ in range(self.layerCount)])
# Creating of RNN network
outputRnn, _ = tf.nn.dynamic_rnn(rnnLayer, self.input, dtype=tf.float32)
inputSize = np.shape(self.input)[2]
# Dense layer
musicHat = tf.layers.dense(inputs=outputRnn, units=inputSize, activation=tf.nn.relu, name='musicHat')
voiceHat = tf.layers.dense(inputs=outputRnn, units=inputSize, activation=tf.nn.relu, name='voiceHat')
# Time-frequency masking layer
retMusic = musicHat / (musicHat + voiceHat + np.finfo(float).eps) * self.input
retVoice = voiceHat / (musicHat + voiceHat + np.finfo(float).eps) * self.input
return retMusic, retVoice
