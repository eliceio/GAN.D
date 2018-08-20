import tensorflow as tf
import numpy as np

class AE_Model:

    def __init__(self):
        pass

    def Encoder(self):
        x=tf.placeholder(self.input_shape)
        x=tf.nn.relu(tf.layers.conv2d(x, 128, kernel_size=3, padding='same'))
        x=tf.layers.max_pooling2d(x, pool_size=2)
        x=tf.nn.relu(tf.layers.conv2d(x, 64, kernel_size=3, padding='same'))
        x=tf.layers.max_pooling2d(x, pool_size=2)
        x = tf.nn.relu(tf.layers.conv2d(x, 32, kernel_size=3, padding='same'))
        x = tf.layers.max_pooling2d(x, pool_size=2)

        shape=tf.shape(x)
        x = tf.layers.flatten(x)
        x = tf.layers.dense(x, 128)

        z_mean = tf.layers.dense(x, self.latent_shape)
        z = sampling(z_mean, z_log_var)

        return z


    def Decoder(self):
        # Encoder
        # Decoder
        pass


class RNN_Model:
    def __init__(self):
        pass

    def network(self):
        pass
