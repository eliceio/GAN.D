import tensorflow as tf
import numpy as np

class AEmodel:

    def __init__(self):
        pass

    def Encoder(self):
        pass

    def Decoder(self):
        latent_inputs = x
        x = tf.layers.dense(latent_inputs, shape[1] * shape[2] * shape[3], activation='relu', kernel_initializer='glorot_uniform')
        x = tf.reshape(x, [shape[1], shape[2], shape[3]])
        x = tf.layers.dense(x, 128, kernel_initializer='glorot_uniform')

        x = tf.layers.conv2d(x, filters=32, kernel_size=3, activation='relu', padding='same')
        x = tf.image.resize_images(x, size=[2, 2])
        x = tf.layers.conv2d(x, filters=64, kernel_size=3, activation='relu', padding='same')
        x = tf.image.resize_images(x, size=[2, 2])
        x = tf.layers.conv2d(x, filters=128, kernel_size=3, activation='relu', padding='same')
        x = tf.image.resize_images(x, size=[2, 2])
        x = tf.layers.conv2d(x, filters=1, kernel_size=3, activation='sigmoid', padding='same')
        return z



class RNN_Model:
    def __init__(self):
        pass

    def network(self):
        pass
