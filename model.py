import tensorflow as tf
import numpy as np

class AE_Model:
    def __init__(self):
        pass

    def network(self,x) :
        x = tf.nn.relu(tf.layers.conv2d(x, 128, kernel_size=3, padding='same'))
        x = tf.layers.max_pooling2d(x, pool_size=2)
        x = tf.nn.relu(tf.layers.conv2d(x, 64, kernel_size=3, padding='same'))
        x = tf.layers.max_pooling2d(x, pool_size=2)
        x = tf.nn.relu(tf.layers.conv2d(x, 32, kernel_size=3, padding='same'))
        x = tf.layers.max_pooling2d(x, pool_size=2)

        shape = tf.shape(x)
        x = tf.layers.flatten(x)
        x = tf.layers.dense(x, 128)

        z_mean = tf.layers.dense(x, self.latent_shape)
        z = sampling(z_mean, z_log_var)

        latent_inputs = z
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
