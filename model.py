import tensorflow as tf
import numpy as np
from module import LSTM, Dropout, Dense
from tensorflow.contrib.distributions import Categorical, Mixture, MultivariateNormalDiag


class AE_Model:
    def __init__(self, latent_dim, batch_size):
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.input_shape = [256, 256, 1]  # w * h * c
        self.latent_shape = 128

    def sampling(self, z_mean, z_log_var):
        z_mean = z_mean
        z_log_var = z_log_var
        batch = self.batch_size
        dim = self.latent_shape
        epsilon = tf.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

    def network(self, x):
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
        z = self.sampling(z_mean, z_mean)

        latent_inputs = z
        x = tf.nn.relu(tf.layers.dense(latent_inputs, shape[1] * shape[2] * shape[3],
                            kernel_initializer='glorot_uniform'))
        x = tf.reshape(x, [shape[1], shape[2], shape[3]])
        x = tf.layers.dense(x, 128, kernel_initializer='glorot_uniform')
   
        x = tf.nn.relu(tf.layers.conv2d(x, filters=32, kernel_size=3, padding='same'))
        x = tf.image.resize_images(x, size=[2*shape[1], 2*shape[2]])
        x = tf.nn.relu(tf.layers.conv2d(x, filters=64, kernel_size=3, padding='same'))
        x = tf.image.resize_images(x, size=[2*shape[1], 2*shape[2]])
        x = tf.nn.relu(tf.layers.conv2d(x, filters=128, kernel_size=3, padding='same'))
        x = tf.image.resize_images(x, size=[2*shape[1], 2*shape[2]])
        x = tf.nn.sigmoid(tf.layers.conv2d(x, filters=1, kernel_size=3, padding='same'))
        return z


class RNN_Model:
    def __init__(self):
        self.input = tf.placeholder(tf.float32, shape=(None, 128))

    def network(self):
        X = LSTM(self.input, 512)
        X = Dropout(X, 0.40)
        X = LSTM(X, 512)
        X = Dropout(X, 0.40)
        X = Dense(X, num_unit=1000, activation=tf.nn.relu)
        outputs = X  # todo mdn 추가 할 것

        self.pred = outputs

    def loss(self):
        return self.input - self.pred


class MDN:
    def __init__(self):
        pass

    def network(self):
        pass
