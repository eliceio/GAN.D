import tensorflow as tf
import numpy as np

class AE_Model:
    def __init__(self, latent_dim, batch_size):
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.input_shape = [256, 256, 1] #w * h * c
        self.latent_shape = 128
    

    def sampling(z_mean,z_log_var):
        z_mean= z_mean
        z_log_var = z_log_var
        batch = self.batch_size
        dim = self.latent_shapeßS
        epsilon = tf.random_normal(shape=(batch,dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

    def encoder(self):
        pass


    def decoder(self):s
        pass


class RNN_Model:
    def __init__(self):
        pass

    def network(self):
        pass
