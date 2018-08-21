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
        x = tf.layers.dense(latent_inputs, shape[1] * shape[2] * shape[3], activation='relu',
                            kernel_initializer='glorot_uniform')
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
    def __init__(self, output_dimension, num_mixtures, **kwargs):
        self.output_dim = output_dimension
        self.num_mix = num_mixtures

    def __call__(self, *args, **kwargs):
        with tf.name_scope('MDN'):
            mdn_out = tf.concat([self.mdn_mus,
                                 self.mdn_sigmas,
                                 self.mdn_pi],
                                name='mdn_outputs')
        return mdn_out

    def elu_plus_one_plus_epsilon(self, x):
        """ELU activation with a very small addition to help prevent NaN in loss."""
        return (tf.nn.elu(x) + 1 + 1e-8)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

    def get_mixture_loss_func(self, output_dim, num_mixes):
        """Construct a loss functions for the MDN layer parametrised by number of mixtures."""

        # Construct a loss function with the right number of mixtures and outputs
        def loss_func(y_true, y_pred):
            out_mu, out_sigma, out_pi = tf.split(y_pred, num_or_size_splits=[num_mixes * output_dim,
                                                                             num_mixes * output_dim,
                                                                             num_mixes],
                                                 axis=1, name='mdn_coef_split')
            cat = Categorical(logits=out_pi)
            component_splits = [output_dim] * num_mixes
            mus = tf.split(out_mu, num_or_size_splits=component_splits, axis=1)
            sigs = tf.split(out_sigma, num_or_size_splits=component_splits, axis=1)
            coll = [MultivariateNormalDiag(loc=loc, scale_diag=scale) for loc, scale
                    in zip(mus, sigs)]
            mixture = Mixture(cat=cat, components=coll)
            loss = mixture.log_prob(y_true)
            loss = tf.negative(loss)
            loss = tf.reduce_mean(loss)
            return loss

        # Actually return the loss_func
        with tf.name_scope('MDN'):
            return loss_func

    def get_mixture_sampling_fun(self, output_dim, num_mixes):
        """Construct a sampling function for the MDN layer parametrised by mixtures and output dimension."""

        # Construct a loss function with the right number of mixtures and outputs
        def sampling_func(y_pred):
            out_mu, out_sigma, out_pi = tf.split(y_pred, num_or_size_splits=[num_mixes * output_dim,
                                                                             num_mixes * output_dim,
                                                                             num_mixes],
                                                 axis=1, name='mdn_coef_split')
            cat = Categorical(logits=out_pi)
            component_splits = [output_dim] * num_mixes
            mus = tf.split(out_mu, num_or_size_splits=component_splits, axis=1)
            sigs = tf.split(out_sigma, num_or_size_splits=component_splits, axis=1)
            coll = [MultivariateNormalDiag(loc=loc, scale_diag=scale) for loc, scale
                    in zip(mus, sigs)]
            mixture = Mixture(cat=cat, components=coll)
            samp = mixture.sample()
            # Todo: temperature adjustment for sampling function.
            return samp

        # Actually return the loss_func
        with tf.name_scope('MDNLayer'):
            return sampling_func

    def get_mixture_mse_accuracy(self, output_dim, num_mixes):
        """Construct an MSE accuracy function for the MDN layer
        that takes one sample and compares to the true value."""

        # Construct a loss function with the right number of mixtures and outputs
        def mse_func(y_true, y_pred):
            out_mu, out_sigma, out_pi = tf.split(y_pred, num_or_size_splits=[num_mixes * output_dim,
                                                                             num_mixes * output_dim,
                                                                             num_mixes],
                                                 axis=1, name='mdn_coef_split')
            cat = Categorical(logits=out_pi)
            component_splits = [output_dim] * num_mixes
            mus = tf.split(out_mu, num_or_size_splits=component_splits, axis=1)
            sigs = tf.split(out_sigma, num_or_size_splits=component_splits, axis=1)
            coll = [MultivariateNormalDiag(loc=loc, scale_diag=scale) for loc, scale
                    in zip(mus, sigs)]
            mixture = Mixture(cat=cat, components=coll)
            samp = mixture.sample()
            mse = tf.reduce_mean(tf.square(samp - y_true), axis=-1)
            # Todo: temperature adjustment for sampling functon.
            return mse

        # Actually return the loss_func
        with tf.name_scope('MDNLayer'):
            return mse_func

    def split_mixture_params(self, params, output_dim, num_mixes):
        """Splits up an array of mixture parameters into mus, sigmas, and pis
        depending on the number of mixtures and output dimension."""
        mus = params[:num_mixes * output_dim]
        sigs = params[num_mixes * output_dim:2 * num_mixes * output_dim]
        pi_logits = params[-num_mixes:]
        return mus, sigs, pi_logits

    def softmax(self, w, t=1.0):
        """Softmax function for a list or numpy array of logits. Also adjusts temperature."""
        e = np.array(w) / t  # adjust temperature
        e -= e.max()  # subtract max to protect from exploding exp values.
        e = np.exp(e)
        dist = e / np.sum(e)
        return dist

    def sample_from_categorical(self, dist):
        """Samples from a categorical model PDF."""
        r = np.random.rand(1)  # uniform random number in [0,1]
        accumulate = 0
        for i in range(0, dist.size):
            accumulate += dist[i]
            if accumulate >= r:
                return i
        tf.logging.info('Error sampling mixture model.')
        return -1

    def sample_from_output(self, params, output_dim, num_mixes, temp=1.0):
        """Sample from an MDN output with temperature adjustment."""
        mus = params[:num_mixes * output_dim]
        sigs = params[num_mixes * output_dim:2 * num_mixes * output_dim]
        pis = self.softmax(params[-num_mixes:], t=temp)
        m = self.sample_from_categorical(pis)
        # Alternative way to sample from categorical:
        # m = np.random.choice(range(len(pis)), p=pis)
        mus_vector = mus[m * output_dim:(m + 1) * output_dim]
        sig_vector = sigs[m * output_dim:(m + 1) * output_dim] * temp  # adjust for temperature
        cov_matrix = np.identity(output_dim) * sig_vector
        sample = np.random.multivariate_normal(mus_vector, cov_matrix, 1)
        return sample

    def network(self):
        with tf.name_scope('MDN'):
            self.mdn_mus = Dense(self.num_mix * self.output_dim)  # mix*output vals, no activation
            self.mdn_sigmas = Dense(self.num_mix * self.output_dim,
                                    activation=self.elu_plus_one_plus_epsilon)  # mix*output vals exp activation
            self.mdn_pi = Dense(self.num_mix)  # mix vals, logits
