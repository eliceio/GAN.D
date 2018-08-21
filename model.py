import tensorflow as tf
import numpy as np
from module import LSTM, Dense
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
        # input
        self.batch = 4
        self.input = tf.placeholder(tf.float32, shape=(self.batch, 128))
        self.y_true = tf.placeholder(tf.float32, shape=(self.batch, 128))

        # make network
        self.network = tf.make_template('net', self._network)
        self.y_pred = self._network()

    def _network(self):
        numComponents = 24
        outputDim = 128
        input = tf.expand_dims(self.input, 1)
        X = LSTM(input, 512, dropout=True, keep_prob=0.4, scope="LSTM_1")
        X = LSTM(X, 512, dropout=True, keep_prob=0.4, scope="LSTM_2")
        X = LSTM(X, 512, dropout=True, keep_prob=0.4, scope="LSTM_3")
        X = tf.reshape(X, [self.batch, -1])
        X = Dense(X, num_unit=1000, activation=tf.nn.relu)
        outputs = MDN(X, outputDim, numComponents).logit

        return outputs

    def loss(self):
        num_mixes = 24
        output_dim = 128
        out_mu, out_sigma, out_pi = tf.split(self.y_pred, num_or_size_splits=[num_mixes * output_dim,
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
        loss = mixture.log_prob(self.y_true)
        loss = tf.negative(loss)
        loss = tf.reduce_mean(loss)
        return loss

    def sample_from_output(self, params, output_dim, num_mixes, temp=1.0):
        """Sample from an MDN output with temperature adjustment."""

        # inner methods
        def softmax(w, t=1.0):
            """Softmax function for a list or numpy array of logits. Also adjusts temperature."""
            e = np.array(w) / t  # adjust temperature
            e -= e.max()  # subtract max to protect from exploding exp values.
            e = np.exp(e)
            dist = e / np.sum(e)
            return dist

        def sample_from_categorical(dist):
            """Samples from a categorical model PDF."""
            r = np.random.rand(1)  # uniform random number in [0,1]
            accumulate = 0
            for i in range(0, dist.size):
                accumulate += dist[i]
                if accumulate >= r:
                    return i
            tf.logging.info('Error sampling mixture model.')
            return -1

        # make output
        mus = params[:num_mixes * output_dim]
        sigs = params[num_mixes * output_dim:2 * num_mixes * output_dim]
        pis = softmax(params[-num_mixes:], t=temp)
        m = sample_from_categorical(pis)
        mus_vector = mus[m * output_dim:(m + 1) * output_dim]
        sig_vector = sigs[m * output_dim:(m + 1) * output_dim] * temp  # adjust for temperature
        cov_matrix = np.identity(output_dim) * sig_vector
        sample = np.random.multivariate_normal(mus_vector, cov_matrix, 1)
        return sample


class MDN:
    def __init__(self, X, output_dimension, num_mixtures):
        self.output_dim = output_dimension
        self.num_mix = num_mixtures

        # make network
        self.network = tf.make_template('net', self._network)
        self.logit = self.network(X)

    def elu_plus_one_plus_epsilon(self, x):
        """ELU activation with a very small addition to help prevent NaN in loss."""
        return (tf.nn.elu(x) + 1 + 1e-8)

    def _network(self, X):
        with tf.name_scope('MDN'):
            self.mdn_mus = Dense(X, self.num_mix * self.output_dim)  # mix*output vals, no activation
            self.mdn_sigmas = Dense(X, self.num_mix * self.output_dim,
                                    activation=self.elu_plus_one_plus_epsilon)  # mix*output vals exp activation
            self.mdn_pi = Dense(X, self.num_mix)  # mix vals, logits

            mdn_out = tf.concat([self.mdn_mus,
                                 self.mdn_sigmas,
                                 self.mdn_pi],
                                axis=1,
                                name='mdn_outputs')

            return mdn_out
