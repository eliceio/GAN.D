import tensorflow as tf
import numpy as np
from module import LSTM, Dense
from tensorflow.contrib.distributions import Categorical, Mixture, MultivariateNormalDiag


class AE_Model:
    def __init__(self, latent_dim, batch_size):
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.input_image = tf.placeholder(tf.float32, shape=(self.batch_size, 256, 256, 1))

        # make network
        self.network = tf.make_template('net', self._network)
        self.y_pred, self.encoder_pred = self.network(self.input_image)

    def sampling(self, z_mean, z_log_var):
        z_mean = z_mean
        z_log_var = z_log_var
        batch = self.batch_size
        dim = self.latent_dim
        epsilon = tf.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

    def _network(self, x):
        with tf.variable_scope('Encoder'):
            enc = tf.layers.conv2d(x, 128, kernel_size=3, activation=tf.nn.relu,
                                   padding='same')  # (batch, 256, 256, 128)
            enc = tf.layers.max_pooling2d(enc, pool_size=2, strides=2)  # (batch, 128, 128, 128)
            enc = tf.layers.conv2d(enc, 64, kernel_size=3, activation=tf.nn.relu,
                                   padding='same')  # (batch, 128, 128, 64)
            enc = tf.layers.max_pooling2d(enc, pool_size=2, strides=2)  # (batch, 64, 64, 64)
            enc = tf.layers.conv2d(enc, 32, kernel_size=3, activation=tf.nn.relu, padding='same')  # (batch, 64, 64, 32)
            enc = tf.layers.max_pooling2d(enc, pool_size=2, strides=2)  # (batch, 32, 32, 32)

            shape = enc.get_shape()
            print(shape)
            enc = tf.layers.flatten(enc)
            enc = tf.layers.dense(enc, 128)

            self.z_mean = tf.layers.dense(enc, self.latent_dim)
            self.z_log_var = tf.layers.dense(enc, self.latent_dim)

            z = self.sampling(self.z_mean, self.z_log_var)

        with tf.variable_scope('Decoder'):
            self.latent_inputs = tf.placeholder_with_default(z, shape=(None, self.latent_dim))
            dec = tf.layers.dense(self.latent_inputs, shape[1] * shape[2] * shape[3], activation=tf.nn.relu)
            dec = tf.reshape(dec, [shape[0], shape[1], shape[2], shape[3]])
            dec = tf.layers.dense(dec, 128, activation=tf.nn.relu)  # (batch, 32,32,128)
            dec = tf.layers.conv2d_transpose(dec, 32, 3, strides=2, activation=tf.nn.relu,
                                             padding='same')  # (batch, 64, 64, 32)
            dec = tf.layers.conv2d_transpose(dec, 64, 3, strides=2, activation=tf.nn.relu,
                                             padding='same')  # (batch, 128, 128, 64)
            dec = tf.layers.conv2d_transpose(dec, 128, 3, strides=2, activation=tf.nn.relu,
                                             padding='same')  # (batch, 256, 256, 128)
            dec = tf.layers.conv2d(dec, filters=1, kernel_size=3, activation=tf.nn.sigmoid,
                                   padding='same')  # (batch, 256, 256, 1)
        return dec, z

    def loss(self):
        reconstruction_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.layers.flatten(self.input_image),
                                                                      logits=tf.layers.flatten(self.y_pred))
        reconstruction_loss *= 256 * 256
        kl_loss = 1 + self.z_log_var - tf.square(self.z_mean) - tf.exp(self.z_log_var)
        kl_loss = tf.reduce_sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        vae_loss = tf.reduce_mean(reconstruction_loss) + tf.reduce_mean(kl_loss)

        return vae_loss

    def __call__(self, *args, **kwargs):
        return self.y_pred

    @staticmethod
    def load(sess, logdir):
        ckpt = tf.train.latest_checkpoint(logdir)
        if ckpt:
            t_vars = tf.trainable_variables()
            restore_vars = [var for var in t_vars if 'rnn_model' not in var.name]
            tf.train.Saver(restore_vars).restore(sess, ckpt)


class RNN_Model:
    def __init__(self, batch_size, numComponents, outputDim):
        # model parameter
        self.batch = batch_size
        self.numComponents = numComponents
        self.outputDim = outputDim

        # model input
        self.input = tf.placeholder(tf.float32, shape=(self.batch, 128))
        self.y_true = tf.placeholder(tf.float32, shape=(self.batch, 128))

        # make network
        self.network = tf.make_template('rnn_model', self._network)
        self.y_pred = self.network()

    def _network(self):
        input = tf.expand_dims(self.input, 1)
        X = LSTM(input, 512, dropout=True, keep_prob=0.4, scope="LSTM_1")
        X = LSTM(X, 512, dropout=True, keep_prob=0.4, scope="LSTM_2")
        X = LSTM(X, 512, dropout=True, keep_prob=0.4, scope="LSTM_3")
        X = tf.reshape(X, [self.batch, -1])
        X = Dense(X, num_unit=1000, activation=tf.nn.relu)
        outputs = MDN(X, self.outputDim, self.numComponents).logit

        return outputs

    def loss(self):
        out_mu, out_sigma, out_pi = tf.split(self.y_pred, num_or_size_splits=[self.numComponents * self.outputDim,
                                                                              self.numComponents * self.outputDim,
                                                                              self.numComponents],
                                             axis=1, name='mdn_coef_split')
        cat = Categorical(logits=out_pi)
        component_splits = [self.outputDim] * self.numComponents
        mus = tf.split(out_mu, num_or_size_splits=component_splits, axis=1)
        sigs = tf.split(out_sigma, num_or_size_splits=component_splits, axis=1)
        coll = [MultivariateNormalDiag(loc=loc, scale_diag=scale) for loc, scale
                in zip(mus, sigs)]
        mixture = Mixture(cat=cat, components=coll)
        loss = mixture.log_prob(self.y_true)
        loss = tf.negative(loss)
        loss = tf.reduce_mean(loss)
        return loss

    @staticmethod
    def load(sess, logdir):
        ckpt = tf.train.latest_checkpoint(logdir)
        if ckpt:
            t_vars = tf.trainable_variables()
            restore_vars = [var for var in t_vars if 'rnn_model/' in var.name]
            tf.train.Saver(restore_vars).restore(sess, ckpt)

    def sample_from_output(self, params, output_dim, num_mixes, temp=1.0):

        # inner methods
        def softmax(w, t=1.0):
            e = np.array(w) / t  # adjust temperature
            e -= e.max()  # subtract max to protect from exploding exp values.
            e = np.exp(e)
            dist = e / np.sum(e)
            return dist

        def sample_from_categorical(dist):
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

    def __call__(self, *args, **kwargs):
        return self.y_pred


class MDN:
    def __init__(self, X, output_dimension, num_mixtures):
        self.output_dim = output_dimension
        self.num_mix = num_mixtures

        # make network
        self.network = tf.make_template('net', self._network)
        self.logit = self.network(X)

    def elu_plus_one_plus_epsilon(self, x):
        return (tf.nn.elu(x) + 1 + 1e-8)

    def _network(self, X):
        with tf.name_scope('MDN'):
            def linear(x):
                return x

            self.mdn_mus = Dense(X, self.num_mix * self.output_dim, activation=linear)  # mix*output vals, no activation
            self.mdn_sigmas = Dense(X, self.num_mix * self.output_dim,
                                    activation=self.elu_plus_one_plus_epsilon)  # mix*output vals exp activation
            self.mdn_pi = Dense(X, self.num_mix, activation=linear)  # mix vals, logits

            mdn_out = tf.concat([self.mdn_mus,
                                 self.mdn_sigmas,
                                 self.mdn_pi],
                                axis=-1,
                                name='mdn_outputs')

            return mdn_out
