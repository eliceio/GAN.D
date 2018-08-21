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
    def __init__(self):
        pass

    def network(self):
        pass


# Todo 테스트용 모델
class TestModel:
    def __init__(self, batch_size):
        # model parameter
        self.batch_size = batch_size
        self.pred_size = 3

        # model input
        self.start_frames = tf.placeholder(tf.float32, shape=(batch_size, 256, 256, 1))
        self.end_frames = tf.placeholder(tf.float32, shape=(batch_size, 256, 256, 1))
        self.target_frames = tf.placeholder(tf.float32, shape=(batch_size, self.pred_size, 256, 256, 1))

        # make network
        self.network = tf.make_template('net', self._network)
        self.logit = self.network()

    def _Encoder(self, inputs, reuse=False):
        with tf.variable_scope('Encoder', reuse=reuse):  # inputs <- (batch, 256, 256, c)
            enc = tf.nn.relu(tf.layers.conv2d(inputs, 128, kernel_size=3, padding='same'))  # (batch, 25, 256, 128)
            enc = tf.layers.max_pooling2d(enc, pool_size=2, strides=2)  # (batch, 128, 128, 128)
            enc = tf.nn.relu(tf.layers.conv2d(enc, 64, kernel_size=3, padding='same'))  # (batch, 128, 128, 64)
            enc = tf.layers.max_pooling2d(enc, pool_size=2, strides=2)  # (batch, 64, 64, 64)
            enc = tf.nn.relu(tf.layers.conv2d(enc, 32, kernel_size=3, padding='same'))  # (batch, 64, 64, 32)
            enc = tf.layers.max_pooling2d(enc, pool_size=2, strides=2)  # (batch, 32, 32, 32)
            enc = tf.layers.flatten(enc)
            enc = tf.layers.dense(enc, 128)  # (batch, 128)

            return enc

    def _Decoder(self, inputs, reuse=False):
        pass

    def _RNN_Predict(self, start_frame_vec, end_frame_vec):
        with tf.variable_scope('RNN_Predict'):
            # make RNN init input - start_frame + ..(mean_vector).. + end_frame
            expand_start = tf.expand_dims(start_frame_vec, 1)
            expand_end = tf.expand_dims(end_frame_vec, 1)
            seq_vector = []
            seq_vector.append(expand_start)
            for i in range(self.pred_size):
                seq_vector.append(expand_start + expand_end / 2)
            seq_vector.append(expand_end)

            # RNN input
            seq_input = tf.concat(seq_vector, axis=1)  # (batch, pred_size + 2, 128)

            # Bi-LSTM
            with tf.variable_scope('Bi_LSTM_1'):
                lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(256, forget_bias=1.0)
                lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(256, forget_bias=1.0)
                (output_fw, output_bw), states = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, seq_input,
                                                                                 dtype=tf.float32)  # (batch, pred_size + 2, 256) x2
                outputs = tf.concat([output_fw, output_bw], axis=2)  # (batch, pred_size + 2, 512)
                outputs = tf.layers.dense(outputs, 128, activation=tf.nn.tanh)  # (batch, pred_size + 2, 128)

            # Drop predicted start, end frame
            pred_ouput = outputs[:, 1:-1, :]  # (batch, pred_size, 128)

            # append residual connect, start, end frame
            residual_input = tf.concat([expand_start, pred_ouput], axis=1)
            residual_input = tf.concat([residual_input, expand_end], axis=1)  # (batch, pred_size +2, 128)

            # Bi-LSTM
            with tf.variable_scope('Bi_LSTM_2'):
                lstm_fw_cell_2 = tf.contrib.rnn.BasicLSTMCell(256, forget_bias=1.0)
                lstm_bw_cell_2 = tf.contrib.rnn.BasicLSTMCell(256, forget_bias=1.0)
                (output_fw_2, output_bw_2), states_2 = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell_2, lstm_bw_cell_2,
                                                                                       residual_input,
                                                                                       dtype=tf.float32)  # (batch, pred_size + 2, 256) x2
                outputs_2 = tf.concat([output_fw_2, output_bw_2], axis=2)  # (batch, pred_size + 2, 512)
                outputs_2 = tf.layers.dense(outputs_2, 128, activation=tf.nn.tanh)  # (batch, pred_size + 2, 128)

            # Drop predicted start, end frame
            pred_ouput_2 = outputs_2[:, 1:-1, :]  # (batch, pred_size, 128)

            # append original start, end frame
            pred_seq_vec = tf.concat([expand_start, pred_ouput_2], axis=1)
            pred_seq_vec = tf.concat([pred_seq_vec, expand_end], axis=1)  # (batch, pred_size +2, 128)

            return pred_seq_vec

    def _Generator(self):
        # encoding input image
        start_frame_vector = self._Encoder(self.start_frames, reuse=False)
        end_frame_vector = self._Encoder(self.end_frames, reuse=True)

        # pred_seq_vec
        pred_seq_vec = self._RNN_Predict(start_frame_vector, end_frame_vector)

        # decoding seq_vec to image
        pred_shots = [self._Decoder(input_tensor, reuse=None if i == 0 else True) for i, input_tensor in
                      enumerate(pred_seq_vec)]

    def _Discriminator(self):
        pass

    def _network(self):
        G = self._Generator()
        D_real = self._Discriminator()
        D_fake = self._Discriminator()

    def __call__(self, *args, **kwargs):
        pass

    def loss(self):
        pass
