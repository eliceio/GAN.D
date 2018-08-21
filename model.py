import tensorflow as tf
import numpy as np
from module import LSTM, Dropout, Dense
from tensorflow.contrib.distributions import Categorical, Mixture, MultivariateNormalDiag


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
        self.G_out, self.D_real, self.D_fake = self.network()

    def _Encoder(self, inputs, reuse=False):
        with tf.variable_scope('Encoder', reuse=reuse):  # inputs <- (batch, 256, 256, c)
            enc = tf.layers.conv2d(inputs, 128, kernel_size=3, activation=tf.nn.relu,
                                   padding='same')  # (batch, 25, 256, 128)
            enc = tf.layers.max_pooling2d(enc, pool_size=2, strides=2)  # (batch, 128, 128, 128)
            enc = tf.layers.conv2d(enc, 64, kernel_size=3, activation=tf.nn.relu,
                                   padding='same')  # (batch, 128, 128, 64)
            enc = tf.layers.max_pooling2d(enc, pool_size=2, strides=2)  # (batch, 64, 64, 64)
            enc = tf.layers.conv2d(enc, 32, kernel_size=3, activation=tf.nn.relu, padding='same')  # (batch, 64, 64, 32)
            enc = tf.layers.max_pooling2d(enc, pool_size=2, strides=2)  # (batch, 32, 32, 32)
            enc = tf.layers.flatten(enc)
            enc = tf.layers.dense(enc, 128)  # (batch, 128)

            return enc

    def _Decoder(self, inputs, reuse=False):
        with tf.variable_scope('Decoder', reuse=reuse):  # inputs <- (batch, 128)
            dec = tf.layers.dense(inputs, 32 * 32 * 32, activation=tf.nn.relu)  # (batch, 32768)
            dec = tf.reshape(dec, [-1, 32, 32, 32])  # (batch, 32, 32, 32)
            dec = tf.layers.conv2d_transpose(dec, 32, 3, strides=2, activation=tf.nn.relu,
                                             padding='same')  # (batch, 64, 64, 32)
            dec = tf.layers.conv2d_transpose(dec, 64, 3, strides=2, activation=tf.nn.relu,
                                             padding='same')  # (batch, 128, 128, 64)
            dec = tf.layers.conv2d_transpose(dec, 128, 3, strides=2, activation=tf.nn.relu,
                                             padding='same')  # (batch, 256, 256, 128)
            dec = tf.layers.conv2d(dec, filters=1, kernel_size=3, activation=tf.nn.sigmoid,
                                   padding='same')  # (batch, 256, 256, 1)

            return dec

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

    # todo : noise를 어떻게 넣어야 할까?, pred rnn 할때? 그떄 동작을 만드니까?
    def _Generator(self):
        with tf.variable_scope('Generator'):
            # encoding input image
            start_frame_vector = self._Encoder(self.start_frames, reuse=False)
            end_frame_vector = self._Encoder(self.end_frames, reuse=True)

            # pred_seq_vec
            pred_seq_vec = self._RNN_Predict(start_frame_vector, end_frame_vector)

            # decoding seq_vec to image
            pred_shots = []
            for i in range(self.pred_size + 2):
                pred_shots.append(self._Decoder(pred_seq_vec[:, i:i + 1, :], reuse=None if i == 0 else True))

            return pred_shots  # (pred_size + 2, ) return list

    def _Discriminator(self, inputs, reuse=False):
        with tf.variable_scope('Discriminator', reuse=reuse):
            def encoder_dis(input, reuse=False):  # todo : pix2pix의 Discriminator 처럼하면 어떨까?
                with tf.variable_scope('Encoder_Dis', reuse=reuse):
                    enc = tf.layers.conv2d(input, 128, kernel_size=3, activation=tf.nn.relu,
                                           padding='same')  # (batch, 25, 256, 128)
                    enc = tf.layers.max_pooling2d(enc, pool_size=2, strides=2)  # (batch, 128, 128, 128)
                    enc = tf.layers.conv2d(enc, 64, kernel_size=3, activation=tf.nn.relu,
                                           padding='same')  # (batch, 128, 128, 64)
                    enc = tf.layers.max_pooling2d(enc, pool_size=2, strides=2)  # (batch, 64, 64, 64)
                    enc = tf.layers.conv2d(enc, 32, kernel_size=3, activation=tf.nn.relu,
                                           padding='same')  # (batch, 64, 64, 32)
                    enc = tf.layers.max_pooling2d(enc, pool_size=2, strides=2)  # (batch, 32, 32, 32)
                    enc = tf.layers.flatten(enc)
                    enc = tf.layers.dense(enc, 128)  # (batch, 128)

                    enc = tf.expand_dims(enc, 1)  # expand dim
                return enc

            encoded_inputs = []
            for i in range(self.pred_size + 2):
                encoded_inputs.append(encoder_dis(inputs[i], reuse=None if i == 0 else True))

            encoded_inputs = tf.concat(encoded_inputs, axis=1)

            # Bi-LSTM
            with tf.variable_scope('Bi_LSTM'):
                lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(256, forget_bias=1.0)
                lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(256, forget_bias=1.0)
                (output_fw, output_bw), states = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell,
                                                                                 encoded_inputs,
                                                                                 dtype=tf.float32)  # (batch, pred_size + 2, 256) x2
                outputs = tf.concat([output_fw, output_bw], axis=2)  # (batch, pred_size + 2, 512)
                outputs = tf.layers.dense(outputs, 128, activation=tf.nn.tanh)  # (batch, pred_size + 2, 128)

            # GRU recurrent
            with tf.variable_scope('GRU'):
                cell_fw = tf.contrib.rnn.GRUCell(64)
                cell_bw = tf.contrib.rnn.GRUCell(64)
                gru_ouputs, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, outputs, dtype=tf.float32)
                gru_ouputs = tf.concat(gru_ouputs, 2)

            ## Disciriminator output
            out = gru_ouputs[:, -1, :]
            W_dis = tf.get_variable("weights", shape=[128, 1])
            b_dis = tf.get_variable("bias", shape=[1])
            out = tf.sigmoid(tf.matmul(out, W_dis) + b_dis)

            return out

    # todo : Generator의 출력을 target 처럼 나오도록 수정하고 Discriminator도 맞도록 수정
    def _network(self):
        G_out = self._Generator()
        D_real = None # 나중에 추가 할 것
        D_fake = self._Discriminator(G_out)

        return G_out, D_real, D_fake

    def discriminator_train_op(self):
        pass

    def generator_train_op(self):
        pass

    def __call__(self, *args, **kwargs):
        return None
