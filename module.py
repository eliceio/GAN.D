import tensorflow as tf


def LSTM(inputs, num_unit, seqlens=None):
    cell = tf.contrib.rnn.BasicLSTMCell(num_unit, forget_bias=1.0)
    outputs, states = tf.nn.dynamic_rnn(cell, inputs,
                                        sequence_length=seqlens,
                                        dtype=tf.float32)

    return outputs


def Dropout(inputs, keep_prob):
    dropout = tf.contrib.rnn.DropoutWrapper(
        inputs, output_keep_prob=keep_prob)

    return dropout

def Dense(inputs, num_unit, activation=tf.nn.relu):
    return tf.layers.dense(inputs, units=num_unit, activation=activation)