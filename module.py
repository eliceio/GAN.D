import tensorflow as tf


def LSTM(inputs, num_unit, seqlens=None, bidirection=False, dropout=False, keep_prob=0.4, scope=None):
    cell = tf.contrib.rnn.BasicLSTMCell(num_unit, forget_bias=1.0)

    if dropout:
        cell = Dropout(cell, keep_prob)

    if (bidirection):
        outputs, states = tf.nn.bidirectional_dynamic_rnn(cell, inputs,
                                                          sequence_length=seqlens,
                                                          dtype=tf.float32,
                                                          scope=scope)
    else:
        outputs, states = tf.nn.dynamic_rnn(cell, inputs,
                                            sequence_length=seqlens,
                                            dtype=tf.float32,
                                            scope=scope)

    return outputs


def Dropout(cell, keep_prob):
    return tf.contrib.rnn.DropoutWrapper(
        cell, output_keep_prob=keep_prob)


def Dense(inputs, num_unit, activation=tf.nn.relu):
    return tf.layers.dense(inputs, units=num_unit, activation=activation)
