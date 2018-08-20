import tensorflow as tf
import numpy as np
from module import LSTM, Dropout, Dense
from tensorflow.contrib.distributions import Categorical, Mixture, MultivariateNormalDiag


class Encoder:

    def __init__(self):
        pass

    def network(self):
        pass


class Decoder:
    def __init__(self):
        pass

    def network(self):
        pass


class RNN_Model:
    def __init__(self):
        self.input = tf.placeholder(tf.float32, shape=(None, 128))

    def network(self):
        X = LSTM(self.input, 512)
        X = Dropout(X, 0.40)
        X = LSTM(X, 512)
        X = Dropout(X, 0.40)
        X = Dense(X, num_unit=1000, activation=tf.nn.relu)
        outputs = X # todo mdn 추가 할 것

        self.pred = outputs

    def loss(self):
        return self.input - self.pred


class MDN:
    def __init__(self):
        pass

    def network(self):
        pass
