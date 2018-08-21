from model import RNN_Model
import tensorflow as tf
import numpy as np

print("Start")

model = RNN_Model()
train_loss = model.loss()

test_input = tf.zeros_like(np.zeros([4, 256, 256, 1]))

print("Done")