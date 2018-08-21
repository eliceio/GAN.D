from model import TestModel
import tensorflow as tf
import numpy as np

print("Start")

model = TestModel(4)
train_loss = model.loss()

test_input = tf.zeros_like(np.zeros([4, 256, 256, 1]))

print("Done")