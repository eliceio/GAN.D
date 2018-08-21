from model import AE_Model
import tensorflow as tf
import numpy as np

print("Start")

model = AE_Model(128, 4)
train_loss = model.loss()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())


print("Done")