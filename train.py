from model import AE_Model
import tensorflow as tf

print("Start")

model = AE_Model(128, 4)
train_loss = model.loss()

optimizer = tf.train.RMSPropOptimizer(learning_rate=0.001)
train_op = optimizer.minimize(train_loss)
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    x = 0
    y = 0
    sess.run([train_op], feed_dict={model.input_image: x, model.y_pred: y})

print("Done")
