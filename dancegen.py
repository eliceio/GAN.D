from model import AE_Model, RNN_Model
from ops import get_lv_dataset_iterator
import numpy as np
import tensorflow as tf
import cv2

# hparameter
ENCODED_DATA_PATH = './lv.npy'
decoder_logdir = './logdir/{}'.format("AE_1/")
RNN_logdir = './logdir/{}'.format("RNN_2/")

train = True  # train false -> generate video

numComponents = 24
latent_dim = 128
batch_size = 1024
num_step = 1000000
save_per_step = 100

batch = batch_size if train else 1

# model load
decoder = AE_Model(latent_dim, batch)
rnn_predict = RNN_Model(batch, numComponents, latent_dim)
saver = tf.train.Saver(tf.trainable_variables())

# loss
rnn_loss = rnn_predict.loss()

# optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=0.0005)
train_op = optimizer.minimize(rnn_loss)

# dataset, scaler
iterator, scaler = get_lv_dataset_iterator(ENCODED_DATA_PATH, batch)
dataset_stacked = iterator.get_next()

# Summary
tf.summary.scalar('net/loss', rnn_loss)
summ_op = tf.summary.merge_all()

session_conf = tf.ConfigProto(
    gpu_options=tf.GPUOptions(
        allow_growth=True,
    ),
)
if train:
    with tf.Session(config=session_conf) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(iterator.initializer)
        decoder.load(sess, logdir=decoder_logdir)
        rnn_predict.load(sess, RNN_logdir)

        writer = tf.summary.FileWriter(RNN_logdir, sess.graph)
        for i in range(num_step):
            X, Y = sess.run(dataset_stacked)
            _, loss, summ = sess.run([train_op, rnn_loss, summ_op],
                                     feed_dict={rnn_predict.input: X, rnn_predict.y_true: Y})
            print("step : " + str(i) + ", loss : " + str(loss))
            writer.add_summary(summ, global_step=i)

            if i % save_per_step == 0:
                saver.save(sess, '{}/checkpoint_step_{}'.format(RNN_logdir, i))

        writer.close()
else:
    with tf.Session(config=session_conf) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(iterator.initializer)
        decoder.load(sess, logdir=decoder_logdir)
        rnn_predict.load(sess, RNN_logdir)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter("out.mp4", fourcc, 30.0, (208, 120))
        X, Y = sess.run(dataset_stacked)
        lv_in = X

        for i in range(500):
            input = np.array(lv_in)
            lv_out = sess.run(rnn_predict(),
                              feed_dict={rnn_predict.input: input})
            shape = np.array(lv_out).shape[1]
            lv_out = np.array(lv_out).reshape(shape)
            lv_out = rnn_predict.sample_from_output(lv_out, 128, numComponents, temp=0.01)
            lv_out = scaler.inverse_transform(lv_out)
            img = sess.run(decoder(), feed_dict={decoder.latent_inputs: np.array(lv_out).reshape(1, 128)})
            img = np.array(img).reshape(120, 208, 1)
            img = img * 255
            img = np.array(img).astype("uint8")
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            lv_in = lv_out
            video.write(img)
        video.release()
