from model import AE_Model, interNet_Not_MDN
from ops import get_interNet_dataset, get_next
import numpy as np
import tensorflow as tf
import cv2

# hparameter
ENCODED_DATA_PATH = './lv.npy'
decoder_logdir = './logdir/{}'.format("AE_3_fine/")
RNN_logdir = './logdir/{}'.format("BI_RNN_NOT_MDN_1/")

train = False  # train false -> generate video

numComponents = 24
latent_dim = 128
pred_size = 15
batch_size = 512
num_step = 100000
save_per_step = 1000

batch = batch_size if train else 1

# model load
decoder = AE_Model(latent_dim, batch)
rnn_predict = interNet_Not_MDN(batch, pred_size, latent_dim, numComponents)
saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=10)

# loss
rnn_loss = rnn_predict.loss()

# optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=0.0005)
t_vars = tf.trainable_variables()
vars = [var for var in t_vars if 'interNet/' in var.name]
train_op = optimizer.minimize(rnn_loss, var_list=vars)

# dataset, scaler
dataset, scaler = get_interNet_dataset(ENCODED_DATA_PATH)

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
        decoder.load(sess, logdir=decoder_logdir)
        rnn_predict.load(sess, RNN_logdir)
        print("summary")
        writer = tf.summary.FileWriter(RNN_logdir, sess.graph)
        print("Training Start")
        for i in range(num_step):
            start, end, target = get_next(dataset, batch_size, pred_size, 0)
            _, loss, summ = sess.run([train_op, rnn_loss, summ_op],
                                     feed_dict={rnn_predict.start_frames: start, rnn_predict.end_frames: end,
                                                rnn_predict.target_frames: target})
            print("step : " + str(i) + ", loss : " + str(loss))
            writer.add_summary(summ, global_step=i)

            if i % save_per_step == 0:
                saver.save(sess, '{}/checkpoint_step_{}'.format(RNN_logdir, i))

        writer.close()
else:
    with tf.Session(config=session_conf) as sess:
        sess.run(tf.global_variables_initializer())
        decoder.load(sess, logdir=decoder_logdir)
        rnn_predict.load(sess, RNN_logdir)

        start, end, target = get_next(dataset, batch, pred_size, 0)
        lv_start = start
        input_start = np.array(lv_start)

        input_start = scaler.inverse_transform(input_start)
        input_start = sess.run(decoder(), feed_dict={decoder.latent_inputs: np.array(input_start).reshape(1, 128)})
        input_start = np.array(input_start).reshape(180, 320, 1)
        input_start = input_start * 255
        input_start = np.array(input_start).astype("uint8")
        # input_start = cv2.cvtColor(input_start, cv2.COLOR_GRAY2RGB)
        cv2.imwrite("./output/frame_start.jpg", input_start)
        input_start = lv_start

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter("out.mp4", fourcc, 30.0, (320, 180))

        for i in range(100):
            print("step : " + str(i))
            _, lv_end, target = get_next(dataset, batch, pred_size, 0)

            input_end = np.array(lv_end)
            lv_out = sess.run(rnn_predict(),
                              feed_dict={rnn_predict.start_frames: input_start, rnn_predict.end_frames: input_end})
            lv_out = np.squeeze(lv_out)
            lv_out = scaler.inverse_transform(lv_out)

            input_end = scaler.inverse_transform(input_end)
            input_end = sess.run(decoder(), feed_dict={decoder.latent_inputs: np.array(input_end).reshape(1, 128)})
            input_end = np.array(input_end).reshape(180, 320, 1)
            input_end = input_end * 255
            input_end = np.array(input_end).astype("uint8")
            input_end = cv2.cvtColor(input_end, cv2.COLOR_GRAY2RGB)
            cv2.imwrite("./output/frame_{}_15.jpg".format(i), input_end)

            for j in range(pred_size):
                img = sess.run(decoder(), feed_dict={decoder.latent_inputs: np.array(lv_out[j]).reshape(1, 128)})
                img = np.array(img).reshape(180, 320, 1)
                img = img * 255
                img = np.array(img).astype("uint8")
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                lv_in = lv_out
                # video.write(img)
                cv2.imwrite("./output/frame_{:02d}_{:02d}.jpg".format(i, j), img)
                video.write(img)
            input_start = lv_end
            video.write(input_end)
        video.release()
