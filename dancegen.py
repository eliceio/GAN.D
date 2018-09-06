# from model import AE_Model, RNN_Model
# from ops import get_lv_dataset_iterator
# import numpy as np
# import tensorflow as tf
# import cv2
#
# # hparameter
# ENCODED_DATA_PATH = './lv.npy'
# decoder_logdir = './logdir/{}'.format("AE_3/")
# RNN_logdir = './logdir/{}'.format("RNN_1/")
#
# train = True  # train false -> generate video
#
# numComponents = 24
# latent_dim = 128
# batch_size = 1024
# num_step = 1000000
# save_per_step = 100
#
# batch = batch_size if train else 1
#
# # model load
# decoder = AE_Model(latent_dim, batch)
# rnn_predict = RNN_Model(batch, numComponents, latent_dim)
# saver = tf.train.Saver(tf.trainable_variables())
#
# # loss
# rnn_loss = rnn_predict.loss()
#
# # optimizer
# optimizer = tf.train.AdamOptimizer(learning_rate=0.0005)
# train_op = optimizer.minimize(rnn_loss)
#
# # dataset, scaler
# iterator, scaler = get_lv_dataset_iterator(ENCODED_DATA_PATH, batch)
# dataset_stacked = iterator.get_next()
#
# # Summary
# tf.summary.scalar('net/loss', rnn_loss)
# summ_op = tf.summary.merge_all()
#
# session_conf = tf.ConfigProto(
#     gpu_options=tf.GPUOptions(
#         allow_growth=True,
#     ),
# )
# if train:
#     with tf.Session(config=session_conf) as sess:
#         sess.run(tf.global_variables_initializer())
#         sess.run(iterator.initializer)
#         decoder.load(sess, logdir=decoder_logdir)
#         rnn_predict.load(sess, RNN_logdir)
#
#         writer = tf.summary.FileWriter(RNN_logdir, sess.graph)
#         for i in range(num_step):
#             X, Y = sess.run(dataset_stacked)
#             _, loss, summ = sess.run([train_op, rnn_loss, summ_op],
#                                      feed_dict={rnn_predict.input: X, rnn_predict.y_true: Y})
#             print("step : " + str(i) + ", loss : " + str(loss))
#             writer.add_summary(summ, global_step=i)
#
#             if i % save_per_step == 0:
#                 saver.save(sess, '{}/checkpoint_step_{}'.format(RNN_logdir, i))
#
#         writer.close()
# else:
#     with tf.Session(config=session_conf) as sess:
#         sess.run(tf.global_variables_initializer())
#         sess.run(iterator.initializer)
#         decoder.load(sess, logdir=decoder_logdir)
#         rnn_predict.load(sess, RNN_logdir)
#
#         fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#         video = cv2.VideoWriter("out.mp4", fourcc, 30.0, (208, 120))
#         X, Y = sess.run(dataset_stacked)
#         lv_in = X
#
#         for i in range(500):
#             input = np.array(lv_in)
#             lv_out = sess.run(rnn_predict(),
#                               feed_dict={rnn_predict.input: input})
#             shape = np.array(lv_out).shape[1]
#             lv_out = np.array(lv_out).reshape(shape)
#             lv_out = rnn_predict.sample_from_output(lv_out, 128, numComponents, temp=0.01)
#             lv_out = scaler.inverse_transform(lv_out)
#             img = sess.run(decoder(), feed_dict={decoder.latent_inputs: np.array(lv_out).reshape(1, 128)})
#             img = np.array(img).reshape(120, 208, 1)
#             img = img * 255
#             img = np.array(img).astype("uint8")
#             img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
#             lv_in = lv_out
#             video.write(img)
#         video.release()

from model import AE_Model, interNet
from ops import get_interNet_dataset, get_next
import numpy as np
import tensorflow as tf
import cv2

# hparameter
ENCODED_DATA_PATH = './lv.npy'
decoder_logdir = './logdir/{}'.format("AE_3_fine/")
RNN_logdir = './logdir/{}'.format("BI_RNN_4/")

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
rnn_predict = interNet(batch, pred_size, latent_dim, numComponents)
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

        # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        # video = cv2.VideoWriter("out.mp4", fourcc, 30.0, (208, 120))
        # start, end, target = get_next(dataset, batch, pred_size, 0)
        # start_image = cv2.imread("./frame_s.jpg")
        # start_image = cv2.cvtColor(start_image, cv2.COLOR_BGR2GRAY) / 255
        # start_image = start_image.reshape(1, 180, 320, 1)
        # end_image = cv2.imread("./frame_s.jpg")
        # end_image = cv2.cvtColor(end_image, cv2.COLOR_BGR2GRAY) / 255
        # end_image = end_image.reshape(1, 180, 320, 1)
        # start = sess.run(decoder.encoder_pred, feed_dict={decoder.input_image: start_image})
        # end = sess.run(decoder.encoder_pred, feed_dict={decoder.input_image: end_image})

        # for i in range(500):
        #     input = np.array(lv_in)
        #     lv_out = sess.run(rnn_predict(),
        #                       feed_dict={rnn_predict.input: input})
        #     shape = np.array(lv_out).shape[1]
        #     lv_out = np.array(lv_out).reshape(shape)
        #     lv_out = rnn_predict.sample_from_output(lv_out, 128, numComponents, temp=0.01)
        #     lv_out = scaler.inverse_transform(lv_out)
        #     img = sess.run(decoder(), feed_dict={decoder.latent_inputs: np.array(lv_out).reshape(1, 128)})
        #     img = np.array(img).reshape(120, 208, 1)
        #     img = img * 255
        #     img = np.array(img).astype("uint8")
        #     img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        #     lv_in = lv_out
        #     video.write(img)
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
            # shape = np.array(lv_out).shape[1]
            lv_out = np.squeeze(lv_out)
            lv_out = np.array(
                list(map(lambda x: rnn_predict.sample_from_output(x, 128, numComponents, temp=0.01), lv_out)))
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
                cv2.imwrite("./output/frame_{}_{}.jpg".format(i, j), img)
                video.write(img)
            input_start = lv_end
            video.write(input_end)
        video.release()
