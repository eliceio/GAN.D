from model import AE_Model
import tensorflow as tf
import cv2
import numpy as np
# hparameter
latent_dim = 128
batch_size = 1
logdir = './logdir/{}'.format("AE_3_fine/")

lv = np.load("lv.npy")
tf.reset_default_graph()
model = AE_Model(latent_dim, batch_size)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
video = cv2.VideoWriter("output.avi", fourcc, 30.0, (320, 180))

session_conf = tf.ConfigProto(
    gpu_options=tf.GPUOptions(
        allow_growth=True,
    ),
)

# Training
with tf.Session(config=session_conf) as sess:
    sess.run(tf.global_variables_initializer())
    model.load(sess, logdir=logdir)
    for i in range(1000):
        data = lv[i].reshape(1,128)
        img = sess.run(model(), feed_dict={model.latent_inputs: data})
        img = np.array(img).reshape(180,320,1)
        img = img * 255
        img = np.array(img).astype("uint8")
        img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
        video.write(img)
    video.release()