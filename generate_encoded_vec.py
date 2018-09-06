import tensorflow as tf
from tensorflow.contrib.keras import backend as K
import os
import numpy as np
import cv2
from model import AE_Model


def main():
    lv_array = []
    latent_dim = 128
    batch_size = 1
    logdir = './logdir/{}'.format("AE_3_fine/")
    limit = len(os.listdir('imgs'))
    model = AE_Model(latent_dim, batch_size)

    with tf.Session() as sess:
        model.load(sess, logdir=logdir)
        for i in range(1, limit):
            img = cv2.imread('imgs/frame{}.jpg'.format(i), cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (320, 180))
            data_np = np.array(img) / 255
            data_np = data_np.reshape(1, 180, 320, 1)
            lv = sess.run(model.encoder_pred, feed_dict={model.input_image: data_np})
            lv = np.resize(lv, (latent_dim,))
            lv_array.append(lv)

    print(np.array(lv_array).shape)
    np.save("./lv.npy", np.array(lv_array))


if __name__ == '__main__':
    main()