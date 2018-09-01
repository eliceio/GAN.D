import tensorflow as tf
from tensorflow.contrib.keras import backend as K
import os
import numpy as np
import cv2
from model import AE_Model
import ops

def main():
    image_list = ops.image_list
    lv_array = []
    latent_dim = 128
    batch_size = 1
    logdir = './logdir/{}'.format("AE_1/")
    model = AE_Model(latent_dim, batch_size)

    # make images flat
    imgs = []
    for i in range(len(image_list)):
        for img in image_list[i] :
            imgs.append(img)
    # imgs = np.array(imgs)
    with tf.Session() as sess:
        model.load(sess, logdir=logdir)
        for img in imgs:
            # img = cv2.imread('imgs/{}.jpg'.format(i), cv2.IMREAD_GRAYSCALE)
            # img = cv2.resize(img, (208, 120))
            data_np = np.array(img) / 255
            # at now, image's dimension (256x256x1)
            data_np = data_np.reshape(1, 256, 256, 1)
            lv = sess.run(model.encoder_pred, feed_dict={model.input_image: data_np})
            lv = np.resize(lv, (latent_dim,))
            lv_array.append(lv)

    print(np.array(lv_array).shape)
    np.save("./lv.npy", np.array(lv_array))


if __name__ == '__main__':
    ops.load_data()
    main()
