import tensorflow as tf
import time
import os
import glob
import cv2
import numpy as np


from utils import (
    checkpoint_dir,
    read_data,
    imsave
)


def lrelu(x, leak=0.2, name="lrelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)

class AE(object):

    def __init__(self, sess, image_size, label_size, c_dim, batch_size, answer):
        self.sess=sess
        self.image_size = image_size
        self.label_size = label_size
        self.c_dim = c_dim
        self.initializer = tf.truncated_normal_initializer(stddev=0.02)
        # self.batch_size = batch_size
        if answer == True:
            self.batch_size = batch_size
        else:
            self.batch_size = 1
        self.build_model()


    def build_model(self):
        self.images = tf.placeholder(tf.float32, [None, 2, self.image_size, self.image_size, self.c_dim], name='images')
        self.labels = tf.placeholder(tf.float32, [None, self.label_size, self.label_size, self.c_dim], name='labels')
        self.shape = tf.shape(self.images)
        self.pred = self.model()
        self.loss = tf.reduce_mean(tf.square(self.labels - self.pred))

        self.saver = tf.train.Saver() # To save checkpoint

    def model(self, reuse = False):
        with tf.variable_scope("autoencoder"):
            self.img = tf.reshape(self.images, shape=[self.batch_size, self.image_size, self.image_size, self.c_dim * 2])
            conv1 = tf.contrib.layers.conv2d(inputs=self.img, num_outputs=16, kernel_size=4, stride=2, padding="SAME", \
                                             reuse=reuse, activation_fn=lrelu, weights_initializer=self.initializer,
                                             scope="d_conv1")  # 128 x 128 x 16
            conv2 = tf.contrib.layers.conv2d(inputs=conv1, num_outputs=32, kernel_size=4, stride=2, padding="SAME", \
                                             reuse=reuse, activation_fn=lrelu, normalizer_fn=tf.contrib.layers.batch_norm, \
                                             weights_initializer=self.initializer, scope="d_conv2")  # 64 x 64 x 32
            conv3 = tf.contrib.layers.conv2d(inputs=conv2, num_outputs=64, kernel_size=4, stride=2, padding="SAME", \
                                             reuse=reuse, activation_fn=lrelu, normalizer_fn=tf.contrib.layers.batch_norm, \
                                             weights_initializer=self.initializer, scope="d_conv3")  # 32 x 32 x 64
            conv4 = tf.contrib.layers.conv2d(inputs=conv3, num_outputs=128, kernel_size=4, stride=2, padding="SAME", \
                                             reuse=reuse, activation_fn=lrelu, normalizer_fn=tf.contrib.layers.batch_norm, \
                                             weights_initializer=self.initializer, scope="d_conv4")  # 16 x 16 x 128
            conv5 = tf.contrib.layers.conv2d(inputs=conv4, num_outputs=256, kernel_size=4, stride=2, padding="SAME", \
                                             reuse=reuse, activation_fn=lrelu,
                                             normalizer_fn=tf.contrib.layers.batch_norm, \
                                             weights_initializer=self.initializer, scope="d_conv5") # 8 x 8 x 256
            conv6 = tf.contrib.layers.conv2d(inputs=conv5, num_outputs=512, kernel_size=4, stride=2, padding="SAME", \
                                             reuse=reuse, activation_fn=lrelu,
                                             normalizer_fn=tf.contrib.layers.batch_norm, \
                                             weights_initializer=self.initializer, scope="d_conv6") # 4 x 4 x 512

            conv_trans1 = tf.contrib.layers.conv2d(conv6, num_outputs=4 * 256, kernel_size=4, stride=1, padding="SAME", \
                                                   activation_fn=tf.nn.relu, normalizer_fn=tf.contrib.layers.batch_norm, \
                                                   weights_initializer=self.initializer, scope="g_conv1")
            conv_trans1 = tf.reshape(conv_trans1, shape=[self.batch_size, 8, 8, 256])
            conv_trans2 = tf.contrib.layers.conv2d(conv_trans1, num_outputs=4 * 128, kernel_size=4, stride=1, padding="SAME", \
                                                   activation_fn=tf.nn.relu, normalizer_fn=tf.contrib.layers.batch_norm, \
                                                   weights_initializer=self.initializer, scope="g_conv2")
            conv_trans2 = tf.reshape(conv_trans2, shape=[self.batch_size, 16, 16, 128])
            conv_trans3 = tf.contrib.layers.conv2d(conv_trans2, num_outputs=4 * 64, kernel_size=4, stride=1, padding="SAME", \
                                                   activation_fn=tf.nn.relu, normalizer_fn=tf.contrib.layers.batch_norm, \
                                                   weights_initializer=self.initializer, scope="g_conv3")
            conv_trans3 = tf.reshape(conv_trans3, shape=[self.batch_size, 32, 32, 64])

            conv_trans4 = tf.contrib.layers.conv2d(conv_trans3, num_outputs=4 * 32, kernel_size=4, stride=1, padding="SAME", \
                                                   activation_fn=tf.nn.relu, normalizer_fn=tf.contrib.layers.batch_norm, \
                                                   weights_initializer=self.initializer, scope="g_conv4")
            conv_trans4 = tf.reshape(conv_trans4, shape=[self.batch_size, 64, 64, 32])

            conv_trans5 = tf.contrib.layers.conv2d(conv_trans4, num_outputs=4 * 16, kernel_size=4, stride=1, padding="SAME", \
                                                   activation_fn=tf.nn.relu, normalizer_fn=tf.contrib.layers.batch_norm, \
                                                   weights_initializer=self.initializer, scope="g_conv5")
            conv_trans5 = tf.reshape(conv_trans5, shape=[self.batch_size, 128, 128, 16])

            conv_trans6 = tf.contrib.layers.conv2d(conv_trans5, num_outputs=4 * 8, kernel_size=4, stride=1, padding="SAME", \
                                                   activation_fn=tf.nn.relu, normalizer_fn=tf.contrib.layers.batch_norm, \
                                                   weights_initializer=self.initializer, scope="g_conv6")
            conv_trans6 = tf.reshape(conv_trans6, shape=[self.batch_size, 256, 256, 8])

            recon_bag = tf.contrib.layers.conv2d(conv_trans6, num_outputs=3, kernel_size=4, stride=1, padding="SAME", \
                                                 activation_fn=tf.nn.relu, scope="g_conv7")
            return recon_bag


    def train(self, config):
        # nx, ny = input_setup(config)
        #
        # data_dir = checkpoint_dir(config)

        # input_, label_ = read_data(data_dir)

        if config.is_train:
            input_bag = sorted(glob.glob("./Train/*.png"))
            input_ = [[cv2.imread(input_bag[i])/ 255.0, cv2.imread(input_bag[i+1])/ 255.0] for i in range(0, len(input_bag) - 1)]
            label_ = [cv2.imread(path)/ 255.0 for path in input_bag[1:-1]]
        else:
            input_bag = sorted(glob.glob("./Test/*.png"))
            input_ = [[cv2.imread(input_bag[0])/ 255.0, cv2.imread(input_bag[2])/ 255.0]]
            label_ = [cv2.imread(input_bag[1]) / 255.0]
            print(np.array(input_).shape)
            print(np.array(label_).shape)

        self.train_op = tf.train.AdamOptimizer(learning_rate=config.learning_rate).minimize(self.loss)
        self.learning_rate = config.learning_rate
        tf.initialize_all_variables().run()

        counter = 0

        time_ = time.time()

        self.load(config.checkpoint_dir)

        if config.is_train:
            print("Now Start Training...")
            for ep in range(config.epoch):
                batch_idxs = len(input_) // config.batch_size
                for idx in range(0, batch_idxs):
                    batch_images = input_[idx * config.batch_size : (idx + 1) * config.batch_size]
                    batch_labels = label_[idx * config.batch_size : (idx + 1) * config.batch_size]
                    counter += 1
                    _, err = self.sess.run([self.train_op, self.loss], feed_dict = {self.images : batch_images, self.labels: batch_labels})

                    if counter % 10 == 0:
                        print("Epoch: [%2d], step: [%2d], time: [%4.4f], loss:[%.8f]"% ((ep+1), counter, time.time()-time_, err))
                    if counter % 200 == 0:
                        self.save(config.checkpoint_dir, counter)
        else:
            print("Now Start Testing...")

            result = self.pred.eval({self.images: input_})[0]
            imsave(result, config.result_dir+'/result.png', config)


    def load(self, checkpoint_dir):

        print("\n Reading Checkpoints... \n")
        model_dir = "%s_learning_rate_%f"%("autoencoder", self.learning_rate)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        # print("\n\n\n")
        # print(ckpt)
        # print(ckpt.model_checkpoint_path)
        # print("\n\n\n")
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_path = str(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(os.getcwd(), ckpt_path))
            print("\n Checkpoint Loading Success! %s\n\n"% ckpt_path)
        else:
            print("\n! Checkpoint Loading Failed \n\n")

    def save(self, checkpoint_dir, step):
        model_name = "Autoencoder.model"
        model_dir = "%s_learning_rate_%f"%("autoencoder", self.learning_rate)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)
