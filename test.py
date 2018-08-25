from model import AE_Model
from ops import get_dataset_iterator
import tensorflow as tf
import cv2

# hparameter
latent_dim = 128
batch_size = 1
logdir = './logdir/{}'.format("AE_1/")


def main():
    tf.reset_default_graph()
    model = AE_Model(latent_dim, batch_size)
    saver = tf.train.Saver(tf.trainable_variables())

    # Dataset
    iterator = get_dataset_iterator('./imgs', batch_size)
    input_image_stacked = iterator.get_next()

    session_conf = tf.ConfigProto(
        gpu_options=tf.GPUOptions(
            allow_growth=True,
        ),
    )

    # Training
    with tf.Session(config=session_conf) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(iterator.initializer)
        model.load(sess, logdir=logdir)

        image = sess.run(input_image_stacked)
        output = sess.run(model(), feed_dict={model.input_image: image})

        output = output[0]
        image = image[0]
        output *= 255  # restore norm
        image *= 255  # restore norm

        cv2.imwrite('./output.jpg', output)
        cv2.imwrite('./input.jpg', image)


if __name__ == '__main__':
    main()
