from model import AE_Model
from ops import get_dataset_iterator
import tensorflow as tf

# hparameter
latent_dim = 128
batch_size = 16
num_step = 9000
save_per_step = 300
logdir = './logdir/{}'.format("AE_3_fine/")


def main():
    tf.reset_default_graph()
    model = AE_Model(latent_dim, batch_size)
    saver = tf.train.Saver(tf.trainable_variables())

    # Loss
    train_loss = model.loss()

    # Optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate=3e-4)
    train_op = optimizer.minimize(train_loss)

    # Dataset
    iterator = get_dataset_iterator('./imgs', batch_size)
    input_image_stacked = iterator.get_next()

    # Summary
    tf.summary.scalar('net/loss', train_loss)
    summ_op = tf.summary.merge_all()

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
        writer = tf.summary.FileWriter(logdir, sess.graph)

        for i in range(num_step):
            image = sess.run(input_image_stacked)
            _, loss, summ = sess.run([train_op, train_loss, summ_op], feed_dict={model.input_image: image})
            print("step : " + str(i) + ", loss : " + str(loss))
            writer.add_summary(summ, global_step=i)

            if i % save_per_step == 0:
                saver.save(sess, '{}/checkpoint_step_{}'.format(logdir, i))

        writer.close()


if __name__ == '__main__':
    main()
