import tensorflow as tf
import os

def _parse_function(filename):
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_string, channels=1)
    image_resized = tf.image.resize_images(image_decoded, [120, 208])
    return image_resized


def get_dataset_iterator(dataset_path, batch_size):
    data = os.walk(dataset_path).__next__()
    imagepaths = list()

    for d in data[2]:
        if d.endswith('.jpg') or d.endswith('.jpeg'):
            imagepaths.append(os.path.join(dataset_path, d))

    imagepaths = tf.convert_to_tensor(imagepaths, dtype=tf.string)

    dataset = tf.data.Dataset.from_tensor_slices(imagepaths)

    dataset = dataset.map(_parse_function)
    dataset = dataset.repeat()
    dataset = dataset.shuffle(256)
    batched_dataset = dataset.batch(batch_size)

    iterator = batched_dataset.make_initializable_iterator()

    return iterator