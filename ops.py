import tensorflow as tf
import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def _parse_function(filename):
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_string, channels=1)
    image_resized = tf.image.resize_images(image_decoded, [120, 208])
    image_norm = image_resized / 255
    return image_norm


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


# X, y 에 대한 itr 와 scaler도 리턴 나중에 복원 해야하니까
def get_lv_dataset_iterator(dataset_path, batch_size):
    data = np.load(dataset_path)

    data = np.array(data).reshape(-1, 128)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler = scaler.fit(data)
    data = scaler.transform(data)

    X = data[0:len(data) - 1]
    Y = data[1:len(data)]

    dataset = tf.data.Dataset.from_tensor_slices((X, Y))
    dataset = dataset.repeat()
    dataset = dataset.shuffle(256)
    batched_dataset = dataset.batch(batch_size)

    iterator = batched_dataset.make_initializable_iterator()

    return iterator, scaler
