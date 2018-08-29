import tensorflow as tf
import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from os.path import join
import cv2

data_path = './datasets'
image_list = []

def load_data():
    global image_list
    dir_list = [d for d in os.listdir(data_path) if not d.startswith('.')]

    for dir in dir_list:
        images = [i for i in os.listdir(join(data_path,dir)) if i.endswith('.png')]
        temp_image_list = []
        alphanum_key = lambda key: int(key.split('_')[2].split('.')[0])
        images = sorted(images, key=alphanum_key)
        # print(images)
        for image in images :
            temp_image_list.append(cv2.imread(join(data_path,join(dir,image)), flags=0))
        image_list.append(np.array(temp_image_list))
    image_list = np.array(image_list)
    # print(type(image_list[0]))


def get_batch(batch_size, k = 3):
    input_data = []
    label_data = []
    for _ in range(batch_size):
        sampled_list = image_list[np.random.choice(len(image_list),size=1)][0]
        # li = [f for f in os.listdir(join(data_path, str(sampled_list[0]))) if not f.startswith('.')]
        # print(sampled_list.shape)
        temp_input = []
        while True:
            idx = np.random.choice(len(sampled_list[0]), size = 1)[0]
            if idx < len(sampled_list) - 30 :
                break

        # print(idx)
        for i in range(k):
            temp_input.append(sampled_list[idx + (i*(30//k-1))])
            # temp_input.append(li[idx+15])
            # temp_input.append(li[idx+30])
        temp_label = [sampled_list[f] for f in range(idx, idx+30) ]

        input_data.append(temp_label)
        label_data.append(temp_label)
        # print(temp_input, end=',')
        # print(temp_label)
    return tf.convert_to_tensor(np.array(input_data)), tf.convert_to_tensor(np.array(label_data))

# 'load_data' & 'get_batch' 테스트 코드 나중에 삭제 예정.
# 일단 전역변수 image_list에다 데이터를 읽어서 ndarray 형태로 저장.
# 그런 다음 get_batch 코드에서 임의로 디렉토리 선택하고 k(interval 이미지 수)에 맞게 알아서 데이터를 가져옴
# tensor 형태로 return 하는데 일단 이렇게 해두고 저장된거 자체를 ndarray가 아니라 Tensor 형태로 해보도록
# 수정할 예정 여전히 리펙토링은 추후에
# load_data()
# input, label = get_batch(batch_size=5)
# print(type(input))
# print(type(label))




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
