import tensorflow as tf
import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from os.path import join
import cv2

data_path = './datasets'
image_list = []

'''
datasets 디렉토리의 데이터를 메모리로 불러오는 메소드
시작할때 한번 실행하면 전역변수인 image_list에 저장이 된다.
이후 get_batch 메소드를 통해 텐서형태로 batch_size 만큼 이미지를 
추출할 수 있음.
'''


def load_data():
    global image_list
    # 디렉토리 리스트 불러오기
    dir_list = [d for d in os.listdir(data_path) if not d.startswith('.')]
    # 디렉토리 1개마다 반복문
    for d in dir_list:
        # 디렉토리 내의 이미지 불러오기 확장자는 '.png'
        images = [i for i in os.listdir(join(data_path, d)) if i.endswith('.png')]
        temp_image_list = []
        # 이미지 파일의 이름 규칙에 맞게 순서대로 정렬하기. 이는 운영체제의 파일 시스템에 따라 불러오는 이미지 순서가 다르기 때문에
        # 디렉토리 상에서 파일이 정렬되어있어도, 불러올때는 다른 순서로 불러오게되는 문제가 생김.
        images = sorted(images, key=lambda key: int(key.split('_')[2].split('.')[0]))
        # GrayScale 이미지 읽어오기
        for image in images:
            temp_image_list.append(cv2.imread(join(data_path, join(d, image)), flags=0))
        # 현재 디렉토리의 이미지 리스트를 전체 이미지 리스트에 더하기.
        image_list.append(np.array(temp_image_list))

    image_list = np.array(image_list)


'''
batch_size : 우리가 아는 그 batch_size
k : 30 프레임을 대표하는 조각의 수 Ex) k = 3이면 1, 15, 30 번째 프레임을 뽑는다. 그리고 해당 프레임 사이를 보간한다. 
'''


def get_batch(batch_size, k=3):
    input_data = []
    label_data = []
    # batch_size 만큼 반복에서 이미지 추출
    for _ in range(batch_size):
        # 디렉토리를 랜덤으로 선택
        sampled_list = image_list[np.random.choice(len(image_list), size=1)][0]
        temp_input = []
        while True:
            # 30 프레임(1초)을 뽑을 수 있도록 시작 프레임을 선택
            idx = np.random.choice(len(sampled_list[0]), size=1)[0]
            if idx < len(sampled_list) - 30:
                break
        # k 값에 맞게 프레임 추출
        for i in range(k):
            temp_input.append(sampled_list[idx + (i * (30 // k - 1))])
        # 시작 인덱스로부터 30개 프레임 추출
        temp_label = [sampled_list[f] for f in range(idx, idx + 30)]

        input_data.append(temp_label)
        label_data.append(temp_label)
    # 텐서 형태로 반환
    return tf.convert_to_tensor(np.array(input_data)), tf.convert_to_tensor(np.array(label_data))


def _parse_function(filename):
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_string, channels=1)
    image_resized = tf.image.resize_images(image_decoded, [180, 320])
    image_norm = image_resized / 255
    return image_norm


def get_dataset_iterator(dataset_path, batch_size):
    data = os.walk(dataset_path).__next__()
    imagepaths = list()

    for d in data[2]:
        if d.endswith('.jpg') or d.endswith('.jpeg') or d.endswith('.png'):
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

def get_interNet_dataset(dataset_path):
    data = np.load(dataset_path)

    data = np.array(data).reshape(-1, 128)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler = scaler.fit(data)
    data = scaler.transform(data)

    # X = data[0:len(data) - 1]
    # Y = data[1:len(data)]

    # dataset = tf.data.Dataset.from_tensor_slices((X, Y))
    # dataset = dataset.repeat()
    # dataset = dataset.shuffle(256)
    # batched_dataset = dataset.batch(batch_size)

    # iterator = batched_dataset.make_initializable_iterator()

    return data, scaler


def get_next(dataset, batch_size, pred_size, skip_size):
    start = []
    end = []
    target = []
    for _ in range(batch_size):
        # 디렉토리를 랜덤으로 선택
        sample_index = np.random.randint(len(dataset) - (pred_size + 2), size=1)
        temp_input = []
        start.append(np.squeeze(dataset[sample_index]))
        for i in range(1, pred_size + 1):
            if i % (skip_size + 1) == 0:
                temp_input.append(np.squeeze(dataset[sample_index + i]))

        target.append(temp_input)
        end.append(np.squeeze(dataset[sample_index + pred_size + 1]))

    return np.asarray(start), np.asarray(end), np.asarray(target)