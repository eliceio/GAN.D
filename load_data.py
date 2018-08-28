#path 관련 라이브러리
import os
from os.path import isdir, join
from pathlib import Path
import random
import re

data_path = './datasets'
def get_batch(batch_size):
    listdir = [d for d in os.listdir(data_path) if not d.startswith('.')]
    lendir = len(listdir)
    input_data = []
    label_data = []
    for _ in range(batch_size):
        sampled_list = random.sample(listdir,1)
        li = [f for f in os.listdir(join(data_path, str(sampled_list[0]))) if not f.startswith('.')]
        temp_input = []
        temp_label = []
        while True:
            idx = random.sample(range(len(li)), k = 1)[0]
            if idx < len(li) - 30 :
                break
        alphanum_key = lambda key : int(key.split('_')[2].split('.')[0])
        li = sorted(li, key=alphanum_key)

        temp_input.append(li[idx])
        temp_input.append(li[idx+15])
        temp_input.append(li[idx+30])
        temp_label = [li[f] for f in range(idx, idx+30) ]

        print(temp_input, end=',')
        print(temp_label)


get_batch(5)