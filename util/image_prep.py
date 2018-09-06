import os
from PIL import Image
import numpy as np
import cv2

path = "./pix2pix/inputs/"
dirs = os.listdir(path)


def black_remove(src):
    src = np.array(src)
    # src = ~src
    gray = cv2.cvtColor(src, cv2.COLOR_GRAY2BGR)
    # bg_index = np.where(np.less(gray, 255))
    # gray[bg_index] = 0

    return gray


for item in dirs:  # Iterates through each picture
    if os.path.isfile(path + item):
        im = Image.open(path + item)
        f, e = os.path.splitext(path + item)
        # imResize = im.resize((320,180), Image.ANTIALIAS)
        imResize = black_remove(im)
        imResize = Image.fromarray(imResize)
        imResize.save(f + '.jpg', 'JPEG')
        # print(item)
