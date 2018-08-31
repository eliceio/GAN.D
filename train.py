#from model import TestModel
import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2

from model import TestModel, AE_train, SRNN_train


#######################
#### for AutoEncoder
####################### 'ctrl+/' => adding # 


batch_size=10
data_size=20000
epoch_num=100000
learning_rate=0.0002

model=TestModel()

sess=tf.Session()

ckpt_addr='./ckpt/model.ckpt'                   #including model.ckpt
img_addr='./data/skeleton/'                    # /directory including images to train/

AE_train(sess,model,batch_size,learning_rate,data_size,epoch_num,ckpt_addr,img_addr,optimizer='RMSProp')


#######################
#### for SRNN
####################### 'ctrl+/' => adding # 

batch_size=10
iter_num=100000
learning_rate=0.0002

model=TestModel()

sess=tf.Session()

ckpt_addr='./ckpt/SRNNckpt/model.ckpt'                   #including model.ckpt
csv_addr='./data/RNNpreTrain/'                    # /directory including images to train/

SRNN_train(sess,model,batch_size,learning_rate,iter_num,ckpt_addr,csv_addr,optimizer='RMSProp')





