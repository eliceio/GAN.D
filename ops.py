import tensorflow as tf
import numpy as np
import csv
import os
import cv2

def lrelu(x,leak=0.2,name='lrelu'):
    with tf.variable_scope(name):
        return tf.maximum(x,leak*x)
    
def lreul6(x,leak=0.2,name='lrelu6'):
    with tf.varialbe_scope(name):
        return tf.minimum(tf.maximum(x,leak*x),6.0)

############################################################################################################
################################### for RNN model ##########################################################
############################################################################################################
# data extraction for RNN from csv files list
# making-up latent list
def bag_of_latent(addr, dir_list):  ### dictionary / keys: Video dirs, items:
    bol = {}
    len_bol = {}
    for video in dir_list:
        with open(addr + video, 'r') as csv_file:
            readCSV = csv.reader(csv_file, delimiter=',')
            data_list = []

            for line in readCSV:
                dum = [float(x) for x in line]
                data_list.append(dum)
        len_bol[video] = len(data_list)
        bol[video] = np.array(data_list)

    return bol, len_bol

# make up seq-input data for RNN training
# random sequence sampling by frame from BOL
def random_batch_seq(BOL, len_BOL, batch_num, frame=15):
    keys = list(len_BOL.keys())
    len_keys = len(keys)
    batch_seq_X = []
    batch_seq_Y = []
    data_frame = 30
    if data_frame % frame != 0:
        print('Because data_frame is 30, choose common denominators')
    pattern_frame = data_frame // frame

    for jmi in range(batch_num):
        dum_next_key = np.random.choice(len_keys)
        next_key = keys[dum_next_key]
        cur_data_len = BOL[next_key].shape[0]
        starting_frame = np.random.choice(int(cur_data_len - frame * pattern_frame))
        single_X = []
        single_Y = BOL[next_key][starting_frame:starting_frame + pattern_frame * frame:pattern_frame]
        batch_seq_Y.append(single_Y)

        single_dum = list((single_Y[0] + single_Y[-1]) / 2)
        single_X.append(single_Y[0])
        for jmii in range(frame - 2):
            single_X.append(single_dum)
        single_X.append(single_Y[-1])

        batch_seq_X.append(single_X)

    return np.array(batch_seq_X), np.array(batch_seq_Y)

### Interface with AutoEncoder
### 
def AE_SRNN_seq_X(img1,img2, frame=15, latent_num=128):
    seq_X=[]
    data_frame = 30
    np_img1=np.array(img1)
    np_img2=np.array(img2)
    
    if data_frame % frame != 0:
        print('Because data_frame is 30, choose common denominators')
    pattern_frame = data_frame // frame    
    
    img_dum=list((np_img1+np_img2)/2)
    seq_X.append(img1)
    for jmi in range(frame-2):
        seq-X.append(img_dum)
    seq_X.append(img2)
    return np.array(seq_X).reshape([1,-1,latent_num])

    
# generating latent data to make RNN model learned
### These are functions below are stated temporarily
def making_arbit_randomCSV(lv_num, total_num, mean=0, std=3):  ### for pre-training
    lv = np.random.normal(mean, std, (total_num, lv_num))
    return lv

def CSVdata_generator(addr='./', name='Video', num_CSVfile=8):
    for jmi in range(num_CSVfile):
        with open(addr + name + '_{:02d}.csv'.format(jmi), 'w', newline='') as fop:
            CSVwriter = csv.writer(fop, delimiter=',')
            dum = np.random.choice(2000) + 500
            dum2 = making_arbit_randomCSV(128, dum)
            CSVwriter.writerows(dum2)

######################################################################################
################################### for Auto-Encoder #######################################################
############################################################################################################
def load_BatchImage(batch_size,num_a,addr_a,list_a):    
    mask_a=np.random.choice(num_a,batch_size)
    A=[]
    for jmi in range(batch_size):
        imgA=cv2.imread(addr_a+list_a[mask_a[jmi]],cv2.IMREAD_GRAYSCALE)
        imgA=imgA.reshape([256,256,1])
        imgA=imgA/255
        A.append(imgA)
    A=np.array(A)
    
    return A

def image_recovery(image):
    img=image*255
    img=img.astype('uint8')
    img=cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
    return img

def strong_image_recovery(image,relax_factor=0.5):
    mark=image<relax_factor
    image[mark]=0
    img=image*255
    img=img.astype('uint8')
    img=cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
    return img

########################################################################################################################
############################################ to exploit models #########################################################
########################################################################################################################

def restore_ckpt(sess,AEckpt_addr,SRNNckpt_addr):
######### 
    sess.run(tf.global_variables_initializer())
    
    AE_En_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='Encoder')
    AE_De_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='Decoder')
    RNN_var=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='SimpleRNN_Predict')
    
    assert AE_En_var!=[],'You should state model first'
    
    saver1=tf.train.Saver(var_list=AE_En_var+AE_De_var)
    saver2=tf.train.Saver(RNN_var)

    try:
        saver1.restore(sess=sess,save_path=AEckpt_addr)
        print('AE_Variables are restored')
    except:
        print('AE_Varabless are not restored')

    try:
        saver2.restore(sess=sess,save_path=SRNNckpt_addr)
        print('SRNN_Variables are restored')
    except:
        print('SRNN_Varabless are not restored')
        
        
