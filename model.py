import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2

from ops import lrelu, lreul6
from ops import load_BatchImage, image_recovery, strong_image_recovery               ## for AutoEncoder
from ops import bag_of_latent, random_batch_seq, making_arbit_randomCSV, CSVdata_generator  ## for SRNN

# Todo 
class TestModel:
    def __init__(self):
        ### model parameter
        ### model input(placeholders)
        #AutoEncoder
        self.X = tf.placeholder(tf.float32, shape=(None, 256, 256, 1))
        self.Y = tf.placeholder(tf.float32, shape=(None, 256, 256, 1))
        self.is_training=tf.placeholder(tf.bool,shape=[])
        self.lv=tf.placeholder(tf.float32,shape=(None,128))        
        
        #SRNN Model
        self.RNN_X=tf.placeholder(tf.float32,shape=(None,None,128))

        ### Statment of models
        # AutoEncoder
        self.Encoder=self._Encoder(self.X,is_training=self.is_training)
        self.AE=self._Decoder(self.Encoder,is_training=self.is_training)
        self.Decoder=self._Decoder(self.lv,is_training=self.is_training,reuse=True)
        
        # SRNN model
        self.SimpleRNN=self.SimpleRNN_Predict(self.RNN_X)
             
        
    def _Encoder(self, inputs, is_training=False,reuse=False):
        with tf.variable_scope('Encoder', reuse=reuse):  # inputs <- (batch, 256, 256, c)
            conv1=tf.contrib.layers.conv2d(inputs,256,3,2,activation_fn=lrelu,padding='SAME') #128*128*256
            pool1=tf.contrib.layers.max_pool2d(conv1,2,2,padding='SAME')                     #64*64*256
            conv2=tf.contrib.layers.conv2d(pool1,128,3,2,activation_fn=lrelu,padding='SAME') #32*32*128
            pool2=tf.contrib.layers.max_pool2d(conv2,2,2,padding='SAME')                     #16*16*128
            conv3=tf.contrib.layers.conv2d(pool2,64,3,2,activation_fn=lrelu,padding='SAME')  #8*8*64
            z=tf.layers.flatten(conv3)
            z=tf.layers.dense(z,1024,activation=tf.nn.relu6)
            #z=tf.contrib.layers.dropout(z,keep_prob=0.9,is_training=is_training)
            z=tf.layers.dense(z,256,activation=tf.nn.relu6)
            z=tf.layers.batch_normalization(z,training=is_training)
            z= tf.layers.dense(z, 128, activation=lrelu)
            return z

    def _Decoder(self, inputs,is_training=False, reuse=False):
        with tf.variable_scope('Decoder', reuse=reuse):  # inputs <- (batch, 128)
            dec = tf.layers.dense(inputs, 16 * 16 * 8, activation=tf.nn.relu6)
            dec = tf.layers.dense(dec,24*24*12,activation=tf.nn.relu6)
            dec = tf.layers.dense(dec, 32 * 32 * 16, activation=tf.nn.relu6)
            #dec = tf.layers.dense(dec, 64 * 64 * 32, activation=tf.nn.relu)
            dec = tf.reshape(dec,[-1,32,32,16])
            deconv1=tf.contrib.layers.conv2d_transpose(dec,32,3,2,activation_fn=lrelu,padding='SAME') # 64*64*32
            #deconv2=tf.contrib.layers.dropout(deconv1,keep_prob=0.9,is_training=is_training)
            deconv3=tf.contrib.layers.conv2d_transpose(deconv1,64,3,2,activation_fn=lrelu,padding='SAME') # 128*128*64
            deconv4=tf.contrib.layers.conv2d_transpose(deconv3,128,3,2,activation_fn=lrelu,padding='SAME') #  256*256*128
            #output=tf.contrib.layers.conv2d_transpose(deconv4,1,3,1,activation_fn=lrelu,padding='SAME') #256*256*1
            output=tf.contrib.layers.conv2d(deconv4,1,3,1,activation_fn=tf.nn.sigmoid,padding='SAME') # 256*256*1
            return output

    def SimpleRNN_Predict(self, seq_input,reuse=False):
        with tf.variable_scope('SimpleRNN_Predict',reuse=reuse):

            # Bi-LSTMr
            with tf.variable_scope('SBi_LSTM_1'):
                lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(512, forget_bias=1.0)
                lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(512, forget_bias=1.0)
                (output_fw, output_bw), states = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, seq_input,
                                                                                 dtype=tf.float32)

                residual_fw = tf.layers.dense(output_fw, 128, activation=tf.nn.tanh)
                residual_bw = tf.layers.dense(output_bw, 128, activation=tf.nn.tanh)
                
                seq_input2=(residual_fw+residual_bw)/2
                
            with tf.variable_scope('SBi_LSTM_2'):
                lstm_fw_cell2 = tf.contrib.rnn.BasicLSTMCell(512, forget_bias=1.0)
                lstm_bw_cell2 = tf.contrib.rnn.BasicLSTMCell(512, forget_bias=1.0)
                (output_fw2, output_bw2), states = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell2, lstm_bw_cell2, seq_input2,
                                                                                 dtype=tf.float32)
                residual_fw2 = tf.layers.dense(output_fw2, 128, activation=tf.nn.tanh)
                residual_bw2 = tf.layers.dense(output_bw2, 128, activation=tf.nn.tanh)
                
                seq_input3=(residual_fw+residual_bw)/2
                
            with tf.variable_scope('SBi_LSTM_3'):
                lstm_fw_cell3 = tf.contrib.rnn.BasicLSTMCell(256, forget_bias=1.0)
                lstm_bw_cell3 = tf.contrib.rnn.BasicLSTMCell(256, forget_bias=1.0)
                (output_fw3, output_bw3), states = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, seq_input3,
                                                                                 dtype=tf.float32)
                residual_fw3 = tf.layers.dense(output_fw3, 128, activation=tf.nn.tanh)
                residual_bw3 = tf.layers.dense(output_bw3, 128, activation=tf.nn.tanh)
            
            pred_seq_vec=(residual_fw3+residual_bw3)/2

            return pred_seq_vec
    
    
    def __call__(self, *args, **kwargs):
        return None
    
    def AE_loss(self,X,Y,_lambda=10):
        loss=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.layers.flatten(X),
                                                                      logits=tf.layers.flatten(Y)))

        loss2=tf.reduce_mean(tf.losses.absolute_difference(X,Y))
        loss=loss+_lambda*loss2
        return loss
    
    def SRNN_loss(self,X,Y,_lambda=10):
        loss2=tf.reduce_mean(tf.losses.mean_squared_error(X,Y))
        #model.latentVariables, model.SimpleRNN
        loss=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=X,labels=Y))
        loss=_lambda*loss2+loss
        return loss


        
################# Training Fcns #####################################################################################################

def AE_train(sess,model,batch_size,learning_rate,data_size,epoch_num,ckpt_addr,img_addr,optimizer='RMSProp'):
    ### sess: Session, model: model class
        #### initial setting

    iter_num=int(data_size/batch_size)
    current_path=os.getcwd()
    addr_a=img_addr
    os.chdir(addr_a)
    list_a=os.listdir()
    os.chdir(current_path)
    num_a=len(list_a)
    
    ############################### loss & optimizer ####################################################
    if optimizer=='RMSProp':
        optimizer=tf.train.RMSPropOptimizer(learning_rate=learning_rate)
    else:
        optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate)
    
    loss=model.AE_loss(model.X,model.AE,_lambda=10)
    
    AE_En_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='Encoder')
    AE_De_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='Decoder')
    ##grads = optimizer.compute_gradients(loss,var_list=enc_var + dec_var)
    grads = optimizer.compute_gradients(loss,var_list=AE_En_var+AE_De_var)
    update= optimizer.apply_gradients(grads)
    
    ############################### Initializing ########################################################
    
    init=tf.global_variables_initializer()
    saver=tf.train.Saver(var_list=AE_En_var+AE_De_var)
    sess.run(init)

    ############################### Learning_process ######################################################
    
    try:
        saver.restore(sess=sess,save_path=ckpt_addr)
        print("\nmodel restored.\n")
    except:
        print('\nmodel Not restored\n')
    

    for epoch in range(epoch_num):
        for jmi in range(iter_num):
            batch_X=load_BatchImage(batch_size,num_a,addr_a,list_a)

            _=sess.run(update,feed_dict={model.X:batch_X, model.is_training:True})

        saver.save(sess,ckpt_addr)

        loss_,images=sess.run([loss,model.AE],feed_dict={model.X:batch_X, model.is_training:False})


        print('{}th Epoch loss:{}'.format(epoch+1,loss_/batch_size))
        fig,ax=plt.subplots(1,2,figsize=(5,5))

        for jmii in range(2):
            ax[jmii].set_axis_off()

        ax[0].imshow(image_recovery(batch_X[0]))
        ax[1].imshow(image_recovery(images[0]))

        plt.savefig('./images/'+'sample-{:05d}.png'.format(epoch),bbox_inches='tight')
        plt.close(fig)
    print('Done')
        
def SRNN_train(sess,model,batch_size,learning_rate,iter_num,ckpt_addr,csv_addr,optimizer='RMSProp'):
    
    current_dir=os.getcwd()
    os.chdir(csv_addr)
    list_dir=os.listdir()
    ######## Bag of latent Variables
    BOL,BOL_len=bag_of_latent('./',list_dir)
    os.chdir(current_dir)
    
    Y=tf.placeholder(tf.float32,shape=(None,None,128))
    
    ############## loss & optimizer ######################################################################
    loss=model.SRNN_loss(Y,model.SimpleRNN,_lambda=10)
    if optimizer=='RMSProp':
        optimizer=tf.train.RMSPropOptimizer(learning_rate=learning_rate)
    else:
        optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate)
    
    SRNN_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='SimpleRNN_Predict')
    grads = optimizer.compute_gradients(loss,var_list=SRNN_var)

    update= optimizer.apply_gradients(grads)
    ############ Initializing ##########################################################################
    init=tf.global_variables_initializer()
    saver=tf.train.Saver(SRNN_var)
    sess.run(init)
    try:
        saver.restore(sess=sess,save_path=ckpt_addr)
        print('Variables are restored')
    except:
        print('Variables are not restored')
        
    ############## learning_Process #####################################################################
    for jmi in range(iter_num):
    
        batch_X,batch_Y=random_batch_seq(BOL,BOL_len,batch_size)

        _=sess.run(update,feed_dict={model.RNN_X:batch_X,Y:batch_Y})

        if jmi%100==0:
            loss_=sess.run(loss,feed_dict={model.RNN_X:batch_X,Y:batch_Y})
            saver.save(sess,save_path=ckpt_addr)
            print('{:04d} iter_loss: {:07f}'.format(jmi,loss_))
    
    
