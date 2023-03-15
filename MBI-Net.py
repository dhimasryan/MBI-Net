"""
@author: Ryandhimas Zezario
ryandhimas@citi.sinica.edu.tw
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import keras
import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import math
import matplotlib.pyplot as plt
from keras.utils import plot_model
from keras.models import Sequential, model_from_json, Model, load_model
from keras.layers import Layer, concatenate
from keras.layers.core import Dense, Dropout, Flatten, Activation, Reshape, Lambda
from keras.activations import softmax
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import GlobalAveragePooling1D
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from keras.backend import squeeze
from keras.layers import LSTM, TimeDistributed, Bidirectional, dot, Input, CuDNNLSTM
from keras.constraints import max_norm
from keras_self_attention import SeqSelfAttention
from SincNet import Sinc_Conv_Layer
import argparse
import tensorflow as tf
import scipy.io
import scipy.stats
import librosa
import time  
import numpy as np
import numpy.matlib
import random
import pdb
random.seed(999)

epoch=25
batch_size = 1

def norm_data(input_x):
    input_x = (input_x-0)/(100-0)
    return input_x
    
def denorm(input_x):
    input_x = input_x*(100-0) + 0
    return input_x
    
def ListRead(filelist):
    f = open(filelist, 'r')
    Path=[]
    for line in f:
        Path=Path+[line[0:-1]]
    return Path

def Sp_and_phase_left(path, Noisy=False):
    audio_data, sr = librosa.load(path, mono=False) 
    audio_data=audio_data/np.max(abs(audio_data))    
    audio_16k = librosa.resample(audio_data, sr, 16000)

    channel_1 = np.asfortranarray(audio_16k[0])  
    F = librosa.stft(channel_1,n_fft=512,hop_length=256,win_length=512,window=scipy.signal.hamming)
      
    Lp=np.abs(F)
    phase=np.angle(F)
    if Noisy==True:    
        meanR = np.mean(Lp, axis=1).reshape((257,1))
        stdR = np.std(Lp, axis=1).reshape((257,1))+1e-12
        NLp = (Lp-meanR)/stdR
    else:
        NLp=Lp
    
    NLp=np.reshape(NLp.T,(1,NLp.shape[1],257))
    end2end = np.reshape(channel_1,(1,channel_1.shape[0],1)) 
    return NLp, end2end

def Sp_and_phase_right(path, Noisy=False):
    audio_data, sr = librosa.load(path, mono=False) 
    audio_data=audio_data/np.max(abs(audio_data))    
    audio_16k = librosa.resample(audio_data, sr, 16000)

    channel_1 = np.asfortranarray(audio_16k[1])     
    F = librosa.stft(channel_1,n_fft=512,hop_length=256,win_length=512,window=scipy.signal.hamming)
      
    Lp=np.abs(F)
    phase=np.angle(F)
    if Noisy==True:    
        meanR = np.mean(Lp, axis=1).reshape((257,1))
        stdR = np.std(Lp, axis=1).reshape((257,1))+1e-12
        NLp = (Lp-meanR)/stdR
    else:
        NLp=Lp
    
    NLp=np.reshape(NLp.T,(1,NLp.shape[1],257))
    end2end = np.reshape(channel_1,(1,channel_1.shape[0],1)) 
    return NLp, end2end

def data_generator(file_list, file_list_ssl_left, file_list_ssl_right):
	index=0
	while True:
         intell_filepath=file_list[index].split(',')
         ssl_filepath_left=file_list_ssl_left[index].split(',')
         ssl_filepath_right=file_list_ssl_right[index].split(',')         
         
         noisy_LP_left,noisy_end2end_left =Sp_and_phase_left(intell_filepath[1])       
         noisy_ssl_left =np.load(ssl_filepath_left[1])    

         noisy_LP_right,noisy_end2end_right =Sp_and_phase_right(intell_filepath[1])       
         noisy_ssl_right =np.load(ssl_filepath_right[1])   

                
         intell=norm_data(np.asarray(float(intell_filepath[0])).reshape([1]))

         feat_length_end2end = math.ceil(float(noisy_end2end_left.shape[1])/256)
         final_len = noisy_LP_left.shape[1] + int(feat_length_end2end) + noisy_ssl_left.shape[1]
         
         index += 1
         if index == len(file_list):
             index = 0
            
             random.Random(7).shuffle(file_list)
             random.Random(7).shuffle(file_list_ssl_left)
             random.Random(7).shuffle(file_list_ssl_right)             

         yield  [noisy_LP_left, noisy_end2end_left, noisy_ssl_left, noisy_LP_right, noisy_end2end_right, noisy_ssl_right], [intell, intell[0]*np.ones([1,final_len,1]), intell[0]*np.ones([1,final_len,1]), intell[0]*np.ones([1,final_len,1])]

def BLSTM_CNN_with_ATT_cross_domain_multi_branched():
    #Left Branch
    input_size =(None,1)
    _input_left = Input(shape=(None, 257))
    _input_end2end_left = Input(shape=(None, 1))
    
    SincNet_left = Sinc_Conv_Layer(input_size, N_filt = 257, Filt_dim = 251, fs = 16000, NAME = "SincNet_left").compute_output(_input_end2end_left)
    merge_input_left = concatenate([_input_left, SincNet_left],axis=1) 
    re_input_left = keras.layers.core.Reshape((-1, 257, 1), input_shape=(-1, 257))(merge_input_left)
    
    conv1_left = (Conv2D(16, (3,3), strides=(1, 1), activation='relu', padding='same'))(re_input_left)
    conv1_left = (Conv2D(16, (3,3), strides=(1, 1), activation='relu', padding='same'))(conv1_left)
    conv1_left = (Conv2D(16, (3,3), strides=(1, 3), activation='relu', padding='same'))(conv1_left)        
    
    conv2_left = (Conv2D(32, (3,3), strides=(1, 1), activation='relu', padding='same'))(conv1_left)
    conv2_left = (Conv2D(32, (3,3), strides=(1, 1), activation='relu', padding='same'))(conv2_left)
    conv2_left = (Conv2D(32, (3,3), strides=(1, 3), activation='relu', padding='same'))(conv2_left)
    
    conv3_left = (Conv2D(64, (3,3), strides=(1, 1), activation='relu', padding='same'))(conv2_left)
    conv3_left = (Conv2D(64, (3,3), strides=(1, 1), activation='relu', padding='same'))(conv3_left)
    conv3_left = (Conv2D(64, (3,3), strides=(1, 3), activation='relu', padding='same'))(conv3_left)
    
    conv4_left = (Conv2D(128, (3,3), strides=(1, 1), activation='relu', padding='same'))(conv3_left)
    conv4_left = (Conv2D(128, (3,3), strides=(1, 1), activation='relu', padding='same'))(conv4_left)
    conv4_left = (Conv2D(128, (3,3), strides=(1, 3), activation='relu', padding='same'))(conv4_left)
    
    re_shape_left = keras.layers.core.Reshape((-1, 4*128), input_shape=(-1, 4, 128))(conv4_left)
    _input_ssl_left = Input(shape=(None, 1024))
    bottleneck_left=TimeDistributed(Dense(512, activation='relu'))(_input_ssl_left)
    concat_with_wave2vec_left = concatenate([re_shape_left, bottleneck_left],axis=1) 
    blstm_left=Bidirectional(CuDNNLSTM(128, return_sequences=True), merge_mode='concat')(concat_with_wave2vec_left)
    
    flatten_left = TimeDistributed(keras.layers.core.Flatten())(blstm_left)
    dense1_left=TimeDistributed(Dense(128, activation='relu'))(flatten_left)
    dense1_left=Dropout(0.3)(dense1_left)
    
    attention_left = SeqSelfAttention(attention_type=SeqSelfAttention.ATTENTION_TYPE_MUL,kernel_regularizer=keras.regularizers.l2(1e-4),bias_regularizer=keras.regularizers.l1(1e-4),attention_regularizer_weight=1e-4, name='Attention_left')(dense1_left)
    Frame_score_left=TimeDistributed(Dense(1, activation='sigmoid'), name='Frame_score_left')(attention_left)
    
    #Right Branch
    _input_right = Input(shape=(None, 257))
    _input_end2end_right = Input(shape=(None, 1))
    
    SincNet_right = Sinc_Conv_Layer(input_size, N_filt = 257, Filt_dim = 251, fs = 16000, NAME = "SincNet_right").compute_output(_input_end2end_right)
    merge_input_right = concatenate([_input_right, SincNet_right],axis=1) 
    re_input_right = keras.layers.core.Reshape((-1, 257, 1), input_shape=(-1, 257))(merge_input_right)
    
    conv1_right = (Conv2D(16, (3,3), strides=(1, 1), activation='relu', padding='same'))(re_input_right)
    conv1_right = (Conv2D(16, (3,3), strides=(1, 1), activation='relu', padding='same'))(conv1_right)
    conv1_right = (Conv2D(16, (3,3), strides=(1, 3), activation='relu', padding='same'))(conv1_right)        
    
    conv2_right = (Conv2D(32, (3,3), strides=(1, 1), activation='relu', padding='same'))(conv1_right)
    conv2_right = (Conv2D(32, (3,3), strides=(1, 1), activation='relu', padding='same'))(conv2_right)
    conv2_right = (Conv2D(32, (3,3), strides=(1, 3), activation='relu', padding='same'))(conv2_right)
    
    conv3_right = (Conv2D(64, (3,3), strides=(1, 1), activation='relu', padding='same'))(conv2_right)
    conv3_right = (Conv2D(64, (3,3), strides=(1, 1), activation='relu', padding='same'))(conv3_right)
    conv3_right = (Conv2D(64, (3,3), strides=(1, 3), activation='relu', padding='same'))(conv3_right)    
    
    conv4_right = (Conv2D(128, (3,3), strides=(1, 1), activation='relu', padding='same'))(conv3_right)
    conv4_right = (Conv2D(128, (3,3), strides=(1, 1), activation='relu', padding='same'))(conv4_right)
    conv4_right = (Conv2D(128, (3,3), strides=(1, 3), activation='relu', padding='same'))(conv4_right)
    
    re_shape_right = keras.layers.core.Reshape((-1, 4*128), input_shape=(-1, 4, 128))(conv4_right)
    _input_ssl_right = Input(shape=(None, 1024))
    bottleneck_right=TimeDistributed(Dense(512, activation='relu'))(_input_ssl_right)
    concat_with_wave2vec_right = concatenate([re_shape_right, bottleneck_right],axis=1) 
    blstm_right=Bidirectional(CuDNNLSTM(128, return_sequences=True), merge_mode='concat')(concat_with_wave2vec_right)
    
    flatten_right = TimeDistributed(keras.layers.core.Flatten())(blstm_right)
    dense1_right=TimeDistributed(Dense(128, activation='relu'))(flatten_right)
    dense1_right=Dropout(0.3)(dense1_right)
    
    attention_right = SeqSelfAttention(attention_type=SeqSelfAttention.ATTENTION_TYPE_MUL,kernel_regularizer=keras.regularizers.l2(1e-4),bias_regularizer=keras.regularizers.l1(1e-4),attention_regularizer_weight=1e-4, name='Attention_right')(dense1_right)
    Frame_score_right=TimeDistributed(Dense(1, activation='sigmoid'), name='Frame_score_right')(attention_right)
    
    #Main Branch
    concat_score = concatenate([Frame_score_left, Frame_score_right]) 
    Frame_score_final=TimeDistributed(Dense(1), name='Frame_score_final')(concat_score)
    Intell_score_final=GlobalAveragePooling1D(name='Intell_score')(Frame_score_final)
    
    model = Model(outputs=[Intell_score_final,Frame_score_final, Frame_score_right, Frame_score_left], inputs=[_input_left,_input_end2end_left, _input_ssl_left, _input_right,_input_end2end_right, _input_ssl_right])
    
    return model
    
def Train(train_list, train_list_ssl_left, train_list_ssl_right, NUM_TRAIN, valid_list_ssl_left, valid_list_ssl_right, NUM_VALID, pathmodel):
    print ('model building...')
    
    adam = Adam(lr=1e-4)
    model = BLSTM_CNN_with_ATT_cross_domain_multi_branched()
    model.compile(loss={'Intell_score': 'mse', 'Frame_score_final': 'mse', 'Frame_score_right': 'mse', 'Frame_score_left': 'mse'}, optimizer=adam)
    plot_model(model, to_file='model_'+pathmodel+'.png', show_shapes=True)
    
    with open(pathmodel+'.json','w') as f:    # save the model
        f.write(model.to_json()) 
    checkpointer = ModelCheckpoint(filepath=pathmodel+'.hdf5', verbose=1, save_best_only=True, mode='min') 
    
    print ('training...')
    g1 = data_generator (train_list, train_list_ssl_left, train_list_ssl_right)
    g2 = data_generator (valid_list, valid_list_ssl_left, valid_list_ssl_right)
    
    hist=model.fit_generator(g1,steps_per_epoch=NUM_TRAIN, epochs=epoch, verbose=1, validation_data=g2, validation_steps=NUM_VALID, max_queue_size=1, workers=1, callbacks=[checkpointer])
    
    model.save(pathmodel+'.h5')
    
    # plotting the learning curve
    TrainERR=hist.history['loss']
    ValidERR=hist.history['val_loss']
    print ('@%f, Minimun error:%f, at iteration: %i' % (hist.history['val_loss'][epoch-1], np.min(np.asarray(ValidERR)),np.argmin(np.asarray(ValidERR))+1))
    print ('drawing the training process...')
    plt.figure(4)
    plt.plot(range(1,epoch+1),TrainERR,'b',label='TrainERR')
    plt.plot(range(1,epoch+1),ValidERR,'r',label='ValidERR')
    plt.xlim([1,epoch])
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('error')
    plt.grid(True)
    plt.show()
    plt.savefig('Learning_curve'+pathmodel+'.png', dpi=150)

def Test(valid_list, valid_list_ssl_left, valid_list_ssl_right,pathmodel): 	 
    print 'testing...'
    
    Intell_Predict=np.zeros([len(valid_list),])
    Intell_true   =np.zeros([len(valid_list),])
    list_predicted_mos_score =[]

    for i in range(len(valid_list)):
       print i
       Asessment_filepath_left=valid_list[i].split(',')
       ssl_filepath_left = valid_list_ssl_left[i].split(',')
       
       noisy_LP_left, noisy_end2end_left =Sp_and_phase_left(Asessment_filepath_left[1]) 
       noisy_ssl_left = np.load(ssl_filepath_left[1])
       
       Asessment_filepath_right=valid_list[i].split(',')
       ssl_filepath_right = valid_list_ssl_right[i].split(',')
       
       noisy_LP_right, noisy_end2end_right =Sp_and_phase_right(Asessment_filepath_right[1]) 
       noisy_ssl_right = np.load(ssl_filepath_right[1])
       
       intell=float(Asessment_filepath_left[0])
       [Intell_score_final,Frame_score_final, Frame_score_right, Frame_score_left]=model.predict([noisy_LP_left, noisy_end2end_left, noisy_ssl_left, noisy_LP_right, noisy_end2end_right, noisy_ssl_right], verbose=0, batch_size=batch_size)
       
       denorm_Intell_predict = denorm(Intell_score_final)
       Intell_Predict[i]=denorm_Intell_predict
       Intell_true[i]   =intell
       list_predicted_mos_score.append(denorm_Intell_predict) 
     
    with open('List_predicted_score_mos'+pathmodel+'.txt','w') as g:
       for item in list_predicted_mos_score:
          g.write("%s\n" % item)
          
    RMSE=np.sqrt(np.mean((Intell_true-Intell_Predict)**2)) 
    print ('Test error= %f' % RMSE)
    LCC=np.corrcoef(Intell_true, Intell_Predict)
    print ('Linear correlation coefficient= %f' % LCC[0][1])
    SRCC=scipy.stats.spearmanr(Intell_true.T, Intell_Predict.T)
    print ('Spearman rank correlation coefficient= %f' % SRCC[0])
    
    # Plotting the scatter plot
    M=np.max([np.max(Intell_Predict),1])
    plt.figure(1)
    plt.scatter(Intell_true, Intell_Predict, s=14)
    plt.xlim([0,M])
    plt.ylim([0,M])
    plt.xlabel('True Intell')
    plt.ylabel('Predicted Intell')
    plt.title('LCC= %f, SRCC= %f, RMSE= %f' % (LCC[0][1], SRCC[0], RMSE))
    plt.show()
    plt.savefig('Scatter_plot_Intell'+pathmodel+'.png', dpi=150)


if __name__ == '__main__':  
     
    parser = argparse.ArgumentParser('')
    parser.add_argument('--gpus', type=str, default='0') 
    parser.add_argument('--mode', type=str, default='train') 
    
    args = parser.parse_args() 
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    
    pathmodel="MBI-Net"

    #################################################################             
    ######################### Training data #########################
    
    Train_list_wav = ListRead('/data/List_Clarity_Challenge_HL.txt')
    Train_list_ssl_feat_left = ListRead('/data/List_Npy_Train_wavLM_Clarity_Challenge_hl_mono_left.txt')
    Train_list_ssl_feat_right = ListRead('/data/List_Npy_Train_wavLM_Clarity_Challenge_hl_mono_right.txt')
    
    NUM_DATA =  len(Train_list_wav)
    NUM_TRAIN = int(NUM_DATA*0.9) 
    NUM_VALID = NUM_DATA-NUM_TRAIN
    
    train_list= Train_list_wav[: NUM_TRAIN]
    random.Random(7).shuffle(train_list)
    valid_list= Train_list_wav[NUM_TRAIN: ]
    
    train_list_ssl_left= Train_list_ssl_feat_left[: NUM_TRAIN]
    random.Random(7).shuffle(train_list_ssl_left)
    valid_list_ssl_left= Train_list_ssl_feat_left[NUM_TRAIN: ]
    
    train_list_ssl_right= Train_list_ssl_feat_right[: NUM_TRAIN]
    random.Random(7).shuffle(train_list_ssl_right)
    valid_list_ssl_right= Train_list_ssl_feat_right[NUM_TRAIN: ]

    if args.mode == 'train':
       print 'training'  
       Train(train_list, train_list_ssl_left, train_list_ssl_right, NUM_TRAIN, valid_list_ssl_left, valid_list_ssl_right, NUM_VALID, pathmodel)
       print 'complete training stage'    
    
    if args.mode == 'test':      
       print 'testing' 
       Test(valid_list, valid_list_ssl_left, valid_list_ssl_right,pathmodel)
       print 'complete testing stage'

