"""
@author: Ryandhimas Zezario
ryandhimas@citi.sinica.edu.tw
"""

import sys
import scipy.io
import librosa
import os
import time  
import numpy as np
import numpy.matlib
import random
import tqdm
import torch
import argparse
from WavLM import WavLM, WavLMConfig

def ListRead(filelist):
    f = open(filelist, 'r')
    Path=[]
    for line in f:
        Path=Path+[line[0:-1]]
    return Path
    
def SSL_extractor(filepath, model, list_new, mode):
    path = filepath [1]
    S=path.split('/')
    wave_name=S[-1]
    name = wave_name[:-4] 
    new_name =  name +'.npy'
 
    audio_data, sr = librosa.load(path, mono=False) 
    audio_data=audio_data/np.max(abs(audio_data))    
    audio_16k = librosa.resample(audio_data, sr, 16000)
    
    if mode == 'left':
       directory ='/data/train_wavLM_hl_left/'
       if not os.path.exists(directory):
          os.system('mkdir -p ' + directory)
          
       cached_path = os.path.join(directory,new_name)    
        
       channel_1 = np.asfortranarray(audio_16k[0])  
       end2end_channel_1 = np.reshape(channel_1,(1,channel_1.shape[0]))      
       end2end_channel_1 = torch.from_numpy(end2end_channel_1).to("cuda")
       
       rep, layer_results = model.extract_features(end2end_channel_1, output_layer=model.cfg.encoder_layers, ret_layer_results=True)[0]             
       causal_1 = rep.detach().to("cpu").numpy()
       np.save(cached_path,causal_1)
       
       info = filepath[0]+','+str(cached_path)
       list_new.append(info)

    else:    
       directory ='/data/train_wavLM_hl_right/'
       if not os.path.exists(directory):
          os.system('mkdir -p ' + directory)
          
       cached_path = os.path.join(directory,new_name)    
          
       channel_2 = np.asfortranarray(audio_16k[1])  
       end2end_channel_2 = np.reshape(channel_2,(1,channel_2.shape[0]))     
       end2end_channel_2 = torch.from_numpy(end2end_channel_2).to("cuda")
       
       rep, layer_results = model.extract_features(end2end_channel_2, output_layer=model.cfg.encoder_layers, ret_layer_results=True)[0]             
       causal_2 = rep.detach().to("cpu").numpy()
       np.save(cached_path,causal_2)
       
       info = filepath[0]+','+str(cached_path)
       list_new.append(info)

    return list_new
    
def train_data_generator(file_list,model,mode, filename):
    list_new=[]
    # for index in range(len(file_list)):
    for index in range(10):    
        wav_filepath = file_list[index].split(',')
        list_new=SSL_extractor(wav_filepath, model, list_new, mode)    
   
    with open('/data/'+filename+'.txt','w') as g:
        for item in list_new:
          g.write("%s\n" % item)
          
def Extract_Feat(Train_data, mode, filename):
    checkpoint = torch.load('/data/WavLM-Large.pt')
    cfg = WavLMConfig(checkpoint['cfg'])
    model = WavLM(cfg)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    
    device = torch.device("cuda")
    model = model.to(device)
    train_data_generator(Train_data,model,mode, filename)

if __name__ == '__main__':	
    parser = argparse.ArgumentParser('')
    parser.add_argument('--mode', type=str, default='left') 
    parser.add_argument('--filename', type=str, default='List_Npy_Train_wavLM_Clarity_Challenge_hl_mono_left')     
    args = parser.parse_args() 
    mode = args.mode
    filename = args.filename
    Train_list= ListRead('/data/List_Clarity_Challenge_HL.txt')
    Extract_Feat(Train_list, mode, filename)
	
