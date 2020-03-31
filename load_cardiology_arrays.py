#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 10:19:19 2019

@author: scro3517

Loading Nature Medicine Cardiology Test Set (Ng)
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import json
import pickle
from scipy.signal import resample
#%%

basepath = '/home/scro3517/Desktop/CARDIOL_MAY_2017'
fs = 200
original_frame_length = 6000
samples_per_frame = 256
resampled_length = 2500
""" Determine Number of Frames Per Original Frame """
nframes = original_frame_length//samples_per_frame
samples_to_take_per_frame = samples_per_frame*nframes

""" All Files in Directory """
files = os.listdir(basepath)
""" Return Unique Patient Ids """
unique_patient_numbers = np.unique([file.split('_')[0] for file in files if not os.path.isdir(os.path.join(basepath,file))])

classification = 'all' #all

#%%
inputs = dict()
outputs = dict()
all_labels = []
for patient_number in unique_patient_numbers:
    inputs[patient_number] = []
    outputs[patient_number] = []

    """ Load Frame Data """
    filename = [file for file in files if patient_number in file and 'ecg' in file][0]
    f = open(os.path.join(basepath,filename),'rb')
    frame = np.fromfile(f,dtype=np.int16) #6000x1
    
    """ Load Group Label File """    
    group_label = [file for file in files if patient_number in file and 'grp' in file][0]
    with open(os.path.join(basepath,group_label)) as json_file:
        data = json.load(json_file)
    
    onsets = [episode['onset']-1 for episode in data['episodes']] #=1 for python start at 0
    offsets = [episode['offset'] for episode in data['episodes']]
    rhythms = [episode['rhythm_name'] for episode in data['episodes']]
    
    for nframe in range(nframes):
        start_sample = nframe * samples_per_frame
        end_sample = start_sample + samples_per_frame
        mini_frame = frame[start_sample:end_sample]
        for i in range(len(rhythms)):
            if onsets[i] <= start_sample < offsets[i]:
                mini_label = rhythms[i]     
                if mini_label == 'AVB_TYPE2':
                    mini_label = 'AVB'
                elif mini_label == 'AFL':
                    mini_label = 'AFIB'
                elif mini_label == 'SUDDEN_BRADY':
                    break
        
        if mini_label == 'SUDDEN_BRADY': #dont record sudden brady
            continue
        
        """ Resample Frame """
        mini_frame = resample(mini_frame,resampled_length)
        
        """ Binarize Labels """
        if classification == 'binary':
            if mini_label == 'NSR':
                mini_label = 0
            else:
                mini_label = 1
        
        all_labels.append(mini_label)
        inputs[patient_number].append(mini_frame)
        outputs[patient_number].append(mini_label)
    
#    """ Take Last Portion of Frame """
#    frame = frame[-samples_to_take_per_frame:]
#    """ Reshape Frame """
#    frames = np.reshape(frame,(-1,samples_per_frame))
#    """ Change dtype of Frame """
#    frames = np.array(frames,dtype=float)
#    """ Return Group JSON File """
#    """ Obtain Label from Group Label File """
#    onset_instance = 0
#    label = data['episodes'][onset_instance]['rhythm_name']
    
#    """ Convert Into Binary Classification """
#    if classification == 'binary':
#        if 'NSR' in label:
#            label = 0
#        else:
#            label = 1
#    labels = np.repeat(label,frames.shape[0]).tolist()
#            
#    inputs[patient_number] = frames
#    outputs[patient_number] = labels
    
    inputs[patient_number] = np.array(inputs[patient_number])
    outputs[patient_number] = np.array(outputs[patient_number])
#%%
""" Retrieve Unique Class Names """
unique_labels = []
for label in all_labels:
    if label not in unique_labels:
        unique_labels.append(label)

""" Convert Drug Names to Labels """
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
label_encoder.fit(unique_labels)
for patient_number,labels in outputs.items():
    outputs[patient_number] = label_encoder.transform(labels)

#%%
""" Make New Directory to Avoid Contamination """
savepath = os.path.join(basepath,'patient_data','%s_classes' % classification)
try:
    os.chdir(savepath)
except:
    os.makedirs(savepath)
#%%
""" Save Inputs and Labels Dicts For Splitting Later """
with open(os.path.join(savepath,'ecg_signal_frames_cardiology.pkl'),'wb') as f:
    pickle.dump(inputs,f)
with open(os.path.join(savepath,'ecg_signal_arrhythmia_labels_cardiology.pkl'),'wb') as f:
    pickle.dump(outputs,f)
