#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 27 15:38:15 2019

@author: Dani Kiyasseh
"""

import os
import wfdb
import pickle
import numpy as np
from scipy import signal
#%%
#basepath = '/home/scro3517/Desktop/ptb-diagnostic-ecg-database-1.0.0'

patient_folders = os.listdir(basepath)
patient_folders = sorted([folder for folder in patient_folders if os.path.isdir(os.path.join(basepath,folder))])
""" Remove patient_data Folder I Created Below """
patient_folders = [folder for folder in patient_folders if 'patient_data' not in folder]

samples = 2500
leads_list = ['i','ii','iii','avr','avl','avf','v1','v2','v3','v4','v5','v6']
#leads = ['ii']

classification = 'binary'

def obtain_arrays(patient_folders,leads,classification):
    inputs = dict()
    outputs = dict()
    all_labels = []
    for folder in patient_folders:
        """ Path for Patient Files """
        folderpath = os.path.join(basepath,folder)
        """ Patient Files """
        files = os.listdir(folderpath)
        """ Keep Unique File Names """
        files = [file.split('.hea')[0] for file in files if '.hea' in file]
        """ Prepare Dicts for Population """
        inputs[folder] = []
        outputs[folder] = []
        for file in files:
            """ Load Data File """
            filepath = os.path.join(folderpath,file)
            record = wfdb.rdsamp(filepath)
            lead_indices = np.where([ld in leads for ld in record[1]['sig_name']])[0]
            data = np.transpose(record[0])
            """ Resample and Store Data """
            lead_data = []
            for lead_index in lead_indices:
                current_data = data[lead_index,:]
                current_data = signal.resample(current_data,len(current_data)//2)
                lead_data.append(current_data)
            lead_data = np.array(lead_data)
            label = record[1]['comments'][4].split(': ')[1]
            
            if classification == 'binary':
                if label != 'Myocardial infarction' and label != 'Healthy control':
                    continue
            
            """ Iterate Through Lead Data Frame by Frame """
            nframes = lead_data.shape[1]//samples
            for nframe in range(nframes):
                frame = lead_data[0,samples*nframe:samples*(nframe+1)]
                inputs[folder].append(frame)
                outputs[folder].append(label)
                all_labels.append(label)
        
        """ Convert List into Array """
        input_array = np.array(inputs[folder])
        output_array = np.array(outputs[folder])
        
        if len(input_array) == 0:
            inputs.pop(folder)
            outputs.pop(folder)
        else:
            inputs[folder] = input_array
            outputs[folder] = output_array
    
    return inputs,outputs,all_labels

#%%
def encode_outputs(outputs,all_labels):
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
        
    return outputs
    
#%%
def save_arrays(inputs,outputs,leads):
    """ Make New Directory to Avoid Contamination """
    savepath = os.path.join(basepath,'patient_data','leads_%s' % leads)
    try:
        os.chdir(savepath)
    except:
        os.makedirs(savepath)

    """ Save Inputs and Labels Dicts For Splitting Later """
    with open(os.path.join(savepath,'ecg_signal_frames_ptb.pkl'),'wb') as f:
        pickle.dump(inputs,f)
    with open(os.path.join(savepath,'ecg_signal_labels_ptb.pkl'),'wb') as f:
        pickle.dump(outputs,f)
        
    print('Leads %s Saved!' % leads)

#%%
if __name__ == '__main__':
    for leads in leads_list:
        leads = [leads]
        inputs,outputs,all_labels = obtain_arrays(patient_folders,leads,classification)
        outputs = encode_outputs(outputs,all_labels)
        save_arrays(inputs,outputs,leads)                        
        
        
