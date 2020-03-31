#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 20:07:07 2020

@author: scro3517
"""

import os 
import numpy as np
import pandas as pd
import random
from tqdm import tqdm 
from scipy.signal import resample
import pickle

from sklearn.preprocessing import LabelEncoder
enc = LabelEncoder()
enc.fit(['Left','Right'])
#%%
dataset = 'chapman_pvc'
basepath = '/mnt/SecondaryHDD/PVCVTECGData/'
files = os.listdir(basepath)
files = [file for file in files if '.csv' in file]
labels = pd.read_csv(os.path.join(basepath,'labels','Diagnosis.csv'))

random.seed(0)
phases = ['train','val','test']
phase_ratios = [0.6,0.2,0.2] 
files_shuffled = random.sample(files,len(files))

""" Place Patients into Phases """
phase_patients = dict()
start_index = 0
for phase,ratio in zip(phases,phase_ratios):
    npatients = int(ratio*len(files_shuffled))
    if 'test' in phase:
        npatients = len(files_shuffled) #to avoid missing any patients due to rounding errors
    phase_patients[phase] = files_shuffled[start_index:start_index+npatients]
    start_index += npatients

#%%

def obtain_resampling_length(trial):
    if trial == 'contrastive_ms':
        resampling_length = 5000
    elif trial == 'contrastive_ml':
        resampling_length = 2500
    elif trial == 'contrastive_msml':
        resampling_length = 5000
    elif trial == 'contrastive_ss':
        resampling_length = 2500
    else: #default setting
        resampling_length = 2500
    return resampling_length   

sampling_rate = 2000
modality_list = ['ecg']
fraction_list = [1]
desired_leads = ['I','II','III','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6'] #['II','V2','aVL','aVR'] #['I','II','III','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6']
inputs_dict = dict()
outputs_dict = dict()
pids = dict()

def obtain_arrays(trial):
    for modality in modality_list:
        inputs_dict[modality] = dict()
        outputs_dict[modality] = dict()
        pids[modality] = dict()
        for fraction in fraction_list:
            inputs_dict[modality][fraction] = dict()
            outputs_dict[modality][fraction] = dict()
            pids[modality][fraction] = dict()
            for phase in tqdm(phases):            
                current_patients = phase_patients[phase]
                current_inputs = [] 
                current_outputs = [] 
                current_pids = []
                for patient in tqdm(current_patients):
                    data = pd.read_csv(os.path.join(basepath,patient)) #SxL
                    leads = data.columns.tolist()
                    
                    resampling_length = obtain_resampling_length(trial)
                    data_resampled = resample(data,data.shape[0]//4) #downsample all ECG from 2000Hz to 500Hz regardless
                    data_resampled = data_resampled.T #12x2500
                    lead_indices = np.where(np.in1d(leads,desired_leads))[0]
                    data_resampled = data_resampled[lead_indices,:]
                    data_resampled = data_resampled.T #2500xL
                    
                    patient_id = int(patient.strip('.csv'))
                    encoded_label = enc.transform(labels[labels['HospitalID']==patient_id]['LeftRight']).item()
                    
                    if trial in ['contrastive_ml','contrastive_msml']:
                        if data_resampled.shape[0] < 5000:
                            data_resampled = resample(data_resampled,resampling_length)
                            current_inputs.append(data_resampled) #2500x12
                            current_outputs.append(encoded_label)
                            current_pids.append(patient)
                        else:
                            nsegments = data_resampled.shape[0]//resampling_length
                            for n in range(nsegments):
                                start = n*resampling_length
                                current_frame = data_resampled[start:start+resampling_length,:]
                                start += resampling_length
                                current_inputs.append(current_frame)
                                current_outputs.append(encoded_label)
                                current_pids.append(patient)
                    else:
                        if data_resampled.shape[0] < 5000:
                            data_resampled = resample(data_resampled,resampling_length)
                            data_resampled = data_resampled.T #12x2500
                            current_inputs.append(data_resampled)
                            current_outputs.append([encoded_label for _ in range(data_resampled.shape[0])])
                            current_pids.append([patient for _ in range(data_resampled.shape[0])])
                        else:
                            nsegments = data_resampled.shape[0]//resampling_length
                            data_resampled = data_resampled.T #12x2500
                            for n in range(nsegments):
                                start = n*resampling_length
                                current_frame = data_resampled[:,start:start+resampling_length]
                                start += resampling_length
                                current_inputs.append(current_frame)
                                current_outputs.append([encoded_label for _ in range(current_frame.shape[0])])
                                current_pids.append([patient for _ in range(current_frame.shape[0])])
                
                inputs_dict[modality][fraction][phase] = np.concatenate(current_inputs)
                outputs_dict[modality][fraction][phase] = np.concatenate(current_outputs)
                pids[modality][fraction][phase] = np.concatenate(current_pids)
    
    return inputs_dict,outputs_dict,pids

#%%
""" Remove Entries with NaNs - Appears to be Entire Rows """
def remove_nan_entries(inputs_dict,outputs_dict,pids):
    for phase in phases:
        good_indices = np.unique(np.where(~np.isnan(inputs_dict['ecg'][1][phase]))[0])
        inputs_dict['ecg'][1][phase] = inputs_dict['ecg'][1][phase][good_indices]
        outputs_dict['ecg'][1][phase] = outputs_dict['ecg'][1][phase][good_indices]
        pids['ecg'][1][phase] = pids['ecg'][1][phase][good_indices]
    return inputs_dict,outputs_dict,pids

#%%
def save_final_frames_and_labels(frames_dict,labels_dict,path,dataset):
    """ Save Frames and Labels Dicts """
    with open(os.path.join(path,'frames_phases_%s.pkl' % dataset),'wb') as f:
        pickle.dump(frames_dict,f)
    
    with open(os.path.join(path,'labels_phases_%s.pkl' % (dataset)),'wb') as g:
        pickle.dump(labels_dict,g)
        
    with open(os.path.join(path,'pid_phases_%s.pkl' % (dataset)),'wb') as h:
        pickle.dump(pids,h)
    print('Final Frames Saved!')

#%%
trial = 'contrastive_ss' # '' | 'contrastive_ms' | 'contrastive_ml' | 'contrastive_msml' | 'contrastive_ss'
savepath = os.path.join(basepath,trial,'leads_%s' % str(desired_leads))
if os.path.isdir(savepath) == False:
    os.makedirs(savepath)
    
if __name__ == '__main__':
    inputs_dict,outputs_dict,pids = obtain_arrays(trial)
    inputs_dict,outputs_dict,pids = remove_nan_entries(inputs_dict,outputs_dict,pids)
    save_final_frames_and_labels(inputs_dict,outputs_dict,savepath,dataset)


