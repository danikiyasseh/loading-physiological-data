#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 12:13:14 2020

@author: scro3517
"""
import numpy as np
import os
import pickle
import wfdb
from tqdm import tqdm
import pandas as pd
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
import ast

#%%
basepath = '/mnt/SecondaryHDD/PTB-XL'

fs = 500
""" File Names to Load """
folders1 = os.listdir(basepath)
pth_to_folders = [folder for folder in folders1 if 'records500' in folder] #records100 contains 100Hz data
pth_to_folders = [os.path.join(pth_to_folder,fldr) for pth_to_folder in pth_to_folders for fldr in os.listdir(os.path.join(basepath,pth_to_folder))]
files = [os.path.join(pth_to_folder,fldr) for pth_to_folder in pth_to_folders for fldr in os.listdir(os.path.join(basepath,pth_to_folder))]
files = [file.split('.hea')[0] for file in files if '.hea' in file]
""" Database with Patient-Specific Info """
df = pd.read_csv(os.path.join(basepath,'ptbxl_database.csv'))
""" Database with Label Information """
codes_df = pd.read_csv(os.path.join(basepath,'scp_statements.csv'))

""" Identify Codes of Interest """
code_of_interest = 'rhythm' # options: 'rhythm' | 'all' 

if code_of_interest == 'rhythm':
    encoder = LabelBinarizer() #because we may still have multi-output rhythm classification #LabelEncoder()
    codes = codes_df[codes_df['rhythm']==1]['Unnamed: 0'].tolist()
elif code_of_interest == 'all':
    encoder = LabelBinarizer()
    codes = codes_df['Unnamed: 0']

""" Fit Binarizer to Codes for Conversion Later """
encoder.fit(codes)
ncodes = len(encoder.classes_)

def generate_inputs_and_outputs(leads_of_interest,samples_to_take,trial):
    inputs = dict()
    outputs = dict()
    for file in tqdm(files):
        data = wfdb.rdsamp(os.path.join(basepath,file))
        leads_data = data[0].transpose()
        leads_names = data[1]['sig_name']

        #patient_name = int(file.split('/')[-1].split('_hr')[0]) #lr for 100 Hz setting
        patient_name = df[df['filename_hr']==file].patient_id.astype(int).item()
        filedata = df[df['filename_hr']==file]        
        labels_and_confidence = filedata['scp_codes'].item()
        """ Convert string of dictionary to dictionary """
        labels_and_confidence = ast.literal_eval(labels_and_confidence)
        labels = list(labels_and_confidence.keys())
        
        """ Only Include Patient if Label Falls Within Set of Labels of Interest """
        label_present_bool = [label in codes for label in labels]
        if any(label_present_bool): #if at least one of the labels are present
            
            """ Keep Labels of Interest """
            if code_of_interest in ['rhythm']:
                label_indices = np.where(label_present_bool)[0].tolist()
                labels = np.array(labels)[label_indices].tolist()
            
            """ Convert Labels into List of Lists of OHE Vectors """
            list_of_ohe = encoder.transform(labels)
            
            """ Generate Multi-Hot Vector """
            multi_label_output = np.zeros((ncodes))
            for entry in list_of_ohe:
                multi_label_output += entry
            
            if patient_name in inputs.keys():
                current_inputs = list(inputs[patient_name])
                current_outputs = list(outputs[patient_name])
            else:
                current_inputs = []
                current_outputs = []
                
            nframes = data[1]['sig_len']//samples_to_take
            lead_indices = np.where(np.in1d(leads_names,leads_of_interest))[0]
            leads_data = leads_data[lead_indices,:]
            
            if trial in ['contrastive_ml','contrastive_msml']:
                for i in range(nframes):
                    lead_frames = leads_data[:,i*samples_to_take:(i+1)*samples_to_take]
                    lead_frames = lead_frames.transpose() #2500x12
                    current_inputs.append(lead_frames)
                    current_outputs.append(multi_label_output)
            else:
                for name,lead in zip(leads_of_interest,leads_data):
                    for i in range(nframes):
                        frame = lead[i*samples_to_take:(i+1)*samples_to_take]
                        current_inputs.append(frame)
                        current_outputs.append(multi_label_output)
            
            inputs[patient_name] = np.array(current_inputs)
            outputs[patient_name] = np.array(current_outputs)
    
    return inputs,outputs

#%%
def obtain_samples_to_take(trial):
    if trial == 'contrastive_ms':
        samples_to_take = 5000
    elif trial == 'contrastive_ml':
        samples_to_take = 2500
    elif trial == 'contrastive_msml':
        samples_to_take = 5000
    elif trial == 'contrastive_ss':
        samples_to_take = 2500
    else: #default setting
        samples_to_take = 2500
    return samples_to_take

#%%
def makepath_and_save_data(inputs,outputs,leads_of_interest,trial='',code_of_interest='all'):
    savepath = os.path.join(basepath,'patient_data',trial,'leads_%s' % str(leads_of_interest),'classes_%s' % code_of_interest)
    try:
        os.chdir(savepath)
    except:
        os.makedirs(savepath)
    
    with open(os.path.join(savepath,'ecg_signal_frames_ptbxl.pkl'),'wb') as f:
        pickle.dump(inputs,f)
    with open(os.path.join(savepath,'ecg_signal_labels_ptbxl.pkl'),'wb') as f:
        pickle.dump(outputs,f)
    print('Saved!')
#%%
leads_of_interest = [['II','V2','aVL','aVR']] #[['I','II','III','aVL','aVR','aVF','V1','V2','V3','V4','V5','V6']] #list of lists regardless of number of leads
trial = 'contrastive_ms' # 'contrastive_ms' | 'contrastive_ml' | 'contrastive_msml' | 'contrastive_ss' | '' #default
for leads in leads_of_interest:
    samples_to_take = obtain_samples_to_take(trial)
    inputs,outputs = generate_inputs_and_outputs(leads,samples_to_take,trial)
    makepath_and_save_data(inputs,outputs,leads,trial=trial,code_of_interest=code_of_interest)

