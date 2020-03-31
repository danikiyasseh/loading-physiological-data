#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 08:08:07 2020

@author: scro3517
"""

import os
import numpy as np
import pandas as pd
import pickle
from scipy.signal import resample
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
enc = LabelEncoder()
#%%
dataset = 'chapman'
basepath = '/mnt/SecondaryHDD/chapman_ecg'
trial = 'contrastive_msml' # '' | 'contrastive_ms' | 'contrastive_ml' | 'contrastive_msml' | 'contrastive_ss'

files = os.listdir(os.path.join(basepath,'ECGDataDenoised'))
database = pd.read_csv(os.path.join(basepath,'Diagnostics.csv'))
dates = database['FileName'].str.split('_',expand=True).iloc[:,1]
dates.name = 'Dates'
dates = pd.to_datetime(dates)
database_with_dates = pd.concat((database,dates),1)
#""" Unique Dates in Database """
#enc.fit(dates)

""" Combine Rhythm Labels """
old_rhythms = ['AF','SVT','ST','AT','AVNRT','AVRT','SAAWR','SI','SA']
new_rhythms = ['AFIB','GSVT','GSVT','GSVT','GSVT','GSVT','GSVT','SR','SR']
database_with_dates['Rhythm'] = database_with_dates['Rhythm'].replace(old_rhythms,new_rhythms)
unique_labels = database_with_dates['Rhythm'].value_counts().index.tolist()
enc.fit(unique_labels)

""" Combine Dates """
def combine_dates(date):
    new_dates = ['All Terms']#use this for continual learning dataset ['Term 1','Term 2','Term 3']
    cutoff_dates = ['2019-01-01']##use this for continual learning dataset ['2018-01-16','2018-02-09','2018-12-30']
    cutoff_dates = [pd.Timestamp(date) for date in cutoff_dates]
    for t,cutoff_date in enumerate(cutoff_dates):
        if date < cutoff_date:
            new_date = new_dates[t]
            break
    return new_date
database_with_dates['Dates'] = database_with_dates['Dates'].apply(combine_dates)

#%%
""" Look at Label Composition """
#""" GroupBy output can be treated as an interable """
groupby_dates = database_with_dates.groupby('Dates')
groupby_dates['Rhythm'].value_counts()

#%%
""" Patients in Each Task and Phase """
phases = ['train','val','test']
phase_fractions = [0.6, 0.2, 0.2]
phase_fractions_dict = dict(zip(phases,phase_fractions))
terms = ['All Terms']##use this for continual learning dataset ['Term 1','Term 2','Term 3']

term_phase_patients = dict()
for term in terms:
    term_phase_patients[term] = dict()
    term_patients = database_with_dates['FileName'][database_with_dates['Dates'] == term]
    random_term_patients = term_patients.sample(frac=1,random_state=0)
    start = 0
    for phase,fraction in phase_fractions_dict.items():
        if phase == 'test':
            phase_patients = random_term_patients.iloc[start:].tolist() #to avoid missing last patient due to rounding
        else:
            npatients = int(fraction*len(term_patients))
            phase_patients = random_term_patients.iloc[start:start+npatients].tolist()
        term_phase_patients[term][phase] = phase_patients
        start += npatients

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

sampling_rate = 500
modality_list = ['ecg']
fraction_list = [1]
leads = ['I','II','III','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6']
desired_leads = ['II','V2','aVL','aVR'] #['I','II','III','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6']
inputs_dict = dict()
outputs_dict = dict()
pids = dict()

for modality in modality_list:
    inputs_dict[modality] = dict()
    outputs_dict[modality] = dict()
    pids[modality] = dict()
    for fraction in fraction_list:
        inputs_dict[modality][fraction] = dict()
        outputs_dict[modality][fraction] = dict()
        pids[modality][fraction] = dict()
        for phase in phases:
            inputs_dict[modality][fraction][phase] = dict()
            outputs_dict[modality][fraction][phase] = dict()
            pids[modality][fraction][phase] = dict()
            for term in terms:
                current_patients = term_phase_patients[term][phase]
                current_inputs = []
                current_outputs = []
                current_pids = []
                for patient in tqdm(current_patients):
                    filename = patient + '.csv'
                    data = pd.read_csv(os.path.join(basepath,'ECGDataDenoised',filename)) #SxL
                    
                    resampling_length = obtain_resampling_length(trial)
                    data_resampled = resample(data,resampling_length)
                    data_resampled = data_resampled.T #12x2500
                    lead_indices = np.where(np.in1d(leads,desired_leads))[0]
                    data_resampled = data_resampled[lead_indices,:] #12x2500
                    
                    label = database_with_dates['Rhythm'][database_with_dates['FileName']==patient]
                    encoded_label = enc.transform(label).item()
                    
                    if trial in ['contrastive_ml','contrastive_msml']:
                        data_resampled = data_resampled.T #2500x12
                        current_inputs.append(data_resampled)
                        current_outputs.append(encoded_label)
                        current_pids.append(patient)
                    else:
                        current_inputs.append(data_resampled)
                        current_outputs.append([encoded_label for _ in range(data_resampled.shape[0])])
                        current_pids.append([patient for _ in range(data_resampled.shape[0])])
                    
                inputs_dict[modality][fraction][phase][term] = np.array(current_inputs)
                outputs_dict[modality][fraction][phase][term] = np.array(current_outputs)
                pids[modality][fraction][phase][term] = np.array(current_pids)

#%%
""" Remove Entries with NaNs - Appears to be Entire Rows """
for phase in phases:
    for term in terms:
        good_indices = np.unique(np.where(~np.isnan(inputs_dict['ecg'][1][phase][term]))[0])
        inputs_dict['ecg'][1][phase][term] = inputs_dict['ecg'][1][phase][term][good_indices]
        outputs_dict['ecg'][1][phase][term] = outputs_dict['ecg'][1][phase][term][good_indices]
        pids['ecg'][1][phase][term] = pids['ecg'][1][phase][term][good_indices]

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

savepath = os.path.join(basepath,trial,'leads_%s' % str(desired_leads))
if os.path.isdir(savepath) == False:
    os.makedirs(savepath)
save_final_frames_and_labels(inputs_dict,outputs_dict,savepath,dataset)



