#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 14:17:07 2019

@author: Dani Kiyasseh

Arrange Desired Dataset into Train/Val/Test Dicts for Training 

Inputs:
    ECG and PPG Frames and Labels

Outputs:
    Dicts of ECG and PPG Split According to Training Phase
"""

import os
import pickle
import random
from operator import itemgetter
import numpy as np
import pandas as pd
from tqdm import tqdm
import ast

from sklearn.decomposition import PCA
#%%
basepath = '/mnt/SecondaryHDD'
""" Database with Patient-Specific Info """
df = pd.read_csv(os.path.join(basepath,'PTB-XL','ptbxl_database.csv'),index_col='patient_id')

""" Binarize Devices to Use for Continual Learning Setting """
df['device'][df['device'].str.contains('AT')] = 'AT'
df['device'][df['device'].str.contains('CS')] = 'CS'

dataset = 'ptbxl'
trial = 'contrastive_ms' #contrastive_msml' # contrastive_ms' | 'contrastive_ml' | 'contrastive_msml' 'contrastive_ss' | '' #default as was used for AL and Cont. Learn Papers
print('Dataset: %s' % dataset)
peak_detection = False

def return_modified_df(df,code_of_interest):
    """ Filter Rows According to code_of_interest - patient noy have contained a certain label """
    codes_df = pd.read_csv(os.path.join(basepath,'PTB-XL','scp_statements.csv'))
    if code_of_interest == 'rhythm':
        codes = codes_df[codes_df['rhythm']==1]['Unnamed: 0'].tolist()
    elif code_of_interest == 'all':
        codes = codes_df['Unnamed: 0']
    """ Convert Str Dict into Dict """
    patient_labels = df['scp_codes'].apply(ast.literal_eval)
    """ Return List of Labels Without Confidence Values """
    patient_labels = patient_labels.apply(lambda x:list(x.keys()))
    """ Return Bool of Patients We Want """
    patient_labels_bool = patient_labels.apply(lambda x:any([entry in codes for entry in x]))
    """ Modified df According to code_of_interest """
    df = df.loc[patient_labels_bool]
    return df

def load_path(basepath,leads=['ii'],code_of_interest='all'):
    leads = leads
    path = os.path.join(basepath,'PTB-XL','patient_data',trial,'leads_%s' % leads,'classes_%s' % code_of_interest)
    label = ''
    return path,label

def determine_classification_setting(dataset_name):
    if dataset_name == 'physionet':
        classification = '5-way'
    elif dataset_name == 'bidmc':
        classification = '2-way'
    elif dataset_name == 'mimic': #change this accordingly
        classification = '2-way'
    elif dataset_name == 'cipa':
        classification = '7-way'
    elif dataset_name == 'cardiology':
        classification = '12-way'
    elif dataset_name == 'physionet2017':
        classification = '4-way'
    elif dataset_name == 'tetanus':
        classification = '2-way'
    elif dataset_name == 'ptb':
        classification = '2-way'
    elif dataset_name == 'fetal':
        classification = '2-way'
    elif dataset_name == 'physionet2016':
        classification = '2-way'
    elif dataset_name == 'physionet2020':
        classification = '9-way' #because binary multilabel
    elif dataset_name == 'ptbxl':
        classification = '71-way'

    return classification

def load_frames_and_labels(path,dataset,peak_detection,label=''):
    try:
        """ Load ECG Frames and Labels """
        with open(os.path.join(path,'ecg_signal_frames_%s.pkl' % dataset),'rb') as f:
            ecg_frames = pickle.load(f)
        
        with open(os.path.join(path,'ecg_signal_%slabels_%s.pkl' % (label,dataset)),'rb') as g:
            ecg_labels = pickle.load(g)
    except:
        ecg_frames = None
        ecg_labels = None

    return ecg_frames, ecg_labels

#%%
def remove_patients_with_empty_frames(ecg_frames):
    patients_with_empty_frames = [name for name,frames in ecg_frames.items() if np.array(frames).shape[0] == 0]

    if ecg_frames is not None:
        [ecg_frames.pop(key) for key in patients_with_empty_frames]
        [ecg_labels.pop(key) for key in patients_with_empty_frames]

#%%
def obtain_default_train_test_split(df,ecg_frames):
    """ Split Patients Into Train, Val, and Test """    
    test_folds = [10]
    val_folds = [9]
    train_folds = [0,1,2,3,4,5,6,7,8]
    
    test_condition0 = df['strat_fold'].isin(test_folds)
    val_condition0 = df['strat_fold'].isin(val_folds)
    train_condition0 = df['strat_fold'].isin(train_folds)
        
    """ Default Fold Split From Original Paper """
    patient_numbers_test = df[test_condition0].index.astype(int).unique().tolist() #bc patient may have multiple ecgs, we are only interested in unique patient ids
    patient_numbers_val = df[val_condition0].index.astype(int).unique().tolist()
    patient_numbers_train = df[train_condition0].index.astype(int).unique().tolist()
    
    return patient_numbers_train,patient_numbers_val,patient_numbers_test

def obtain_continual_train_test_split(df,devices):
    """ Split Patients Into Train, Val, and Test """    
    test_folds = [10]
    val_folds = [9]
    train_folds = [0,1,2,3,4,5,6,7,8]
    
    test_condition0 = df['strat_fold'].isin(test_folds)
    val_condition0 = df['strat_fold'].isin(val_folds)
    train_condition0 = df['strat_fold'].isin(train_folds)
    
    device_phase_patients = dict()
    device_phase_patients['train'] = dict()
    device_phase_patients['val'] = dict()
    device_phase_patients['test'] = dict()
    
    for device in devices:
        condition1 = df['device'] == device
        combined_test_condition = test_condition0 & condition1
        combined_val_condition = val_condition0 & condition1
        combined_train_condition = train_condition0 & condition1        
        
        patient_numbers_test = df[combined_test_condition].index.astype(int).tolist()
        patient_numbers_val = df[combined_val_condition].index.astype(int).tolist()
        patient_numbers_train = df[combined_train_condition].index.astype(int).tolist()
        
        device_phase_patients['train'][device] = patient_numbers_train
        device_phase_patients['val'][device] = patient_numbers_val
        device_phase_patients['test'][device] = patient_numbers_test
        
    return device_phase_patients

#%%
def obtain_patient_number_fraction_dict(fractions,patient_numbers_train,patient_numbers_val,patient_numbers_test):
    """ Obtain Patient-Level Fraction of Training Set As Labelled """
    labelled_patient_dict = {}
    unlabelled_patient_dict = {}
    labelled_patient_numbers_prev = 0
    for n,fraction in enumerate(fractions):
        if n == 0:
            labelled_length = int(len(patient_numbers_train)*fraction)
            random.seed(0)
            labelled_patient_numbers = random.sample(patient_numbers_train,labelled_length)
            unlabelled_patient_numbers = list(set(patient_numbers_train) - set(labelled_patient_numbers))
        else:
            patient_numbers_to_choose = list(set(patient_numbers_train) - set(labelled_patient_numbers_prev))
            current_fraction = fraction - fractions[n-1]
            labelled_length = int(len(patient_numbers_train)*current_fraction)
            random.seed(0)
            labelled_patient_numbers = random.sample(patient_numbers_to_choose,labelled_length)
            labelled_patient_numbers = labelled_patient_numbers + labelled_patient_numbers_prev
            unlabelled_patient_numbers = list(set(patient_numbers_train) - set(labelled_patient_numbers))
    
        labelled_patient_dict[fraction] = labelled_patient_numbers
        unlabelled_patient_dict[fraction] = unlabelled_patient_numbers
        
        labelled_patient_numbers_prev = labelled_patient_numbers
    
    return fractions, labelled_patient_dict, unlabelled_patient_dict

def change_labels(dataset_name,header,noise_level,noise_type,frames,labels):
    """ Introduce Noise to Labels @ Different Intensity Levels 
    Frames represent all frames for the 'unlabelled' dataset
    Labels represent all labels for the 'unlabelled' dataset """
    if header == 'unlabelled' and noise_type is not None:
        nlabels = labels.shape[0]
        nlabels_to_switch = int(nlabels*noise_level)
        random.seed(0)
        label_indices_to_switch = random.sample(list(np.arange(nlabels)),nlabels_to_switch)
        for index in label_indices_to_switch:
            original_label = labels[index]
            classification = determine_classification_setting(dataset_name)
            nclasses = int(classification.split('-')[0])
            class_set = set(np.arange(nclasses))
            remaining_class_set = list(class_set - set([original_label]))
            if noise_type == 'random':
                random.seed(0)
                new_label = random.sample(remaining_class_set,1)[0]
            elif noise_type == 'nearest_neighbour':
                pca = PCA(n_components=2)
                pca_frames = pca.fit_transform(frames)
                distance_matrix = np.linalg.norm(pca_frames - pca_frames[:,None], axis=-1)
                distance_matrix[distance_matrix==0] = 1e9 #to avoid choosing diagonal entry
                closest_indices = np.argmin(distance_matrix,1)
                new_labels = labels[closest_indices]
                new_label = new_labels[index]
                
            labels[index] = new_label

    return labels

#%%
def obtain_default_arrays(dataset_name,fractions,ecg_frames,ecg_labels,labelled_patient_dict,unlabelled_patient_dict,noise_type=None,noise_level=None):
    """ Split Data Into Phases and Save Into Dicts """
    modalities = ['ecg']#,'ppg'] #change depending on modality (or both)
    #phases = ['train','val','test']
    
    frames_dict = dict()
    labels_dict = dict()
    pid_dict = dict()
    
    for modality in modalities:
        
        frames_dict[modality] = dict()
        labels_dict[modality] = dict()
        pid_dict[modality] = dict()
        
        modality_frames = ecg_frames
        modality_labels = ecg_labels
            
        nframes_per_patient = [array.shape[0] for array in list(modality_labels.values())]
        nframes_per_patient_dict = dict(zip(modality_labels.keys(),nframes_per_patient))
            
        for fraction in tqdm(fractions):
            
            train_labelled_patients = labelled_patient_dict[fraction]
            train_unlabelled_patients = unlabelled_patient_dict[fraction]
                                    
            frames_dict[modality][fraction] = dict()
            labels_dict[modality][fraction] = dict()
            pid_dict[modality][fraction] = dict()
            
            frames_dict[modality][fraction]['train'] = dict()
            labels_dict[modality][fraction]['train'] = dict()
            pid_dict[modality][fraction]['train'] = dict()
            
            train_headers = ['labelled','unlabelled']
            train_patients = [train_labelled_patients,train_unlabelled_patients]
            for header,patient_numbers in zip(train_headers,train_patients):
                if len(patient_numbers) == 1:
                    frames = np.array(modality_frames[patient_numbers[0]])
                    frames_dict[modality][fraction]['train'][header] = frames
                    
                    labels = np.array(modality_labels[patient_numbers[0]])
                    labels = change_labels(dataset_name,header,noise_level,noise_type,frames,labels)
                    labels_dict[modality][fraction]['train'][header] = labels
                    
                    pid = [patient_numbers[0] for _ in range(nframes_per_patient_dict[patient_numbers[0]])]
                    pid_dict[modality][fraction]['train'][header] = pid
                elif len(patient_numbers) > 1: 
                    frames = np.concatenate(list(itemgetter(*patient_numbers)(modality_frames)))
                    frames_dict[modality][fraction]['train'][header] = frames
                    
                    labels = np.concatenate(list(itemgetter(*patient_numbers)(modality_labels)))
                    labels = change_labels(dataset_name,header,noise_level,noise_type,frames,labels)
                    labels_dict[modality][fraction]['train'][header] = labels
                    
                    pid = [patient_number for patient_number in patient_numbers for _ in range(nframes_per_patient_dict[patient_number])]
                    pid_dict[modality][fraction]['train'][header] = pid
            
            remaining_phases = ['val','test']
            remaining_patients = [patient_numbers_val,patient_numbers_test]
            remaining_content = dict(zip(remaining_phases,remaining_patients))
            
            for phase,patient_numbers in remaining_content.items():
                if len(patient_numbers) == 1:
                    frames_dict[modality][fraction][phase] = np.array(modality_frames[patient_numbers[0]])
                    labels_dict[modality][fraction][phase] = np.array(modality_labels[patient_numbers[0]])
                    pid = [patient_numbers[0] for _ in range(nframes_per_patient_dict[patient_numbers[0]])]
                    pid_dict[modality][fraction][phase] = pid
                elif len(patient_numbers) > 1:                
                    frames_dict[modality][fraction][phase] = np.concatenate(list(itemgetter(*patient_numbers)(modality_frames))) #indices #list(itemgetter(*indices)(patient_number_list))
                    labels_dict[modality][fraction][phase] = np.concatenate(list(itemgetter(*patient_numbers)(modality_labels)))
                    pid = [patient_number for patient_number in patient_numbers for _ in range(nframes_per_patient_dict[patient_number])]
                    pid_dict[modality][fraction][phase] = pid
    
    return frames_dict,labels_dict,pid_dict

#%%

def obtain_continual_arrays(devices,ecg_frames,ecg_labels,device_phase_patients):
    #sampling_rate = 500
    phases = ['train','val','test']
    modality_list = ['ecg']
    fraction_list = [1]
    inputs_dict = dict()
    outputs_dict = dict()
    pids = dict()

    for modality in modality_list:
        inputs_dict[modality] = dict()
        outputs_dict[modality] = dict()
        pids[modality] = dict()
        
        """ Rename Inputs and Outputs """
        modality_frames = ecg_frames
        modality_labels = ecg_labels
        
        for fraction in fraction_list:
            inputs_dict[modality][fraction] = dict()
            outputs_dict[modality][fraction] = dict()
            pids[modality][fraction] = dict()
            for phase in phases:
                inputs_dict[modality][fraction][phase] = dict()
                outputs_dict[modality][fraction][phase] = dict()
                pids[modality][fraction][phase] = dict()
                for device in tqdm(devices):
                    current_patients = device_phase_patients[phase][device]
                    current_inputs = list(itemgetter(*current_patients)(modality_frames)) #might have to concatenate
                    current_outputs = list(itemgetter(*current_patients)(modality_labels)) #might have to concatenate

                    inputs_dict[modality][fraction][phase][device] = np.concatenate(current_inputs)
                    outputs_dict[modality][fraction][phase][device] = np.concatenate(current_outputs)
                    pids[modality][fraction][phase][device] = np.array(current_patients)
    
    return inputs_dict,outputs_dict,pids

#%%
def make_directory(path,noise_level=None):
    if noise_level is not None:
        path = os.path.join(path,'noise_level_%.2f' % noise_level)
        try:
            os.chdir(path)
        except:
            os.makedirs(path)
    return path

def save_final_frames_and_labels(frames_dict,labels_dict,pid_dict,path,peak_detection,noise_level=None,setting='default'):
    if setting == 'continual':
        path2 = setting
    else:
        path2 = ''
    
    savepath = os.path.join(path,path2)
    if os.path.isdir(savepath) == False:
        os.makedirs(savepath)
    """ Save Frames and Labels Dicts """
    with open(os.path.join(savepath,'frames_phases_%s.pkl' % dataset),'wb') as f:
        pickle.dump(frames_dict,f,protocol=4) #4 allows you to save larger files
    
    with open(os.path.join(savepath,'labels_phases_%s.pkl' % (dataset)),'wb') as g:
        pickle.dump(labels_dict,g,protocol=4)

    with open(os.path.join(savepath,'pid_phases_%s.pkl' % (dataset)),'wb') as h:
        pickle.dump(pid_dict,h,protocol=4)
    
    print('Final Frames Saved!')
#%%
if __name__ == '__main__':
    setting = 'default' #default' # 'default' | 'continual' 
    fractions = [1] #[0.1,0.3,0.5,0.7,0.9]
    leads_list = [['II','V2','aVL','aVR']] #[['I','II','III','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6']] #['i','ii','iii','avr','avl','avf','v1','v2','v3','v4','v5','v6'] #for ptb dataset
    code_of_interest = 'rhythm'
    df = return_modified_df(df,code_of_interest)
    """ Noisy Label Formulation """
    noise_type = None #random' OR 'nearest_neighbour' OR None
    leads = None 
    """ End Noisy Label Formulation """
    for leads in leads_list:
    #for noise_level in noise_level_list:
        path, label = load_path(basepath,leads,code_of_interest)
        ecg_frames, ecg_labels = load_frames_and_labels(path,dataset,peak_detection,label)
        
        remove_patients_with_empty_frames(ecg_frames)
        if setting == 'default': #generate data for default training (includes contrastive)
            patient_numbers_train, patient_numbers_val, patient_numbers_test = obtain_default_train_test_split(df,ecg_frames)
            fractions, labelled_patient_dict, unlabelled_patient_dict = obtain_patient_number_fraction_dict(fractions,patient_numbers_train,patient_numbers_val,patient_numbers_test)
            frames_dict, labels_dict, pid_dict = obtain_default_arrays(dataset,fractions,ecg_frames,ecg_labels,labelled_patient_dict,unlabelled_patient_dict)#noise_type,noise_level
        elif setting == 'continual': #generate data for CL setting
            devices = ['CS','AT']
            device_phase_patients = obtain_continual_train_test_split(df,devices)
            frames_dict, labels_dict, pid_dict = obtain_continual_arrays(devices,ecg_frames,ecg_labels,device_phase_patients)
        """ Save Data """
        path = make_directory(path)#,noise_level)
        save_final_frames_and_labels(frames_dict,labels_dict,pid_dict,path,peak_detection,setting=setting)#,noise_level)

