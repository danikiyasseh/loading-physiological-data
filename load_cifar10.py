#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 09:56:52 2020

@author: Dani Kiyasseh

Prepare CIFAR10 Data for ALPS
"""

""" Download CIFAR10 """
#import torchvision
#torchvision.datasets.CIFAR10('/mnt/SecondaryHDD', train=True, transform=None, target_transform=None, download=True)

import os
import numpy as np
import pickle
import random
#%%
dataset = 'cifar10'
basepath = '/mnt/SecondaryHDD/cifar-10-python/cifar-10-batches-py'
files = os.listdir(basepath)
files = [file for file in files if 'data' in file]

all_inputs = []
all_outputs = []
for file in files:
    with open(os.path.join(basepath,file),'rb') as f:
        data_dict = pickle.load(f,encoding='bytes')
    
    data_array = data_dict[b'data']
    image_array = data_array.reshape((-1,3,32,32))
    label_array = data_dict[b'labels']
    
    all_inputs.extend(image_array)
    all_outputs.extend(label_array)

""" Convert List to Array """
all_inputs = np.array(all_inputs)
all_outputs = np.array(all_outputs)

#%%
""" Testing Data """
files = os.listdir(basepath)
files = [file for file in files if 'test' in file]

test_inputs = []
test_outputs = []

for file in files:
    with open(os.path.join(basepath,file),'rb') as f:
        data_dict = pickle.load(f,encoding='bytes')
    
    data_array = data_dict[b'data']
    image_array = data_array.reshape((-1,3,32,32))
    label_array = data_dict[b'labels']
    
    test_inputs.extend(image_array)
    test_outputs.extend(label_array)

""" Convert List to Array """
test_inputs = np.array(test_inputs)
test_outputs = np.array(test_outputs)  

#%%
train_ratio,val_ratio = 0.80,0.20
train_nsamples = int(train_ratio*all_inputs.shape[0])
random.seed(0)
shuffled_indices = random.sample(list(np.arange(all_inputs.shape[0])),all_inputs.shape[0])
train_indices,val_indices = shuffled_indices[:train_nsamples], shuffled_indices[train_nsamples:]
phases = ['train','val','test']

modality_list = ['image']
fraction_list = [0.1,0.3,0.5,0.7,0.9]

frames_dict = dict()
labels_dict = dict()
for modality in modality_list:
    frames_dict[modality] = dict()
    labels_dict[modality] = dict()
    #pids_dict[modality] = dict()
    for fraction in fraction_list:
        frames_dict[modality][fraction] = dict()
        labels_dict[modality][fraction] = dict()
        #pids_dict[modality][fraction] = dict()      
        for phase in phases:
            if phase == 'train':
                frames_dict[modality][fraction][phase] = dict()
                labels_dict[modality][fraction][phase] = dict()
                #pids_dict[modality][fraction][phase] = dict()      
                """ Training Subset """
                train_data = all_inputs[train_indices,:] #take first X amount
                train_labels = all_outputs[train_indices]
                #train_pids = all_pids[:nsamples]
                """ Laballed Subset """
                nsamples = len(train_indices)
                labelled_fraction = int(nsamples*fraction)
                labelled_data = train_data[:labelled_fraction]
                labelled_labels = train_labels[:labelled_fraction]
                #labelled_pids = train_pids[:labelled_fraction]
                frames_dict[modality][fraction][phase]['labelled'] = labelled_data
                labels_dict[modality][fraction][phase]['labelled'] = labelled_labels
                #pids_dict[modality][fraction][phase]['labelled'] = labelled_pids
                """ Unlabelled Subset """
                unlabelled_data = train_data[labelled_fraction:]
                unlabelled_labels = train_labels[labelled_fraction:]
                #unlabelled_pids = train_pids[labelled_fraction:]
                frames_dict[modality][fraction][phase]['unlabelled'] = unlabelled_data
                labels_dict[modality][fraction][phase]['unlabelled'] = unlabelled_labels
                #pids_dict[modality][fraction][phase]['unlabelled'] = unlabelled_pids
            elif phase == 'val':
                val_data = all_inputs[val_indices,:] #take final Y amount
                val_labels = all_outputs[val_indices]
                #val_pids = all_pids[-nsamples:]
                
                frames_dict[modality][fraction][phase] = val_data
                labels_dict[modality][fraction][phase] = val_labels
                #pids_dict[modality][fraction][phase] = val_pids            
                
            elif phase == 'test':
                frames_dict[modality][fraction][phase] = test_inputs
                labels_dict[modality][fraction][phase] = test_outputs
                #pids_dict[modality][fraction][phase] = test_pids

#%%
""" Save Dicts """
names = ['frames','labels']
final_dicts = dict(zip(names,[frames_dict,labels_dict]))

for item_name,item_dict in final_dicts.items():
    with open(os.path.join(basepath,'%s_phases_%s.pkl' % (item_name,dataset)),'wb') as f:
        pickle.dump(item_dict,f)
print('Final Frames Saved!')





















