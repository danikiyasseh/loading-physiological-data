#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 12:57:20 2019

@author: Dani Kiyasseh

Load CIPA Data into Patient-Specific Input and Output Dicts 
"""

import wfdb
import numpy as np
import pandas as pd
import pickle
import os
from scipy.signal import resample
#%%
#basepath = '/home/scro3517/Desktop/cipa-ecg-validation-study-1.0.0'

""" Clinical Database """
adeg = pd.read_csv(os.path.join(basepath,'adeg.csv'))
""" ECG ID """
ecg_id = adeg['EGREFID']
fs = 1000
old_samples = 10000
samples_to_take = 2500
new_fs = fs/(old_samples/samples_to_take)

nframes = old_samples//samples_to_take

""" Desired Lead """
lead = ['II','aVR'] #'all'
""" Return Patients Who Have Actually Taken Drug (No Placebo) """
#adeg = adeg[adeg['TRTA'] != 'Placebo']
""" Remove Rows with nan in ECG Reference ID """
adeg = adeg[adeg['EGREFID'].notna()]
""" Return Unique EGREFID and Drug """
adeg_drug_and_ecg = adeg.groupby(['EGREFID','TRTA','USUBJID'],as_index=False)['APERIOD'].sum()
""" Return Unique Patient Numbers """
unique_patient_numbers = adeg_drug_and_ecg['USUBJID'].unique()

""" Save Frames and Labels in Patient-Specific Dict """
inputs = dict()
outputs = dict()
all_labels = []
for patient_number in unique_patient_numbers:
    patient_specific_ecg_ids = adeg_drug_and_ecg[adeg_drug_and_ecg['USUBJID'] == patient_number]
    inputs[patient_number] = []
    outputs[patient_number] = []
    for ecg_id in patient_specific_ecg_ids['EGREFID']:
        path = os.path.join(basepath,'raw',str(patient_number))
        record_info = wfdb.rdsamp(os.path.join(path,ecg_id))

        """ Obtain Drug for This Sitting """
        drug = patient_specific_ecg_ids['TRTA'][patient_specific_ecg_ids['EGREFID'] == ecg_id].item()
        
        ecg_leads = record_info[0] #10000x12
        """ Transpose Data to Get 12x10000 """
        ecg_leads = np.transpose(ecg_leads)
        
#        """ Choose Specific Lead or All Leads """
#        if lead != 'all':
#            lead_index = np.where([lead == el for el in record_info[1]['sig_name']])[0].item()
#            ecg_leads = ecg_leads[lead_index,:]
#            """ Downsample Frame """
#            ecg_leads = resample(ecg_leads,2500) #axis=1
#            drugs = drug
#        else:
#            ecg_leads = resample(ecg_leads,2500,axis=1)
#            drugs = np.repeat(drug,ecg_leads.shape[0])
        """ Determine Lead Indices """
        if lead != 'all':
            lead_indices = [np.where([ld == el for el in record_info[1]['sig_name']])[0].item() for ld in lead]
        else:
            lead_indices = np.arange(ecg_leads.shape[0])
        
        """ Obtain Frames and Labels from Each Lead and Frame """
        for lead_index in lead_indices:
            current_ecg_lead = ecg_leads[lead_index,:]
            for nframe in range(nframes):
                start_sample = nframe * samples_to_take
                end_sample = start_sample + samples_to_take
                mini_frame = current_ecg_lead[start_sample:end_sample]
                mini_label = drug
                
                """ Append Data to Inputs and Outputs """
                inputs[patient_number].append(mini_frame)
                outputs[patient_number].append(mini_label)
                all_labels.append(mini_label)
        
    """ Convert to Arrays """
    inputs[patient_number] = np.array(inputs[patient_number])
    outputs[patient_number] = np.array(outputs[patient_number])
#%%
""" Retrieve Unique Drug Names """
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
savepath = os.path.join(basepath,'leads_%s' % lead)
try:
    os.chdir(savepath)
except:
    os.makedirs(savepath)

#%%
""" Save Inputs and Labels Dicts For Splitting Later """
with open(os.path.join(savepath,'ecg_signal_frames_cipa.pkl'),'wb') as f:
    pickle.dump(inputs,f)
with open(os.path.join(savepath,'ecg_signal_drug_labels_cipa.pkl'),'wb') as f:
    pickle.dump(outputs,f)

print('Final Dicts Saved!')




 
