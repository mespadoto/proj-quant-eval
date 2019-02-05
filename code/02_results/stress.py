#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
from glob import glob
from sklearn.preprocessing import MinMaxScaler

def normalize(D):
    scaler = MinMaxScaler()
    D = scaler.fit_transform(D.reshape((-1, 1)))
    D = D.squeeze()
    return D

def normalized_stress(D_high, D_low):
    return np.sqrt(np.sum((D_high - D_low)**2) / np.sum(D_high**2))

base_dir = '/data/src/proj-survey-data'
datasets = []
projections = []
id_runs = []
stress_values = []

output_dirs = glob(base_dir + '/output_*')

for output_dir in output_dirs:
    if not os.path.isdir(output_dir):
        continue
    
    D_high_files = glob(output_dir + '/*_D_high_list.npy')
    
    for D_high_f in D_high_files:
        dataset = os.path.basename(D_high_f).split('D')[0][:-1]
        print(dataset)
    
        D_high = normalize(np.load(D_high_f))
    
        D_low_files = glob(output_dir + '/%s_D_low_list*.npy' % dataset)
    
        for D_low_f in D_low_files:
            id_run_complete = os.path.basename(D_low_f).replace('%s_D_low_list_' % dataset, '').replace('.npy', '')

            projection = id_run_complete.split('|')[0]            
            id_run     = id_run_complete.split('|')[1]
            D_low = np.load(D_low_f)
            D_low[np.isnan(D_low)] = -1.0
            D_low = normalize(D_low)
            
            datasets.append(dataset)
            projections.append(projection)
            id_runs.append(id_run)
            stress_values.append(normalized_stress(D_high, D_low))


df = pd.DataFrame.from_dict({'dataset': datasets, 'projection': projections, 
                        'id_run': id_runs, 'normalized_stress': stress_values})

df.to_csv('stress.csv', index=None, sep=';', decimal=',')
