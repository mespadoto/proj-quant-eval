#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib as mpl
mpl.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

param_best = pd.read_csv('param_best.csv')

algos = ['IDMAP', 'T-SNE', 'UMAP', 'PBC']
datasets = ['fashion_mnist', 'har', 'cnae9', 'coil20']

dfs = []

for a in algos:
    for d in datasets:
        dfs.append(param_best[(param_best['dataset_name'] == d) & (param_best['new_proj_name'] == a)])

best = pd.concat(dfs) # [['new_proj_name', 'dataset_name', 'projection_parameters']].reset_index(drop=True)

def cleanup_id_run(id_run):
    return id_run.replace(' ', '').replace("'", "").replace(':', '').replace('{', '').replace('}', '').replace('/', '').replace(',', '').replace('-', '').replace('.', '')

best['clean_params'] = best['projection_parameters'].apply(cleanup_id_run)
best['file_name'] = best['dataset_name'] + '_D_low_list_' + best['new_proj_name'] + '|' + best['clean_params'] + '.npy'
best['file_name'] = best['file_name'].str.replace('_PBC', '_ProjectionByClustering')
best['file_name'] = best['file_name'].str.replace('_T-SNE', '_MTSNE')

fig, ax = plt.subplots(4, 4, figsize=(20, 20))
fig.tight_layout()

figl, axl = plt.subplots(4, 4, figsize=(20, 20))

for D_low_file, corr, coord in zip(list(best.sort_values(['new_proj_name', 'dataset_name'])['file_name']), 
                                   list(best.sort_values(['new_proj_name', 'dataset_name'])['metric_pq_shepard_diagram_correlation']), 
                                   [(0,0),(0,1),(0,2),(0,3), (1,0),(1,1),(1,2),(1,3), (2,0),(2,1),(2,2),(2,3), (3,0),(3,1),(3,2),(3,3)]):
    ds = D_low_file.split('D')[0][:-1]
    algo = D_low_file.split('D_low_list')[1].split('|')[0][1:]
    D_high_file = '%s_D_high_list.npy' % ds

    D_low = np.load(D_low_file)
    D_high = np.load(D_high_file)

    scaler = MinMaxScaler()
    D_low = scaler.fit_transform(D_low.reshape(-1, 1)).flatten()
    
    scaler = MinMaxScaler()
    D_high = scaler.fit_transform(D_high.reshape(-1, 1)).flatten()
    
    selection = np.random.randint(0, D_high.shape[0], 2000, random_state=420)
    D_high = D_high[selection]
    D_low = D_low[selection]

    ax[coord[0], coord[1]].scatter(D_high, D_low, s=3, alpha=0.01, c='grey')
    ax[coord[0], coord[1]].plot([0, 1], [0, 1], alpha=0.3, c='red')
    ax[coord[0], coord[1]].axis('off')

    axl[coord[0], coord[1]].scatter(D_high, D_low, s=3, alpha=0.01, c='grey')
    axl[coord[0], coord[1]].plot([0, 1], [0, 1], alpha=0.3, c='red')
    axl[coord[0], coord[1]].axis('off')
    axl[coord[0], coord[1]].text(0, 0, 'R = %s' % str(corr))
    axl[coord[0], coord[1]].set_title('%s %s' % (algo, ds))


fig.savefig('shepard.png')
figl.savefig('shepard_labeled.png')
