#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.linear_model import LinearRegression

samples = np.linspace(500, 50000, 30).astype('int32')
dims = np.linspace(50, 1000, 30).astype('int32')
proj_names = ['UMAP', 'PBC', 'T-SNE', 'IDMAP']

results = dict()

for proj in proj_names:
    tmp = joblib.load('time_eval_chkpt_%s.pkl' % proj)
    results[proj] = tmp[proj]

fig, ax = plt.subplots(2, 2, figsize=(12, 10))
fig.tight_layout()

figl, axl = plt.subplots(2, 2, figsize=(12, 10))

for proj, coord in zip(proj_names, [(0,0), (0,1), (1,0), (1,1)]):
    if proj == 'IDMAP':
        proj2 = 'T-SNE'
        results[proj2]['x']
        results[proj2]['y']
        results[proj2]['c']
        
        m = LinearRegression()
        X = np.vstack((np.array(results[proj]['x'])[400:], np.array(results[proj]['y'])[400:])).T
        y = np.array(results[proj]['c'])[400:]
        m.fit(X, y)
        
        X_new = np.vstack((np.array(results[proj2]['x']), np.array(results[proj2]['y']))).T
        c_new = m.predict(X_new)
        
        results[proj]['x'] = results[proj]['x'] + results[proj2]['x'][810:]
        results[proj]['y'] = results[proj]['y'] + results[proj2]['y'][810:]
        results[proj]['c'] = results[proj]['c'] + list(c_new[810:])
    
    vmin = np.min(results[proj]['c'])
    vmax = np.max(results[proj]['c'])

    p = ax[coord[0], coord[1]].scatter(results[proj]['x'], results[proj]['y'], c=results[proj]['c'], s=220, marker='s', vmin=vmin, vmax=vmax, cmap='viridis')
    pl = axl[coord[0], coord[1]].scatter(results[proj]['x'], results[proj]['y'], c=results[proj]['c'], s=220, marker='s', vmin=vmin, vmax=vmax, cmap='viridis')

    ax[coord[0], coord[1]].spines['top'].set_visible(False)
    ax[coord[0], coord[1]].spines['right'].set_visible(False)
    ax[coord[0], coord[1]].spines['bottom'].set_visible(False)
    ax[coord[0], coord[1]].spines['left'].set_visible(False)

    axl[coord[0], coord[1]].spines['top'].set_visible(False)
    axl[coord[0], coord[1]].spines['right'].set_visible(False)
    axl[coord[0], coord[1]].spines['bottom'].set_visible(False)
    axl[coord[0], coord[1]].spines['left'].set_visible(False)
    
    if coord == (0,0) or coord == (1,0):
        axl[coord[0], coord[1]].set_ylabel('dims')

    if coord == (1,0) or coord == (1,1):
        axl[coord[0], coord[1]].set_xlabel('samples')

    axl[coord[0], coord[1]].set_title('%s' % proj)
    fig.colorbar(p, ax=ax[coord[0], coord[1]])
    figl.colorbar(p, ax=axl[coord[0], coord[1]])

fig.savefig('time_eval_local.png')
figl.savefig('time_eval_local_labeled.png')



