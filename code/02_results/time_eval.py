#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import joblib
import numpy as np
import matplotlib.pyplot as plt
import mtsne
import vp
import umap
from time import perf_counter

results = dict()

samples = np.linspace(500, 50000, 30).astype('int32')
dims = np.linspace(50, 1000, 30).astype('int32')

projs = dict()
projs['UMAP'] = umap.UMAP(n_components=2, random_state=420, n_neighbors=10, min_dist=0.5)
projs['T-SNE'] = mtsne.MTSNE(n_components=2, random_state=420, perplexity=30.0, n_iter=500, n_iter_without_progress=100, n_jobs=1)
projs['PBC'] = vp.ProjectionByClustering(fraction_delta=8.0, n_iterations=50, init_type='random', dissimilarity_type='euclidean', cluster_factor=4.5)
projs['IDMAP'] = vp.IDMAP(fraction_delta=8.0, n_iterations=50, init_type='random', dissimilarity_type='euclidean')

proj_names = ['UMAP', 'PBC', 'T-SNE', 'IDMAP']

for proj in proj_names:
    results[proj] = dict()

    p = projs[proj]

    x = []
    y = []
    c = []
    
    for s in samples:
        for d in dims:
            data = np.random.normal(0, 1, s*d).reshape((s, d))

            x.append(s)
            y.append(d)

            print(s, d, proj)
            
            t0 = perf_counter()
            ret = p.fit_transform(data)
            elapsed_time = perf_counter() - t0

            if type(ret) == tuple:
                elapsed_time = ret[1]
            
            c.append(elapsed_time)
            print(elapsed_time)

        results[proj]['x'] = x
        results[proj]['y'] = y
        results[proj]['c'] = c

        joblib.dump(results, 'time_eval_chkpt_%s.pkl' % proj)
