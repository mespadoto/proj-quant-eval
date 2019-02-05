#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys

import joblib
from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

import drtoolbox
import mtsne
import projections
import runner
import tapkee
import vp
from metrics import *

for dl in [datasets.load_digits]: #, datasets.load_wine, datasets.load_digits]:
    X, y = dl(return_X_y=True)

    # X = np.hstack((X, X, X, X, X, X, X, X))
    # X = np.vstack((X, X, X, X, X, X, X, X))
    # y = np.hstack((y, y, y, y, y, y, y, y))

    mm = MinMaxScaler()
    X = mm.fit_transform(X)

    if X.shape[0] > 100:
        _, X, _, y = train_test_split(X, y, test_size=0.5, random_state=420)

    print(X.shape)

    dataset_name = dl.__name__.split('_')[1]
    output_dir = 'test'

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for p in runner.list_projections():
        runner.run_eval(dataset_name, p, X, y, output_dir)
