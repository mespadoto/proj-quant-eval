#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from glob import glob
import numpy as np
import pandas as pd
from metrics import *
import joblib
from sklearn.model_selection import train_test_split
from multiprocessing import Process

def load_dataset(dataset_name):
    data_dir = os.path.join('data', dataset_name)
    X = np.load(os.path.join(data_dir, 'X.npy'))
    y = np.load(os.path.join(data_dir, 'y.npy'))

    return X, y

def measure_dataset(dataset_name):
    print(dataset_name)
    X, y = load_dataset(dataset_name)

    # if X.shape[0] > 10000:
    #     test_size = 1 / (X.shape[0] / 10000)
    #     _, X_test, _, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
    #     X = X_test
    #     y = y_test

    results = eval_metrics(X=X, y=y, k=7)
    joblib.dump(results, 'results_' + dataset_name + '.pkl')

if __name__ == '__main__':
    datasets = ['seismic', 'spambase', 'cnae9', 'secom', 'epileptic', 
                'bank', 'sentiment', 'fmd', 'har', 'gene', 'efigi',
                'fashion_mnist', 'mnist', 'sms', 'cifar10', 'p53',
                'svhn', 'hatespeech', 'imdb', 'coil20',
                'hiva', 'orl']  # 'hepmass'
    plist = []

    for d in datasets:
        plist.append(Process(target=measure_dataset, args=(d,)))

    for p in plist:
        p.start()

    for p in plist:
        p.join()

    results = dict()

    for dataset_name in datasets:
        file_name = 'results_%s.pkl' % dataset_name
        results[dataset_name] = joblib.load(file_name)
        os.unlink(file_name)

    df = pd.DataFrame.from_dict(results).transpose()
    column_list = df.columns
    df['dataset_name'] = df.index
    df = df.reset_index(drop=True)
    df = df.loc[:, ['dataset_name'] + list(column_list)]

    df.to_csv('measure.csv', index=None)
