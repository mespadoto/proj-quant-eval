#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import argparse
import os
from glob import glob

import joblib
import numpy as np
from sklearn import datasets
from sklearn.model_selection import ParameterGrid

import mtsne
import projections
import tapkee
import drtoolbox
import vp
from metrics import *


def list_projections():
    return sorted(projections.all_projections.keys())


def list_datasets():
    return [os.path.basename(d) for d in sorted(glob('data/*'))]


def print_projections():
    print('Projections:' + ' '.join(list_projections()))


def print_datasets():
    print('Datasets:' + ' '.join(list_datasets()))


def load_dataset(dataset_name):
    data_dir = os.path.join('data', dataset_name)
    X = np.load(os.path.join(data_dir, 'X.npy'))
    y = np.load(os.path.join(data_dir, 'y.npy'))

    return X, y


def run_eval(dataset_name, projection_name, X, y, output_dir):
    # TODO: add noise adder
    global DISTANCES

    dc_results = dict()
    pq_results = dict()
    projected_data = dict()

    dc_results['original'] = eval_dc_metrics(
        X=X, y=y, dataset_name=dataset_name, output_dir=output_dir)

    proj_tuple = projections.all_projections[projection_name]
    proj = proj_tuple[0]
    grid_params = proj_tuple[1]

    grid = ParameterGrid(grid_params)

    for params in grid:
        id_run = proj.__class__.__name__ + '|' + str(params)
        proj.set_params(**params)

        print('-----------------------------------------------------------------------')
        print(projection_name, id_run)

        X_new, y_new, result = projections.run_projection(
            proj, X, y, id_run, dataset_name, output_dir)
        pq_results[id_run] = result
        projected_data[id_run] = dict()
        projected_data[id_run]['X'] = X_new
        projected_data[id_run]['y'] = y_new

    results_to_dataframe(dc_results, dataset_name).to_csv(
        '%s/%s_%s_dc_results.csv' % (output_dir, dataset_name, projection_name), index=None)
    results_to_dataframe(pq_results, dataset_name).to_csv(
        '%s/%s_%s_pq_results.csv' % (output_dir, dataset_name, projection_name), index=None)
    joblib.dump(projected_data, '%s/%s_%s_projected.pkl' %
                (output_dir, dataset_name, projection_name))
    joblib.dump(DISTANCES, '%s/%s_%s_distance_files.pkl' %
                (output_dir, dataset_name, projection_name))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Projection Survey Runner')

    parser.add_argument('-d', type=str, help='dataset name')
    parser.add_argument('-p', type=str, help='projection name')
    parser.add_argument('-o', type=str, help='output directory (must exist)')
    args, unknown = parser.parse_known_args()

    if args.d is None:
        print_datasets()
        print_projections()
        exit()

    dataset_name = args.d
    projection_name = args.p
    output_dir = args.o

    if not os.path.exists(output_dir):
        print('Directory %s not found' % output_dir)
        exit(1)

    X, y = load_dataset(dataset_name)
    run_eval(dataset_name, projection_name, X, y, output_dir)
