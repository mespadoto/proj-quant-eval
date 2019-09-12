#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from scipy import spatial
from scipy import stats

N_SAMPLES = 0
DISTANCES = dict()


def cleanup_id_run(id_run):
    return id_run.replace(' ', '').replace("'", "").replace(':', '').replace('{', '').replace('}', '').replace('/', '').replace(',', '').replace('-', '').replace('.', '')


def eval_dc_metrics(**kwargs):
    global N_SAMPLES
    global DISTANCES

    X = kwargs.get('X', None)
    dataset_name = kwargs.get('dataset_name', None)
    output_dir = kwargs.get('output_dir', None)

    N_SAMPLES = metric_dc_num_samples(X)

    D_high_list = spatial.distance.pdist(X, 'euclidean')
    D_high_matrix = spatial.distance.squareform(D_high_list)

    DISTANCES['D_high_list'] = '%s/%s_D_high_list.npy' % (
        output_dir, dataset_name)
    DISTANCES['D_high_matrix'] = '%s/%s_D_high_matrix.npy' % (
        output_dir, dataset_name)

    np.save(DISTANCES['D_high_list'], D_high_list)
    np.save(DISTANCES['D_high_matrix'], D_high_matrix)

    D_high_list = None
    D_high_matrix = None

    results = dict()

    for func in [f for f in globals() if 'metric_dc_' in f]:
        param_dict = {p: kwargs.get(p, None) for p in globals(
        )[func].__code__.co_varnames[:globals()[func].__code__.co_argcount]}
        results[func] = globals()[func](**param_dict)

    results['elapsed_time'] = kwargs.get('elapsed_time', 0.0)

    return results


def empty_pq_metrics():
    results = dict()

    for func in [f for f in globals() if 'metric_pq_' in f]:
        results[func] = 0.0

    results['elapsed_time'] = 0.0

    return results


def eval_pq_metrics(**kwargs):
    global DISTANCES

    X = kwargs.get('X', None)
    id_run = kwargs.get('id_run', None)
    dataset_name = kwargs.get('dataset_name', None)
    output_dir = kwargs.get('output_dir', None)

    D_low_list = spatial.distance.pdist(X, 'euclidean')
    D_low_matrix = spatial.distance.squareform(D_low_list)

    clean_id_run = cleanup_id_run(id_run)

    DISTANCES[id_run] = dict()
    DISTANCES[id_run]['D_low_list'] = '%s/%s_D_low_list_%s.npy' % (
        output_dir, dataset_name, clean_id_run)
    DISTANCES[id_run]['D_low_matrix'] = '%s/%s_D_low_matrix_%s.npy' % (
        output_dir, dataset_name, clean_id_run)

    np.save(DISTANCES[id_run]['D_low_list'], D_low_list)
    np.save(DISTANCES[id_run]['D_low_matrix'], D_low_matrix)

    D_low_list = None
    D_low_matrix = None

    D_high = np.load(DISTANCES['D_high_matrix'], mmap_mode='c')
    D_low = np.load(DISTANCES[id_run]['D_low_matrix'], mmap_mode='c')

    Q = metric_coranking_matrix(D_high, D_low)

    DISTANCES[id_run]['Q'] = '%s/%s_Q_%s.npy' % (
        output_dir, dataset_name, clean_id_run)
    np.save(DISTANCES[id_run]['Q'], Q)

    results = dict()

    for func in [f for f in globals() if 'metric_pq_' in f]:
        param_dict = {p: kwargs.get(p, None) for p in globals(
        )[func].__code__.co_varnames[:globals()[func].__code__.co_argcount]}
        results[func] = globals()[func](**param_dict)

    results['elapsed_time'] = kwargs.get('elapsed_time', 0.0)

    return results


def results_to_dataframe(results, dataset_name):
    df = pd.DataFrame.from_dict(results).transpose()
    column_list = df.columns

    df['proj'] = df.index + '|'
    df['projection_name'] = df['proj'].apply(
        lambda x: pd.Series(str(x).split('|')))[0]
    df['projection_parameters'] = df['proj'].apply(
        lambda x: pd.Series(str(x).split('|')))[1]
    df['dataset_name'] = dataset_name
    df = df.drop(['proj'], axis=1)
    df = df.reset_index(drop=True)
    df = df.loc[:, ['dataset_name', 'projection_name',
                    'projection_parameters'] + list(column_list)]

    return df


def metric_dc_dataset_is_balanced(y):
    counts = np.zeros((len(np.unique(y)),))

    for i, l in enumerate(np.unique(y)):
        counts[i] = np.count_nonzero(y == l)

    return np.min(counts) / np.max(counts) > 0.5


def metric_dc_num_samples(X):
    return X.shape[0]


def metric_dc_num_features(X):
    return X.shape[1]


def metric_dc_num_classes(y):
    return len(np.unique(y))


def metric_dc_sparsity_ratio(X):
    return 1.0 - (np.count_nonzero(X) / float(X.shape[0] * X.shape[1]))


def metric_dc_intrinsic_dim(X):
    pca = PCA()
    pca.fit(X)

    return np.where(pca.explained_variance_ratio_.cumsum() >= 0.95)[0][0] + 1


def metric_coranking_matrix(D_high, D_low):
    global N_SAMPLES

    high_rank = D_high.argsort(axis=1).argsort(axis=1)
    low_rank = D_low.argsort(axis=1).argsort(axis=1)

    Q, _, _ = np.histogram2d(
        high_rank.flatten(), low_rank.flatten(), bins=N_SAMPLES)
    Q = Q[1:, 1:]

    return Q


def metric_neighborhood_hit(X, y, k):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X, y)

    neighbors = knn.kneighbors(X, return_distance=False)
    return np.mean(np.mean((y[neighbors] == np.tile(y.reshape((-1, 1)), k)).astype('uint8'), axis=1))


def metric_dc_neighborhood_hit_k_03(X, y):
    return metric_neighborhood_hit(X, y, 3)


def metric_dc_neighborhood_hit_k_05(X, y):
    return metric_neighborhood_hit(X, y, 5)


def metric_dc_neighborhood_hit_k_07(X, y):
    return metric_neighborhood_hit(X, y, 7)


def metric_dc_neighborhood_hit_k_11(X, y):
    return metric_neighborhood_hit(X, y, 11)


def metric_pq_neighborhood_hit_k_03(X, y):
    return metric_neighborhood_hit(X, y, 3)


def metric_pq_neighborhood_hit_k_05(X, y):
    return metric_neighborhood_hit(X, y, 5)


def metric_pq_neighborhood_hit_k_07(X, y):
    return metric_neighborhood_hit(X, y, 7)


def metric_pq_neighborhood_hit_k_11(X, y):
    return metric_neighborhood_hit(X, y, 11)


def metric_trustworthiness(k, id_run):
    global N_SAMPLES
    global DISTANCES

    D_high = np.load(DISTANCES['D_high_matrix'], mmap_mode='c')
    D_low = np.load(DISTANCES[id_run]['D_low_matrix'], mmap_mode='c')

    n = N_SAMPLES

    nn_orig = D_high.argsort()
    nn_proj = D_low.argsort()

    knn_orig = nn_orig[:, :k + 1][:, 1:]
    knn_proj = nn_proj[:, :k + 1][:, 1:]

    sum_i = 0

    for i in range(n):
        U = np.setdiff1d(knn_proj[i], knn_orig[i])

        sum_j = 0
        for j in range(U.shape[0]):
            sum_j += np.where(nn_orig[i] == U[j])[0] - k

        sum_i += sum_j

    return float((1 - (2 / (n * k * (2 * n - 3 * k - 1)) * sum_i)).squeeze())


def metric_continuity(k, id_run):
    global N_SAMPLES
    global DISTANCES

    D_high = np.load(DISTANCES['D_high_matrix'], mmap_mode='c')
    D_low = np.load(DISTANCES[id_run]['D_low_matrix'], mmap_mode='c')

    n = N_SAMPLES

    nn_orig = D_high.argsort()
    nn_proj = D_low.argsort()

    knn_orig = nn_orig[:, :k + 1][:, 1:]
    knn_proj = nn_proj[:, :k + 1][:, 1:]

    sum_i = 0

    for i in range(N_SAMPLES):
        V = np.setdiff1d(knn_orig[i], knn_proj[i])

        sum_j = 0
        for j in range(V.shape[0]):
            sum_j += np.where(nn_proj[i] == V[j])[0] - k

        sum_i += sum_j

    return float((1 - (2 / (n * k * (2 * n - 3 * k - 1)) * sum_i)).squeeze())


def metric_pq_trustworthiness_k_03(id_run):
    return metric_trustworthiness(3, id_run)


def metric_pq_trustworthiness_k_05(id_run):
    return metric_trustworthiness(5, id_run)


def metric_pq_trustworthiness_k_07(id_run):
    return metric_trustworthiness(7, id_run)


def metric_pq_trustworthiness_k_11(id_run):
    return metric_trustworthiness(11, id_run)


def metric_pq_continuity_k_03(id_run):
    return metric_continuity(3, id_run)


def metric_pq_continuity_k_05(id_run):
    return metric_continuity(5, id_run)


def metric_pq_continuity_k_07(id_run):
    return metric_continuity(7, id_run)


def metric_pq_continuity_k_11(id_run):
    return metric_continuity(11, id_run)


def metric_pq_normalized_stress(id_run):
    global DISTANCES

    D_high = np.load(DISTANCES['D_high_list'], mmap_mode='c')
    D_low = np.load(DISTANCES[id_run]['D_low_list'], mmap_mode='c')

    return np.sum((D_high - D_low)**2) / np.sum(D_high**2)


def metric_pq_shepard_diagram_correlation(id_run, dataset_name):
    global DISTANCES

    D_high = np.load(DISTANCES['D_high_list'], mmap_mode='c')
    D_low = np.load(DISTANCES[id_run]['D_low_list'], mmap_mode='c')

    return stats.spearmanr(D_high, D_low)[0]
