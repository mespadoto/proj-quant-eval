#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import subprocess
import sys
import tempfile
import traceback
from distutils.spawn import find_executable
from glob import glob

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class VispipelineProjection(BaseEstimator, TransformerMixin):
    def __init__(self, projection, command, verbose):
        self.known_projections = [
            'plmp',
            'idmap',
            'lsp',
            'plsp',
            'lamp',
            'fastmap',
            'lisomap',
            'pekalska',
            'projclus']

        self.projection = projection
        self.command = command
        self.verbose = verbose

        if self.projection not in self.known_projections:
            raise ValueError('Invalid projection name: %s. Valid values are %s'
                             % (self.projection, ','.join(self.known_projections)))

    def fit_transform(self, X, y=None):
        raise Exception('Not implemented')

    def _send_data(self, X, y):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.tmp_file = tempfile.NamedTemporaryFile(
            mode='w', dir=self.tmp_dir.name, suffix='.data', delete=False)

        try:
            if y is None:
                y = np.zeros((X.shape[0],))

            n_samples = X.shape[0]
            n_features = X.shape[1]

            with open(self.tmp_file.name, 'w') as f:
                f.write('DY\n')
                f.write(str(n_samples))
                f.write('\n')
                f.write(str(n_features))
                f.write('\n')
                f.write(';'.join([str(i) for i in range(n_features)]))
                f.write('\n')

                for i in range(n_samples):
                    f.write('%s;%s;%s\n' %
                            (i, ';'.join([str(i) for i in X[i, :]]), y[i]))
        except:
            raise Exception('Error converting file to vp-run')

        return self.tmp_dir.name

    def _receive_data(self):
        proj_files = glob(self.tmp_dir.name + '/*-%s*.prj' % self.projection)

        if len(proj_files) != 1:
            raise ValueError(
                'Error looking for projection file inside %s' % self.tmp_dir.name)

        with open(proj_files[0], 'r') as f:
            f.readline()
            n_samples = int(f.readline())
            n_features = int(f.readline())
            f.readline()

            if self.verbose:
                print(n_samples, n_features)

            X_new = np.zeros((n_samples, n_features))
            y_new = np.zeros((n_samples,))

            for i in range(n_samples):
                row = f.readline()
                rowvals = row.split(';')

                y_new[i] = rowvals[n_features+1]

                for j in range(n_features):
                    X_new[i, j] = rowvals[j+1]

                # if self.verbose:
                #     print(row.strip('\n'))

        return X_new

    def _run(self, X, y, cmdargs):
        if not find_executable(self.command):
            raise ValueError('Command %s not found' % self.command)

        self._send_data(X, y)

        cmdline = [self.command, self.projection,
                   self.tmp_dir.name] + [str(x) for x in cmdargs]

        if self.verbose:
            print('#################################################')
            print(' '.join(cmdline))

        rc = subprocess.run(cmdline, universal_newlines=True, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, timeout=86400, check=True)

        if self.verbose:
            print('return code: ', rc.returncode)
            print('stdout:')
            print('_________________________________________________')
            print(rc.stdout)
            print('_________________________________________________')
            print('stderr:')
            print('_________________________________________________')
            print(rc.stderr)
            print('#################################################')

        try:
            X_new = self._receive_data()
            return X_new
        except:
            print('Error running projection')
            print('return code: ', rc.returncode)
            print('stdout:')
            print('_________________________________________________')
            print(rc.stdout)
            print('_________________________________________________')
            print('stderr:')
            print('_________________________________________________')
            print(rc.stderr)

            reason, _, tb = sys.exc_info()
            print(reason)
            traceback.print_tb(tb)
            raise('Error running projection')


class PLMP(VispipelineProjection):
    # 1. Fraction Delta (float, default: 8.0)
    # 2. Number of Iterations (int, default: 100)
    # 3. Sample Type (SampleType, default: 0)
    #    0. Random sampling
    #    1. Clustering centroid sampling
    #    2. Clustering medoid sampling
    #    3. Max-min sampling
    #    4. Spam
    # 4. Dissimilarity Type (DissimilarityType, default: 2)
    #    0. City-block
    #    1. Cosine-based dissimilarity
    #    2. Euclidean
    #    3. Extended Jaccard
    #    4. Infinity norm
    #    5. Dynamic Time Warping (DTW)
    #    6. Max Moving Euclidean
    #    7. Min Moving Euclidean
    def __init__(self, command=os.getcwd() + '/vispipeline/vp-run',
                 fraction_delta=8.0,
                 n_iterations=100,
                 sample_type='random',
                 dissimilarity_type='euclidean',
                 verbose=False):
        super(PLMP, self).__init__(projection='plmp',
                                   command=command, verbose=verbose)

        self.sample_types = ['random',
                             'clustering_centroid',
                             'clustering_medoid',
                             'maxmin',
                             'spam']
        self.dissimilarity_types = ['cityblock',
                                    'cosine',
                                    'euclidean',
                                    'extended_jaccard',
                                    'infinity_norm',
                                    'dtw',
                                    'max_moving_euclidean',
                                    'min_moving_euclidean']
        self.set_params(command, fraction_delta, n_iterations,
                        sample_type, dissimilarity_type, verbose)

    def set_params(self, command=os.getcwd() + '/vispipeline/vp-run',
                   fraction_delta=8.0,
                   n_iterations=100,
                   sample_type='random',
                   dissimilarity_type='euclidean',
                   verbose=False):

        self.command = command
        self.verbose = verbose
        self.fraction_delta = fraction_delta
        self.n_iterations = n_iterations

        try:
            self.sample_type_index = self.sample_types.index(sample_type)
        except:
            raise ValueError('Invalid sample type: %s. Valid values are %s'
                             % (sample_type, ','.join(self.sample_types)))
        try:
            self.dissimilarity_type_index = self.dissimilarity_types.index(
                dissimilarity_type)
        except:
            raise ValueError('Invalid dissimilarity index: %s. Valid values are %s'
                             % (dissimilarity_type, ','.join(self.dissimilarity_types)))

        # TODO: fill with valid ranges
        if self.fraction_delta < 0.0:
            raise ValueError('Invalid fraction delta')

        # TODO: fill with valid ranges
        if self.n_iterations < 1:
            raise ValueError('Invalid n_iterations')

    def fit_transform(self, X, y=None):
        return super(PLMP, self)._run(X, y,
                                      [self.fraction_delta,
                                       self.n_iterations,
                                       self.sample_type_index,
                                       self.dissimilarity_type_index])


class IDMAP(VispipelineProjection):
    # Interactive Document Map (IDMAP)
    # ---------------
    # Parameters:
    # 1. Fraction Delta (float, default: 8.0)
    # 2. Number of Iterations (int, default: 100)
    # 3. Initialization Type (InitializationType, default: 0)
    #    0. Fastmap
    #    1. Nearest Neighbor Projection (NNP)
    #    2. Random
    # 4. Dissimilarity Type (DissimilarityType, default: 2)
    #    0. City-block
    #    1. Cosine-based dissimilarity
    #    2. Euclidean
    #    3. Extended Jaccard
    #    4. Infinity norm
    #    5. Dynamic Time Warping (DTW)
    #    6. Max Moving Euclidean
    #    7. Min Moving Euclidean
    def __init__(self, command=os.getcwd() + '/vispipeline/vp-run',
                 fraction_delta=8.0,
                 n_iterations=100,
                 init_type='fastmap',
                 dissimilarity_type='euclidean',
                 verbose=False):
        super(IDMAP, self).__init__(projection='idmap',
                                    command=command, verbose=verbose)

        self.init_types = ['fastmap', 'nnp', 'random']
        self.dissimilarity_types = ['cityblock',
                                    'cosine',
                                    'euclidean',
                                    'extended_jaccard',
                                    'infinity_norm',
                                    'dtw',
                                    'max_moving_euclidean',
                                    'min_moving_euclidean']
        self.set_params(command, fraction_delta, n_iterations,
                        init_type, dissimilarity_type, verbose)

    def set_params(self, command=os.getcwd() + '/vispipeline/vp-run',
                   fraction_delta=8.0,
                   n_iterations=100,
                   init_type='fastmap',
                   dissimilarity_type='euclidean',
                   verbose=False):
        self.command = command
        self.verbose = verbose
        self.fraction_delta = fraction_delta
        self.n_iterations = n_iterations

        try:
            self.init_type_index = self.init_types.index(init_type)
        except:
            raise ValueError('Invalid init type: %s. Valid values are %s'
                             % (init_type, ','.join(self.init_types)))

        try:
            self.dissimilarity_type_index = self.dissimilarity_types.index(
                dissimilarity_type)
        except:
            raise ValueError('Invalid dissimilarity index: %s. Valid values are %s'
                             % (dissimilarity_type, ','.join(self.dissimilarity_types)))

        # TODO: fill with valid ranges
        if self.fraction_delta < 0.0:
            raise ValueError('Invalid fraction delta')

        # TODO: fill with valid ranges
        if self.n_iterations < 1:
            raise ValueError('Invalid n_iterations')

    def fit_transform(self, X, y=None):
        return super(IDMAP, self)._run(X, y,
                                       [self.fraction_delta,
                                        self.n_iterations,
                                        self.init_type_index,
                                        self.dissimilarity_type_index])


class LSP(VispipelineProjection):
    # 1. Number of Neighbors (int, default: 8)
    # 2. Control Points Choice (ControlPointsType, default: 0)
    #    0. Random
    #    1. K-medoids
    #    2. K-means
    # 3. Fraction Delta (float, default: 8.0)
    # 4. Number of Iterations (int, default: 100)
    # 5. Dissimilarity Type (DissimilarityType, default: 2)
    #    0. City-block
    #    1. Cosine-based dissimilarity
    #    2. Euclidean
    #    3. Extended Jaccard
    #    4. Infinity norm
    #    5. Dynamic Time Warping (DTW)
    #    6. Max Moving Euclidean
    #    7. Min Moving Euclidean
    def __init__(self, command=os.getcwd() + '/vispipeline/vp-run',
                 fraction_delta=8.0,
                 n_iterations=100,
                 n_neighbors=8,
                 control_point_type='random',
                 dissimilarity_type='euclidean',
                 verbose=False):
        super(LSP, self).__init__(projection='lsp',
                                  command=command, verbose=verbose)

        self.control_point_types = ['random', 'kmedoids', 'kmeans']
        self.dissimilarity_types = ['cityblock',
                                    'cosine',
                                    'euclidean',
                                    'extended_jaccard',
                                    'infinity_norm',
                                    'dtw',
                                    'max_moving_euclidean',
                                    'min_moving_euclidean']
        self.set_params(command, fraction_delta, n_iterations,
                        n_neighbors, control_point_type, dissimilarity_type, verbose)

    def set_params(self, command=os.getcwd() + '/vispipeline/vp-run',
                   fraction_delta=8.0,
                   n_iterations=100,
                   n_neighbors=8,
                   control_point_type='random',
                   dissimilarity_type='euclidean',
                   verbose=False):
        self.command = command
        self.verbose = verbose
        self.fraction_delta = fraction_delta
        self.n_iterations = n_iterations
        self.n_neighbors = n_neighbors

        try:
            self.control_point_type_index = self.control_point_types.index(
                control_point_type)
        except:
            raise ValueError('Invalid control point type: %s. Valid values are %s'
                             % (control_point_type, ','.join(self.control_point_types)))

        try:
            self.dissimilarity_type_index = self.dissimilarity_types.index(
                dissimilarity_type)
        except:
            raise ValueError('Invalid dissimilarity index: %s. Valid values are %s'
                             % (dissimilarity_type, ','.join(self.dissimilarity_types)))

        # TODO: fill with valid ranges
        if self.fraction_delta < 0.0:
            raise ValueError('Invalid fraction delta')

        # TODO: fill with valid ranges
        if self.n_iterations < 1:
            raise ValueError('Invalid n_iterations')

        # TODO: fill with valid ranges
        if self.n_neighbors < 1:
            raise ValueError('Invalid n_neighbors')

    def fit_transform(self, X, y=None):
        return super(LSP, self)._run(X, y,
                                     [self.n_neighbors,
                                      self.control_point_type_index,
                                      self.fraction_delta,
                                      self.n_iterations,
                                      self.dissimilarity_type_index])


class PLSP(VispipelineProjection):
    # 1. Sample Type (SampleType, default: 1)
    #    0. Random sampling
    #    1. Clustering sampling
    # 2. Dissimilarity Type (DissimilarityType, default: 2)
    #    0. City-block
    #    1. Cosine-based dissimilarity
    #    2. Euclidean
    #    3. Extended Jaccard
    #    4. Infinity norm
    #    5. Dynamic Time Warping (DTW)
    #    6. Max Moving Euclidean
    #    7. Min Moving Euclidean
    def __init__(self, command=os.getcwd() + '/vispipeline/vp-run',
                 sample_type='clustering',
                 dissimilarity_type='euclidean',
                 verbose=False):
        super(PLSP, self).__init__(projection='plsp',
                                   command=command, verbose=verbose)

        self.sample_types = ['random', 'clustering']
        self.dissimilarity_types = ['cityblock',
                                    'cosine',
                                    'euclidean',
                                    'extended_jaccard',
                                    'infinity_norm',
                                    'dtw',
                                    'max_moving_euclidean',
                                    'min_moving_euclidean']
        self.set_params(command, sample_type, dissimilarity_type, verbose)

    def set_params(self, command=os.getcwd() + '/vispipeline/vp-run',
                   sample_type='clustering',
                   dissimilarity_type='euclidean',
                   verbose=False):
        self.command = command
        self.verbose = verbose

        try:
            self.sample_type_index = self.sample_types.index(sample_type)
        except:
            raise ValueError('Invalid sample type: %s. Valid values are %s'
                             % (sample_type, ','.join(self.sample_types)))

        try:
            self.dissimilarity_type_index = self.dissimilarity_types.index(
                dissimilarity_type)
        except:
            raise ValueError('Invalid dissimilarity index: %s. Valid values are %s'
                             % (dissimilarity_type, ','.join(self.dissimilarity_types)))

    def fit_transform(self, X, y=None):
        return super(PLSP, self)._run(X, y,
                                      [self.sample_type_index,
                                       self.dissimilarity_type_index])


class LAMP(VispipelineProjection):
    # 1. Fraction Delta (float, default: 8.0)
    # 2. Number of Iterations (int, default: 100)
    # 3. Sample Type (enum, default: 2)
    #    0. RANDOM
    #    1. CLUSTERING_CENTROID
    #    2. CLUSTERING_MEDOID
    #    3. MAXMIN
    #    4. SPAM
    def __init__(self, command=os.getcwd() + '/vispipeline/vp-run',
                 fraction_delta=8.0,
                 n_iterations=100,
                 sample_type='random',
                 verbose=False):
        super(LAMP, self).__init__(projection='lamp',
                                   command=command, verbose=verbose)

        self.sample_types = ['random',
                             'clustering_centroid',
                             'clustering_medoid',
                             'maxmin',
                             'spam']
        self.set_params(command, fraction_delta, n_iterations,
                        sample_type, verbose=False)

    def set_params(self, command=os.getcwd() + '/vispipeline/vp-run',
                   fraction_delta=8.0,
                   n_iterations=100,
                   sample_type='random',
                   verbose=False):
        self.command = command
        self.verbose = verbose
        self.fraction_delta = fraction_delta
        self.n_iterations = n_iterations

        try:
            self.sample_type_index = self.sample_types.index(sample_type)
        except:
            raise ValueError('Invalid sample type: %s. Valid values are %s'
                             % (sample_type, ','.join(self.sample_types)))

        # TODO: fill with valid ranges
        if self.fraction_delta < 0.0:
            raise ValueError('Invalid fraction delta')

        # TODO: fill with valid ranges
        if self.n_iterations < 1:
            raise ValueError('Invalid n_iterations')

    def fit_transform(self, X, y=None):
        return super(LAMP, self)._run(X, y,
                                      [self.fraction_delta,
                                       self.n_iterations,
                                       self.sample_type_index])


class Fastmap(VispipelineProjection):
    # 1. Dissimilarity Type(DissimilarityType, default: 2)
    #     0. City - block
    #     1. Cosine - based dissimilarity
    #     2. Euclidean
    #     3. Extended Jaccard
    #     4. Infinity norm
    #     5. Dynamic Time Warping(DTW)
    #     6. Max Moving Euclidean
    #     7. Min Moving Euclidean
    def __init__(self, command=os.getcwd() + '/vispipeline/vp-run',
                 dissimilarity_type='euclidean',
                 verbose=False):
        super(Fastmap, self).__init__(
            projection='fastmap', command=command, verbose=verbose)

        self.dissimilarity_types = ['cityblock',
                                    'cosine',
                                    'euclidean',
                                    'extended_jaccard',
                                    'infinity_norm',
                                    'dtw',
                                    'max_moving_euclidean',
                                    'min_moving_euclidean']
        self.set_params(command, dissimilarity_type, verbose)

    def set_params(self, command=os.getcwd() + '/vispipeline/vp-run',
                   dissimilarity_type='euclidean',
                   verbose=False):
        self.command = command
        self.verbose = verbose

        try:
            self.dissimilarity_type_index = self.dissimilarity_types.index(
                dissimilarity_type)
        except:
            raise ValueError('Invalid dissimilarity index: %s. Valid values are %s'
                             % (dissimilarity_type, ','.join(self.dissimilarity_types)))

    def fit_transform(self, X, y=None):
        return super(Fastmap, self)._run(X, y, [self.dissimilarity_type_index])


class RapidSammon(VispipelineProjection):
    # 1. Dissimilarity Type(DissimilarityType, default: 2)
    #     0. City - block
    #     1. Cosine - based dissimilarity
    #     2. Euclidean
    #     3. Extended Jaccard
    #     4. Infinity norm
    #     5. Dynamic Time Warping(DTW)
    #     6. Max Moving Euclidean
    #     7. Min Moving Euclidean
    def __init__(self, command=os.getcwd() + '/vispipeline/vp-run',
                 dissimilarity_type='euclidean',
                 verbose=False):
        super(RapidSammon, self).__init__(
            projection='pekalska', command=command, verbose=verbose)

        self.dissimilarity_types = ['cityblock',
                                    'cosine',
                                    'euclidean',
                                    'extended_jaccard',
                                    'infinity_norm',
                                    'dtw',
                                    'max_moving_euclidean',
                                    'min_moving_euclidean']
        self.set_params(command, dissimilarity_type, verbose)

    def set_params(self, command=os.getcwd() + '/vispipeline/vp-run',
                   dissimilarity_type='euclidean',
                   verbose=False):
        self.command = command
        self.verbose = verbose

        try:
            self.dissimilarity_type_index = self.dissimilarity_types.index(
                dissimilarity_type)
        except:
            raise ValueError('Invalid dissimilarity index: %s. Valid values are %s'
                             % (dissimilarity_type, ','.join(self.dissimilarity_types)))

    def fit_transform(self, X, y=None):
        return super(RapidSammon, self)._run(X, y, [self.dissimilarity_type_index])


class ProjectionByClustering(VispipelineProjection):
    # 1. Fraction Delta (float, default: 8.0)
    # 2. Number of Iterations (int, default: 50)
    # 3. Initialization Type (InitializationType, default: 0)
    #    0. Fastmap
    #    1. Nearest Neighbor Projection (NNP)
    #    2. Random
    # 4. Dissimilarity Type (DissimilarityType, default: 2)
    #    0. City-block
    #    1. Cosine-based dissimilarity
    #    2. Euclidean
    #    3. Extended Jaccard
    #    4. Infinity norm
    #    5. Dynamic Time Warping (DTW)
    #    6. Max Moving Euclidean
    #    7. Min Moving Euclidean
    # 5. Cluster Factor(float, default: 4.5)
    def __init__(self, command=os.getcwd() + '/vispipeline/vp-run',
                 fraction_delta=8.0,
                 n_iterations=50,
                 init_type='fastmap',
                 dissimilarity_type='euclidean',
                 cluster_factor=4.5,
                 verbose=False):
        super(ProjectionByClustering, self).__init__(
            projection='projclus', command=command, verbose=verbose)

        self.init_types = ['fastmap', 'nnp', 'random']
        self.dissimilarity_types = ['cityblock',
                                    'cosine',
                                    'euclidean',
                                    'extended_jaccard',
                                    'infinity_norm',
                                    'dtw',
                                    'max_moving_euclidean',
                                    'min_moving_euclidean']
        self.set_params(command, fraction_delta, n_iterations,
                        init_type, dissimilarity_type, cluster_factor, verbose)

    def set_params(self, command=os.getcwd() + '/vispipeline/vp-run',
                   fraction_delta=8.0,
                   n_iterations=50,
                   init_type='fastmap',
                   dissimilarity_type='euclidean',
                   cluster_factor=4.5,
                   verbose=False):
        self.command = command
        self.verbose = verbose
        self.fraction_delta = fraction_delta
        self.n_iterations = n_iterations
        self.cluster_factor = cluster_factor

        try:
            self.init_type_index = self.init_types.index(init_type)
        except:
            raise ValueError('Invalid init type: %s. Valid values are %s'
                             % (init_type, ','.join(self.init_types)))

        try:
            self.dissimilarity_type_index = self.dissimilarity_types.index(
                dissimilarity_type)
        except:
            raise ValueError('Invalid dissimilarity index: %s. Valid values are %s'
                             % (dissimilarity_type, ','.join(self.dissimilarity_types)))

        # TODO: fill with valid ranges
        if self.fraction_delta < 0.0:
            raise ValueError('Invalid fraction delta')

        # TODO: fill with valid ranges
        if self.n_iterations < 1:
            raise ValueError('Invalid n_iterations')

        # TODO: fill with valid ranges
        if self.cluster_factor < 0.0:
            raise ValueError('Invalid cluster factor')

    def fit_transform(self, X, y=None):
        return super(ProjectionByClustering, self)._run(X, y,
                                                        [self.dissimilarity_type_index,
                                                         self.cluster_factor,
                                                         self.fraction_delta,
                                                         self.n_iterations,
                                                         self.init_type_index])


class LandmarkIsomap(VispipelineProjection):
    # 1. Number of Neighbors (int, default: 8)
    # 5. Dissimilarity Type (DissimilarityType, default: 2)
    #    0. City-block
    #    1. Cosine-based dissimilarity
    #    2. Euclidean
    #    3. Extended Jaccard
    #    4. Infinity norm
    #    5. Dynamic Time Warping (DTW)
    #    6. Max Moving Euclidean
    #    7. Min Moving Euclidean
    def __init__(self, command=os.getcwd() + '/vispipeline/vp-run',
                 n_neighbors=8,
                 dissimilarity_type='euclidean',
                 verbose=False):
        super(LandmarkIsomap, self).__init__(
            projection='lisomap', command=command, verbose=verbose)

        self.dissimilarity_types = ['cityblock',
                                    'cosine',
                                    'euclidean',
                                    'extended_jaccard',
                                    'infinity_norm',
                                    'dtw',
                                    'max_moving_euclidean',
                                    'min_moving_euclidean']
        self.set_params(command, n_neighbors, dissimilarity_type, verbose)

    def set_params(self, command=os.getcwd() + '/vispipeline/vp-run',
                   n_neighbors=8,
                   dissimilarity_type='euclidean',
                   verbose=False):
        self.command = command
        self.verbose = verbose
        self.n_neighbors = n_neighbors

        try:
            self.dissimilarity_type_index = self.dissimilarity_types.index(
                dissimilarity_type)
        except:
            raise ValueError('Invalid dissimilarity index: %s. Valid values are %s'
                             % (dissimilarity_type, ','.join(self.dissimilarity_types)))

        # TODO: fill with valid ranges
        if self.n_neighbors < 1:
            raise ValueError('Invalid n_neighbors')

    def fit_transform(self, X, y=None):
        return super(LandmarkIsomap, self)._run(X, y,
                                                [self.dissimilarity_type_index,
                                                 self.n_neighbors])
