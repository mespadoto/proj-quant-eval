#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import umap
import numpy as np
from sklearn import (decomposition, discriminant_analysis, manifold,
                     random_projection)
from time import perf_counter
import mtsne
import tapkee
import drtoolbox
import ae
import vp
import metrics
import traceback


def run_projection(proj, X, y, id_run, dataset_name, output_dir):
    t0 = perf_counter()

    try:
        X_new = proj.fit_transform(X, y)
    except:
        print('----------------------------------------------------')
        print('Error running %s: ' % id_run)
        reason, _, tb = sys.exc_info()
        print('Reason:')
        print(reason)
        print('Traceback:')
        traceback.print_tb(tb, file=sys.stdout)
        print('----------------------------------------------------')
        return np.zeros((X.shape[0], 2)), y, metrics.empty_pq_metrics()

    elapsed_time = perf_counter() - t0

    if X_new.shape[0] != X.shape[0]:
        print('----------------------------------------------------')
        print("Error running %s: Projection returned %d rows when %d rows were expected" % (id_run, X_new.shape[0], X.shape[0]))
        print('----------------------------------------------------')
        return np.zeros((X.shape[0], 2)), y, metrics.empty_pq_metrics()

    if len(X_new.shape) != 2 or X_new.shape[1] != 2:
        print('----------------------------------------------------')
        print("Error running %s: Projection did not return 2 columns: " % id_run, X_new.shape)
        print('----------------------------------------------------')
        return np.zeros((X.shape[0], 2)), y, metrics.empty_pq_metrics()

    return X_new, y, metrics.eval_pq_metrics(X=X_new, y=y, elapsed_time=elapsed_time, id_run=id_run, dataset_name=dataset_name, output_dir=output_dir)


all_projections = dict()

all_projections['AE']      = (ae.AutoencoderProjection(), {'n_components': [2], 'model_size': [ae.ModelSize.SMALL, ae.ModelSize.MEDIUM, ae.ModelSize.LARGE]})
all_projections['DM']      = (tapkee.DiffusionMaps(), {'t': [2, 5, 10], 'width': [1.0, 5.0, 10.0], 'verbose': [False]})
all_projections['FA']      = (decomposition.FactorAnalysis(), {'n_components': [2], 'max_iter': [1000, 2000], 'random_state': [42]})
all_projections['FICA']    = (decomposition.FastICA(), {'n_components': [2], 'fun': ['logcosh', 'exp'], 'max_iter': [200, 400], 'random_state': [42]})
all_projections['FMAP']    = (vp.Fastmap(), {'verbose': [False], 'dissimilarity_type': ['euclidean']})
all_projections['FMVU']    = (drtoolbox.FastMVU(), {'k': [8, 12, 15], 'verbose': [False]})
all_projections['GDA']     = (drtoolbox.GDA(), {'kernel': ['gauss', 'linear'], 'verbose': [False]})
all_projections['GPLVM']   = (drtoolbox.GPLVM(), {'sigma': [0.5, 1.0, 2.0], 'verbose': [False]})
all_projections['GRP']     = (random_projection.GaussianRandomProjection(), {'n_components': [2], 'random_state': [42]})
all_projections['HLLE']    = (manifold.LocallyLinearEmbedding(), {'n_components': [2], 'n_neighbors': [7, 11], 'max_iter': [100, 200], 'reg': [0.001, 0.01, 0.1], 'method': ['hessian'], 'eigen_solver': ['dense'], 'random_state': [42]})
all_projections['IDMAP']   = (vp.IDMAP(), {'verbose': [False], 'fraction_delta': [2.0, 8.0, 12.0], 'n_iterations': [100, 200], 'init_type': ['fastmap', 'random'], 'dissimilarity_type': ['euclidean']})
all_projections['IPCA']    = (decomposition.IncrementalPCA(), {'n_components': [2]})
all_projections['ISO']     = (manifold.Isomap(), {'n_components': [2], 'n_neighbors': [3, 5, 7], 'eigen_solver': ['dense']})
all_projections['KPCAPol'] = (decomposition.KernelPCA(), {'n_components': [2], 'gamma': [None] + [0.05, 0.05, 0.5], 'degree': [2, 3, 5], 'kernel': ['poly'], 'max_iter': [None], 'random_state': [42]})
all_projections['KPCARbf'] = (decomposition.KernelPCA(), {'n_components': [2], 'gamma': [None] + [0.05, 0.05, 0.5], 'kernel': ['rbf'], 'max_iter': [None], 'random_state': [42]})
all_projections['KPCASig'] = (decomposition.KernelPCA(), {'n_components': [2], 'gamma': [None] + [0.05, 0.05, 0.5], 'degree': [3], 'kernel': ['sigmoid'], 'max_iter': [None], 'random_state': [42]})
all_projections['LAMP']    = (vp.LAMP(), {'verbose': [False], 'fraction_delta': [2.0, 8.0, 12.0], 'n_iterations': [100, 200], 'sample_type': ['random', 'clustering_centroid']})
all_projections['LE']      = (manifold.SpectralEmbedding(), {'n_components': [2], 'affinity': ['nearest_neighbors'], 'random_state': [42]})
all_projections['LISO']    = (vp.LandmarkIsomap(), {'verbose': [False], 'n_neighbors': [4, 8, 16], 'dissimilarity_type': ['euclidean']})
all_projections['LLC']     = (drtoolbox.LLC(), {'k': [8, 12], 'n_analyzers': [10, 20], 'max_iter': [200, 400], 'verbose': [False]})
all_projections['LLE']     = (manifold.LocallyLinearEmbedding(), {'n_components': [2], 'n_neighbors': [5, 7, 11], 'max_iter': [100, 200], 'reg': [0.001, 0.01, 0.1], 'method': ['standard'], 'eigen_solver': ['dense'], 'random_state': [42]})
all_projections['LLTSA']   = (tapkee.LinearLocalTangentSpaceAlignment(), {'n_neighbors': [4, 7, 11], 'verbose': [False]}) # subject to "eigendecomposition failed" errors (Eigen's NoConvergence)
all_projections['LMDS']    = (tapkee.LandmarkMDS(), {'n_neighbors': [4, 7, 11], 'verbose': [False]})
all_projections['LMNN']    = (drtoolbox.LMNN(), {'k': [3, 5, 7], 'verbose': [False]})
all_projections['LMVU']    = (drtoolbox.LandmarkMVU(), {'k1': [3, 5, 7], 'k2': [8, 12, 15], 'verbose': [False]})
all_projections['LPP']     = (tapkee.LocalityPreservingProjections(), {'n_neighbors': [4, 7, 11], 'verbose': [False]}) # subject to "eigendecomposition failed" errors (Eigen's NoConvergence)
all_projections['LSP']     = (vp.LSP(), {'verbose': [False], 'fraction_delta': [2.0, 8.0, 12.0], 'n_iterations': [100, 200], 'n_neighbors': [4, 8, 16], 'control_point_type': ['random', 'kmeans'], 'dissimilarity_type': ['euclidean']})
all_projections['LTSA']    = (manifold.LocallyLinearEmbedding(), {'n_components': [2], 'n_neighbors': [5, 7, 11], 'max_iter': [100, 200], 'reg': [0.001, 0.01, 0.1], 'method': ['ltsa'], 'eigen_solver': ['dense'], 'random_state': [42]})
all_projections['MC']      = (drtoolbox.ManifoldChart(), {'n_analyzers': [10, 20], 'max_iter': [200, 400], 'verbose': [False]})
all_projections['MCML']    = (drtoolbox.MCML(), {'verbose': [False]})
all_projections['MDS']     = (manifold.MDS(), {'n_components': [2], 'n_init': [2, 4], 'metric': [True], 'max_iter': [300, 500], 'random_state': [42]})
all_projections['MLLE']    = (manifold.LocallyLinearEmbedding(), {'n_components': [2], 'n_neighbors': [5, 7, 11], 'max_iter': [100, 200], 'reg': [0.001, 0.01, 0.1], 'method': ['modified'], 'eigen_solver': ['dense'], 'random_state': [42]})
all_projections['MVU']     = (drtoolbox.MVU(), {'k': [8, 12, 15], 'verbose': [False]})
all_projections['NMDS']    = (manifold.MDS(), {'n_components': [2], 'n_init': [2, 4], 'metric': [False], 'max_iter': [300, 500], 'random_state': [42]})
all_projections['NMF']     = (decomposition.NMF(), {'n_components': [2], 'init': ['random', 'nndsvdar'], 'beta_loss': ['frobenius'], 'max_iter': [200, 400], 'alpha': [0, 0.5], 'l1_ratio': [0.0, 0.5], 'random_state': [42]})
all_projections['PBC']     = (vp.ProjectionByClustering(), {'verbose': [False], 'fraction_delta': [2.0, 8.0, 12.0], 'n_iterations': [100, 200], 'init_type': ['fastmap', 'random'], 'dissimilarity_type': ['euclidean'], 'cluster_factor': [1.5, 4.5, 9.0]})
all_projections['PCA']     = (decomposition.PCA(), {'n_components': [2], 'random_state': [42]})
all_projections['PLSP']    = (vp.PLSP(), {'dissimilarity_type': ['euclidean'], 'verbose': [False], 'sample_type': ['clustering']})
all_projections['PPCA']    = (drtoolbox.ProbPCA(), {'max_iter': [200, 400], 'verbose': [False]})
all_projections['RSAM']    = (vp.RapidSammon(), {'verbose': [False], 'dissimilarity_type': ['euclidean']})
all_projections['SPCA']    = (decomposition.SparsePCA(), {'n_components': [2], 'alpha': [0.01, 0.1, 0.5], 'ridge_alpha': [0.05, 0.05, 0.5], 'max_iter': [1000, 2000], 'tol': [1e-08], 'method': ['lars'], 'random_state': [42], 'normalize_components': [True]})
all_projections['SPE']     = (tapkee.StochasticProximityEmbedding(), {'n_neighbors': [6, 12, 18], 'n_updates': [20, 70], 'max_iter': [0], 'verbose': [False]})
all_projections['SRP']     = (random_projection.SparseRandomProjection(), {'n_components': [2], 'density': ['auto'], 'random_state': [42]})
all_projections['TSNE']    = (mtsne.MTSNE(), {'n_components': [2], 'perplexity': [5.0, 15.0, 30.0, 50.0], 'early_exaggeration': [6.0, 12.0, 18.0], 'learning_rate': [200.0], 'n_iter': [1000, 3000], 'n_iter_without_progress': [300], 'min_grad_norm': [1e-07], 'metric': ['euclidean'], 'init': ['random'], 'random_state': [42], 'method': ['barnes_hut'], 'angle': [0.5], 'n_jobs': [4]})
all_projections['TSVD']    = (decomposition.TruncatedSVD(), {'n_components': [2], 'algorithm': ['randomized'], 'n_iter': [5, 10], 'random_state': [42]})
all_projections['UMAP']    = (umap.UMAP(), {'n_components': [2], 'random_state': [42], 'n_neighbors': [5, 10, 15], 'metric': ['euclidean'], 'init': ['spectral', 'random'], 'min_dist': [0.001, 0.01, 0.1, 0.5], 'spread': [1.0], 'angular_rp_forest': [False]})
