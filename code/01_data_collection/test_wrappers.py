#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from sklearn import datasets
import mtsne
import tapkee
import vp
import drtoolbox
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import NMF

X, y = datasets.load_iris(True)

mm = MinMaxScaler()
X = mm.fit_transform(X)

projections = [vp.PLMP(),
               vp.IDMAP(),
               vp.LSP(),
               vp.PLSP(),
               vp.LAMP(),
               vp.Fastmap(),
               vp.LandmarkIsomap(),
               vp.RapidSammon(),
               vp.ProjectionByClustering()]

for p in projections:
    proj_name = p.__class__.__name__
    print('---------------------------------------------------')
    print(proj_name)
    p.set_params(command=os.getcwd() + '/vispipeline/vp-run', verbose=True)
    X_new = p.fit_transform(X)
    print(proj_name, X_new.shape)

projections = [tapkee.DiffusionMaps(),
               tapkee.StochasticProximityEmbedding(),
               tapkee.LandmarkMDS(),
               tapkee.LocalityPreservingProjections(),
               tapkee.LinearLocalTangentSpaceAlignment(),
               tapkee.NeighborhoodPreservingEmbedding()]

for p in projections:
    proj_name = p.__class__.__name__
    print('---------------------------------------------------')
    print(proj_name)
    p.set_params(verbose=True)
    X_new = p.fit_transform(X)
    print(proj_name, X_new.shape)

projections = [drtoolbox.CCA(),
               drtoolbox.FastMVU(),
               drtoolbox.GDA(),
               drtoolbox.GPLVM(),
               drtoolbox.LLC(),
               drtoolbox.LMNN(),
               drtoolbox.LandmarkMVU(),
               drtoolbox.ManifoldChart(),
               drtoolbox.MCML(),
               drtoolbox.MVU(),
               drtoolbox.NCA(),
               drtoolbox.ProbPCA(),
               drtoolbox.Sammon()]

for p in projections:
    proj_name = p.__class__.__name__
    print('---------------------------------------------------')
    print(proj_name)
    p.set_params(verbose=True)
    X_new = p.fit_transform(X)
    print(proj_name, X_new.shape)

p = mtsne.MTSNE()
proj_name = p.__class__.__name__
print('---------------------------------------------------')
print(proj_name)
p.set_params(n_jobs=4, n_iter=2000)
X_new = p.fit_transform(X)
print(proj_name, X_new.shape)
