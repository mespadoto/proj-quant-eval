#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import collections
import joblib
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

proj = dict()
proj['AutoencoderProjection'] = 'AE'
proj['CCA'] = 'CoCA'
proj['DiffusionMaps'] = 'DM'
proj['FactorAnalysis'] = 'FA'
proj['Fastmap'] = 'FMAP'
proj['GDA'] = 'GDA'
proj['GPLVM'] = 'GPLVM'
proj['FastICA'] = 'F-ICA'
proj['FastMVU'] = 'F-MVU'
proj['IDMAP'] = 'IDMAP'
proj['Isomap'] = 'ISO'
proj['LandmarkIsomap'] = 'L-ISO'
proj['LAMP'] = 'LAMP'
proj['LinearDiscriminantAnalysis'] = 'LDA'
proj['SpectralEmbedding'] = 'LE'
proj['LLC'] = 'LLC'
proj['LocallyLinearEmbedding'] = 'LLE'
proj['LocallyLinearEmbedding'] = 'H-LLE'
proj['LocallyLinearEmbedding'] = 'M-LLE'
proj['LMNN'] = 'LMNN'
proj['LocalityPreservingProjections'] = 'LPP'
proj['LSP'] = 'LSP'
proj['LocallyLinearEmbedding'] = 'LTSA'
proj['LinearLocalTangentSpaceAlignment'] = 'L-LTSA'
proj['ManifoldChart'] = 'MC'
proj['MCML'] = 'MCML'
proj['MDS'] = 'MDS'
proj['LandmarkMDS'] = 'L-MDS'
proj['MDS'] = 'N-MDS'
proj['MVU'] = 'MVU'
proj['LandmarkMVU'] = 'L-MVU'
proj['NMF'] = 'NMF'
proj['NeighborhoodPreservingEmbedding'] = 'NPE'
proj['ProjectionByClustering'] = 'PBC'
proj['PCA'] = 'PCA'
proj['IncrementalPCA'] = 'I-PCA'
proj['KernelPCA'] = 'K-PCA'
proj['ProbPCA'] = 'P-PCA'
proj['SparsePCA'] = 'S-PCA'
proj['PLSP'] = 'PLSP'
proj['GaussianRandomProjection'] = 'G-RP'
proj['SparseRandomProjection'] = 'S-RP'
proj['RapidSammon'] = 'R-SAM'
proj['MTSNE'] = 'T-SNE'
proj['StochasticProximityEmbedding'] = 'SPE'
proj['TruncatedSVD'] = 'T-SVD'
proj['UMAP'] = 'UMAP'

sort_key = dict()
sort_key['AE'] = 'AE'
sort_key['CCA'] = 'CCA'
sort_key['CHL'] = 'CHL'
sort_key['CLM'] = 'CLM'
sort_key['CoCA'] = 'CoCA'
sort_key['CuCA'] = 'CuCA'
sort_key['DM'] = 'DM'
sort_key['DML'] = 'DML'
sort_key['EM'] = 'EM'
sort_key['FA'] = 'FA'
sort_key['FD'] = 'FD'
sort_key['FMAP'] = 'FMAP'
sort_key['FS'] = 'FS'
sort_key['GDA'] = 'GDA'
sort_key['GPLVM'] = 'GPLVM'
sort_key['GTM'] = 'GTM'
sort_key['ICA'] = 'ICA'
sort_key['F-ICA'] = 'ICA-F'
sort_key['NL-ICA'] = 'ICA-NL'
sort_key['IDMAP'] = 'IDMAP'
sort_key['ISO'] = 'ISO'
sort_key['L-ISO'] = 'ISO-L'
sort_key['KECA'] = 'KECA'
sort_key['KLP'] = 'KLP'
sort_key['LAMP'] = 'LAMP'
sort_key['LDA'] = 'LDA'
sort_key['LE'] = 'LE'
sort_key['LLC'] = 'LLC'
sort_key['LLE'] = 'LLE'
sort_key['H-LLE'] = 'LLE-H'
sort_key['M-LLE'] = 'LLE-M'
sort_key['LMNN'] = 'LMNN'
sort_key['LoCH'] = 'LoCH'
sort_key['LPP'] = 'LPP'
sort_key['LR'] = 'LR'
sort_key['LSP'] = 'LSP'
sort_key['LTSA'] = 'LTSA'
sort_key['L-LTSA'] = 'LTSA-L'
sort_key['MAF'] = 'MAF'
sort_key['MC'] = 'MC'
sort_key['MCA'] = 'MCA'
sort_key['MCML'] = 'MCML'
sort_key['MDS'] = 'MDS'
sort_key['L-MDS'] = 'MDS-L'
sort_key['MG-MDS'] = 'MDS-MG'
sort_key['N-MDS'] = 'MDS-N'
sort_key['ML'] = 'ML'
sort_key['MVU'] = 'MVU'
sort_key['FMVU'] = 'MVU-F'
sort_key['L-MVU'] = 'MVU-L'
sort_key['NeRV'] = 'NeRV'
sort_key['t-NeRV'] = 'NeRV-T'
sort_key['NMF'] = 'NMF'
sort_key['NLM'] = 'NLM'
sort_key['NN'] = 'NN'
sort_key['NPE'] = 'NPE'
sort_key['PBC'] = 'PBC'
sort_key['PC'] = 'PC'
sort_key['PCA'] = 'PCA'
sort_key['I-PCA'] = 'PCA-I'
sort_key['K-PCA-P'] = 'PCA-K-P'
sort_key['K-PCA-R'] = 'PCA-K-R'
sort_key['K-PCA-S'] = 'PCA-K-S'
sort_key['L-PCA'] = 'PCA-L'
sort_key['NL-PCA'] = 'PCA-NL'
sort_key['P-PCA'] = 'PCA-P'
sort_key['R-PCA'] = 'PCA-R'
sort_key['S-PCA'] = 'PCA-S'
sort_key['PLMP'] = 'PLMP'
sort_key['PLP'] = 'PLP'
sort_key['PLSP'] = 'PLSP'
sort_key['PM'] = 'PM'
sort_key['PP'] = 'PP'
sort_key['RBF-MP'] = 'RBF-MP'
sort_key['RP'] = 'RP'
sort_key['G-RP'] = 'RP-G'
sort_key['S-RP'] = 'RP-S'
sort_key['SAM'] = 'SAM'
sort_key['R-SAM'] = 'SAM-R'
sort_key['SDR'] = 'SDR'
sort_key['SFA'] = 'SFA'
sort_key['SMA'] = 'SMA'
sort_key['SNE'] = 'SNE'
sort_key['T-SNE'] = 'SNE-T'
sort_key['SOM'] = 'SOM'
sort_key['ViSOM'] = 'SOM-VI'
sort_key['SPE'] = 'SPE'
sort_key['G-SVD'] = 'SVD-G'
sort_key['T-SVD'] = 'SVD-T'
sort_key['TF'] = 'TF'
sort_key['UMAP'] = 'UMAP'
sort_key['VQ'] = 'VQ'

unsort_key = dict()
unsort_key['AE'] = 'AE'
unsort_key['CCA'] = 'CCA'
unsort_key['CHL'] = 'CHL'
unsort_key['CLM'] = 'CLM'
unsort_key['CoCA'] = 'CoCA'
unsort_key['CuCA'] = 'CuCA'
unsort_key['DM'] = 'DM'
unsort_key['DML'] = 'DML'
unsort_key['EM'] = 'EM'
unsort_key['FA'] = 'FA'
unsort_key['FD'] = 'FD'
unsort_key['FMAP'] = 'FMAP'
unsort_key['FS'] = 'FS'
unsort_key['GDA'] = 'GDA'
unsort_key['GPLVM'] = 'GPLVM'
unsort_key['GTM'] = 'GTM'
unsort_key['ICA'] = 'ICA'
unsort_key['ICA-F'] = 'F-ICA'
unsort_key['ICA-NL'] = 'NL-ICA'
unsort_key['IDMAP'] = 'IDMAP'
unsort_key['ISO'] = 'ISO'
unsort_key['ISO-L'] = 'L-ISO'
unsort_key['KECA'] = 'KECA'
unsort_key['KLP'] = 'KLP'
unsort_key['LAMP'] = 'LAMP'
unsort_key['LDA'] = 'LDA'
unsort_key['LE'] = 'LE'
unsort_key['LLC'] = 'LLC'
unsort_key['LLE'] = 'LLE'
unsort_key['LLE-H'] = 'H-LLE'
unsort_key['LLE-M'] = 'M-LLE'
unsort_key['LMNN'] = 'LMNN'
unsort_key['LoCH'] = 'LoCH'
unsort_key['LPP'] = 'LPP'
unsort_key['LR'] = 'LR'
unsort_key['LSP'] = 'LSP'
unsort_key['LTSA'] = 'LTSA'
unsort_key['LTSA-L'] = 'L-LTSA'
unsort_key['MAF'] = 'MAF'
unsort_key['MC'] = 'MC'
unsort_key['MCA'] = 'MCA'
unsort_key['MCML'] = 'MCML'
unsort_key['MDS'] = 'MDS'
unsort_key['MDS-L'] = 'L-MDS'
unsort_key['MDS-MG'] = 'MG-MDS'
unsort_key['MDS-N'] = 'N-MDS'
unsort_key['ML'] = 'ML'
unsort_key['MVU'] = 'MVU'
unsort_key['MVU-F'] = 'FMVU'
unsort_key['MVU-L'] = 'L-MVU'
unsort_key['NeRV'] = 'NeRV'
unsort_key['NeRV-T'] = 't-NeRV'
unsort_key['NMF'] = 'NMF'
unsort_key['NLM'] = 'NLM'
unsort_key['NN'] = 'NN'
unsort_key['NPE'] = 'NPE'
unsort_key['PBC'] = 'PBC'
unsort_key['PC'] = 'PC'
unsort_key['PCA'] = 'PCA'
unsort_key['PCA-I'] = 'I-PCA'
unsort_key['PCA-K-P'] = 'K-PCA-P'
unsort_key['PCA-K-R'] = 'K-PCA-R'
unsort_key['PCA-K-S'] = 'K-PCA-S'
unsort_key['PCA-L'] = 'L-PCA'
unsort_key['PCA-NL'] = 'NL-PCA'
unsort_key['PCA-P'] = 'P-PCA'
unsort_key['PCA-R'] = 'R-PCA'
unsort_key['PCA-S'] = 'S-PCA'
unsort_key['PLMP'] = 'PLMP'
unsort_key['PLP'] = 'PLP'
unsort_key['PLSP'] = 'PLSP'
unsort_key['PM'] = 'PM'
unsort_key['PP'] = 'PP'
unsort_key['RBF-MP'] = 'RBF-MP'
unsort_key['RP'] = 'RP'
unsort_key['RP-G'] = 'G-RP'
unsort_key['RP-S'] = 'S-RP'
unsort_key['SAM'] = 'SAM'
unsort_key['SAM-R'] = 'R-SAM'
unsort_key['SDR'] = 'SDR'
unsort_key['SFA'] = 'SFA'
unsort_key['SMA'] = 'SMA'
unsort_key['SNE'] = 'SNE'
unsort_key['SNE-T'] = 'T-SNE'
unsort_key['SOM'] = 'SOM'
unsort_key['SOM-VI'] = 'ViSOM'
unsort_key['SPE'] = 'SPE'
unsort_key['SVD-G'] = 'G-SVD'
unsort_key['SVD-T'] = 'T-SVD'
unsort_key['TF'] = 'TF'
unsort_key['UMAP'] = 'UMAP'
unsort_key['VQ'] = 'VQ'


df = pd.read_csv('full.csv')
df_rerun = pd.read_csv('full_rerun.csv')
df_svhn = pd.read_csv('full_svhn.csv')
df = pd.concat([df, df_rerun, df_svhn]).reset_index(drop=True)

df['params'] = df['projection_parameters'].copy(deep=True)

df['projection_parameters'] = df['projection_parameters'].str.replace('<ModelSize\.(\w+?)\:\s\d>', lambda m: "'" + m.group(1) + "'")
df = df[df['elapsed_time'] > 0.0]

df = df.reset_index(drop=True)

new_params = []
sel_params = []
new_proj_names = []

#    perl -pi -e "s/<ModelSize\.(\w+?)\:\s\d>/'\1'/g" full.csv
#    "s/\'random_state\': 42//g" full.csv
#    perl -pi -e "s/\'verbose\': False//g" full.csv
#perl -pi -e "s/\'max_iter\': None//g" full.csv
#perl -pi -e "s/\'max_iter\': 0//g" full.csv
#    perl -pi -e "s/\'n_components\': 2//g" full.csv
#perl -pi -e "s/\s,\s//g" full.csv
#perl -pi -e "s/,,\s}/}/g" full.csv
#perl -pi -e "s/{,\s/{/g" full.csv

params_remove = ['n_components', 'verbose', 'random_state', 'density', 'normalize_components', 'angle', 'beta_loss', 'dissimilarity_type', 'eigen_solver', 'method', 'metric', 'min_grad_norm', 'n_jobs', 'n_iter_without_progress', 'angular_rp_forest', 'algorithm']

for row in df.iterrows():
    proj_name = row[1]['projection_name']
    param_dict = eval(row[1]['projection_parameters'])
    
    new_proj_name = proj[proj_name]
    
    if proj_name == 'LocallyLinearEmbedding':
        if param_dict['method'] == 'ltsa':
            new_proj_name = 'LTSA'
        elif param_dict['method'] == 'hessian':
            new_proj_name = 'H-LLE'
        elif param_dict['method'] == 'modified':
            new_proj_name = 'M-LLE'
        else:
            new_proj_name = 'LLE'
    elif proj_name == 'MDS':
        if not param_dict['metric']:
            new_proj_name = 'N-MDS'
        else:
            new_proj_name = 'MDS'
    elif proj_name == 'KernelPCA':
        del param_dict['max_iter']
        
        if param_dict['kernel'] == 'poly':
            new_proj_name = 'K-PCA-P'
        elif param_dict['kernel'] == 'rbf':
            new_proj_name = 'K-PCA-R'
        elif param_dict['kernel'] == 'sigmoid':
            new_proj_name = 'K-PCA-S'
        del param_dict['kernel']
    elif proj_name == 'MTSNE':
        del param_dict['init']
        del param_dict['learning_rate']
    elif proj_name == 'StochasticProximityEmbedding':
        del param_dict['max_iter']

    new_param_dict = collections.OrderedDict()
    sel_param_dict = collections.OrderedDict()

    for k in params_remove:
        if param_dict.get(k) != None:
            del param_dict[k]

    by_val = str in [type(t) for t in param_dict.values()]

    if by_val:
        d = collections.OrderedDict(sorted(param_dict.items(), key=lambda kv: '_' + str(kv[1]) if type(kv[1]) == str else kv[0]))
    else:
        d = collections.OrderedDict(sorted(param_dict.items()))

    i=1
    for k, v in d.items():
        new_param_dict[i] = v
        sel_param_dict[k] = v
        
        i+=1

    new_params.append(dict(new_param_dict))
    sel_params.append(dict(sel_param_dict))
    new_proj_names.append(new_proj_name)


df['new_proj_name'] = new_proj_names
df['new_proj_name_sort'] = [sort_key[k] for k in new_proj_names]
df['new_params'] = new_params
df['sel_params'] = sel_params

#remove NPE, LDA
df = df[~(df['new_proj_name'] == 'NPE')]
df = df[~(df['new_proj_name'] == 'LDA')]
df = df.reset_index(drop=True)

df['id_run'] = df['params'].apply(lambda id_run: id_run.replace(' ', '').replace("'", "").replace(':', '').replace('{', '').replace('}', '').replace('/', '').replace(',', '').replace('-', '').replace('.', ''))

ret = pd.concat([df, pd.DataFrame((d for idx, d in df['new_params'].iteritems()))], axis=1)
ret = pd.concat([ret, pd.DataFrame((d for idx, d in df['sel_params'].iteritems()))], axis=1)

ret['key'] = ret['dataset_name'] + '|' + ret['projection_name'] + '|' + df['id_run']

results = ret[[  'key',
                 'dataset_name',
                 'id_run',
                 'new_proj_name',
                 'new_proj_name_sort',
                 'projection_name',
                 'projection_parameters',
                 'metric_pq_continuity_k_07',
                 'metric_pq_trustworthiness_k_07',
                 'metric_pq_neighborhood_hit_k_07',
                 'metric_pq_shepard_diagram_correlation',
                 'metric_pq_normalized_stress', 1,2,3,4]]

stress = pd.read_csv('stress.csv', sep=';', decimal=',')
stress_rerun = pd.read_csv('stress_rerun.csv', sep=';', decimal=',')
stress_svhn = pd.read_csv('stress_svhn.csv', sep=';', decimal=',')
stress = pd.concat([stress, stress_rerun, stress_svhn]).reset_index(drop=True)

stress['key'] = stress['dataset'] + '|' + stress['projection'] + '|' + stress['id_run']

results = results.merge(stress, on='key', how='left')

results = results[[  'key',
                     'dataset_name',
                     'id_run_x',
                     'new_proj_name',
                     'new_proj_name_sort',
                     'projection_name',
                     'projection_parameters',
                     'metric_pq_continuity_k_07',
                     'metric_pq_trustworthiness_k_07',
                     'metric_pq_neighborhood_hit_k_07',
                     'metric_pq_shepard_diagram_correlation',
                     'normalized_stress', 1,2,3,4]]

def normalize_col(col):
    scaler = MinMaxScaler()
    X = np.array(np.array(col).reshape((-1, 1)))
    X[np.isnan(X)] = -1
    X = scaler.fit_transform(X)
    return X.squeeze()


results['metric_pq_shepard_diagram_correlation'] = normalize_col(results['metric_pq_shepard_diagram_correlation'])
results['normalized_stress'] = normalize_col(results['normalized_stress'])

results['mu'] = (results['metric_pq_continuity_k_07'] + results['metric_pq_trustworthiness_k_07'] + results['metric_pq_neighborhood_hit_k_07'] + results['metric_pq_shepard_diagram_correlation'] + (1 - results['normalized_stress']))/5

results = results.sort_values(['dataset_name', 'new_proj_name_sort'])
results.to_csv('results_mu.csv', index=None, sep=';', decimal=',')

idx = results.groupby(['dataset_name', 'new_proj_name_sort'])['mu'].idxmax()
param_var = results.loc[idx][['new_proj_name', 'new_proj_name_sort', 'dataset_name', 1, 2, 3, 4]]

param_best = results.loc[idx][['new_proj_name', 'new_proj_name_sort', 'dataset_name', 'projection_parameters', 1, 2, 3, 4, 'metric_pq_shepard_diagram_correlation', 'metric_pq_neighborhood_hit_k_07']]
param_best.to_csv('param_best.csv', index=None)

def custom_hist(a):
    return np.histogram(a, bins=(0.0, 0.62, 0.75, 0.87, 1.0))

tmp = results[['dataset_name', 'new_proj_name_sort', 'mu']].sort_values(['dataset_name', 'new_proj_name_sort']).groupby(['dataset_name', 'new_proj_name_sort']).agg(custom_hist)
param_hist = pd.DataFrame(pd.DataFrame(tmp.iloc[:,0].tolist())[0].tolist())

tmp2 = results[['dataset_name', 'new_proj_name_sort']].drop_duplicates().sort_values(['dataset_name', 'new_proj_name_sort']).reset_index(drop=True)

param_hist['dataset_name'] = tmp2['dataset_name']
param_hist['new_proj_name_sort'] = tmp2['new_proj_name_sort']

param_hist['b1'] = param_hist[0]
param_hist['b2'] = param_hist[1]
param_hist['b3'] = param_hist[2]
param_hist['b4'] = param_hist[3]

param_hist = param_hist[['dataset_name', 'new_proj_name_sort', 'b1', 'b2', 'b3', 'b4']]

def normalize_row(r):
    minval = min(r['b1'], r['b2'], r['b3'], r['b4'])
    maxval = max(r['b1'], r['b2'], r['b3'], r['b4'])

    return [(r['b1']-minval)/(maxval-minval + 0.0001),
            (r['b2']-minval)/(maxval-minval + 0.0001),
            (r['b3']-minval)/(maxval-minval + 0.0001),
            (r['b4']-minval)/(maxval-minval + 0.0001)]

tmp3 = param_hist.apply(normalize_row, axis=1, result_type='expand')

param_hist['nb1'] = tmp3[0]
param_hist['nb2'] = tmp3[1]
param_hist['nb3'] = tmp3[2]
param_hist['nb4'] = tmp3[3]

param_hist['new_proj_name'] = [unsort_key[k] for k in param_hist['new_proj_name_sort']]
param_hist[['dataset_name', 'new_proj_name', 'b1', 'b2', 'b3', 'b4', 'nb1', 'nb2', 'nb3', 'nb4']].to_csv('heatmap_11_param_hist.csv', index=None)

def custom_var(c):
    if c.dtype == 'O':
        x = np.array(c[~c.isna()])
        
        cat = np.unique(x) #[str(s) for s in x]

        l = len(x)
        ss = 0

        if len(x) == 0:
            return 0.0

        for c in cat:
            ss += (np.sum(np.isin(x, c).astype('int32'))/(l))**2

        return (1 - np.sqrt(ss)) / (1 - (1 / (l-1)))
    else:
        x = np.array(c)

        if np.std(x) == 0:
            return 0.0

        x[np.isnan(x)] = 0.0
        x = (x - np.min(x)) / (np.max(x) - np.min(x) + 0.0001)
        return np.std(x)

#heat map 1
pvt = results.pivot_table(index='new_proj_name_sort', columns='dataset_name', aggfunc=max)['mu']

#heat map 2, param variance
pmap = param_var[['new_proj_name_sort', 1, 2, 3, 4]].groupby('new_proj_name_sort').agg(custom_var)

pvt['new_proj_name'] = [unsort_key[k] for k in pvt.index]
pmap['new_proj_name'] = [unsort_key[k] for k in pmap.index]

pvt[['new_proj_name', 'bank', 'cifar10', 'cnae9', 'coil20', 'epileptic', 'fashion_mnist', 'fmd', 'har', 'hatespeech', 'hiva', 'imdb', 'orl', 'secom', 'seismic', 'sentiment', 'sms', 'spambase', 'svhn']].to_csv('heatmap_11.csv', index=None)
pmap[['new_proj_name', 1, 2, 3, 4]].to_csv('heatmap_12.csv', index=None)

param_var['preset'] = param_var[1].astype(str) + '|' + param_var[2].astype(str) + '|' + param_var[3].astype(str) + '|' + param_var[4].astype(str)
results['preset'] = results[1].astype(str) + '|' + results[2].astype(str) + '|' + results[3].astype(str) + '|' + results[4].astype(str)

param_var = param_var.reset_index(drop=True)

#useful to check how many datasets didnt work with some projection
dataset_count = param_var[['new_proj_name_sort', 'preset']].groupby('new_proj_name_sort').count()

def custom_mode(c):
    return pd.DataFrame.mode(c).iloc[0]

common_presets = param_var[['new_proj_name_sort', 'preset']].groupby('new_proj_name_sort').apply(custom_mode).reset_index(drop=True)

results_preset = results.merge(common_presets, on='preset', suffixes=['', '_y'])
pvt_preset = results_preset.pivot_table(index='new_proj_name_sort', columns='dataset_name', aggfunc=max)['mu']

pvt_preset['new_proj_name'] = [unsort_key[k] for k in pvt_preset.index]
pvt_preset[['new_proj_name', 'bank', 'cifar10', 'cnae9', 'coil20', 'epileptic', 'fashion_mnist', 'fmd', 'har', 'hatespeech', 'hiva', 'imdb', 'orl', 'secom', 'seismic', 'sentiment', 'sms', 'spambase', 'svhn']].to_csv('heatmap_21.csv', index=None)

cols = common_presets['preset'].str.split('|', expand=True)
common_presets[1] = cols[0]
common_presets[2] = cols[1]
common_presets[3] = cols[2]
common_presets[4] = cols[3]

common_presets['new_proj_name'] = [unsort_key[k] for k in common_presets['new_proj_name_sort']]
common_presets[['new_proj_name', 1, 2, 3, 4]].to_csv('heatmap_22.csv', index=None)

#Star plot data
#TODO: filter projections and datasets?
idx = results.reset_index(drop=True).groupby(['dataset_name', 'new_proj_name'])['mu'].idxmax()

starplot = results.iloc[idx][[ 'new_proj_name',
                     'dataset_name',
                     'metric_pq_continuity_k_07',
                     'metric_pq_trustworthiness_k_07',
                     'metric_pq_neighborhood_hit_k_07',
                     'metric_pq_shepard_diagram_correlation',
                     'normalized_stress']]

X = starplot[[   'metric_pq_continuity_k_07',
                 'metric_pq_trustworthiness_k_07',
                 'metric_pq_neighborhood_hit_k_07',
                 'metric_pq_shepard_diagram_correlation',
                 'normalized_stress']]

X = np.array(X)
y = starplot['new_proj_name']

starplot.to_csv('pcp.csv', index=None)

lenc = LabelEncoder()
y_cat = lenc.fit_transform(y)

from umap import UMAP

umap = UMAP(n_components=2, random_state=42, min_dist=0.0, n_neighbors=30)
X_new = umap.fit_transform(X)

scaler = MinMaxScaler()
X_new = scaler.fit_transform(X_new)

np.save('X_starplot.npy', X_new)
np.save('y_starplot.npy', y_cat)
np.save('y_starplot_names.npy', np.array(y))

##SPLOM data
##TODO: how to rank?
##TODO: use stress.py script?
#idx = results.groupby(['dataset_name', 'new_proj_name'])['metric_pq_shepard_diagram_correlation'].idxmax()
#
#splom = results.iloc[idx]
##seismic_D_high_list.npy
##seismic_D_low_list_UMAP|angular_rp_forestFalseinitspectralmetriceuclideanmin_dist01n_components2n_neighbors10random_state42spread10.npy
#
#
#
#
##TODO: finding top projections for each dataset
#datasets = list(pvt.columns)
#proj_per_dataset = dict()
#
#for d in datasets:
#    tmp = pvt[d].sort_values(ascending=False)[:5]
#
#    res = []    
#    for i in range(5):
#        res.append(str(tmp[i]) + ',' + tmp.index[i])
#    
#    proj_per_dataset[d] = res
#
#proj_dataset = pd.DataFrame.from_dict(proj_per_dataset)
