#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage import io

import matplotlib as mpl
import matplotlib.colors as colors

#/home/mespadoto/anaconda3/pkgs/matplotlib-2.2.2-py36h0e671d2_1/lib/python3.6/site-packages/matplotlib/_cm.py

def get_custom_YlOrRd():
    spec = (
    (1.0                , 1.0                 , 1.0                ),
    (1.0                , 0.92941176470588238 , 0.62745098039215685),
    (0.99607843137254903, 0.85098039215686272 , 0.46274509803921571),
    (0.99607843137254903, 0.69803921568627447 , 0.29803921568627451),
    (0.99215686274509807, 0.55294117647058827 , 0.23529411764705882),
    (0.9882352941176471 , 0.30588235294117649 , 0.16470588235294117),
    (0.8901960784313725 , 0.10196078431372549 , 0.10980392156862745),
    (0.74117647058823533, 0.2                 , 0.14901960784313725),
    (0.60196078431372548, 0.2                 , 0.14901960784313725)
    )

    lutsize = mpl.rcParams['image.lut']
    return colors.LinearSegmentedColormap.from_list('CustomYlOrRd', spec, lutsize)


def get_custom_RdYlGn():
    spec = (
        (0.7470588235294118 , 0.0                 , 0.14901960784313725),
        (0.84313725490196079, 0.18823529411764706 , 0.15294117647058825),
        (0.95686274509803926, 0.42745098039215684 , 0.2627450980392157 ),
        (0.99215686274509807, 0.68235294117647061 , 0.38039215686274508),
        (0.99607843137254903, 0.8784313725490196  , 0.54509803921568623),
        (1.0                , 1.0                 , 0.4), #0.74901960784313726),
        (0.85098039215686272, 0.93725490196078431 , 0.54509803921568623),
        (0.65098039215686276, 0.85098039215686272 , 0.41568627450980394),
        (0.4                , 0.74117647058823533 , 0.38823529411764707),
        (0.10196078431372549, 0.59607843137254901 , 0.31372549019607843),
        (0.0                , 0.60784313725490196 , 0.21568627450980393)
        )

    lutsize = mpl.rcParams['image.lut']
    return colors.LinearSegmentedColormap.from_list('CustomRdYlGn', spec, lutsize)

def heatmap_with_bars(rows, cols, values, bars, colormap, figname, cell_size=100, visible_zeros=True, annotations=True):
    h = rows*cell_size
    w = cols*cell_size
    
    hmap = np.zeros((h+2, w+2, 3)).astype('uint8')
    
    fig = plt.figure(figsize=(cols, rows))
    fig.tight_layout()
    ax = fig.add_subplot(111)
    
    for r in range(rows):
        for c in range(cols):
            cell_value = values[r, c]
            params = bars[r, c]
            
            r_beg = r*cell_size+1
            r_end = (r+1)*cell_size+1
            
            c_beg = c*cell_size+1
            c_end = (c+1)*cell_size+1
            
            if not colormap:
                bg_color = (255, 255, 255)
                fg_color = (255, 255, 255)
                fmt = '{:g}'
            else:
                bg_color = (np.array(colormap(cell_value))[:3]*255).astype('uint8')
                fg_color = (bg_color * 0.8).astype('uint8')
                fmt = '{:5.2f}'
            
            hmap[r_beg:r_end,c_beg:c_end,:] = bg_color

            if annotations:
                if type(cell_value) == str:
                    ax.annotate('{:s}'.format(cell_value), xy=(c_beg+7, r_beg+35), fontsize=16, alpha=0.8)
                elif cell_value > 0.0 or visible_zeros:
                    ax.annotate(fmt.format(cell_value), xy=(c_beg+7, r_beg+35), fontsize=16, alpha=0.8)
            
            for i, p in enumerate((params*cell_size*0.75).astype('uint32')):
                r_bar = r_end - p
                c_bar = c_beg+(i*cell_size//len(params))
                
                hmap[r_bar:r_end,c_bar:c_bar+cell_size//len(params),:] = fg_color
    
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.set_frame_on(False)
    ax.axis('off')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.tick_params(top=False, bottom=False, left=False, right=False, labelleft=False, labelbottom=False, labeltop=False)
    ax.imshow(hmap)
    fig.savefig(figname, dpi=300, bbox_inches='tight', pad_inches=0)


def concat_images(a, b, output):
    tmp1 = io.imread(a)
    tmp2 = io.imread(b)

    tmp = np.zeros((tmp1.shape[0], tmp1.shape[1] + tmp2.shape[1], 4)).astype('uint8')
    tmp[:,:tmp1.shape[1],:] = tmp1
    tmp[:,tmp1.shape[1]:tmp1.shape[1]+tmp2.shape[1],:] = tmp2

    io.imsave(output, tmp)



#7.1: heatmap 1
pvt = pd.read_csv('heatmap_11.csv')
param_hist = pd.read_csv('heatmap_11_param_hist.csv')

bars = np.zeros((pvt.shape[0], pvt.shape[1]-1, 4))

for i, c in enumerate(['nb1', 'nb2', 'nb3', 'nb4']):
    bars[:,:,i] = param_hist[['new_proj_name', 'dataset_name', c]].set_index(['new_proj_name', 'dataset_name']).unstack().values

projections = list(pvt['new_proj_name'])
datasets    = list(pvt.columns)[1:]

data = np.array(pvt)[:,1:].astype('float32')
data[np.isnan(data)] = 0.0

heatmap1_data = pd.DataFrame(data)
heatmap1_data.columns = datasets
heatmap1_data['projection_name'] = projections

RdYlGn = get_custom_RdYlGn()

heatmap_with_bars(data.shape[0], data.shape[1], data, bars, RdYlGn, 'heatmap1_bar_labeled.png', cell_size=100, visible_zeros=True)
heatmap_with_bars(data.shape[0], data.shape[1], data, bars, RdYlGn, 'heatmap1_bar.png', cell_size=100, annotations=False)

#7.1: heatmap 2
pmap = pd.read_csv('heatmap_12.csv')
#param_hist = pd.read_csv('heatmap_11_param_hist.csv')

bars = np.zeros((pmap.shape[0], pmap.shape[1]-1, 4))

projections = list(pmap['new_proj_name'])
datasets    = list(pmap.columns)[1:]

data = np.array(pmap)[:,1:].astype('float32')
data[np.isnan(data)] = 0.0

for i in range(4):
    heatmap1_data[str(i+1)] = data[:,i]

cmap = get_custom_YlOrRd()
heatmap_with_bars(data.shape[0], data.shape[1], data, bars, cmap, 'heatmap1_param_labeled.png', cell_size=100, visible_zeros=False)
heatmap_with_bars(data.shape[0], data.shape[1], data, bars, cmap, 'heatmap1_param.png', cell_size=100, annotations=False)

concat_images('heatmap1_bar.png', 'heatmap1_param.png', 'heatmap1.png')
concat_images('heatmap1_bar_labeled.png', 'heatmap1_param_labeled.png', 'heatmap1_labeled.png')


#7.2: heatmap 1
pvt = pd.read_csv('heatmap_21.csv')
bars = np.zeros((pvt.shape[0], pvt.shape[1]-1, 4))

projections = list(pvt['new_proj_name'])
datasets    = list(pvt.columns)[1:]

data = np.array(pvt)[:,1:].astype('float32')
data[np.isnan(data)] = 0.0

heatmap2_data = pd.DataFrame(data)
heatmap2_data.columns = datasets
heatmap2_data['projection_name'] = projections

RdYlGn = get_custom_RdYlGn()
heatmap_with_bars(data.shape[0], data.shape[1], data, bars, RdYlGn, 'heatmap2_bar_labeled.png', cell_size=100, visible_zeros=True)
heatmap_with_bars(data.shape[0], data.shape[1], data, bars, RdYlGn, 'heatmap2_bar.png', cell_size=100, annotations=False)

#7.2: heatmap 2
pmap = pd.read_csv('heatmap_22.csv')
bars = np.zeros((pmap.shape[0], pmap.shape[1]-1, 4))

projections = list(pmap['new_proj_name'])
datasets    = list(pmap.columns)[2:]

data = np.zeros_like(np.array(pmap)[:,1:]).astype('float32')
labels = np.array(pmap)[:,1:]

labels[labels == 'LARGE'] = 'large'
labels[labels == 'clustering_centroid'] = 'cl_cnt'
labels[labels == 'nearest_neighbors'] = 'nn'
labels[labels == 'random'] = 'rand'
labels[labels == 'clustering'] = 'clu'
labels[labels == 'kmeans'] = 'kmns'
labels[labels == 'fastmap'] = 'fmap'

data[np.isnan(data)] = 0.0

for i in range(4):
    heatmap2_data[str(i+1)] = labels[:,i]

heatmap_with_bars(labels.shape[0], labels.shape[1], labels, bars, None, 'heatmap2_param_labeled.png', cell_size=100, visible_zeros=False)
heatmap_with_bars(labels.shape[0], labels.shape[1], labels, bars, None, 'heatmap2_param.png', cell_size=100, annotations=False)

concat_images('heatmap2_bar.png', 'heatmap2_param.png', 'heatmap2.png')
concat_images('heatmap2_bar_labeled.png', 'heatmap2_param_labeled.png', 'heatmap2_labeled.png')

heatmap1_data[['projection_name', 'bank', 'cifar10', 'cnae9', 'coil20', 'epileptic', 'fashion_mnist', 'fmd', 'har', 'hatespeech', 'hiva', 'imdb', 'orl', 'secom', 'seismic', 'sentiment', 'sms', 'spambase', 'svhn', '1', '2', '3', '4']].to_csv('heatmap1_labels.csv', index=None)
heatmap2_data[['projection_name', 'bank', 'cifar10', 'cnae9', 'coil20', 'epileptic', 'fashion_mnist', 'fmd', 'har', 'hatespeech', 'hiva', 'imdb', 'orl', 'secom', 'seismic', 'sentiment', 'sms', 'spambase', 'svhn', '1', '2', '3', '4']].to_csv('heatmap2_labels.csv', index=None)


########## Colorbars
import matplotlib as mpl
import matplotlib.colors as colors
import matplotlib.pyplot as plt

def get_custom_YlOrRd():
    spec = (
    (1.0                , 1.0                 , 1.0                ),
    (1.0                , 0.92941176470588238 , 0.62745098039215685),
    (0.99607843137254903, 0.85098039215686272 , 0.46274509803921571),
    (0.99607843137254903, 0.69803921568627447 , 0.29803921568627451),
    (0.99215686274509807, 0.55294117647058827 , 0.23529411764705882),
    (0.9882352941176471 , 0.30588235294117649 , 0.16470588235294117),
    (0.8901960784313725 , 0.10196078431372549 , 0.10980392156862745),
    (0.74117647058823533, 0.2                 , 0.14901960784313725),
    (0.60196078431372548, 0.2                 , 0.14901960784313725)
    )

    lutsize = mpl.rcParams['image.lut']
    return colors.LinearSegmentedColormap.from_list('CustomYlOrRd', spec, lutsize)


def get_custom_RdYlGn():
    spec = (
        (0.7470588235294118 , 0.0                 , 0.14901960784313725),
        (0.84313725490196079, 0.18823529411764706 , 0.15294117647058825),
        (0.95686274509803926, 0.42745098039215684 , 0.2627450980392157 ),
        (0.99215686274509807, 0.68235294117647061 , 0.38039215686274508),
        (0.99607843137254903, 0.8784313725490196  , 0.54509803921568623),
        (1.0                , 1.0                 , 0.4), #0.74901960784313726),
        (0.85098039215686272, 0.93725490196078431 , 0.54509803921568623),
        (0.65098039215686276, 0.85098039215686272 , 0.41568627450980394),
        (0.4                , 0.74117647058823533 , 0.38823529411764707),
        (0.10196078431372549, 0.59607843137254901 , 0.31372549019607843),
        (0.0                , 0.60784313725490196 , 0.21568627450980393)
        )

    lutsize = mpl.rcParams['image.lut']
    return colors.LinearSegmentedColormap.from_list('CustomRdYlGn', spec, lutsize)

RdYlGn = get_custom_RdYlGn()
YlOrRd = get_custom_YlOrRd()

fig = plt.figure(figsize=(20, 6))
fig.tight_layout()
ax1 = fig.add_axes([0.05, 0.80, 0.9, 0.15])
ax1.axis('off')
cmap = RdYlGn
norm = mpl.colors.Normalize(vmin=0.0, vmax=1.0)

cb1 = mpl.colorbar.ColorbarBase(ax1, cmap=cmap,
                                norm=norm,
                                orientation='horizontal')
fig.savefig('heatmap_colorbar1.png', dpi=300, bbox_inches='tight')


fig = plt.figure(figsize=(20, 6))
fig.tight_layout()
ax1 = fig.add_axes([0.05, 0.80, 0.9, 0.15])
ax1.axis('off')
cmap = YlOrRd
norm = mpl.colors.Normalize(vmin=0.0, vmax=1.0)

cb1 = mpl.colorbar.ColorbarBase(ax1, cmap=cmap,
                                norm=norm,
                                orientation='horizontal')
fig.savefig('heatmap_colorbar2.png', dpi=300, bbox_inches='tight')


