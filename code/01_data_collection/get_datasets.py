#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import gzip
import os
import shutil
import tarfile
import tempfile
import zipfile
from glob import glob
import numpy as np
import pandas as pd
import scipy.io as sio
import wget
from keras import datasets as kdatasets
from keras import applications
from scipy.io import arff
from skimage import io, transform
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
import metrics


def download_file(urls, base_dir, name):
    dir_name = os.path.join(base_dir, name)

    if not os.path.exists(dir_name):
        os.makedirs(dir_name, exist_ok=True)

        for url in urls:
            wget.download(url, out=dir_name)


def save_dataset(name, X, y):
    n_samples = metrics.metric_dc_num_samples(X)
    n_features = metrics.metric_dc_num_features(X)
    n_classes = metrics.metric_dc_num_classes(y)
    balanced = metrics.metric_dc_dataset_is_balanced(y)

    print(name, n_samples, n_features, n_classes, balanced, X.shape)

    for l in np.unique(y):
        print('-->', l, np.count_nonzero(y == l))

    dir_name = os.path.join(base_dir, name)

    if not os.path.exists(dir_name):
        os.makedirs(dir_name, exist_ok=True)

    scaler = MinMaxScaler()
    X = scaler.fit_transform(X.astype('float32'))

    np.save(os.path.join(dir_name, 'X.npy'), X)
    np.save(os.path.join(dir_name, 'y.npy'), y)

    np.savetxt(os.path.join(dir_name, 'X.csv.gz'), X, delimiter=',')
    np.savetxt(os.path.join(dir_name, 'y.csv.gz'), y, delimiter=',')


def remove_all_datasets(base_dir):
    if os.path.exists(base_dir):
        shutil.rmtree(base_dir)


def process_cnae9():
    df = pd.read_csv('data/cnae9/CNAE-9.data', header=None)
    y = np.array(df[0])
    X = np.array(df.drop(0, axis=1))
    save_dataset('cnae9', X, y)


def process_bank():
    bank = zipfile.ZipFile('data/bank/bank-additional.zip')
    tmp_dir = tempfile.TemporaryDirectory()
    bank.extractall(tmp_dir.name)

    df = pd.read_csv(os.path.join(
        tmp_dir.name, 'bank-additional', 'bank-additional-full.csv'), sep=';')

    y = np.array(df['y'] == 'yes').astype('uint8')
    X = np.array(pd.get_dummies(df.drop('y', axis=1)))

    X_train, _, y_train, _ = train_test_split(X, y, train_size=0.05, random_state=42, stratify=y)
    X = X_train
    y = y_train

    save_dataset('bank', X, y)


# def process_gene():
#     ge = tarfile.open('data/gene/TCGA-PANCAN-HiSeq-801x20531.tar.gz', 'r:gz')

#     tmp_dir = tempfile.TemporaryDirectory()
#     ge.extractall(tmp_dir.name)

#     df = pd.read_csv(os.path.join(
#         tmp_dir.name, 'TCGA-PANCAN-HiSeq-801x20531/', 'data.csv'))
#     labels = pd.read_csv(os.path.join(
#         tmp_dir.name, 'TCGA-PANCAN-HiSeq-801x20531/', 'labels.csv'))

#     y = np.array(labels['Class'])
#     enc = LabelEncoder()
#     y = enc.fit_transform(y)
#     X = np.array(df.drop('Unnamed: 0', axis=1))
#     save_dataset('gene', X, y)


def process_imdb():
    imdb = tarfile.open('data/imdb/aclImdb_v1.tar.gz', 'r:gz')
    tmp_dir = tempfile.TemporaryDirectory()
    imdb.extractall(tmp_dir.name)

    pos_files = glob(os.path.join(
        tmp_dir.name, 'aclImdb/train/pos') + '/*.txt')
    pos_comments = []

    neg_files = glob(os.path.join(
        tmp_dir.name, 'aclImdb/train/neg') + '/*.txt')
    neg_comments = []

    for pf in pos_files:
        with open(pf, 'r') as f:
            pos_comments.append(' '.join(f.readlines()))

    for nf in neg_files:
        with open(nf, 'r') as f:
            neg_comments.append(' '.join(f.readlines()))

    comments = pos_comments + neg_comments
    y = np.zeros((len(comments),)).astype('uint8')
    y[:len(pos_comments)] = 1

    tfidf = TfidfVectorizer(strip_accents='ascii',
                            stop_words='english', max_features=700)
    X = tfidf.fit_transform(comments).todense()

    X_train, _, y_train, _ = train_test_split(X, y, train_size=0.13, random_state=42, stratify=y)
    X = X_train
    y = y_train

    save_dataset('imdb', X, y)


def process_sentiment():
    sent = zipfile.ZipFile('data/sentiment/sentiment labelled sentences.zip')
    tmp_dir = tempfile.TemporaryDirectory()
    sent.extractall(tmp_dir.name)

    files = ['amazon_cells_labelled.txt',
             'imdb_labelled.txt', 'yelp_labelled.txt']
    dfs = []

    for f in files:
        dfs.append(pd.read_table(os.path.join(
            tmp_dir.name, 'sentiment labelled sentences', f), sep='\t', header=None))

    df = pd.concat(dfs)
    tfidf = TfidfVectorizer(strip_accents='ascii',
                            stop_words='english', max_features=200)

    y = np.array(df[1]).astype('uint8')
    X = tfidf.fit_transform(list(df[0])).todense()
    save_dataset('sentiment', X, y)


def process_cifar10():
    (X, y), (_, _) = kdatasets.cifar10.load_data()

    X_train, _, y_train, _ = train_test_split(X, y, train_size=0.065, random_state=42, stratify=y)
    X = X_train
    y = y_train

    X = X[:,:,:,1]

    save_dataset('cifar10', X.reshape((-1, 32 * 32)), y.squeeze())


def process_fashionmnist():
    (X, y), (_, _) = kdatasets.fashion_mnist.load_data()

    X_train, _, y_train, _ = train_test_split(X, y, train_size=0.05, random_state=42, stratify=y)
    X = X_train
    y = y_train

    save_dataset('fashion_mnist', X.reshape((-1, 28 * 28)), y.squeeze())


# def process_fashionmnist():
#     (X, y), (_, _) = kdatasets.fashion_mnist.load_data()
#     save_dataset('fashion_mnist', X.reshape((-1, 28 * 28)), y.squeeze())


def process_zfmd():
    fmd = zipfile.ZipFile('data/fmd/FMD.zip')
    tmp_dir = tempfile.TemporaryDirectory()
    fmd.extractall(tmp_dir.name)

    fmd_shape = (384, 512, 3)

    images = dict()

    for d in sorted(glob(os.path.join(tmp_dir.name, 'image') + '/*')):
        class_name = os.path.basename(d)

        images[class_name] = []

        for img in glob(d + '/*.jpg'):
            im = io.imread(img)
            if im.shape == fmd_shape:
                images[class_name].append(im)

    image_arrays = []
    label_arrays = []

    for i, c in enumerate(sorted(images.keys())):
        image_arrays.append(np.array(images[c]))
        labels = np.zeros((len(images[c]),)).astype('uint8')
        labels[:] = i
        label_arrays.append(labels)

    model = applications.inception_resnet_v2.InceptionResNetV2(
        include_top=False, weights='imagenet', input_shape=fmd_shape, pooling='max')

    X = np.vstack(image_arrays)
    y = np.hstack(label_arrays)

    X = X / 255.0
    X = model.predict(X)

    save_dataset('fmd', X, y)


def process_svhn():
    data = sio.loadmat('data/svhn/train_32x32.mat')

    X = np.rollaxis(data['X'], 3, 0)
    X = X[:,:,:,1].reshape((-1, 32*32))
    y = data['y'].squeeze()

    X_train, _, y_train, _ = train_test_split(X, y, train_size=0.01, random_state=42, stratify=y)
    X = X_train
    y = y_train

    save_dataset('svhn', X, y)


def process_seismic():
    data, _ = arff.loadarff('data/seismic/seismic-bumps.arff')
    df = pd.DataFrame.from_records(data)

    df['seismic'] = df['seismic'].str.decode("utf-8")
    df['seismoacoustic'] = df['seismoacoustic'].str.decode("utf-8")
    df['shift'] = df['shift'].str.decode("utf-8")
    df['ghazard'] = df['ghazard'].str.decode("utf-8")
    df['class'] = df['class'].str.decode("utf-8")

    y = np.array((df['class'] == '1').astype('uint8'))
    X = np.array(pd.get_dummies(df.drop('class', axis=1)))

    X_train, _, y_train, _ = train_test_split(X, y, train_size=0.25, random_state=42, stratify=y)
    X = X_train
    y = y_train

    save_dataset('seismic', X, y)


# def process_hepmass():
#     tmp_dir = tempfile.TemporaryDirectory()
#     in_file_name = 'data/hepmass/all_train.csv.gz'
#     out_file_name = os.path.join(
#         tmp_dir.name, os.path.basename(in_file_name)[:-3])

#     with gzip.open(in_file_name, 'rb') as infile:
#         with open(out_file_name, 'wb') as outfile:
#             outfile.write(infile.read())

#     df = pd.read_csv(os.path.join(tmp_dir.name, 'all_train.csv'))

#     y = np.array(df['# label'].astype('uint8'))
#     X = np.array(df.drop('# label', axis=1))
#     save_dataset('hepmass', X, y)


def process_epileptic():
    df = pd.read_csv('data/epileptic/data.csv', index_col=None)
    y = np.array(df['y'])
    X = np.array(df.drop(['y', 'Unnamed: 0'], axis=1))

    X_train, _, y_train, _ = train_test_split(X, y, train_size=0.5, random_state=42, stratify=y)
    X = X_train
    y = y_train

    save_dataset('epileptic', X, y)


def process_spambase():
    df = pd.read_csv('data/spambase/spambase.data',
                     header=None, index_col=None)
    y = np.array(df[57]).astype('uint8')
    X = np.array(df.drop(57, axis=1))
    
    save_dataset('spambase', X, y)


def process_sms():
    sms = zipfile.ZipFile('data/sms/smsspamcollection.zip')
    tmp_dir = tempfile.TemporaryDirectory()
    sms.extractall(tmp_dir.name)

    df = pd.read_csv(os.path.join(
        tmp_dir.name, 'SMSSpamCollection'), sep='\t', header=None)
    tfidf = TfidfVectorizer(strip_accents='ascii',
                            stop_words='english', max_features=500)

    y = np.array(df[0] == 'spam').astype('uint8')
    X = tfidf.fit_transform(list(df[1])).todense()

    X_train, _, y_train, _ = train_test_split(X, y, train_size=0.15, random_state=42, stratify=y)
    X = X_train
    y = y_train

    save_dataset('sms', X, y)


def process_hatespeech():
    df = pd.read_csv('data/hatespeech/labeled_data.csv')
    tfidf = TfidfVectorizer(strip_accents='ascii',
                            stop_words='english', max_features=100)

    y = np.array(df['class']).astype('uint8')
    X = tfidf.fit_transform(list(df['tweet'])).todense()
    
    X_train, _, y_train, _ = train_test_split(X, y, train_size=0.13, random_state=42, stratify=y)
    X = X_train
    y = y_train

    save_dataset('hatespeech', X, y)


def process_secom():
    df = pd.read_csv('data/secom/secom.data', sep=' ', header=None)
    labels = pd.read_csv('data/secom/secom_labels.data', sep=' ', header=None)

    y = np.array(labels[0])
    X = np.array(df)
    X[np.isnan(X)] = 0.0
    save_dataset('secom', X, y)


def process_har():
    har = zipfile.ZipFile('data/har/UCI HAR Dataset.zip')
    tmp_dir = tempfile.TemporaryDirectory()
    har.extractall(tmp_dir.name)

    df = pd.read_csv(os.path.join(tmp_dir.name, 'UCI HAR Dataset', 'train',
                                  'X_train.txt'), header=None, delim_whitespace=True)
    labels = pd.read_csv(os.path.join(tmp_dir.name, 'UCI HAR Dataset', 'train',
                                      'y_train.txt'), header=None, delim_whitespace=True)

    y = np.array(labels[0]).astype('uint8')
    X = np.array(df)

    X_train, _, y_train, _ = train_test_split(X, y, train_size=0.1, random_state=42, stratify=y)
    X = X_train
    y = y_train

    save_dataset('har', X, y)


# def process_p53():
#     p53 = zipfile.ZipFile('data/p53/p53_new_2012.zip')
#     tmp_dir = tempfile.TemporaryDirectory()
#     p53.extractall(tmp_dir.name)

#     df = pd.read_csv(os.path.join(tmp_dir.name, 'Data Sets',
#                                   'K9.data'), header=None, na_values='?')

#     y = np.array(df[5408] == 'active').astype('uint8')
#     X = np.array(df.drop([5408, 5409], axis=1))
#     X[np.isnan(X)] = 0.0
#     save_dataset('p53', X, y)


def process_coil20():
    coil20 = zipfile.ZipFile('data/coil20/coil-20-proc.zip')
    tmp_dir = tempfile.TemporaryDirectory()
    coil20.extractall(tmp_dir.name)

    file_list = sorted(glob(tmp_dir.name + '/coil-20-proc/*.png'))

    img_side = 20

    X = np.zeros((len(file_list), img_side, img_side))
    y = np.zeros((len(file_list),)).astype('uint8')

    for i, file_name in enumerate(file_list):
        label = int(os.path.basename(file_name).split(
            '__')[0].replace('obj', ''))

        tmp = io.imread(file_name)
        tmp = transform.resize(tmp, (img_side, img_side), preserve_range=True)

        X[i] = tmp / 255.0
        y[i] = label

    save_dataset('coil20', X.reshape((-1, img_side * img_side)), y)


def process_orl():
    orl = zipfile.ZipFile('data/orl/att_faces.zip')
    tmp_dir = tempfile.TemporaryDirectory()
    orl.extractall(tmp_dir.name)

    subjects = sorted(glob(tmp_dir.name + '/s*'))

    img_h = 112 // 5
    img_w = 92 // 5

    X = np.zeros((len(subjects * 10), img_h, img_w))
    y = np.zeros((len(subjects * 10),)).astype('uint8')

    for i, dir_name in enumerate(subjects):
        label = int(os.path.basename(dir_name).replace('s', ''))

        for j in range(10):
            tmp = io.imread(dir_name + '/%d.pgm' % (j + 1))
            tmp = transform.resize(tmp, (img_h, img_w), preserve_range=True)
            X[i] = tmp / 255.0
            y[i] = label

    save_dataset('orl', X.reshape((-1, img_h * img_w)), y)


def process_hiva():
    hiva = zipfile.ZipFile('data/hiva/HIVA.zip')
    tmp_dir = tempfile.TemporaryDirectory()
    hiva.extractall(tmp_dir.name)

    X = np.loadtxt(tmp_dir.name + '/HIVA/hiva_train.data')
    y = np.loadtxt(tmp_dir.name + '/HIVA/hiva_train.labels')

    X_train, _, y_train, _ = train_test_split(X, y, train_size=0.8, random_state=42, stratify=y)
    X = X_train
    y = y_train

    save_dataset('hiva', X, y)


# def process_zefigi():
#     efigi_img = tarfile.open('data/efigi/efigi_png_gri-1.6.tgz', 'r:gz')
#     efigi_tab = tarfile.open('data/efigi/efigi_tables-1.6.2.tgz', 'r:gz')

#     efigi_shape = (255, 255, 3)

#     tmp_dir = tempfile.TemporaryDirectory()
#     efigi_img.extractall(tmp_dir.name)
#     efigi_tab.extractall(tmp_dir.name)

#     df = pd.read_table(os.path.join(tmp_dir.name, 'efigi-1.6', 'EFIGI_attributes.txt'),
#                        delim_whitespace=True, comment='#', header=None)
#     images = list(os.path.join(
#         tmp_dir.name, 'efigi-1.6', 'png/') + df[0] + '.png')

#     X = np.zeros((len(images),) + efigi_shape).astype('float32')

#     for i, img_name in enumerate(images):
#         X[i] = io.imread(img_name)/255.0

#     y = np.zeros((X.shape[0], 5)).astype('uint8')

#     # elliptical
#     y[:, 0] = np.array(df[1].isin([-6, -5, -4])).astype('uint8')
#     # spiral
#     y[:, 1] = np.array(df[1].isin(
#         [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])).astype('uint8')
#     # lenticular
#     y[:, 2] = np.array(df[1].isin([-3, -2, -1])).astype('uint8')
#     # irregular
#     y[:, 3] = np.array(df[1].isin([10])).astype('uint8')
#     # dwarf
#     y[:, 4] = np.array(df[1].isin([11])).astype('uint8')

#     model = applications.inception_resnet_v2.InceptionResNetV2(
#         include_top=False, weights='imagenet', input_shape=efigi_shape, pooling='max')
#     X /= 255.0
#     X = model.predict(X)
#     y = np.argmax(y, axis=1)
#     save_dataset('efigi', X, y)


if __name__ == '__main__':
    base_dir = './data'

    datasets = dict()

    datasets['cnae9'] = [
        'http://archive.ics.uci.edu/ml/machine-learning-databases/00233/CNAE-9.data']
    datasets['fmd'] = ['http://people.csail.mit.edu/celiu/CVPR2010/FMD/FMD.zip']
    datasets['svhn'] = [
        'http://ufldl.stanford.edu/housenumbers/train_32x32.mat']
    datasets['bank'] = [
        'http://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank-additional.zip']
    datasets['seismic'] = [
        'http://archive.ics.uci.edu/ml/machine-learning-databases/00266/seismic-bumps.arff']
    # datasets['hepmass'] = [
    #     'http://archive.ics.uci.edu/ml/machine-learning-databases/00347/all_train.csv.gz']
    datasets['epileptic'] = [
        'http://archive.ics.uci.edu/ml/machine-learning-databases/00388/data.csv']
    # datasets['gene'] = [
    #     'https://archive.ics.uci.edu/ml/machine-learning-databases/00401/TCGA-PANCAN-HiSeq-801x20531.tar.gz']
    datasets['spambase'] = ['https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data',
                            'https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.names']
    datasets['sms'] = [
        'http://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip']
    datasets['hatespeech'] = [
        'https://github.com/t-davidson/hate-speech-and-offensive-language/blob/master/data/labeled_data.csv?raw=true']
    # datasets['efigi'] = ['https://www.astromatic.net/download/efigi/efigi_png_gri-1.6.tgz',
    #                      'https://www.astromatic.net/download/efigi/efigi_tables-1.6.2.tgz']
    datasets['imdb'] = [
        'http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz']
    datasets['sentiment'] = [
        'https://archive.ics.uci.edu/ml/machine-learning-databases/00331/sentiment labelled sentences.zip']
    datasets['secom'] = ['http://archive.ics.uci.edu/ml/machine-learning-databases/secom/secom.data',
                         'http://archive.ics.uci.edu/ml/machine-learning-databases/secom/secom_labels.data']
    datasets['har'] = [
        'http://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI HAR Dataset.zip']
    # datasets['p53'] = [
    #     'http://archive.ics.uci.edu/ml/machine-learning-databases/p53/p53_new_2012.zip']
    datasets['coil20'] = [
        'http://www.cs.columbia.edu/CAVE/databases/SLAM_coil-20_coil-100/coil-20/coil-20-proc.zip']
    datasets['hiva'] = [
        'http://www.agnostic.inf.ethz.ch/datasets/DataAgnos/HIVA.zip']
    datasets['orl'] = [
        'http://www.cl.cam.ac.uk/Research/DTG/attarchive/pub/data/att_faces.zip']

    parser = argparse.ArgumentParser(
        description='Projection Survey Dataset Downloader')

    parser.add_argument('-d', action='store_true', help='delete all datasets')
    parser.add_argument('-s', action='store_true',
                        help='skip download, assume files are in place')
    args, unknown = parser.parse_known_args()

    if args.d:
        print('Removing all datasets')
        remove_all_datasets(base_dir)

    if not os.path.exists(base_dir):
        os.makedirs(base_dir, exist_ok=True)

    if not args.s:
        print('Downloading all datasets')
        for name, url in datasets.items():
            print('')
            print(name)
            download_file(url, base_dir, name)

    print('')
    print('Processing all datasets')

    for func in sorted([f for f in dir() if f[:8] == 'process_']):
        print(str(func))
        globals()[func]()
