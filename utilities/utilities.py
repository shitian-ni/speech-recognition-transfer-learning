"""
Contributions from:
Luis Andre Dutra e Silva
https://www.kaggle.com/mindcool/lb-0-77-keras-gru-with-filter-banks-features
"""

import pandas as pd
import numpy as np
import re

L = 16000
legal_labels = 'yes no up down left right on off stop go silence unknown'.split()

def list_wavs_fname(fpaths, ext='wav'):
    pat = r'.+/(\w+)/\w+\.' + ext + '$'
    labels = []
    for fpath in fpaths:
        r = re.match(pat, fpath)
        if r:
            labels.append(r.group(1))
    pat = r'.+/(\w+\.' + ext + ')$'
    fnames = []
    for fpath in fpaths:
        r = re.match(pat, fpath)
        if r:
            fnames.append(r.group(1))
    return labels, fnames

def pad_audio(samples):
    if len(samples) >= L: return samples
    else: return np.pad(samples, pad_width=(L - len(samples), 0), mode='constant', constant_values=(0, 0))

def chop_audio(samples, L=16000, num=1000):
    for i in range(num):
        beg = np.random.randint(0, len(samples) - L)
        yield samples[beg: beg + L]

def label_transform_audio(labels):
    return pd.get_dummies(pd.Series(labels))

def label_transform_speech(labels):
    nlabels = []
    for label in labels:
        if label == '_background_noise_':
            nlabels.append('silence')
        elif label not in legal_labels:
            nlabels.append('unknown')
        else:
            nlabels.append(label)
    return pd.get_dummies(pd.Series(nlabels))


def test_data_generator(batch=32):
    fpaths = glob.glob(os.path.join(test_data_path, '*wav'))
    i = 0
    for path in fpaths:
        if i == 0:
            imgs = []
            fnames = []
        i += 1
        rate, samples = wavfile.read(path)
        samples = pad_audio(samples)
        filter_banks = logfbank(samples)
        filter_banks -= (np.mean(filter_banks, axis=0) + 1e-8)
        imgs.append(filter_banks)
        fnames.append(path.split('/')[-1])
        if i == batch:
            i = 0
            imgs = np.array(imgs)
            yield fnames, imgs
    if i < batch:
        imgs = np.array(imgs)

        yield fnames, imgs
    raise StopIteration()
