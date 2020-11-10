import os
import numpy as np
import pickle
import argparse
from sklearn.svm import SVC

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", required=True, type=str, help='path to pickled file. For example, data/dev.pkl')
args = parser.parse_args()

X = []
y = []
max_len = 50
lens = []
with open(args.data_path, 'rb') as infile:
    data = pickle.load(infile)
    for mfccs, label in data:
        #mfcc = mfccs.mean(0) # sum over all timesteps for now    # timesteps X 13
        #lens.append(mfccs.shape[0])
        if len(mfccs) > max_len:
            mfccs = mfccs[:max_len]
        elif len(mfccs) < max_len:
            mfccs = np.concatenate((mfccs, np.array([[0.]*13]*(max_len-len(mfccs)))), axis=0)
        X.append(mfccs.reshape(-1))
        y.append(label)

with open('svm_{}.pkl'.format(max_len), 'rb') as infile:
    clf = pickle.load(infile)

# predict acc
print ('predict accuracy:', clf.score(X, y))
y_score = clf.decision_function(X)

from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(y, y_score, pos_label='spoof')

eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
thresh = interp1d(fpr, thresholds)(eer)
print ('EER', eer)
