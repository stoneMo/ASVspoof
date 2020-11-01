import os
import numpy as np
import pickle
import argparse
from sklearn.svm import SVC

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", required=True, type=str, help='path to pickled file. For example, data/train.pkl')
args = parser.parse_args()

X = []
y = []
with open(args.data_path, 'rb') as infile:
    data = pickle.load(infile)
    for mfccs, label in data:
        mfcc = mfccs.mean(0)
        X.append(mfcc)
        y.append(label)

with open('svm.pkl', 'rb') as infile:
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
