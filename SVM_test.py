import os
import numpy as np
import pickle
import argparse
from sklearn.svm import SVC
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", required=True, type=str, help='path to pickled file. For example, data/dev_mfcc.pkl')
parser.add_argument("--feature_type", required=True, type=str, help='select the feature type. cqcc or mfcc')
args = parser.parse_args()

X = []
y = []
max_len = 50
lens = []
with open(args.data_path, 'rb') as infile:
    data = pickle.load(infile)
    for feat_cqcc, feat_mfcc, label in data:
        #mfcc = mfccs.mean(0) # sum over all timesteps for now    # timesteps X num_dim
        #lens.append(mfccs.shape[0])
        if args.feature_type == "cqcc":
            feats = feat_cqcc
        elif args.feature_type == "mfcc":
            feats = feat_mfcc
        num_dim = feats.shape[1]
        if len(feats) > max_len:
            feats = feats[:max_len]
        elif len(feats) < max_len:
            feats = np.concatenate((feats, np.array([[0.]*num_dim]*(max_len-len(feats)))), axis=0)
        X.append(feats.reshape(-1))
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

# plot roc curve
# plt.figure()
# plt.plot(fpr, tpr)
# plt.xlabel("fpr")
# plt.ylabel("tpr")
# plt.title("SVM test ROC using cqcc")
# plt.savefig("SVM_CQCC_ROC.pdf")

eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
thresh = interp1d(fpr, thresholds)(eer)
print ('EER', eer)


# improvments

# feature combinations 
# boosting (svm mfcc + svm cqcc)