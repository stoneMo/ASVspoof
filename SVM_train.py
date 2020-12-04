import os
import numpy as np
import pickle
import argparse
from sklearn.svm import SVC
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", required=True, type=str, help='path to pickled file. For example, data/train.pkl')
parser.add_argument("--feature_type", required=True, type=str, help='select the feature type. cqcc or mfcc')
args = parser.parse_args()

X = []
y = []

max_len = 50  # 1.25 seconds  # check the timesteps of cqcc and mfcc 
lens = []
with open(args.data_path, 'rb') as infile:
    data = pickle.load(infile)
    for feat_cqcc, feat_mfcc, label in data:
        #mfcc = mfccs.mean(0) # sum over all timesteps for now    # timesteps X 13
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
X = np.array(X)
#clf = SVC(class_weight='balanced')
clf = SVC()
clf.fit(X, y)

# train acc
print ('train accuracy:', clf.score(X, y))

with open('svm_{}.pkl'.format(max_len), 'wb') as outfile:
    pickle.dump(clf, outfile)
