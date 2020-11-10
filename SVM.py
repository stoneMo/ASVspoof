import os
import numpy as np
import pickle
import argparse
from sklearn.svm import SVC
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", required=True, type=str, help='path to pickled file. For example, data/train.pkl')
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
X = np.array(X)
#clf = SVC(class_weight='balanced')
clf = SVC()
clf.fit(X, y)

# train acc
print ('train accuracy:', clf.score(X, y))

with open('svm_{}.pkl'.format(max_len), 'wb') as outfile:
    pickle.dump(clf, outfile)
