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
        mfcc = mfccs.mean(0) # sum over all timesteps for now    # timesteps X 13
        X.append(mfcc)
        y.append(label)

clf = SVC(class_weight='balanced')
#clf = SVC()
clf.fit(X[:4000], y[:4000])

# train acc
print ('train accuracy:', clf.score(X, y))

with open('svm.pkl', 'wb') as outfile:
    pickle.dump(clf, outfile)
