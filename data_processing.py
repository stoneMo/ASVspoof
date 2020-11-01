from python_speech_features import mfcc
import scipy.io.wavfile as wav
import soundfile as sf
import os
import numpy as np
import pickle
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", required=True, type=str, help='path to ASVSpoof data directory. For example, LA/ASVspoof2019_LA_train/flac/')
parser.add_argument("--label_path", required=True, type=str, help='path to label file. For example, LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt')
parser.add_argument("--output_path", required=True, type=str, help='path to output pickle file. For example, ./data/train.pkl')
args = parser.parse_args()

# read in labels
filename2label = {}
for line in open(args.label_path):
    line = line.split()
    filename, label = line[1], line[-1]
    filename2label[filename] = label

mfcc_feats = []
for filepath in os.listdir(args.data_path):
    filename = filepath.split('.')[0]
    if filename not in filename2label: # we skip speaker enrollment stage
        continue
    label = filename2label[filename]
    sig, rate = sf.read(os.path.join(args.data_path, filepath))
    mfcc_feat = mfcc(sig, rate)
    mfcc_feats.append((mfcc_feat, label))

with open(args.output_path, 'wb') as outfile:
    pickle.dump(mfcc_feats, outfile)
