from python_speech_features import mfcc
from CQCC.cqcc import cqcc
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
parser.add_argument("--feature_type", required=True, type=str, help='select the feature type. cqcc or mfcc')
args = parser.parse_args()

def extract_cqcc(x, fs):
    # INPUT SIGNAL
    x = x.reshape(x.shape[0], 1)  # for one-channel signal 
    # print(x.shape)
    # fs: 16000
    # x: (64244,)
    # PARAMETERS
    B = 96
    fmax = fs/2
    fmin = fmax/2**9
    d = 16
    cf = 19
    ZsdD = 'ZsdD'
    # COMPUTE CQCC FEATURES
    CQcc, LogP_absCQT, TimeVec, FreqVec, Ures_LogP_absCQT, Ures_FreqVec, absCQT = cqcc(x, fs, B, fmax, fmin, d, cf, ZsdD)
    return CQcc

# read in labels
filename2label = {}
for line in open(args.label_path):
    line = line.split()
    filename, label = line[1], line[-1]
    filename2label[filename] = label

feats = []
for filepath in os.listdir(args.data_path):
    filename = filepath.split('.')[0]
    if filename not in filename2label: # we skip speaker enrollment stage
        continue
    label = filename2label[filename]
    sig, rate = sf.read(os.path.join(args.data_path, filepath))
    if args.feature_type == "cqcc":
        feat = extract_cqcc(sig, rate)
    elif args.feature_type == "mfcc":
        feat = mfcc(sig, rate)
    print("feat:", feat.shape)
    feats.append((feat, label))

print("number of instances:", len(feats))

with open(args.output_path, 'wb') as outfile:
    pickle.dump(feats, outfile)


