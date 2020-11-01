from python_speech_features import mfcc
import scipy.io.wavfile as wav
import soundfile as sf
import os
import numpy as np
import pickle
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", required=True, type=str, help='dir path to ASVSpoof data dir. For example, LA/ASVspoof2019_LA_train/flac/')
parser.add_argument("--output_path", required=True, type=str, help='file path to output data dir. For example, ./data/train.pkl')
args = parser.parse_args()

mfcc_feats = []
for filename in os.listdir(args.data_path):
    sig, rate = sf.read(os.path.join(args.data_path, filename))
    mfcc_feat = mfcc(sig,rate)
    mfcc_feats.append(mfcc_feat)

with open(args.output_path, 'wb') as outfile:
    pickle.dump(mfcc_feats, outfile)

