import os
import numpy as np
import pickle
import soundfile as sf
import librosa
import time
import gc

from python_speech_features import mfcc
from sklearn.mixture import GaussianMixture as GMM
from sklearn import preprocessing
from pdb import set_trace
from scipy import stats

def testgmm(dest, test_path):
    # training data accuracy
    gmm_bon = pickle.load(open(dest + 'bon' + '.gmm','rb'))
    gmm_sp  = pickle.load(open(dest + 'sp' + '.gmm','rb'))

    bondata = []
    spdata = []
    # debug
    #j = 0
    with open(test_path, 'rb') as infile:
        data = pickle.load(infile)
        for feats, label in data:
            if (label == 'bonafide'):
                j += 1
                bondata.append(feats)
            elif(label == 'spoof'):
                spdata.append(feats)
            # debug
            #if (j > 10):
            #    break
    print(len(bondata), bondata[0].shape)
    print(len(spdata), spdata[0].shape)

    predb = []
    preds = []
    j_bon = len(bondata)
    k_sp  = len(spdata)


    for i in range(j_bon):
        if (i % 50 == 0):
            print('Evaluating Bon sample at',i/j_bon * 100, '%')
        X = bondata[i]
        bscore = gmm_bon.score(X)
        sscore = gmm_sp.score(X)

        #predb.append(np.exp(bscore)-np.exp(sscore))
        predb.append(bscore-sscore)

    for i in range(k_sp):
        if (i % 50 == 0):
            print('Evaluating Sp sample at',i/k_sp * 100, '%')
        X = spdata[i]
        bscore = gmm_bon.score(X)
        sscore = gmm_sp.score(X)

        #preds.append(np.exp(bscore)-np.exp(sscore))
        preds.append(bscore-sscore)

    predb1 = np.asarray(predb)
    preds1 = np.asarray(preds)

    predb1[predb1 < 0] = 0
    predb1[predb1 > 0] = 1
    predbresult1 = np.sum(predb1)
    print(predbresult1, 'Bon samples were CORRECTLY evaluated out of', j_bon,'samples. Bon_Accuracy = ', predbresult1/j_bon )# 0.7356


    preds1[preds1 > 0] = 0
    preds1[preds1 < 0] = 1
    predsresult = np.sum(preds1)
    print(predsresult, 'Sp samples were CORRECTLY evaluated out of', k_sp, 'samples. Sp_Accuracy = ', predsresult/k_sp)# 0.4092

    print('Total GMM Classifier Accuracy = ',(predbresult1 + predsresult)/(j_bon + k_sp))

if __name__ == '__main__':
    train_path = './data/train_mfcc.pkl'
    dev_path = './data/dev_mfcc.pkl'
    test_path = './data/eval_mfcc.pkl'

    dest = './data/'
    testgmm(dest, dev_path)