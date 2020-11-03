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

# https://github.com/MohamadMerchant/Voice-Authentication-and-Face-Recognition
def calculate_delta(array):
    rows,cols = array.shape
    deltas = np.zeros((rows,20))
    N = 2
    for i in range(rows):
        index = []
        j = 1
        while j <= N:
            if i-j < 0:
                first = 0
            else:
                first = i-j
            if i+j > rows -1:
                second = rows -1
            else:
                second = i+j
            index.append((second,first))
            j+=1
        deltas[i] = (array[index[0][0]]-array[index[0][1]] + (2 * (array[index[1][0]]-array[index[1][1]]))) / 10
    return deltas

def extract_features(audio,rate):    
    #mfcc_feat = mfcc(audio,rate, 0.025, 0.01,20,appendEnergy = True, nfft=1103)
    mfcc_feat = mfcc(audio,rate,winlen=0.025,winstep=0.01,numcep=20,
        nfilt=30,nfft=512,appendEnergy=True)
    mfcc_feat = preprocessing.scale(mfcc_feat)
    delta = calculate_delta(mfcc_feat)

    #combining both mfcc features and delta
    combined = np.hstack((mfcc_feat,delta)) 
    return combined

if __name__ == '__main__':
    source = './ASVspoof2019_LA_train/flac/'
    #source = './ASVspoof2019_LA_train/remain/'
    dest = './gmm_models/'

    bon_fold = './bon_features/'
    sp_fold = './sp_features/'

    bname = 'bon'
    sname = 'sp'

    '''bon, sp = txtsplit('./ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt')
    datasplit(bon, sp, source, dest, bon_fold, sp_fold)'''
    

    # training data accuracy
    gmm_bon = pickle.load(open(dest + bname + '.gmm','rb'))
    gmm_sp  = pickle.load(open(dest + sname + '.gmm','rb'))

    predb = []
    preds = []

    samplenum = 2500 # max to 2580
    for i in range(samplenum):
        file = 'bon_features_'+str(i+1)+'.npy'

        file = os.path.join(bon_fold, file)
        X = np.load(file)
        bscore = gmm_bon.score(X)
        sscore = gmm_sp.score(X)

        predb.append(bscore-sscore)
        file = 'sp_features_'+str(i+1)+'.npy'

        file = os.path.join(sp_fold, file)
        X = np.load(file)
        bscore = gmm_bon.score(X)
        sscore = gmm_sp.score(X)

        preds.append(bscore-sscore)

    predb1 = np.asarray(predb)
    predb2 = np.asarray(predb)

    preds1 = np.asarray(preds)
    preds2 = np.asarray(preds)

    predb1[predb1 < 0] = 0
    predb1[predb1 > 0] = 1
    predbresult = np.sum(predb1)
    print(predbresult, 'Accuracy = ', predbresult/samplenum )# 0.7356


    preds1[preds1 > 0] = 0
    preds1[preds1 < 0] = 1
    predsresult = np.sum(preds1)
    print(predsresult, 'Accuracy = ', predsresult/samplenum)# 0.4092
    # 1. some feature did not get because nframe too small(64)
    # 2. could try implement feature extractor for more connections, PCA maybe?
    # 3. train using features that stack together?