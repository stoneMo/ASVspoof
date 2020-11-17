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

def traingmm(train_path, dest):
    '''#print(len(lenth), np.max(lenth), np.mean(lenth), stats.mode(lenth)[0][0])
    gmm_bon = GMM(n_components = 512, covariance_type='diag',n_init = 50) # min shape[0] = 135 # max = 1112
    # 2580 1112 337.8709302325581 289
    gmm_sp  = GMM(n_components = 512, covariance_type='diag',n_init = 50)  # min shape[0] = 64  # max = 1318
    # 22800 1318 341.9821929824561 297'''


    '''# debug
    gmm_bon = GMM(n_components = 5, covariance_type='diag',n_init = 50)
    gmm_sp  = GMM(n_components = 5, covariance_type='diag',n_init = 50)
    i = 0'''

    bondata = []
    spdata = []
    
    with open(train_path, 'rb') as infile:
        data = pickle.load(infile)
        for feats, label in data:
            if (label == 'bonafide'):
                #i += 1
                bondata.append(feats)
            elif(label == 'spoof'):
                spdata.append(feats)
            '''# debug
            if (i > 10):
                break'''
    Xbon = np.vstack(bondata)
    print('Bon feature stacked, shape as: ', Xbon.shape)
    Xsp = np.vstack(spdata)
    print('Sp feature stacked, shape as: ', Xsp.shape)

    t0 = time.time()
    gmm_bon.fit(Xbon)
    print('Bon gmm trained, time spend:', time.time() - t0)

    t0 = time.time()
    gmm_sp.fit(Xsp)
    print('Sp gmm trained, time spend:', time.time() - t0)

    pickle.dump(gmm_bon, open(dest + 'bon' + '.gmm', 'wb'))
    pickle.dump(gmm_sp, open(dest + 'sp' + '.gmm', 'wb'))
    print('GMM model created')



if __name__ == '__main__':
    train_path = './data/train_mfcc.pkl'
    dev_path = './data/dev_mfcc.pkl'
    test_path = './data/eval_mfcc.pkl'

    dest = './data/'
    traingmm(train_path, dest)