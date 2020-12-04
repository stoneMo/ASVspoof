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

def padvec(vector, maxpad):
    needpad = maxpad - vector.shape[0]
    needfront = needpad // 2
    needback = needpad - needfront

    padfront = np.zeros((needfront, vector.shape[1]))
    padback  = np.zeros((needback,  vector.shape[1]))

    vec = np.vstack((padfront, vector))
    vec2 = np.vstack((vec, padback))
    return vec2

def datasplit(bon, sp, source, dest, bon_fold, sp_fold, maxbon, maxsp):
    # feature
    bon_features = np.array([])
    sp_features = np.array([])
    i = 0
    j_bon = 0
    k_sp = 0

    for file in os.listdir(source):
        if (i == 0):
            t0 = time.time()
        i += 1
        if (i %50 == 0):
            t = time.time() - t0
            remain = t/i * (25380 - i)
            print(i/253.8, '%,remain:', remain ,'s, i = ',i)
        f = file

        if (f in bon):
            file = os.path.join(source, file)

            audio, sr = sf.read(file)
            # extract 40 dimensional MFCC & delta MFCC features
            vector  = extract_features(audio,sr)
            #vector = padvec(vector, maxbon)
            j_bon += 1
            np.save(bon_fold+'bon_features_'+ str(j_bon) +'.npy', vector)
            if (j_bon % 100 == 0):                
                print(j_bon,'bon saved')
                #gc.collect() # clear up memory

        if (f in sp):
            file = os.path.join(source, file)

            audio, sr = sf.read(file)
            # extract 40 dimensional MFCC & delta MFCC features
            vector  = extract_features(audio,sr)
            #vector = padvec(vector, maxsp)
            k_sp += 1
            np.save(sp_fold+'sp_features_'+ str(k_sp) +'.npy', vector)
            if (k_sp % 100 == 50):
                print(k_sp,'sp saved')
                #gc.collect() # clear up memory
    print('Final saved')

def txtsplit(dest):
    with open(dest) as file:
        lines = file.readlines()
        bon = []
        sp = []
        for line in lines:
            word = line[:-1].split(' ')
            if (word[-1] == 'bonafide'):
                bon.append(word[1]+'.flac')
            if (word[-1] == 'spoof'):
                sp.append(word[1]+'.flac')
    return bon, sp

def traingmm(gmm, data_fold, tot, name, dest):
    t0 = time.time()
    gmmmodel = gmm
    i = 0
    for file in os.listdir(data_fold):
        if (i == 0):
            t0 = time.time()
        i += 1
        file = os.path.join(data_fold, file)
        features = np.load(file)
        gmmmodel.fit(features)   
        if (i % 100 == 0):
            tnow = time.time() - t0
            print(i/tot, '%', i, 'remain: ',tnow/(i) * (tot-i),'s ' )
    # saving the trained gaussian model
    pickle.dump(gmmmodel, open(dest + name + '.gmm', 'wb'))

def test(data_fold):
    lenth = []
    i = 0
    for file in os.listdir(data_fold):
        i += 1
        file = os.path.join(data_fold, file)
        features = np.load(file)
        lenth.append(features.shape[0])
        if (i % 50 == 49):
            print('Calculating on #',i)
    print(len(lenth), np.max(lenth), np.mean(lenth), stats.mode(lenth)[0][0])

if __name__ == '__main__':
    
    source = './ASVspoof2019_LA_train/flac/'
    #source = './ASVspoof2019_LA_train/remain/'
    dest = './gmm_models/'

    bon_fold = './bon_features/'
    sp_fold = './sp_features/'

    bname = 'bon'
    sname = 'sp'


    bon, sp = txtsplit('./ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt')

    maxbon = 1120
    maxsp = 1320
    datasplit(bon, sp, source, dest, bon_fold, sp_fold, maxbon, maxsp)

    

    #print(len(lenth), np.max(lenth), np.mean(lenth), stats.mode(lenth)[0][0])
    '''gmm_bon = GMM(n_components = 289, covariance_type='diag',n_init = 50,warm_start= True) # min shape[0] = 135 # max = 1112
    # 2580 1112 337.8709302325581 289
    gmm_sp  = GMM(n_components = 297, covariance_type='diag',n_init = 50,warm_start= True)  # min shape[0] = 64  # max = 1318
    # 22800 1318 341.9821929824561 297



    DIR = bon_fold
    bonnum = len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))])
    DIR = sp_fold
    spnum = len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))])
    
    traingmm(gmm_bon, bon_fold, bonnum , bname, dest)
    traingmm(gmm_sp , sp_fold , spnum, sname, dest)'''
    # Took about 1 h for this setup


    '''
    test(bon_fold)
    test(sp_fold)
    '''

    