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

def padvec(vector, maxpad):
    needpad = maxpad - vector.shape[0]
    needfront = needpad // 2
    needback = needpad - needfront

    padfront = np.zeros((needfront, vector.shape[1]))
    padback  = np.zeros((needback,  vector.shape[1]))

    vec = np.vstack((padfront, vector))
    vec2 = np.vstack((vec, padback))
    return vec2

def datasplit(bon, sp, source, bon_fold, sp_fold, maxpad):
    # feature
    i = 0
    j_bon = 0
    k_sp = 0

    totalfilenum = len([name for name in os.listdir(source) if os.path.isfile(os.path.join(source, name))])

    for file in os.listdir(source):
        '''
        # Debug
        if ((j_bon > 10) and (k_sp > 50)) :  
            break
        '''
        if (i == 0):
            t0 = time.time()
        i += 1
        if (i %50 == 0):
            t = time.time() - t0
            remain = t/i * (totalfilenum - i)
            print(i/totalfilenum*100, '%, processed file num =', i, ' remain time:', remain ,'s')
        f = file

        if (f in bon):
            file = os.path.join(source, file)

            audio, sr = sf.read(file)

            # extract 40 dimensional MFCC & delta MFCC features
            vector  = extract_features(audio,sr)
            vector = padvec(vector, maxpad).reshape(1, -1)
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
            vector = padvec(vector, maxpad).reshape(1, -1)
            k_sp += 1
            np.save(sp_fold+'sp_features_'+ str(k_sp) +'.npy', vector)
            if (k_sp % 100 == 50):
                print(k_sp,'sp saved')
                #gc.collect() # clear up memory
    return j_bon, k_sp

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

if __name__ == '__main__':
    source = './ASVspoof2019_LA_dev/flac/'
    #source = './ASVspoof2019_LA_train/remain/'

    bon_fold = './bon_features/'
    sp_fold = './sp_features/'

    dest = './gmm_models/'

    bname = 'bon'
    sname = 'sp'

    bon, sp = txtsplit('./ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt')
    maxpad = 1320
    j_bon, k_sp = datasplit(bon, sp, source, bon_fold, sp_fold, maxpad)
    print()
    print('###################################################################')
    

    # training data accuracy
    gmm_bon = pickle.load(open(dest + bname + '.gmm','rb'))
    gmm_sp  = pickle.load(open(dest + sname + '.gmm','rb'))

    predb = []
    preds = []

    for i in range(j_bon):
        if (i % 50 == 0):
            print('Evaluating Bon sample at',i/j_bon * 100, '%')
        file = 'bon_features_'+str(i+1)+'.npy'

        file = os.path.join(bon_fold, file)
        X = np.load(file)
        bscore = gmm_bon.score(X)
        sscore = gmm_sp.score(X)

        #predb.append(np.exp(bscore)-np.exp(sscore))
        predb.append(bscore-sscore)

    for i in range(k_sp):
        if (i % 50 == 0):
            print('Evaluating Sp sample at',i/k_sp * 100, '%')
        file = 'sp_features_'+str(i+1)+'.npy'

        file = os.path.join(sp_fold, file)
        X = np.load(file)
        bscore = gmm_bon.score(X)
        sscore = gmm_sp.score(X)

        #preds.append(np.exp(bscore)-np.exp(sscore))
        preds.append(bscore-sscore)
    print()
    print('###################################################################')

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