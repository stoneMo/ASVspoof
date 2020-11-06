import os
import numpy as np
import pickle
import soundfile as sf

from python_speech_features import mfcc
from sklearn.mixture import GMM
from sklearn import preprocessing
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
    mfcc_feat = mfcc.mfcc(audio,rate, 0.025, 0.01,20,appendEnergy = True, nfft=1103)
    mfcc_feat = preprocessing.scale(mfcc_feat)
    delta = calculate_delta(mfcc_feat)
    #combining both mfcc features and delta
    combined = np.hstack((mfcc_feat,delta))   # timestep * 13 
    return combined

source = './LA/ASVspoof2019_LA_train/flac/'
dest = './gmm_models/'

# features in true/false
features = np.array([])
for file in os.listdir(source):
    file = os.path.join(source, file)

    (sr, audio) = sf.read(file)
    # extract 40 dimensional MFCC & delta MFCC features
    vector  = extract_features(audio,sr)

    if features.size == 0:
        features = vector
    else:
        features = np.vstack((features, vector))

gmm = GMM(n_components = 16, n_iter = 200, covariance_type='diag',n_init = 3)
gmm.fit(features)

name = 'model'
# saving the trained gaussian model
pickle.dump(gmm, open(dest + name + '.gmm', 'wb'))

# gmm_models = pickle.load(open(dest + name + '.gmm','rb'))
# labels = gmm_models.predict(X)