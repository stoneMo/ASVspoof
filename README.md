# ASVspoof

## Timelines
**[20/10 - 26/10]**

[1] Shentong/Haofan: python reimplementation of CQCC

[2] Chi: X-vector feature extraction

[3] Pinxu: GMM/SVM classifier build 


## Reports
proposal overleaf: https://www.overleaf.com/project/5f6954ebb68ba500013c78f1

## Dataset (LA)
Google drive: https://drive.google.com/file/d/1UGs1o2mDiBO9_iaN-0FupS8x0Tkb4xmt/view?usp=sharing

## Feature Extraction
### CQCC (baseline)
TO DO python reimplementation of the baseline.

### MFCC
python3 data_processing.py --data_path ./LA/ASVspoof2019_LA_train/flac --output_path ./data/train.pkl --label_path ./LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt
python3 data_processing.py --data_path ./LA/ASVspoof2019_LA_dev/flac --output_path ./data/dev.pkl --label_path ./LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt
python3 data_processing.py --data_path ./LA/ASVspoof2019_LA_eval/flac --output_path ./data/eval.pkl --label_path ./LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt

The saved pickle file has the format: [(mfcc_vec[timestepx13], label[bonafide/spoof]) x N instances]

## Classifier

### GMM


### SVM
python3 SVM.py --data_path ./data/train.pkl
python3 SVM_test.py --data_path ./data/dev.pkl
python3 SVM_test.py --data_path ./data/eval.pkl


## Referecne
[1] Ensemble Models for Spoofing Detection in Automatic Speaker Verification, 2019 Interspeech. 

[2] IIIT-H Spoofing Countermeasures for Automatic Speaker Verification, 2019 Interspeech. 
