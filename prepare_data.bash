echo "preparing training data"
python3 data_processing.py --data_path ./LA/ASVspoof2019_LA_train/flac \
                        --output_path ./data/train.pkl \
                        --label_path ./LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt

echo "preparing dev data"
python3 data_processing.py --data_path ./LA/ASVspoof2019_LA_dev/flac \
                        --output_path ./data/dev.pkl \
                        --label_path ./LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt 

# echo "preparing test data"
# python3 data_processing.py --data_path ./LA/ASVspoof2019_LA_eval/flac \
#                         --output_path ./data/eval.pkl \
#                         --label_path ./LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt 