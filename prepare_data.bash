echo "preparing training data"
python3 data_processing.py --data_path ./LA/ASVspoof2019_LA_train/flac \
                        --output_path ./data/train_cqcc.pkl \
                        --label_path ./LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt \
                        --feature_type cqcc

echo "preparing dev data"
python3 data_processing.py --data_path ./LA/ASVspoof2019_LA_dev/flac \
                        --output_path ./data/dev_cqcc.pkl \
                        --label_path ./LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt \
                        --feature_type cqcc

echo "preparing test data"
python3 data_processing.py --data_path ./LA/ASVspoof2019_LA_eval/flac \
                        --output_path ./data/eval_cqcc.pkl \
                        --label_path ./LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt \
                        --feature_type cqcc