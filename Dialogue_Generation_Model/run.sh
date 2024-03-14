python train.py --pretrained --gpt2 --model_checkpoint pretrained/gpt2-chinese --data_path data/CSDS/CSDS_dial.json --scheduler linear --n_epochs 10
python get_idf.py --model_checkpoint pretrained/CSDS/xxx/ --datapath data/CSDS/CSDS_idf_dial.json --out_path data/CSDS/CSDS_idf.json --dataset CSDS
python test.py --model_checkpoint pretrained/CSDS/xxx/ --datapath data/CSDS/CSDS_dial_ppl.json --out_path result/CSDS_ppl_test.json --gpt2 --dataset CSDS --idf_path data/CSDS/CSDS_idf_dial.json
python test.py --model_checkpoint pretrained/CSDS/xxx/ --datapath data/CSDS/CSDS_dial_lm.json --out_path result/CSDS_lm_test.json --gpt2 --dataset CSDS --idf_path data/CSDS/CSDS_idf_dial.json
