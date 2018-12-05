#source activate py36_inex

python train.py --inputPath '../project_data/model_input/en-zh_v100K_emb300K/' --dataLang 'zh' --dataLength 50 --targetLength 50 --modelName 'RNNseq2seq' --n_iter 10 --lr 0.001 --n_batch 10 --savePath '../model/test/' --batchSize 32
