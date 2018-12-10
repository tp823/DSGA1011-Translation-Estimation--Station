#python train.py --inputPath '../project_data/model_input/en-vi_v100K_emb300K/' --dataLang 'vi' --dataLength 50 --targetLength 50 --modelName 'RNNseq2seq' --n_iter 200 --lr 0.001 --n_batch 200 --savePath '../model/rnn-vi-lr-001/' --batchSize 32

#python train.py --inputPath '../project_data/model_input/en-vi_v100K_emb300K/' --dataLang 'vi' --dataLength 50 --targetLength 50 --modelName 'RNNseq2seq' --n_iter 200 --lr 0.01 --n_batch 200 --savePath '../model/rnn-vi-lr-01/' --batchSize 32

#python train.py --inputPath '../project_data/model_input/en-vi_v100K_emb300K/' --dataLang 'vi' --dataLength 50 --targetLength 50 --modelName 'RNNseq2seq' --n_iter 200 --lr 0.0001 --n_batch 200 --savePath '../model/rnn-vi-lr-1/' --batchSize 32

#python train.py --inputPath '../project_data/model_input/en-vi_v100K_emb300K/' --dataLang 'vi' --dataLength 50 --p_dropOut 0 --targetLength 50 --modelName 'RNNseq2seq' --n_iter 200 --lr 0.001 --n_batch 200 --savePath '../model/rnn-vi-dr-0/' --batchSize 32

#python train.py --inputPath '../project_data/model_input/en-vi_v100K_emb300K/' --dataLang 'vi' --dataLength 50 --p_dropOut 0.3 --targetLength 50 --modelName 'RNNseq2seq' --n_iter 200 --lr 0.001 --n_batch 200 --savePath '../model/rnn-vi-dr-3/' --batchSize 32

#python train.py --inputPath '../project_data/model_input/en-vi_v100K_emb300K/' --dataLang 'vi' --dataLength 50 --p_dropOut 0.6 --targetLength 50 --modelName 'RNNseq2seq' --n_iter 200 --lr 0.001 --n_batch 200 --savePath '../model/rnn-vi-dr-6/' --batchSize 32

python train.py --inputPath '../project_data/model_input/en-vi_v100K_emb300K/' --dataLang 'vi' --dataLength 50 --p_dropOut 0 --dimLSTM_enc 128 --dimLSTM_dec 128 --targetLength 50 --modelName 'RNNseq2seq' --n_iter 200 --lr 0.001 --n_batch 200 --savePath '../model/rnn-vi-hs-128/' --batchSize 32

python train.py --inputPath '../project_data/model_input/en-vi_v100K_emb300K/' --dataLang 'vi' --dataLength 50 --p_dropOut 0 --dimLSTM_enc 256 --dimLSTM_dec 256 --targetLength 50 --modelName 'RNNseq2seq' --n_iter 200 --lr 0.001 --n_batch 200 --savePath '../model/rnn-vi-hs-256/' --batchSize 32
