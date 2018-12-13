#python train.py --inputPath '../project_data/model_input/en-vi_v100K_emb300K/' --dataLang 'vi' --dataLength 50 --targetLength 50 --modelName 'RNNseq2seq' --n_iter 200 --lr 0.001 --n_batch 200 --savePath '../model/rnn-vi-lr-001/' --batchSize 32

#python train.py --inputPath '../project_data/model_input/en-vi_v100K_emb300K/' --dataLang 'vi' --dataLength 50 --targetLength 50 --modelName 'RNNseq2seq' --n_iter 200 --lr 0.01 --n_batch 200 --savePath '../model/rnn-vi-lr-01/' --batchSize 32

#python train.py --inputPath '../project_data/model_input/en-vi_v100K_emb300K/' --dataLang 'vi' --dataLength 50 --targetLength 50 --modelName 'RNNseq2seq' --n_iter 200 --lr 0.0001 --n_batch 200 --savePath '../model/rnn-vi-lr-1/' --batchSize 32

#python train.py --inputPath '../project_data/model_input/en-vi_v100K_emb300K/' --dataLang 'vi' --dataLength 50 --p_dropOut 0 --targetLength 50 --modelName 'RNNseq2seq' --n_iter 200 --lr 0.001 --n_batch 200 --savePath '../model/rnn-vi-dr-0/' --batchSize 32

#python train.py --inputPath '../project_data/model_input/en-vi_v100K_emb300K/' --dataLang 'vi' --dataLength 50 --p_dropOut 0.3 --targetLength 50 --modelName 'RNNseq2seq' --n_iter 200 --lr 0.001 --n_batch 200 --savePath '../model/rnn-vi-dr-3/' --batchSize 32

#python train.py --inputPath '../project_data/model_input/en-vi_v100K_emb300K/' --dataLang 'vi' --dataLength 50 --p_dropOut 0.6 --targetLength 50 --modelName 'RNNseq2seq' --n_iter 200 --lr 0.001 --n_batch 200 --savePath '../model/rnn-vi-dr-6/' --batchSize 32

#python train.py --inputPath '../project_data/model_input/en-vi_v100K_emb300K/' --dataLang 'vi' --dataLength 50 --p_dropOut 0 --dimLSTM_enc 128 --dimLSTM_dec 128 --targetLength 50 --modelName 'RNNseq2seq' --n_iter 200 --lr 0.001 --n_batch 200 --savePath '../model/rnn-vi-hs-128/' --batchSize 32

#python train.py --inputPath '../project_data/model_input/en-vi_v100K_emb300K/' --dataLang 'vi' --dataLength 50 --p_dropOut 0 --dimLSTM_enc 256 --dimLSTM_dec 256 --targetLength 50 --modelName 'RNNseq2seq' --n_iter 200 --lr 0.001 --n_batch 200 --savePath '../model/rnn-vi-hs-256/' --batchSize 32

#python train.py --inputPath '../project_data/model_input/en-vi_v100K_emb300K/' --dataLang 'vi' --dataLength 50 --p_dropOut 0 --flgNoPretrain --dimLSTM_enc 128 --dimLSTM_dec 128 --targetLength 50 --modelName 'RNNseq2seq' --n_iter 200 --lr 0.001 --n_batch 200 --savePath '../model/rnn-vi-nopretrained/' --batchSize 32

#python train.py --inputPath '../project_data/model_input/en-vi_v100K_emb300K/' --dataLang 'vi' --dataLength 50 --p_dropOut 0 --dimLSTM_enc 512 --dimLSTM_dec 512 --targetLength 50 --modelName 'RNNseq2seq' --n_iter 100 --lr 0.001 --n_batch 200 --savePath '../model/rnn-vi-hs-512/' --batchSize 32

#python train.py --inputPath '../project_data/model_input/en-vi_v100K_emb300K/' --dataLang 'vi' --dataLength 50 --p_dropOut 0 --dimLSTM_enc 1024 --dimLSTM_dec 1024 --targetLength 50 --modelName 'RNNseq2seq' --n_iter 100 --lr 0.001 --n_batch 200 --savePath '../model/rnn-vi-hs-1024/' --batchSize 32

#python train.py --inputPath '../project_data/model_input/en-vi_v100K_emb300K/' --dataLang 'vi' --dataLength 50 --p_dropOut 0 --teacher_forcing_ratio 0.2 --dimLSTM_enc 256 --dimLSTM_dec 256 --targetLength 50 --modelName 'RNNseq2seq' --n_iter 100 --lr 0.001 --n_batch 200 --savePath '../model/rnn-vi-tf-2/' --batchSize 32

#python train.py --inputPath '../project_data/model_input/en-vi_v100K_emb300K/' --dataLang 'vi' --dataLength 50 --p_dropOut 0 --teacher_forcing_ratio 0.8 --dimLSTM_enc 256 --dimLSTM_dec 256 --targetLength 50 --modelName 'RNNseq2seq' --n_iter 100 --lr 0.001 --n_batch 200 --savePath '../model/rnn-vi-tf-8/' --batchSize 32

python train.py --inputPath '../project_data/model_input/en-vi_v100K_emb300K/' --dataLang 'vi' --dataLength 50 --p_dropOut 0 --teacher_forcing_ratio 0.8 --flg_bidirectional_enc --dimLSTM_enc 256 --dimLSTM_dec 512 --targetLength 50 --modelName 'RNNseq2seq' --n_iter 100 --lr 0.001 --n_batch 200 --savePath '../model/rnn-vi-bidir/' --batchSize 32

python train.py --inputPath '../project_data/model_input/en-vi_v100K_emb300K/' --dataLang 'vi' --dataLength 50 --p_dropOut 0 --dimLSTM_enc 512 --dimLSTM_dec 512 --flg_updateEmb --targetLength 50 --modelName 'RNNseq2seq' --n_iter 100 --lr 0.001 --n_batch 200 --savePath '../model/rnn-vi-updateemb/' --batchSize 32
