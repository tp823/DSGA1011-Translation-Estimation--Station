source activate py36_inex

Path=../model/att_vi/v_lr-4_Dim[1-3]_2/

python eval.py --inputPath ../project_data/model_input/en-vi_v25K_emb300K/ --modelPath ${Path} --dataLang vi --dataLength 50 --dimLSTM_dec 1024 --dimLSTM_enc 512 --modelName AttRNNseq2seq --targetLength 50 --vocab_size_data 25000 --vocab_size_target 25000 --batchSize 64 > ${Path}/test_log.txt