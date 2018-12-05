#source activate py36_inex

python data_process.py --inputPath '../project_data/en-vi/' --language 'en' --outPath '../project_data/model_input/en-vi_v100K_emb300K/' --vocab_size_limit 100000 --embFile '../project_data/word_embeds/wiki.en.vec' --emb_size_limit 300000 > ../project_data/dp_en-vi_v100K_emb300K_log_en.txt

python data_process.py --inputPath '../project_data/en-vi/' --language 'vi' --outPath '../project_data/model_input/en-vi_v100K_emb300K/' --vocab_size_limit 100000 --embFile '../project_data/word_embeds/wiki.vi.vec' --emb_size_limit 300000 > ../project_data/dp_en-vi_v100K_emb300K_log_vi.txt

python data_process.py --inputPath '../project_data/en-zh/' --language 'en' --outPath '../project_data/model_input/en-zh_v100K_emb300K/' --vocab_size_limit 100000 --embFile '../project_data/word_embeds/wiki.en.vec' --emb_size_limit 300000 > ../project_data/dp_en-zh_v100K_emb300K_log_en.txt

python data_process.py --inputPath '../project_data/en-zh/' --language 'zh' --outPath '../project_data/model_input/en-zh_v100K_emb300K/' --vocab_size_limit 100000 --embFile '../project_data/word_embeds/wiki.zh.vec' --emb_size_limit 300000 > ../project_data/dp_en-zh_v100K_emb300K_log_zh.txt
