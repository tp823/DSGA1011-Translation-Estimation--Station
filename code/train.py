'''
Script to train models and tune parameter
'''

import time
import math
import model as m

import numpy as np
import torch
from torch.utils.data import Dataset
from torch import optim
import pickle
import argparse
from pathlib import Path
import pdb

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--inputPath") # Should contain embedding, dictionary, train/dev/test for both input and output language
    parser.add_argument("--dataLang", default = 'vi')
    parser.add_argument("--targetLang", default = 'en')
    parser.add_argument("--dataLength", type=int, default = 50)
    parser.add_argument("--targetLength", type=int, default=50)

    parser.add_argument("--emb_dim_data", type=int, default=300)
    parser.add_argument("--emb_dim_target", type=int, default=300)
    parser.add_argument("--vocab_size_data", type=int, default=100000)
    parser.add_argument("--vocab_size_target", type=int, default=100000)
    parser.add_argument("--flgNoPretrain", action='store_true')

    parser.add_argument("--modelName", default="RNNseq2seq")
    parser.add_argument("--dimLSTM_enc", type=int, default=128)
    parser.add_argument("--dimLSTM_dec", type=int, default=128)
    parser.add_argument("--i", type=int, default=1)  # Index of the element in the parameter set to be tuned
    parser.add_argument("--p_dropOut", type=float, default=0.5)
    parser.add_argument("--teacher_forcing_ratio", type=float, default=0.5)
    parser.add_argument("--nLSTM_enc", type=int, default=1)
    parser.add_argument("--flg_bidirectional_enc", action='store_true')
    parser.add_argument("--flg_updateEmb", action='store_true')


    parser.add_argument("--n_iter", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--lr_decay_rate", type=float, default=0.9)  # Rate of learning rate decay
    parser.add_argument("--lr_decay3", type=int, default=5)  # Decay learning rate every lr_decay3 epochs

    parser.add_argument("--batchSize", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--n_batch", type=int, default=1)
    parser.add_argument("--optType", default='Adam')  # optimizer
    parser.add_argument("--logInterval", type=int, default=1)  # Print test accuracy every n epochs
    parser.add_argument("--flgSave", action='store_true')
    parser.add_argument("--savePath", default='./')
    parser.add_argument("--randSeed", type=int, default=42)
    args = parser.parse_args()

    path = Path(args.savePath)
    path.mkdir(parents=True, exist_ok=True)

    flg_cuda = torch.cuda.is_available()
    torch.manual_seed(args.randSeed)

    print('General parameters: ', args)

    print('Load data: ')
    if not args.flgNoPretrain:
        data_emb = pickle.load(open(args.inputPath + 'embeddings_' + args.dataLang + '.p', 'rb'))
        args.vocab_size_data, args.emb_dim_data = data_emb.shape
        data_emb = torch.from_numpy(data_emb).float()

        target_emb = pickle.load(open(args.inputPath + 'embeddings_' + args.targetLang + '.p', 'rb'))
        args.vocab_size_target, args.emb_dim_target = target_emb.shape
        target_emb = torch.from_numpy(target_emb).float()

    else:
        data_emb = None
        target_emb = None

    train = m.translationDataset(args.inputPath, 'train_' + args.dataLang, 'train_' + args.targetLang, args.dataLength, args.targetLength)
    test = m.translationDataset(args.inputPath, 'dev_' + args.dataLang, 'dev_' + args.targetLang, args.dataLength, args.targetLength)
    dict_target = pickle.load(open(args.inputPath + 'dict_' + args.targetLang + '.p', 'rb'))

    print('To Loader')
    if flg_cuda:
        train_loader = torch.utils.data.DataLoader(train, batch_size=args.batchSize, shuffle=True, pin_memory=True)
        test_loader = torch.utils.data.DataLoader(test, batch_size=args.batchSize, shuffle=False, pin_memory=True)
    else:
        train_loader = torch.utils.data.DataLoader(train, batch_size=args.batchSize, shuffle=True, pin_memory=False)
        test_loader = torch.utils.data.DataLoader(test, batch_size=args.batchSize, shuffle=False, pin_memory=False)

    model_paras = {'emb_dim_enc': args.emb_dim_data, 'emb_dim_dec': args.emb_dim_target, 'vocab_size_enc': args.vocab_size_data, 'vocab_size_dec': args.vocab_size_target,
                   'teacher_forcing_ratio': args.teacher_forcing_ratio, 'dimLSTM_enc': args.dimLSTM_enc, 'dimLSTM_dec': args.dimLSTM_dec, 'nLSTM_enc': args.nLSTM_enc,
                   'flg_bidirectional_enc': args.flg_bidirectional_enc, 'flg_updateEmb': args.flg_updateEmb}

    model = getattr(m, args.modelName)(model_paras, data_emb, target_emb)

    if flg_cuda:
        model = model.cuda()

    print(model)

    if args.optType == 'Adam':
        opt = optim.Adam(model.params, lr=args.lr)
    elif args.optType == 'SGD':
        opt = optim.SGD(model.params, lr=args.lr)


    train_paras = {'n_iter': args.n_iter, 'log_interval': [args.logInterval, 1000],
                   'lr_decay': [args.lr, args.lr_decay_rate, args.lr_decay3, 1e-5],
                   'flgSave': args.flgSave, 'savePath': args.savePath, 'n_batch': args.n_batch, 'randSeed': args.randSeed}

    print("Beginning Training")
    m = m.trainModel(train_paras, train_loader, test_loader, model, opt, dict_target['id2token'])

    start = time.time()
    _, lsTrainLoss, lsTestAccuracy = m.run()
    print('Test Acc max: %.3f' % (np.max(lsTestAccuracy)))
    print('Test Acc final: %.3f' % (lsTestAccuracy[-1]))
    stopIdx = min(lsTestAccuracy.index(np.max(lsTestAccuracy)) * args.logInterval, args.n_iter)
    print('Stop at: %d' % (stopIdx))
    end = time.time()
    print('Training time: ', asMinutes(end - start))


