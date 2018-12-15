

import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
from importlib import reload

# Global Variables
PAD_IDX = 0
UNK_IDX = 1
SOS_IDX = 2
EOS_IDX = 3

def idx2words(id2token, idx):
    '''
    Given a matrix of word idx, convert to words
    '''
    out = []
    di, dj = idx.shape
    for i in range(di):
        idx_i = idx[i]
        out_i = []
        for key in idx_i:
            if key == EOS_IDX:
                break
            else:
                out_i.append(id2token[key])
        out.append(out_i)
    return out

def plot(lsResults, lsNames, lsShape, outPath, legendPos1 = (0.6, 0.1, 1., .3), legendPos2 = (0.6, 0.1, 1., .3), ylimTrain=[0,150], ylimVal = [0, 20]):
    lsTrain = [x[1] for x in lsResults]
    lsTest = [x[2] for x in lsResults]
    lsEpoch = [x[0] for x in lsResults]
    maxEpoch = max([x[-1] for x in lsEpoch])
    figure1 = plt.figure(figsize=(16, 6))
    gs1 = gridspec.GridSpec(1, 2)

    ax1 = figure1.add_subplot(gs1[0])
    for i in range(len(lsResults)):
        ax1.plot(np.asarray(lsEpoch[i]), np.asarray(lsTrain[i]), lsShape[i], label=lsNames[i])
    plt.legend(bbox_to_anchor=legendPos1, loc=len(lsTrain), ncol=1, borderaxespad=0.)
    ax1.set_xlim([0, maxEpoch + 1])
    ax1.set_ylim(ylimTrain)
    ax1.set_ylabel('Training loss', fontsize=12)
    ax1.set_xlabel('Number of Epochs', fontsize=12)
    #ax1.set_title('Training loss by number of epochs', fontsize=16)

    ax2 = figure1.add_subplot(gs1[1])
    for i in range(len(lsResults)):
        ax2.plot(np.asarray(lsEpoch[i]), np.asarray(lsTest[i]), lsShape[i], label=lsNames[i])
    plt.legend(bbox_to_anchor=legendPos2, loc=len(lsTest), ncol=1, borderaxespad=0.)
    ax2.set_xlim([0, maxEpoch + 1])
    ax2.set_ylim(ylimVal)
    ax2.set_ylabel('Validation BLEU', fontsize=12)
    ax2.set_xlabel('Number of Epochs', fontsize=12)
    #ax2.set_title('Validation BLEU by number of epochs', fontsize=16)

    figure1.savefig(outPath + '_plot.png', bbox_inches='tight')
    plt.close(figure1)


# 1. Hyper-parameter - Att, learning rate x dim
dir1 = '../model/att_vi/v_lr-3_Dim[1-3]_'
dir2 = '../model/att_vi/v_lr-4_Dim[1-3]_'
lsResult_Att = []
for i in range(1,4):
    lsResult_Att.append(pickle.load(open(dir1 + str(i) + '/_accuracy.p', 'rb')))
    lsResult_Att.append(pickle.load(open(dir2 + str(i) + '/_accuracy.p', 'rb')))
lsNames = ['LR 1e-3, DimEnc 256', 'LR 1e-4, DimEnc 256',
           'LR 1e-3, DimEnc 512', 'LR 1e-4, DimEnc 512',
           'LR 1e-3, DimEnc 1024', 'LR 1e-4, DimEnc 1024',]
lsShape = ['c-', 'm-', 'b-', 'k-', 'r-', 'g-']

plot(lsResult_Att, lsNames, lsShape, outPath='../plots/Att_lrDim', legendPos1 = (0.6, 0.7, 1., .3), legendPos2 = (0.6, 0.05, 1., .3))


# 2. Hyper - Att, zh
dirs = ['../model/att_zh/z_lr-3_Dim[1-3]_1', '../model/att_zh/z_lr-4_Dim[1-3]_1', '../model/att_zh/z_lr-3_Dim[1-3]_2',
        '../model/att_zh/lr-4V2_Dim[2]_2', '../model/att_zh/z_lr-3_Dim[1-3]_3', '../model/att_zh/lr-4V2_uE_Dim[2]_2']

lsResult_Att_zh_lr = []
for d in dirs:
    lsResult_Att_zh_lr.append(pickle.load(open(d + '/_accuracy.p', 'rb')))
lsNames = ['LR 1e-3, DimEnc 256', 'LR 1e-4, DimEnc 256',
           'LR 1e-3, DimEnc 512', 'LR 1e-4, DimEnc 512',
           'LR 1e-3, DimEnc 1024', 'LR 1e-4, DimEnc 512, trainEmb']
lsShape = ['c-', 'm-', 'b-', 'k-', 'r-', 'g-']

plot(lsResult_Att_zh_lr, lsNames, lsShape, outPath='../plots/Att_lrDim_zh', legendPos1 = (0.3, 0.7, .1, .3), legendPos2 = (0.3, 0.7, .1, .3))




# 3. Hyper - Self-Att, vi - LR and dim

dirs = ['../model/self_att_vi/v_dimD1024F512S6_LR_1', '../model/self_att_vi/v_dimD1024F512S6_LR_2', '../model/self_att_vi/v_dimD1024F512S6_LR_3',
        '../model/self_att_vi/v_dim512S6_LR_2', '../model/self_att_vi/v_dim512S6_LR_3', '../model/self_att_vi/v_dim512S6_LR_4',]

lsResult_sAtt_vi_lr = []
for d in dirs:
    lsResult_sAtt_vi_lr.append(pickle.load(open(d + '/_accuracy.p', 'rb')))
lsNames = ['LR 1e-2, DimDec 1024', 'LR 1e-3, DimDec 1024',
           'LR 1e-4, DimDec 1024', 'LR 1e-3, DimEnc 512',
           'LR 1e-4, DimEnc 512', 'LR 1e-5, DimEnc 512']
lsShape = ['c-', 'm-', 'b-', 'k-', 'r-', 'g-']

plot(lsResult_sAtt_vi_lr, lsNames, lsShape, outPath='../plots/sAtt_lrDim_vi', ylimTrain = [0, 1500], legendPos1 = (0.3, 0.4, .1, .3), legendPos2 = (0.3, 0.7, .1, .3))


# 4. Hyper, Self-Att, vi - Stack
dirs = ['../model/self_att_vi/v_dimD1024F512_LR-4_Stack_1', '../model/self_att_vi/v_dimD1024F512_LR-4_Stack_2', '../model/self_att_vi/v_dimD1024F512_LR-4_Stack_3',
        '../model/self_att_vi/v_dimD1024F512_LR-4_Stack_4', '../model/self_att_vi/v_dimD1024F512_LR-4_Stack_5']

lsResult_sAtt_vi_stack = []
for d in dirs:
    lsResult_sAtt_vi_stack.append(pickle.load(open(d + '/_accuracy.p', 'rb')))
lsNames = ['Stack: 2', 'Stack: 4', 'Stack: 6', 'Stack: 8', 'Stack: 10']
lsShape = ['c-', 'm-', 'b-', 'k-', 'r-']

plot(lsResult_sAtt_vi_stack, lsNames, lsShape, outPath='../plots/sAtt_stack_vi', legendPos1 = (0.6, 0.2, .1, .3), legendPos2 = (0.3, 0.7, .1, .3))








# 2. Att model - attention example
import model as m
import torch
from torch.autograd import Variable

inputPath = '../project_data/model_input/en-vi_v25K_emb300K/'
modelPath = '../model/att_vi/v_lr-4_Dim[1-3]_2/'
dataLang = 'vi'
dataLength=50
dimLSTM_dec=1024
dimLSTM_enc=512
flgNoPretrain=False
optType='Adam'
p_dropOut=0.5
targetLang='en'
targetLength=50
teacher_forcing_ratio=1.0
vocab_size_data=25000
vocab_size_target=25000

data_emb = pickle.load(open(inputPath + 'embeddings_' + dataLang + '.p', 'rb'))
vocab_size_data, emb_dim_data = data_emb.shape
data_emb = torch.from_numpy(data_emb).float()

target_emb = pickle.load(open(inputPath + 'embeddings_' + targetLang + '.p', 'rb'))
vocab_size_target, emb_dim_target = target_emb.shape
target_emb = torch.from_numpy(target_emb).float()

test = m.translationDataset(inputPath, 'dev_' + dataLang, 'dev_' + targetLang, dataLength, targetLength)
dict_target = pickle.load(open(inputPath + 'dict_' + targetLang + '.p', 'rb'))
dict_data = pickle.load(open(inputPath + 'dict_' + dataLang + '.p', 'rb'))

model_paras = {'emb_dim_enc': emb_dim_data, 'emb_dim_dec': emb_dim_target, 'vocab_size_enc': vocab_size_data, 'vocab_size_dec': vocab_size_target,
               'teacher_forcing_ratio': teacher_forcing_ratio, 'dimLSTM_enc': dimLSTM_enc, 'dimLSTM_dec': dimLSTM_dec, 'nLSTM_enc': 1,
               'flg_updateEmb': False, 'p_dropOut': p_dropOut}

model = m.AttRNNseq2seq(model_paras, data_emb, target_emb)

model0 = torch.load(modelPath + '_model.pt', map_location=lambda storage, loc: storage)
model.load_state_dict(model0)
print(model)


from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import rcParams
rcParams['text.latex.unicode']=True
from matplotlib.font_manager import FontProperties

def showAttention(input_list, output_list, attentions, outPath, fullSize = True):
    ChineseFont2 = FontProperties('SimHei')

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)
    if not fullSize:
        leng_out = len(output_list) + 5
        leng_in = len(input_list) + 5
        img = ax.matshow(attentions.numpy()[:leng_out,:leng_in], cmap='bone')
    else:
        img = ax.matshow(attentions.numpy(), cmap='bone')
    divider = make_axes_locatable(ax)
    cax1 = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(img, cax=cax1)

    # Set up axes
    ax.set_xticklabels([''] + input_list +
                       ['<EOS>'], rotation=90, fontsize=9, fontproperties = ChineseFont2)
    ax.set_yticklabels([''] + output_list + ['<EOS>'], fontsize=9)

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    fig.savefig(outPath + '_attViz.png', bbox_inches='tight')
    plt.close(fig)



key = 295
sample = test[key]
data, target, data_len, target_len = sample['data'], sample['target'], sample['data_len'], sample['target_len']
data, target, data_len, target_len = Variable(data).long().view(1,-1), Variable(target).long().view(1,-1), Variable(data_len).long(), Variable(target_len).long()
model.eval()
output, idx, mask, att_weights = model.inference(data, target, data_len, target_len)

pred_txt = idx2words(dict_target['id2token'], idx.data.numpy().astype(int))
target_txt = idx2words(dict_target['id2token'], target.data.numpy().astype(int))
data_txt = idx2words(dict_data['id2token'], data.data.numpy().astype(int))

showAttention(data_txt[0], pred_txt[0], att_weights[0].detach(), outPath = '../plots/att_' + str(key) )


# Att - ZH example
inputPath = '../project_data/model_input/en-zh_v25K_emb300K/'
modelPath = '../model/att_zh/lr-4V2_uE_Dim[2]_2/'
dataLang = 'zh'
dataLength=50
dimLSTM_dec=1024
dimLSTM_enc=512
flgNoPretrain=False
optType='Adam'
p_dropOut=0.5
targetLang='en'
targetLength=50
teacher_forcing_ratio=1.0
vocab_size_data=25000
vocab_size_target=25000

data_emb = pickle.load(open(inputPath + 'embeddings_' + dataLang + '.p', 'rb'))
vocab_size_data, emb_dim_data = data_emb.shape
data_emb = torch.from_numpy(data_emb).float()

target_emb = pickle.load(open(inputPath + 'embeddings_' + targetLang + '.p', 'rb'))
vocab_size_target, emb_dim_target = target_emb.shape
target_emb = torch.from_numpy(target_emb).float()

test = m.translationDataset(inputPath, 'dev_' + dataLang, 'dev_' + targetLang, dataLength, targetLength)
dict_target = pickle.load(open(inputPath + 'dict_' + targetLang + '.p', 'rb'))
dict_data = pickle.load(open(inputPath + 'dict_' + dataLang + '.p', 'rb'))

model_paras = {'emb_dim_enc': emb_dim_data, 'emb_dim_dec': emb_dim_target, 'vocab_size_enc': vocab_size_data, 'vocab_size_dec': vocab_size_target,
               'teacher_forcing_ratio': teacher_forcing_ratio, 'dimLSTM_enc': dimLSTM_enc, 'dimLSTM_dec': dimLSTM_dec, 'nLSTM_enc': 1,
               'flg_updateEmb': True, 'p_dropOut': p_dropOut}

model = m.AttRNNseq2seq(model_paras, data_emb, target_emb)

model0 = torch.load(modelPath + '_model.pt', map_location=lambda storage, loc: storage)
model.load_state_dict(model0)


key = 1123
sample = test[key]
data, target, data_len, target_len = sample['data'], sample['target'], sample['data_len'], sample['target_len']
data, target, data_len, target_len = Variable(data).long().view(1,-1), Variable(target).long().view(1,-1), Variable(data_len).long(), Variable(target_len).long()
model.eval()
output, idx, mask, att_weights = model.inference(data, target, data_len, target_len)

pred_txt = idx2words(dict_target['id2token'], idx.data.numpy().astype(int))
target_txt = idx2words(dict_target['id2token'], target.data.numpy().astype(int))
data_txt = idx2words(dict_data['id2token'], data.data.numpy().astype(int))

showAttention(data_txt[0], pred_txt[0], att_weights[0].detach(), outPath = '../plots/att_zh_' + str(key), fullSize = False )




# 4. Self-att model - attention example