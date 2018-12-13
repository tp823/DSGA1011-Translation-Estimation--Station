'''
Script with different type of models

Hyper: language, pretrain emb or not, beam search or not
'''
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import pickle
import torch.nn.functional as F
from torch.utils.data import Dataset
import random
import sacrebleu
import pdb
import math
import copy

# Global Variables
PAD_IDX = 0
UNK_IDX = 1
SOS_IDX = 2
EOS_IDX = 3


#==== Data Loader ======

class translationDataset(Dataset):
    def __init__(self, root_dir, data_name, target_name, dataLength, targetLength):
        self.data_list=pickle.load(open(root_dir + data_name + '.p', 'rb'))
        self.target_list=pickle.load(open(root_dir + target_name + '.p', 'rb'))
        assert (len(self.data_list) == len(self.target_list))
        self.dataLength = dataLength
        self.targetLength = targetLength

    def __len__(self):
        return len(self.target_list)

    def __getitem__(self, key):
        data = self.data_list[key]
        target = self.target_list[key]

        data_len = max(1, min(len(data), self.dataLength))
        target_len = max(1, min(len(target), self.targetLength))
        # Pad sequence to length of self.dataLength + 1, including the EOS token
        padded_data = np.pad(np.array(data[:self.dataLength] + [EOS_IDX]),
                                 pad_width=((0, self.dataLength - data_len)),
                                 mode="constant", constant_values=0)

        padded_target = np.pad(np.array(target[:self.targetLength] + [EOS_IDX]),
                             pad_width=((0, self.targetLength - target_len)),
                             mode="constant", constant_values=0)

        sample = {'data': torch.from_numpy(padded_data).long(),
                  'target': torch.from_numpy(padded_target).long(),
                  'data_len': torch.from_numpy(np.asarray(data_len)).long(),
                  'target_len': torch.from_numpy(np.asarray(target_len)).long(),}
        return sample



#===== Encoder / Decoders ========
class EncoderRNN(nn.Module):
    def __init__(self, embedding, dimLSTM, nLSTM, vocab_size, emb_dim, flg_bidirectional, p_dropOut, flg_updateEmb):
        super(EncoderRNN, self).__init__()
        self.dimLSTM = dimLSTM
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD_IDX)
        self.flg_bidirectional = flg_bidirectional

        if embedding is not None:
            self.embedding.weight = nn.Parameter(embedding, requires_grad=flg_updateEmb)

        self.gru = nn.GRU(emb_dim, self.dimLSTM, nLSTM, batch_first=True, bidirectional=flg_bidirectional, dropout=p_dropOut)
        self.dropout = nn.Dropout(p_dropOut)

        self.params = list(self.gru.parameters())
        if flg_updateEmb:
            self.params += list(self.embedding.parameters())

    def forward(self, input):
        batchSize = input.shape[0]
        embedded = self.embedding(input)

        if self.flg_bidirectional:
            h0 = self.init_hidden(batchSize, 2)
        else:
            h0 = self.init_hidden(batchSize, 1)

        output = self.dropout(embedded)
        output, hidden = self.gru(output, h0) # output.shape: batch x seq_len x dimLSTM(x2 if bi), hidden.shape: 1(2 if bi) x batch x dimLSTM

        if self.flg_bidirectional:
            hidden = torch.cat([hidden[0], hidden[1]], 1).view(1, batchSize, -1)
        return output, hidden.transpose(0,1)

    def init_hidden(self, batchSize, nlayers):
        if torch.cuda.is_available():
            return Variable(torch.zeros(nlayers, batchSize, self.dimLSTM)).cuda()
        else:
            return Variable(torch.zeros(nlayers, batchSize, self.dimLSTM))


class DecoderRNN(nn.Module):
    def __init__(self, embedding, dimLSTM, vocab_size, emb_dim, p_dropOut, flg_updateEmb):
        super(DecoderRNN, self).__init__()
        self.dimLSTM = dimLSTM # Note: if encoder is bi-directional this should be twice as large as encoder's dimLSTM
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD_IDX)

        if embedding is not None:
            self.embedding.weight = nn.Parameter(embedding, requires_grad=flg_updateEmb)

        self.dropout = nn.Dropout(p=p_dropOut)
        self.gru = nn.GRU(self.dimLSTM + emb_dim, self.dimLSTM, batch_first=True)
        self.out = nn.Linear(self.dimLSTM, vocab_size)
        self.softmax = nn.LogSoftmax(dim=2)

        self.params = list(self.gru.parameters()) + list(self.out.parameters())
        if flg_updateEmb:
            self.params += list(self.embedding.parameters())

    def forward(self, input, hidden, enc_output):
        # Batch front
        embedded = self.dropout(self.embedding(input))
        embedded_concat = torch.cat((embedded, enc_output), dim=2)
        output, hidden = self.gru(embedded_concat, hidden)
        output = self.softmax(self.out(output))
        return output, hidden




#======= Completed models, combining both encoder and decoder ==============


class RNNseq2seq(nn.Module):
    def __init__(self, model_paras, data_emb, target_emb):
        super(RNNseq2seq, self).__init__()
        self.model_paras = model_paras
        self.flg_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.flg_cuda else "cpu")
        self.teacher_forcing_ratio = model_paras.get('teacher_forcing_ratio', 0.0)

        vocab_size_enc = model_paras.get('vocab_size_enc', 100000)
        self.vocab_size_dec = model_paras.get('vocab_size_dec', 100000)

        emb_dim_enc = model_paras.get('emb_dim_enc', 300)
        emb_dim_dec = model_paras.get('emb_dim_dec', 300)

        dimLSTM_enc = model_paras.get('dimLSTM_enc', 128)
        dimLSTM_dec = model_paras.get('dimLSTM_dec', 128)

        nLSTM_enc = model_paras.get('nLSTM_enc', 1)
        flg_bidirectional_enc = model_paras.get('flg_bidirectional_enc', False)
        flg_updateEmb = model_paras.get('flg_updateEmb', False)
        p_dropOut = model_paras.get('p_dropOut', 0.5)

        if flg_bidirectional_enc:
            assert(dimLSTM_enc *2 == dimLSTM_dec ) ##TODO: add msg
        else:
            assert(dimLSTM_enc  == dimLSTM_dec)

        self.encoder = EncoderRNN(data_emb, dimLSTM_enc, nLSTM_enc, vocab_size_enc, emb_dim_enc, flg_bidirectional_enc, p_dropOut, flg_updateEmb)
        self.decoder = DecoderRNN(target_emb, dimLSTM_dec, self.vocab_size_dec, emb_dim_dec, p_dropOut, flg_updateEmb)

        self.params = self.encoder.params + self.decoder.params

    def forward(self, data, target, data_len, target_len):
        encoder_output, encoder_hidden = self.encoder(data)
        use_teacher_forcing = True if random.random() < self.teacher_forcing_ratio else False
        decoder_output, decoder_idx, mask, _ = self._decode(encoder_hidden, target, use_teacher_forcing)
        return decoder_output, decoder_idx, mask, []

    def _decode(self, encoder_hidden, target, use_teacher_forcing):
        batchSize, max_length = target.shape
        decoder_hidden = encoder_hidden.transpose(0,1)
        mask = torch.ones(target.shape)
        decoder_input = torch.tensor([SOS_IDX]*batchSize, device=self.device).view(batchSize, 1)
        decoder_idx = []

        if use_teacher_forcing:
            decoder_input = torch.cat([decoder_input, target[:,0:-1]], 1) # All inputs known, don't need to iterate
            encoder_hidden_seq = encoder_hidden.repeat(1, max_length, 1) # Repeated encoder last hidden layer as the context vector
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_hidden_seq)

        else:
            decoder_output = torch.zeros(batchSize, max_length, self.vocab_size_dec).to(self.device)
            for di in range(max_length):
                decoder_output_i, decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_hidden)
                topv, topi = decoder_output_i.topk(1)
                decoder_input = topi.squeeze(-1).detach()
                decoder_output[:,di,:] = decoder_output_i.squeeze()
                decoder_idx.append(decoder_input)

            # Create a mask, set value to 0 for tokens generated after the first EOS_IDX. Modify loss calculation during training
            decoder_idx = torch.cat(decoder_idx, 1)
            for i in range(batchSize):
                for j in range(max_length):
                    if decoder_idx[i][j].item() == EOS_IDX:
                        mask[i,j:] = 0
        if self.flg_cuda:
            mask = mask.cuda()
        return decoder_output, decoder_idx, mask, []

    def inference(self, data, target, data_len, target_len):
        encoder_output, encoder_hidden = self.encoder(data)
        decoder_output, decoder_idx, mask, _ = self._decode(encoder_hidden, target, use_teacher_forcing = False)
        return decoder_output, decoder_idx, mask, []


#========== Attention enc-dec ==============
class AttnDecoderRNN(nn.Module):
    def __init__(self, embedding, dim_enc, dimLSTM_dec, vocab_size, emb_dim, p_dropOut, flg_updateEmb):
        super(AttnDecoderRNN, self).__init__()

        self.dimLSTM_dec = dimLSTM_dec
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD_IDX)

        if embedding is not None:
            self.embedding.weight = nn.Parameter(embedding, requires_grad=flg_updateEmb)

        self.out = nn.Linear(dimLSTM_dec, vocab_size)
        self.attn = nn.Linear(dimLSTM_dec + emb_dim, dim_enc) # dimLSTM_enc should be the entire encoder output size, so double the size if bi-Directional
        self.dropout = nn.Dropout(p=p_dropOut)

        self.gru = nn.GRU(dim_enc + emb_dim, self.dimLSTM_dec, batch_first=True)
        self.softmax = nn.LogSoftmax(dim=2)

        self.params = list(self.gru.parameters()) + list(self.out.parameters()) + list(self.attn.parameters())
        if flg_updateEmb:
            self.params += list(self.embedding.parameters())

    def forward(self, input, hidden, encoder_outputs):
        # encoder_outputs.shape: batch x seq_len x dimLSTM_enc*2
        # Get the embedding of the current input
        embedded = self.dropout(self.embedding(input)) # Batch x seq_len=1 x emb_dim
        # Get weights from dot product
        query = self.attn(torch.cat((embedded, hidden.transpose(0,1)), dim=2)) # Batch x seq_len=1 x dim_enc
        attn_weights = torch.bmm(query, encoder_outputs.transpose(1, 2)) # Batch x 1 x seq_len_enc

        # Softmax to normalize
        attn_weights = F.softmax(attn_weights, dim = 2) # Batch x 1 x seq_len_enc

        # apply to encoder outputs to get weighted average
        context = attn_weights.bmm(encoder_outputs)
        rnn_input = torch.cat((embedded, context), 2)
        output, hidden = self.gru(rnn_input, hidden)
        output = F.log_softmax(self.out(output.squeeze()), dim=1)
        return output, hidden, attn_weights

    def init_hidden(self, batchSize, nlayers):
        if torch.cuda.is_available():
            return Variable(torch.zeros(nlayers, batchSize, self.dimLSTM_dec)).cuda()
        else:
            return Variable(torch.zeros(nlayers, batchSize, self.dimLSTM_dec))



class AttRNNseq2seq(nn.Module):
    def __init__(self, model_paras, data_emb, target_emb):
        super(AttRNNseq2seq, self).__init__()
        self.model_paras = model_paras
        self.flg_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.flg_cuda else "cpu")
        self.teacher_forcing_ratio = model_paras.get('teacher_forcing_ratio', 0.0)

        vocab_size_enc = model_paras.get('vocab_size_enc', 100000)
        self.vocab_size_dec = model_paras.get('vocab_size_dec', 100000)

        emb_dim_enc = model_paras.get('emb_dim_enc', 300)
        emb_dim_dec = model_paras.get('emb_dim_dec', 300)

        dimLSTM_enc = model_paras.get('dimLSTM_enc', 128)
        dimLSTM_dec = model_paras.get('dimLSTM_dec', 256)
        assert (dimLSTM_enc * 2 == dimLSTM_dec) # Use last enc hidden as dec start

        nLSTM_enc = model_paras.get('nLSTM_enc', 1)
        flg_bidirectional_enc = True
        flg_updateEmb = model_paras.get('flg_updateEmb', False)
        p_dropOut = model_paras.get('p_dropOut', 0.5)

        self.encoder = EncoderRNN(data_emb, dimLSTM_enc, nLSTM_enc, vocab_size_enc, emb_dim_enc, flg_bidirectional_enc, p_dropOut, flg_updateEmb)
        self.decoder = AttnDecoderRNN(target_emb, dimLSTM_enc*2, dimLSTM_dec, self.vocab_size_dec, emb_dim_dec, p_dropOut, flg_updateEmb) # Assume encoder is bi-directional

        self.params = self.encoder.params + self.decoder.params

    def forward(self, data, target, data_len, target_len):
        encoder_output, encoder_hidden = self.encoder(data)
        use_teacher_forcing = True if random.random() < self.teacher_forcing_ratio else False
        decoder_output, decoder_idx, mask, att_weights = self._decode(encoder_hidden, encoder_output, target, use_teacher_forcing)
        return decoder_output, decoder_idx, mask, att_weights

    def _decode(self, encoder_hidden, encoder_output, target, use_teacher_forcing):
        batchSize, max_length = target.shape
        enc_len = encoder_output.shape[1]
        decoder_hidden = encoder_hidden.transpose(0,1)
        mask = torch.ones(target.shape)
        decoder_input = torch.tensor([SOS_IDX]*batchSize, device=self.device).view(batchSize, 1)
        decoder_idx = []
        att_weights = torch.zeros(batchSize, max_length, enc_len).to(self.device)

        decoder_output = torch.zeros(batchSize, max_length, self.vocab_size_dec).to(self.device)
        for di in range(max_length):
            decoder_output_i, decoder_hidden, att_w_i = self.decoder(decoder_input, decoder_hidden, encoder_output)
            if use_teacher_forcing:
                decoder_input = target[:, di].view(-1,1)
            else:
                topv, topi = decoder_output_i.topk(1)
                decoder_input = topi.squeeze(-1).detach().view(-1,1)
            decoder_output[:, di, :] = decoder_output_i.squeeze()
            decoder_idx.append(decoder_input)
            att_weights[:, di, :] = att_w_i.squeeze()

            # Create a mask, set value to 0 for tokens generated after the first EOS_IDX. Modify loss calculation during training
        decoder_idx = torch.cat(decoder_idx, 1)
        for i in range(batchSize):
            for j in range(max_length):
                if decoder_idx[i][j].item() == EOS_IDX:
                    mask[i,j:] = 0
        if self.flg_cuda:
            mask = mask.cuda()
        return decoder_output, decoder_idx, mask, att_weights

    def inference(self, data, target, data_len, target_len):
        encoder_output, encoder_hidden = self.encoder(data)
        decoder_output, decoder_idx, mask, att_weights = self._decode(encoder_hidden, encoder_output, target, use_teacher_forcing = False)
        return decoder_output, decoder_idx, mask, att_weights


#=========== Self-att encoder ===============


## Code based on http://nlp.seas.harvard.edu/2018/04/03/attention.html
def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class LayerNorm(nn.Module):
    "Construct a layernorm module "
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True) ##TODO: double check dim
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class Embeddings(nn.Module):
    def __init__(self, emb_dim, vocab, flg_updateEmb, embedding = None):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, emb_dim, padding_idx=PAD_IDX)
        self.emb_dim = emb_dim
        self.params = []
        if embedding is not None:
            self.embedding.weight = nn.Parameter(embedding, requires_grad=flg_updateEmb)
            self.params += list(self.embedding.parameters())

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.emb_dim)



def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, nHeads, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % nHeads == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // nHeads
        self.nHeads = nHeads
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [l(x).view(nbatches, -1, self.nHeads, self.d_k).transpose(1, 2) for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear. Assume output dim is also d_model
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.nHeads * self.d_k)
        return self.linears[-1](x)



class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp( (torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)).float() )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)


class SublayerConnection(nn.Module):
    '''
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    '''
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class Encoder(nn.Module):
    "Core encoder is a stack of N layers"

    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class SelfAttEncRNNDec(nn.Module):
    '''
    Model using self-attention encoder, and decoder with attention from encoder output
    '''
    def __init__(self, model_paras, data_emb, target_emb):
        super(SelfAttEncRNNDec, self).__init__()

        self.model_paras = model_paras
        self.flg_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.flg_cuda else "cpu")
        self.teacher_forcing_ratio = model_paras.get('teacher_forcing_ratio', 0.0)

        vocab_size_enc = model_paras.get('vocab_size_enc', 100000)
        self.vocab_size_dec = model_paras.get('vocab_size_dec', 100000)

        emb_dim_enc = model_paras.get('emb_dim_enc', 300)
        emb_dim_dec = model_paras.get('emb_dim_dec', 300)

        dimLSTM_dec = model_paras.get('dimLSTM_dec', 300)

        flg_updateEmb = model_paras.get('flg_updateEmb', False)
        p_dropOut = model_paras.get('p_dropOut', 0.5)

        nHead = model_paras.get('nHead', 6) # Number of heads in multi-headed att
        nStack = model_paras.get('nStack', 2) # Number of encoder stacks
        dimFF = model_paras.get('dimFF', 128) # Feed-forward hidden layer size
        self.flg_initAvg = model_paras.get('flg_initAvg', False) # Init hidden layer with average encoder output

        c = copy.deepcopy
        attn = MultiHeadedAttention(nHead, emb_dim_enc)
        ff = PositionwiseFeedForward(emb_dim_enc, dimFF, p_dropOut)
        position = PositionalEncoding(emb_dim_enc, p_dropOut)
        emb = Embeddings(emb_dim_enc, vocab_size_enc, flg_updateEmb)
        self.init_FF = nn.Linear(emb_dim_enc, dimLSTM_dec) # Convert average encoder output to decoder hidden init

        self.emb_enc = nn.Sequential(emb, c(position))
        self.encoder = Encoder(EncoderLayer(emb_dim_enc, c(attn), c(ff), p_dropOut), nStack)
        self.decoder = AttnDecoderRNN(target_emb, emb_dim_enc, dimLSTM_dec, self.vocab_size_dec, emb_dim_dec, p_dropOut, flg_updateEmb)

        self.params = emb.params + list(self.encoder.parameters()) + self.decoder.params + list(self.init_FF.parameters())# Fix this

    def forward(self, data, target, data_len, target_len):
        batchSize = data.shape[0]
        src_mask = (data != PAD_IDX).unsqueeze(-2)
        if self.flg_cuda:
            src_mask = src_mask.cuda()
        encoder_output = self.encoder(self.emb_enc(data), src_mask)
        if self.flg_initAvg:
            encoder_hidden = self.init_FF(encoder_output.mean(dim=1)).view(1, batchSize, -1)
        else:
            encoder_hidden = self.decoder.init_hidden(batchSize, nlayers = 1)
        use_teacher_forcing = True if random.random() < self.teacher_forcing_ratio else False
        decoder_output, decoder_idx, mask, att_weights = self._decode(encoder_hidden, encoder_output, target, use_teacher_forcing)
        return decoder_output, decoder_idx, mask, att_weights

    def _decode(self, encoder_hidden, encoder_output, target, use_teacher_forcing):
        batchSize, max_length = target.shape
        enc_len = encoder_output.shape[1]
        decoder_hidden = encoder_hidden
        mask = torch.ones(target.shape)
        decoder_input = torch.tensor([SOS_IDX]*batchSize, device=self.device).view(batchSize, 1)
        decoder_idx = []
        att_weights = torch.zeros(batchSize, max_length, enc_len).to(self.device) # from decoder

        decoder_output = torch.zeros(batchSize, max_length, self.vocab_size_dec).to(self.device)
        for di in range(max_length):
            decoder_output_i, decoder_hidden, att_w_i = self.decoder(decoder_input, decoder_hidden, encoder_output)
            if use_teacher_forcing:
                decoder_input = target[:, di].view(-1,1)
            else:
                topv, topi = decoder_output_i.topk(1)
                decoder_input = topi.squeeze(-1).detach().view(-1,1)
            decoder_output[:, di, :] = decoder_output_i.squeeze()
            decoder_idx.append(decoder_input)
            att_weights[:, di, :] = att_w_i.squeeze()

            # Create a mask, set value to 0 for tokens generated after the first EOS_IDX. Modify loss calculation during training
        decoder_idx = torch.cat(decoder_idx, 1)
        for i in range(batchSize):
            for j in range(max_length):
                if decoder_idx[i][j].item() == EOS_IDX:
                    mask[i,j:] = 0
        if self.flg_cuda:
            mask = mask.cuda()
        return decoder_output, decoder_idx, mask, att_weights

    def inference(self, data, target, data_len, target_len):
        batchSize = data.shape[0]
        src_mask = (data != PAD_IDX).unsqueeze(-2)
        if self.flg_cuda:
            src_mask = src_mask.cuda()
        encoder_output = self.encoder(self.emb_enc(data), src_mask)
        if self.flg_initAvg:
            encoder_hidden = self.init_FF(encoder_output.mean(dim=1)).view(1, batchSize, -1)
        else:
            encoder_hidden = self.decoder.init_hidden(batchSize, nlayers = 1)
        decoder_output, decoder_idx, mask, att_weights = self._decode(encoder_hidden, encoder_output, target, use_teacher_forcing = False)
        return decoder_output, decoder_idx, mask, att_weights



#=========== Training Class ==========

class trainModel(object):
    def __init__(self, train_paras, train_loader, test_loader, model, optimizer, id2token):
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.model = model
        self.optimizer = optimizer
        self.flg_cuda = torch.cuda.is_available()
        self.id2token = id2token


        # self.train_paras = train_paras
        self.n_batch = train_paras.get('n_batch', 1) # Use this to adjust how many batches as one epoch
        self.n_iter = train_paras.get('n_iter', 1)
        self.log_interval = train_paras.get('log_interval', 1)
        self.lr_decay = train_paras.get('lr_decay', None)  # List of 4 numbers: [init_lr, lr_decay_rate, lr_decay_interval, min_lr]
        self.flgSave = train_paras.get('flgSave', False)  # Set to true if save model
        self.savePath = train_paras.get('savePath', './')

        if self.lr_decay:
            assert len(self.lr_decay) == 4  # Elements include: [starting_lr, decay_multiplier, decay_per_?_epoch, min_lr]
        self.criterion = torch.nn.NLLLoss(reduction='none')
        self.cnt_iter = 0

        self.lsTrainLoss = []
        self.lsTestAccuracy = []
        self.lsEpochNumber = []
        self.bestAccuracy = 0.0
        self.acc = 0.0
        np.random.seed(train_paras.get('randSeed', 42))



    def run(self):
        for epoch in range(self.n_iter):
            self._train(epoch)
            self._test(epoch)
            pickle.dump([self.lsEpochNumber, self.lsTrainLoss, self.lsTestAccuracy], open(self.savePath + '_accuracy.p', 'wb'))
            if self.acc > self.bestAccuracy:
                self.bestAccuracy = self.acc
                if self.flgSave:
                    self._saveModel()
                    self._savePrediction()
        return self.model, self.lsTrainLoss, self.lsTestAccuracy

    def _train(self, epoch):
        train_loss = 0
        self.model.train()
        if self.lr_decay:
            lr = max(self.lr_decay[0] * (self.lr_decay[1] ** (epoch // self.lr_decay[2])), self.lr_decay[3]) # Exponential decay
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        j, nRec = 0, 0

        self.Y_train = []
        self.target_train = []

        while j <= self.n_batch: # Use this to adjust how many batches as one epoch
            # for batch_idx, sample in enumerate(self.train_loader):
            sample = self.train_loader.__iter__().__next__()
            data, target, data_len, target_len = sample['data'], sample['target'], sample['data_len'], sample['target_len']

            nRec += data.size()[0]
            data, target, data_len, target_len = Variable(data).long(), Variable(target).long(), Variable(data_len).long(), Variable(target_len).long()

            self.cnt_iter += 1

            if self.flg_cuda:
                data, target, data_len, target_len = data.cuda(), target.cuda(), data_len.cuda(), target_len.cuda()

            self.optimizer.zero_grad()
            output, idx, mask, att_weights = self.model(data, target, data_len, target_len)
            loss = self.criterion(output.transpose(1,2), target)
            loss = (loss * mask).sum() / mask.sum()
            loss.backward()
            self.optimizer.step()

            train_loss += loss.data.item() * data.size()[0]
            j += 1
            if (j % self.log_interval[1] == 0):
                train_loss_temp = train_loss / nRec
                print('Train Epoch: {}, Batch: {}, Loss: {:.4f}'.format(epoch, j, train_loss_temp))

        if (epoch == 0) | (epoch % self.log_interval[0] == 0) | (epoch == self.n_iter - 1):
            train_loss /= nRec
            self.lsTrainLoss.append(train_loss)
            print('\nTrain Epoch: {} Loss: {:.4f} '.format(epoch, train_loss))


    def _test(self, epoch):
        if (epoch == 0) | (epoch % self.log_interval[0] == 0) | (epoch == self.n_iter - 1):

            self.model.eval()
            test_loss, nRec = 0, 0

            self.pred = []
            if epoch == 0:
                self.target = []

            for batch_idx, sample in enumerate(self.test_loader):
                data, target, data_len, target_len = sample['data'], sample['target'], sample['data_len'], sample['target_len']

                nRec += data.size()[0]
                data, target, data_len, target_len = Variable(data).long(), Variable(target).long(), Variable(data_len).long(), Variable(target_len).long()

                if self.flg_cuda:
                    data, target, data_len, target_len = data.cuda(), target.cuda(), data_len.cuda(), target_len.cuda()

                with torch.no_grad():
                    output, idx, mask, att_weights = self.model.inference(data, target, data_len, target_len)
                    test_loss_step = self.criterion(output.transpose(1, 2), target)
                    test_loss_step = (test_loss_step * mask).sum() / mask.sum()
                    test_loss += test_loss_step * data.size()[0]

                pred_txt = self._idx2words(idx.data.cpu().numpy().astype(int))
                self.pred.extend(pred_txt)

                if epoch == 0: # Only need to process target once
                    target_txt = self._idx2words(target.data.cpu().numpy().astype(int))
                    self.target.extend(target_txt)

            self.acc = self._getBleu(self.pred, self.target)
            test_loss /= nRec
            print('Test set: Average loss: {:.4f}, Bleu: {:.4f}'.format(test_loss, self.acc))
            self.lsTestAccuracy.append(self.acc)
            self.lsEpochNumber.append(epoch)
            self._getSampleEval()

    def _idx2words(self, idx):
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
                    out_i.append(self.id2token[key])
            out.append(out_i)
        return out

    def _getBleu(self, output, target):
        hypotheses = [' '.join(sentence) for sentence in output]
        refs = [' '.join(sentence) for sentence in target]
        score = sacrebleu.corpus_bleu(hypotheses, [refs], force=True)
        return score.score


    def _getSampleEval(self):
        '''
        Provide a single prediction example
        '''
        i = np.random.choice(len(self.pred))
        try:
            print('Dev sample: ', i)
            print('Reference: ', ' '.join(self.target[i]))
            print('Prediction: ', ' '.join(self.pred[i]))
        except:
            pass

    def _saveModel(self):
        torch.save(self.model, self.savePath + '_model.pt')

    def _savePrediction(self, saveName=''):
        pickle.dump([self.pred, self.target], open(self.savePath + str(saveName) + '_pred.p', 'wb'))

