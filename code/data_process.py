'''
Script to process data
1. Create counter based on training set, and save the counter file (including all tokens)
2. Create vocabulary dictionary of the most frequent vocab_size_limit tokens
3. Re-arrange pre-trained embedding file based on dictionary
4. Convert train/dev/test files into (list of) indices
'''

import numpy as np
import argparse
import os
import pickle
import io
from collections import Counter
from pathlib import Path

# Global Variables
PAD_IDX = 0
UNK_IDX = 1
SOS_IDX = 2
EOS_IDX = 3

def getToken(path, iszh = False):
    out = []
    i = 0
    with open(path) as inputfile:
        for line in inputfile:
            if (iszh == False) | (i%2 ==0):
                out.append(line.strip().lower().split(' '))
            i+=1
    return out


def load_embeddings(word2vec, word2id, embedding_dim):
    # Re-organize pretrained embeddings by data dictionary index, and fill non existing ones with random value
    embeddings = np.zeros((len(word2id), embedding_dim))
    cnt_noEmb = 0
    for word, index in word2id.items():
        if word in word2vec:
            embeddings[index] = np.asarray(word2vec[word])
        else:
            embeddings[index] = np.random.normal(scale=0.6, size=(embedding_dim,))
            cnt_noEmb += 1
    print('Number of token with no pretrained embedding: ', cnt_noEmb)
    return embeddings


def load_vectors(fname, num_vecs=None):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    # n, d = map(int, fin.readline().split())
    data = {}
    i = -1
    for line in fin:
        if i != -1: # First line of embedding file is not pretrained vector
            tokens = line.rstrip().split(' ')
            data[tokens[0]] = list(map(float, tokens[1:]))
        i += 1
        if num_vecs is None:
            pass
        else:
            if i > num_vecs:
                break
    emb_dim = len(data[tokens[0]])
    return data, emb_dim

def getCounter(listSentence):
    token_counter = Counter()
    for tokens in listSentence:
        for token in tokens:
            token_counter[token] += 1
    print('Full vocabulary size: ', len(token_counter))
    return token_counter


def data_dictionary(token_counter, vocab_size_limit):
    vocab, count = zip(*token_counter.most_common(vocab_size_limit))
    id2token = list(vocab)
    token2id = dict(zip(vocab, range(4, 4 + len(vocab))))
    id2token = ['<pad>', '<unk>','<sos>','<eos>'] + id2token
    token2id['<pad>'] = PAD_IDX
    token2id['<unk>'] = UNK_IDX
    token2id['<sos>'] = SOS_IDX
    token2id['<eos>'] = EOS_IDX
    return token2id, id2token


def encoding_tokens(sentence, token2id):
    tokens = [token2id[token] if token in token2id else UNK_IDX for token in sentence]
    return tokens

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Should have train/dev/test in this directory, file names should follow convention 'train.tok.language'
    parser.add_argument("--inputPath", default='../project_data/en-vi/')
    parser.add_argument("--language", default='en')
    parser.add_argument("--outPath")  # Output Path
    parser.add_argument("--vocab_size_limit", type=int, default=100000) # Max number of most frequent tokens
    parser.add_argument("--embFile")  # Path and name of embedding file
    parser.add_argument("--emb_size_limit", type=int) # Max number of pre-trained embedding vectors
    parser.add_argument("--counterFile")  # Path and name of previously created counter file, if exists

    args = parser.parse_args()

    path = Path(args.outPath)
    path.mkdir(parents=True, exist_ok=True)

    print("Data processing parameters: ", args)

    print('Loading data')
    emb, embedding_dim = load_vectors(args.embFile, num_vecs=args.emb_size_limit)
    train = getToken(args.inputPath + 'train.tok.' + args.language)
    dev = getToken(args.inputPath + 'dev.tok.' + args.language)
    test = getToken(args.inputPath + 'test.tok.' + args.language)

    print('Creating/loading data dictionary')
    if args.counterFile is not None:
        token_counter = pickle.load(open(args.counterFile, 'rb'))
    else:
        token_counter = getCounter(train)
        # Dump counter file into inputPath by default
        pickle.dump(token_counter, open(args.inputPath + 'token_counter_' + args.language + '.p', 'wb'))

    token2id, id2token = data_dictionary(token_counter, args.vocab_size_limit)
    embeddings = load_embeddings(emb, token2id, embedding_dim)

    # Tokenize data
    print('Tokenize data')
    train2 = [encoding_tokens(s, token2id) for s in train]
    dev2 = [encoding_tokens(s, token2id) for s in dev]
    test2 = [encoding_tokens(s, token2id) for s in test]

    lsLen = [len(x) for x in train2]
    print('Median sentence length: ', np.percentile(lsLen, 50))
    print('90th percentile: ', np.percentile(lsLen, 90))
    print('95th percentile: ', np.percentile(lsLen, 95))
    print('Max: ', max(lsLen))

    pickle.dump({'token2id': token2id, 'id2token': id2token}, open(args.outPath + 'dict_' + args.language + '.p', 'wb'))
    pickle.dump(embeddings, open(args.outPath + 'embeddings_' + args.language + '.p', 'wb'))
    pickle.dump(train2, open(args.outPath + 'train_' + args.language + '.p', 'wb'))
    pickle.dump(dev2, open(args.outPath + 'dev_' + args.language + '.p', 'wb'))
    pickle.dump(test2, open(args.outPath + 'test_' + args.language + '.p', 'wb'))
