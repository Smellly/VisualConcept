# coding: utf-8
import nltk
from tqdm import tqdm
from nltk.corpus import stopwords
stopwords = set(stopwords.words("english"))
import numpy as np
import glob
import json
import pickle as pkl

with open('../self-critical.pytorch/data/dataset_coco.json', 'r') as f:
    rawdata = json.load(f)
    
with open('pkls/stopwords.pkl', 'r') as f:
    tmp = pkl.load(f)
    
def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()
def sigmoid(x):
    return 1 / ( 1 + np.exp(-x))
        
vocab = dict()

train = {}
for item in tqdm(rawdata['images']):
    if item['split'] != 'train':
        continue
    freq = {}
    for sen in item['sentences']:
        for t, pos in nltk.pos_tag(sen['tokens']):
            if t not in stopwords and pos == 'NN':
                vocab[t] = vocab.get(t, 0) + 1
                freq[t] = freq.get(t, 0) + 1
    
    for k,v in freq.items():
        freq[k] = sigmoid(v)
    train[item['filename']] = freq

val = {}
for item in tqdm(rawdata['images']):
    if item['split'] != 'val':
        continue
    freq = {}
    for sen in item['sentences']:
        for t, pos in nltk.pos_tag(sen['tokens']):
            if t not in stopwords and pos == 'NN':
                freq[t] = freq.get(t, 0) + 1
    
    for k,v in freq.items():
        freq[k] = sigmoid(v)
    val[item['filename']] = freq
        
test = {}
for item in tqdm(rawdata['images']):
    if item['split'] != 'test':
        continue
    freq = {}
    for sen in item['sentences']:
        for t, pos in nltk.pos_tag(sen['tokens']):
            if t not in stopwords and pos == 'NN':
                freq[t] = freq.get(t, 0) + 1
    
    for k,v in freq.items():
        freq[k] = sigmoid(v)
    test[item['filename']] = freq
        
with open('slabels_nouns/train_label.pkl', 'w') as f:
    pkl.dump(train, f)

with open('slabels_nouns/val_label.pkl', 'w') as f:
    pkl.dump(val, f)
    
with open('slabels_nouns/test_label.pkl', 'w') as f:
    pkl.dump(test, f)
    
vocab_5 = set()
for x in vocab:
    if vocab[x] > 4:
        vocab_5.add(x)
vocab_5 = list(vocab_5)
vocab_5_set = set(vocab_5)

for img in tqdm(train):
    label = np.zeros([len(vocab_5)])
    for word in train[img]:
        if word not in vocab_5_set:
            continue
        ind = vocab_5.index(word)
        label[ind] = train[img][word]
    np.save('slabels_nouns/train/' + img, label)
    
for img in tqdm(val):
    label = np.zeros([len(vocab_5)])
    for word in val[img]:
        if word not in vocab_5_set:
            continue
        ind = vocab_5.index(word)
        label[ind] = val[img][word]
        # print ind
    np.save('slabels_nouns/val/' + img, label)

for img in tqdm(test):
    label = np.zeros([len(vocab_5)])
    for word in test[img]:
        if word not in vocab_5_set:
            continue
        ind = vocab_5.index(word)
        label[ind] = test[img][word]
        # print ind
    np.save('slabels_nouns/test/' + img, label)
    # break

with open('slabels_nouns/vocab_5.pkl', 'w') as f:
    pkl.dump(vocab_5, f)
    
