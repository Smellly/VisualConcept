# -*- encoding: utf-8 -*-

from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from torch.autograd import Variable as V

from visualConceptDataset import VisualConceptDataset
from myNets import myResnet

import time
import os
import copy
from tqdm import tqdm

from PIL import Image
import pickle as pkl

def getImglist(path):
    p = 'imgs/train/'
    l = [
        'COCO_train2014_000000000009.jpg',
        'COCO_train2014_000000000025.jpg',
        'COCO_train2014_000000000030.jpg',
        'COCO_train2014_000000000034.jpg',
        'COCO_train2014_000000000049.jpg',
        'COCO_train2014_000000000061.jpg',
        'COCO_train2014_000000000064.jpg',
        'COCO_train2014_000000000071.jpg',
        'COCO_train2014_000000000072.jpg'
        ]
    imglist = [p + x for x in l]
    return imglist

def loadModel(modelPath):
    model = models.resnet101()
    fc_features = model.fc.in_features
    model.fc = nn.Linear(fc_features, 9360)
    model = model.cuda()
    model.eval()

    # load best model weights
    model.load_state_dict(torch.load(modelPath))
    return model

def loadVocab(vocabPath):
    with open(vocabPath, 'r') as f:
        vocab = pkl.load(f)
    return vocab

def getLabel(imgname):
    path = '/media/disk0/jay/workspace/places365/mscoco_train2014/'
    filename = imgname.split('/')[-1].replace('.jpg', '.txt')
    with open(path + filename, 'r') as f:
        raw = f.read().splitlines()

    return raw

def returnTF():
    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    return tf

def main():
    imglist = getImglist(None)
    modelPath = 'slogs/smodel-best.pth'
    vocabPath = 'pkls/vocab_clean.pkl'
    model = loadModel(modelPath)
    vocab = loadVocab(vocabPath)
    tf = returnTF()

    for imgname in imglist:
        try:
            img = Image.open(imgname).convert('RGB')
        except IOError:
            print('failed to load image %s'%imgname)
            continue

        input_img = V(tf(img).unsqueeze(0)).cuda()
        # logit = model.forward(input_img)
        logit = model(input_img)
        # pred = nn.functional.softmax(logit).cpu().data.numpy()
        pred = nn.functional.sigmoid(logit)
        
        captions = getLabel(imgname)
        
        words = []
        # print('logit:', logit.shape, logit[0], logit[0].shape)
        # print('prod:', pred.shape, pred[0], pred[0].shape)
        
        topn = 10 
        topvalue = torch.topk(pred, topn)
        # print('topn', topvalue)
        # print('topvalue:', topvalue2)

        for ind, val in enumerate(topvalue[1][0]):
            if val > 0:
                words.append(vocab[val])
        print('imgname:', imgname)
        print('words:', words)
        print('captions:')
        for cap in captions:
            print(cap)
        print()

if __name__ == '__main__':
    main()
