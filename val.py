# -*- encoding: utf-8 -*-

from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms

from visualConceptDataset import VisualConceptDataset
from myNets import myResnet

import time
import os
import copy
from tqdm import tqdm

from PIL import Image
import pickle as pkl

def getImglist(path):
    pass

def loadModel(modelPath):
    model = models.resnet101()
    fc_features = model.fc.in_features
    model.fc = nn.Linear(fc_features, 9360)
    model = model.cuda()
    model.eval()

    # load best model weights
    model.load_state_dict(modelPath)
    return model

def loadVocab(vocabPath):
    with open(vocabPath, 'r') as f:
        vocab = pkl.load(f)
    return vocab

def main():
    imglist = getImglist()
    modelPath = 'log/model-best.pth'
    vocabPath = 'pkls/vocab_clean.pkl'
    model = loadModel(modelPath)
    vocab = loadVocab(vocabPath)

    for imgname in tqdm(imglist):
        try:
            img = Image.open(imgname).convert('RGB')
        except IOError:
            print('failed to load image %s'%imgname)
            continue

        input_img = V(tf(img).unsqueeze(0)).cuda()
        logit = model.forward(input_img)
        pred = torch.ceil(output)
        
        words = []
        for ind, val in enumerate(pred):
            if val == 1:
                words.append(vocab.index(ind))
        print words

if __name__ == '__main__':
    main()

if __name__ == '__main__':
    main()
