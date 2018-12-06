# -*- encoding: utf-8 -*-

import torch.nn as nn

class myResnet(nn.Module):

    def __init__(self , model, classes=1000):
        super(myResnet, self).__init__()
        #取掉model的后两层
        self.resnet_layer = nn.Sequential(*list(model.children())[:-2])
        
        self.transion_layer = nn.ConvTranspose2d(2048, 2048, kernel_size=14, stride=3)
        self.pool_layer = nn.MaxPool2d(32)  
        self.Linear_layer = nn.Linear(2048, classes)
        
    def forward(self, x):
        x = self.resnet_layer(x)
        x = self.transion_layer(x)
        x = self.pool_layer(x)
        x = x.view(x.size(0), -1) 
        x = self.Linear_layer(x)
        
        return x
