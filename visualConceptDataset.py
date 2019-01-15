# -*- encoding: utf-8 -*-

import torch
from torch.utils.data import Dataset
import skimage.io
from PIL import Image
import glob
import numpy as np
import os
import opts

class VisualConceptDataset(Dataset):
    def __init__(self, root, labels_dir, transform=None):
        super(Dataset, self).__init__()
        self.root = root # eg: /media/disk0/jay/workspace/visual-concept/imgs/trains
        self.transform = transform
        self.imglist = glob.glob(
                            os.path.join(self.root, "*.jpg"))
        self.labels_dir = labels_dir

    # 第二步装载数据，返回[img,label]，idx就是一张一张地读取
    def __getitem__(self, idx):
        # get item
        imgname = self.imglist[idx]
        image = Image.open(imgname).convert('RGB')
        # print 'img:', imgname, image.size

        # skimage way
        # image = skimage.io.imread(imgname)
        # # handle grayscale input images
        # if len(image.shape) == 2:
        #     image = image[:, :, np.newaxis]
        #     image = np.concatenate((image,image,image), axis=2)
        # image = torch.from_numpy(image.transpose([2, 0, 1]))

        if self.transform is not None:
            image = self.transform(image)

        # label = np.load(imgname.replace('imgs', self.labels_dir) + ".npy")
        label = np.load(
                os.path.join(self.labels_dir, imgname.split('/')[-1])+  ".npy")
        # print 'label:', label.shape, image.size()
        return image, label

    def __len__(self):
        return len(self.imglist)

