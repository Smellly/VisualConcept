# -*- encoding: utf-8 -*-

from torch.utils.data import Dataset
from skimage import io
import glob
import numpy as np
import os

class VisualConceptDataset(Dataset):
    def __init__(self, root, transform=None):
        super(Dataset, self).__init__()
        self.root = root # eg: /media/disk0/jay/workspace/visual-concept/imgs/trains
        self.transform = transform
        self.imglist = glob.glob(
                            os.path.join(self.root, "*.jpg"))
        print root
        print len(self.imglist), self.imglist[0]

    # 第二步装载数据，返回[img,label]，idx就是一张一张地读取
    def __getitem__(self, idx):
        # get item
        imgname = self.trainlist[idx]
        image = self.transform(io.imread(imgname))

        label = np.load(imgname.replace('imgs', 'labels') + ".npy")
        return image, label

    def __len__(self):
        return len(self.imglist)



