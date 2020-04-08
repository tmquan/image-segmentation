import tensorpack.dataflow 
from tensorpack.dataflow import RNGDataFlow
from tensorpack.utils.argtools import shape2d
from tensorpack.utils.utils import get_rng
import os
import cv2
import numpy as np

import sklearn.metrics
import glob2
import skimage.io
# import tensorflow as tf
from natsort import natsorted

class CustomDataSet(RNGDataFlow):
    """ Produce images read from a list of files as (h, w, c) arrays. """
    def __init__(self, folder, size=None, train_or_valid='train', channel=1, resize=None, debug=False, shuffle=False, hparams=None):
        super(CustomDataSet, self).__init__()
        self.folder = folder
        self.is_train = True if train_or_valid=='train' else False
        self.channel = int(channel)
        assert self.channel in [1, 3], self.channel
        if resize is not None:
            resize = shape2d(resize)
        self.resize = resize
        self.shuffle = shuffle
        self.hparams = hparams
        
        self.images = []
        self.labels = []

        if self.is_train:
            self.imageDir = os.path.join(folder, 'train', 'images')
            self.labelDir = os.path.join(folder, 'train', 'labels')
        else:
            self.imageDir = os.path.join(folder, 'test', 'images')
            self.labelDir = os.path.join(folder, 'test', 'labels')

        self.imageFiles = natsorted (glob2.glob(self.imageDir + '/*.*'))
        self.labelFiles = natsorted (glob2.glob(self.labelDir + '/*.*'))
  
        self._size = min(size, len(self.imageFiles))
        print(self._size)
  
    def reset_state(self):
        self.rng = get_rng(self)   

    def __len__(self):
        return self._size

    def __iter__(self):
        # TODO
        indices = list(range(self.__len__()))
        if self.is_train:
            self.rng.shuffle(indices)

        for idx in indices:

            # image = self.images[idx].copy()
            # label = self.labels[idx].copy()
            image = skimage.io.imread (self.imageFiles[idx])
            label = skimage.io.imread (self.labelFiles[idx])

            label[label < 128] = 0
            label[label >= 128] = 255
            # print(image.shape, label.shape)
            if self.hparams.types==1:
                yield [image, label]
            elif self.hparams.types==6:
                yield [image, label[0,...], label[1,...], label[2,...], label[3,...], label[4,...], label[5,...]]
