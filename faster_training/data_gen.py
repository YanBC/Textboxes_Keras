import os
import sys
sys.path.append(os.getcwd())

from utils.data_butler import Data_Butler

import yaml
import numpy as np
import cv2 as cv
from keras.utils import Sequence

import pickle

class Datagenerator(Sequence):
    """
    my data generator class for keypoint detection; 
    see keras.utils.Sequence definition (https://github.com/keras-team/keras/blob/master/keras/utils/data_utils.py#L305) for more detials 
    """
    def __init__(self, dataset, batch_size, input_shape=(512,512,3), output_shape=(7680,6), shuffle=True):

        self.input_shape = input_shape
        self.output_shape = output_shape
        self.batch_size = batch_size
        
        self.shuffle = shuffle
        self.annoRecord = []
        with open(dataset) as f:
            for line in f.readlines():
                annoFile = line.strip()
                self.annoRecord.append(annoFile)

        self.annoRecord = sorted(self.annoRecord)
        self.dataIds = [x for x in range(len(self.annoRecord))]

        self.on_epoch_end()

    def on_epoch_end(self):
        if self.shuffle == True:
            np.random.shuffle(self.dataIds)

    def __len__(self):
        return int(np.floor(len(self.dataIds) / self.batch_size))


    def __getitem__(self, index):
        index_dataIds = self.dataIds[index*self.batch_size:(index+1)*self.batch_size]
        input_v = np.empty((self.batch_size, *(self.input_shape)))
        output_v = np.empty((self.batch_size, *(self.output_shape)))

        for i, index in enumerate(index_dataIds):

            annoPath = self.annoRecord[index]
            with open(annoPath, 'rb') as f:
                x, y_true = pickle.load(f)

            input_v[i,:], output_v[i,:] = x, y_true

        return input_v, output_v



if __name__ == '__main__':
    with open('./configs/densenet_model.yml') as f:
        config = yaml.safe_load(f.read())

    dataset = os.path.join(config['data_dir'], 'train.txt')

    p = Data_Master(dataset, 2)

    x, y_true = p[0]

