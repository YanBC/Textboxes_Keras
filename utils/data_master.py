import os
import sys
sys.path.append(os.getcwd())

from utils.data_butler import Data_Butler

import yaml
import numpy as np
import cv2 as cv
from keras.utils import Sequence

class Data_Master(Sequence):
    """
    my data generator class for keypoint detection; 
    see keras.utils.Sequence definition (https://github.com/keras-team/keras/blob/master/keras/utils/data_utils.py#L305) for more detials 
    """
    def __init__(self, dataset, batch_size, img_h, img_w, nfeat, offsets, aspect_ratios, shuffle=True):

        self.Alfred = Data_Butler(img_h, img_w, nfeat, offsets, aspect_ratios)
        self.output_shape = self.Alfred.output_shape()

        self.batch_size = batch_size
        self.image_height = img_h
        self.image_width = img_w
        
        self.shuffle = shuffle
        self.annoRecord = dict()
        with open(dataset) as f:
            for line in f.readlines():
                imageFile, annoFile = line.strip().split(',')
                self.annoRecord[imageFile] = annoFile

        self.images = sorted(list(self.annoRecord.keys()))
        self.dataIds = [x for x in range(len(self.images))]

        # self.on_epoch_end()

    def on_epoch_end(self):
        # update indexes after each epoch
        if self.shuffle == True:
            np.random.shuffle(self.dataIds)

    def __len__(self):
        # gives the total number of batches
        return int(np.floor(len(self.dataIds) / self.batch_size))


    def _data_gen(self, indices):
        input_v = np.empty((self.batch_size, self.image_height, self.image_width, 3))
        output_v = np.empty((self.batch_size, *(self.output_shape)))

        for i, index in enumerate(indices):
            imagePath = self.images[index]
            annoPath = self.annoRecord[imagePath]

            image = cv.imread(imagePath)
            annos = []
            with open(annoPath) as f:
                for line in f.readlines():
                    line = line.strip().split(',')
                    left = int(line[0])
                    top = int(line[1])
                    right = int(line[2])
                    bottom = int(line[5])

                    cx = float(right + left) / 2
                    cy = float(bottom + top) / 2
                    width = float(right - left + 1)
                    height = float(bottom - top + 1)
                    annos.append([cx, cy, width, height])            

            input_v[i,:], output_v[i,:] = self.Alfred.encode(image, annos)
        return input_v, output_v


    def __getitem__(self, index):
        # generate one batch of data
        index_dataIds = self.dataIds[index*self.batch_size:(index+1)*self.batch_size]

        x, y_true = self._data_gen(index_dataIds)

        return x, y_true


if __name__ == '__main__':
    import yaml

    with open('./configs/densenet_model.yml') as f:
        config = yaml.safe_load(f.read())

    # dataset = './train.txt'
    dataset = os.path.join(config['data_dir'], 'train.txt')
    p = Data_Master(dataset, 128, config['img_height'], config['img_width'], config['nfeat'], config['offsets'], config['aspect_ratios'])

    x, y_true = p[0]

