import os
import shutil
import cv2 as cv
import numpy as np

from keras.utils import multi_gpu_model
from keras.models import load_model

import sys
sys.path.append(os.getcwd())
from models.keras_layer_L2Normalization import L2Normalization
from models.loss_function import TextBoxes_Loss
from models.textboxes import network as TextBoxes
from utils.data_butler import Data_Butler


class Detector():
    def __init__(self, config):
        n_gpus = config['n_gpus']
        modelPath = config['best_model']

        self.conf_thres = config['conf_thres']
        self.iou_thres = config['iou_thres']
        self.top_k = config['top_k']
        self.img_height = config['img_height']
        self.img_width = config['img_width']

        neg_pos_ratio = config['loss_neg_pos_ratio']
        alpha = config['loss_alpha']
        loss_f = TextBoxes_Loss(neg_pos_ratio=neg_pos_ratio, alpha=alpha, n_neg_min=10)

        # create model
        if n_gpus > 1:
            model = load_model(modelPath, custom_objects={'L2Normalization': L2Normalization, 'compute_loss': loss_f.compute_loss})
            tmpDir = './tmp'
            weightPath = os.path.join(tmpDir, 'tmp_weights')
            assert not os.path.isdir(tmpDir)
            os.mkdir(tmpDir)
            model.save_weights(weightPath)

            self.model = TextBoxes(config)
            parallel_model = multi_gpu_model(self.model, gpus=n_gpus)
            parallel_model.load_weights(weightPath, by_name=True)

            shutil.rmtree(tmpDir)
        else:
            self.model = load_model(modelPath)

        # image preprocessor
        # Alfred is here to help
        self.Alfred = Data_Butler(config['img_height'], \
                                  config['img_width'], \
                                  config['layers'], \
                                  config['nfeat'], \
                                  config['offsets'], \
                                  config['aspect_ratios'], \
                                  config['subtract_mean'])


    def predict(self, image):
        img = self.Alfred.image_preprocessing(image)
        img = np.expand_dims(img, axis=0)

        output_v = self.model.predict(img)

        boxes = self.Alfred.decode(output_v, conf_thres=self.conf_thres, iou_thres=self.iou_thres, top_k=self.top_k, image_width=self.img_width, image_height=self.img_height)

        return boxes




# test
if __name__ == '__main__':
    import yaml

    configPath = './configs/textboxes_myDataSet_include_text.yml'
    with open(configPath) as f:
        config = yaml.safe_load(f.read())

    imagePath = './samples/generated_samples/image/1.jpg'
    annoPath = './samples/generated_samples/label/1.txt'

    image = cv.imread(imagePath)
    d = Detector(config)

    tmp = d.predict(image)