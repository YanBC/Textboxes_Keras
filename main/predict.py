import os
import shutil
import cv2 as cv
import numpy as np

from keras.utils import multi_gpu_model
from keras.models import load_model

import sys
sys.path.append(os.getcwd())
from models.loss_function import TextBoxes_Loss
from utils.data_butler import Data_Butler

from main_config import get_model


class Detector():
    def __init__(self, modelPath, img_height, img_width, top_k, iou_thres, conf_thres, nfeat, offsets, aspect_ratios, n_gpus=1, neg_pos_ratio=3, alpha=1.0):

        self.img_height = img_height
        self.img_width = img_width

        self.top_k = top_k
        self.iou_thres = iou_thres
        self.conf_thres = conf_thres

        loss_f = TextBoxes_Loss(neg_pos_ratio=neg_pos_ratio, alpha=alpha)

        # create model
        if n_gpus > 1:
            model = load_model(modelPath, custom_objects={'compute_loss': loss_f.compute_loss})
            tmpDir = './tmp'
            weightPath = os.path.join(tmpDir, 'tmp_weights')
            assert not os.path.isdir(tmpDir)
            os.mkdir(tmpDir)
            model.save_weights(weightPath)

            self.model = get_model(self.img_height, self.img_width)
            parallel_model = multi_gpu_model(self.model, gpus=n_gpus)
            parallel_model.load_weights(weightPath, by_name=True)

            shutil.rmtree(tmpDir)
        else:
            self.model = load_model(modelPath, custom_objects={'compute_loss': loss_f.compute_loss})

        # image preprocessor
        # Alfred is here to help
        self.Alfred = Data_Butler(img_height, img_width, nfeat, offsets, aspect_ratios)


    def predict(self, image):
        image_height, image_width, _ = image.shape

        img = self.Alfred.image_preprocessing(image)
        img = np.expand_dims(img, axis=0)

        output_v = self.model.predict(img)[0]

        boxes = self.Alfred.decode(output_v, conf_thres=self.conf_thres, iou_thres=self.iou_thres, top_k=self.top_k, image_width=image_width, image_height=image_height)

        return boxes




# test
if __name__ == '__main__':
    import yaml

##########################################################################
    # imagePath = './samples/generated_samples/detect_for_textboxes_test/image/0.jpg'
    # annoPath = './samples/generated_samples/detect_for_textboxes_test/image/0.txt'

    imagePath = './samples/real_samples/images/20252.jpg'
    configPath = './configs/densenet_model.yml'
##########################################################################

    with open(configPath) as f:
        config = yaml.safe_load(f.read())

    n_gpus = config['n_gpus']
    modelPath = config['best_model']

    conf_thres = config['conf_thres']
    iou_thres = config['iou_thres']
    top_k = config['top_k']
    img_height = config['img_height']
    img_width = config['img_width']
    nfeat = config['nfeat']
    offsets = config['offsets']
    aspect_ratios = config['aspect_ratios']

    neg_pos_ratio = config['loss_neg_pos_ratio']
    alpha = config['loss_alpha']

    image = cv.imread(imagePath)

    d = Detector(modelPath, img_height, img_width, top_k, iou_thres, conf_thres, nfeat, offsets, aspect_ratios, n_gpus, neg_pos_ratio, alpha)

    boxes = d.predict(image)
    for i, box in enumerate(boxes):
        conf, cx, cy, w, h = box
        left = cx - w / 2
        right = cx + w / 2
        top = cy - h /2
        bottom = cy + h / 2

        cv.rectangle(image, (int(left), int(top)), (int(right), int(bottom)), (0, 255, 0), 1)

    cv.namedWindow('show', cv.WINDOW_NORMAL)
    cv.imshow('show', image)
    cv.waitKey()