import yaml
import argparse
import time
import os

import sys
sys.path.append(os.getcwd())
from main.predict import Detector

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('config', help='path to config yaml file')
    p.add_argument('des', help='where to save the model weight file')
    args = p.parse_args()

    if not os.path.isdir(args.des):
        os.mkdir(args.des)

    with open(args.config) as f:
        config = yaml.safe_load(f)

    # get keras model through Detector
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

    d = Detector(modelPath, img_height, img_width, top_k, iou_thres, conf_thres, nfeat, offsets, aspect_ratios, n_gpus, neg_pos_ratio, alpha)

    # save model weights
    weightName = str(int(time.time())) + '_weights.h5'
    weightPath = os.path.join(args.des, weightName)
    d.model.save_weights(weightPath, overwrite=False)

    print('Model weight files is saved to {}'.format(weightPath))