import os
import sys
sys.path.append(os.getcwd())

import yaml
import argparse
import cv2 as cv
import numpy as np
from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TerminateOnNaN, CSVLogger, TensorBoard
from keras import backend as K
from keras.models import load_model
from keras.utils import multi_gpu_model

from models.textboxes import textboxes, pixel
from models.loss_function import TextBoxes_Loss
from utils.data_master import Data_Master as data_gen
from utils.data_master import Data_Butler
from models.keras_layer_L2Normalization import L2Normalization




def get_weight_grad(model, inputs, outputs):
    """ Gets gradient of model for given inputs and outputs for all weights"""
    grads = model.optimizer.get_gradients(model.total_loss, model.trainable_weights)
    symb_inputs = (model._feed_inputs + model._feed_targets + model._feed_sample_weights)
    f = K.function(symb_inputs, grads)
    x, y, sample_weight = model._standardize_user_data(inputs, outputs)
    output_grad = f(x + y + sample_weight)
    return output_grad


def get_layer_output_grad(model, inputs, outputs, layer=-1):
    """ Gets gradient a layer output for given inputs and outputs"""
    grads = model.optimizer.get_gradients(model.total_loss, model.layers[layer].output)
    symb_inputs = (model._feed_inputs + model._feed_targets + model._feed_sample_weights)
    f = K.function(symb_inputs, grads)
    x, y, sample_weight = model._standardize_user_data(inputs, outputs)
    output_grad = f(x + y + sample_weight)
    return output_grad


def get_layer_output(model, input_v, layer=-1):
    input_t = model.inputs
    output_t = [model.layers[layer].output]
    f = K.function(input_t, output_t)
    ret = f(input_v)

    return ret



if __name__ == '__main__':
    import yaml

    configPath = './configs/pixel_test.yml'
    with open(configPath) as f:
        config = yaml.safe_load(f.read())


    pretrain_weights = config['backbone']
    neg_pos_ratio = config['loss_neg_pos_ratio']
    alpha = config['loss_alpha']
    weightsDir = config['weights_dir']
    csvPath = config['csv_path']
    n_times = config['epochs']
    tbDir = config['tbDir']
    n_gpus = config['n_gpus']
    dataDir = config['data_dir']
    trainset = os.path.join(dataDir, 'train.txt')
    valset = os.path.join(dataDir, 'val.txt')
    phase1 = config['phase1']
    phase2 = config['phase2']
    steps_per_epoch = config['steps_per_epoch']
    best_model = config['best_model']

    # build model
    sgd = SGD(lr=0.01, momentum=0.9, decay=5e-4, nesterov=False, clipvalue=0.3)
    adam = Adam(lr=0.01, decay=5e-4)
    loss_f = TextBoxes_Loss(neg_pos_ratio=neg_pos_ratio, alpha=alpha)

    # if n_gpus == 1 and os.path.isfile(best_model):
    #     model = load_model(best_model, custom_objects={'L2Normalization': L2Normalization, 'compute_loss': loss_f.compute_loss})
    # else:
    #     model = TextBoxes(config)
    #     model.load_weights(pretrain_weights, by_name=True)
    #     if n_gpus > 1:
    #         model = multi_gpu_model(model, gpus=n_gpus)

    # model = load_model(best_model, custom_objects={'L2Normalization': L2Normalization, 'compute_loss': loss_f.compute_loss})
    

    # model = pixel(config)
    model = load_model('./logs/pixel_test/weights/epoch-11_loss-2.3927_val_loss-1.1830.h5', custom_objects={'L2Normalization': L2Normalization, 'compute_loss': loss_f.compute_loss})

    model.compile(optimizer=adam, loss=loss_f.compute_loss)


    # get sample input
    p = Data_Butler(config['img_height'], \
                    config['img_width'], \
                    config['layers'], \
                    config['nfeat'], \
                    config['offsets'], \
                    config['aspect_ratios'], \
                    config['subtract_mean'])

    imagePath = './samples/generated_samples/image/2.jpg'
    annoPath = './samples/generated_samples/label/2.txt'
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
            width = float(right - left + 1) / 2
            height = float(bottom - top + 1) / 2
            annos.append([cx, cy, width, height])

    x, y_true = p.encode(image, annos)    
    input_v = np.expand_dims(x, axis=0)
    gt_v = np.expand_dims(y_true, axis=0)

    sess = K.get_session()

    # # check gradient
    # grads_t = K.gradients(model.total_loss, model.trainable_weights)
    # f = K.function(model.inputs, grads_t)

    # grads_v = f([input_v])

    weight_grads = get_weight_grad(model, input_v, gt_v)
    output_grad = get_layer_output_grad(model, input_v, gt_v)
    
    output_v = get_layer_output(model, [input_v], -1)