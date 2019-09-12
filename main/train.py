import os
import sys
sys.path.append(os.getcwd())

import yaml
import argparse
import cv2 as cv
from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TerminateOnNaN, CSVLogger
from keras import backend as K
from keras.models import load_model

from models.textboxes import network as TextBoxes
from models.loss_function import TextBoxes_Loss
from utils.data_master import Data_Master as data_gen





def lr_schedule(epoch):
    if epoch < 80:
        return 0.0001
    elif epoch < 100:
        return 0.0001
    else:
        return 0.00001


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='path to configuration file')
    parser.add_argument('--trainset', default='./train.txt', help='path to train set data file')
    parser.add_argument('--valset', default='./val.txt', help='path to validation set data file')
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f.read())

    pretrain_weights = config['backbone']
    neg_pos_ratio = config['loss_neg_pos_ratio']
    alpha = config['loss_alpha']
    weightsDir = config['weights_dir']
    csvPath = config['csv_path']
    n_times = config['epochs']


    # build model
    model = TextBoxes(config)
    model.load_weights(pretrain_weights, by_name=True)

    sgd = SGD(lr=0.001, momentum=0.9, decay=0.0, nesterov=False)
    loss_f = TextBoxes_Loss(neg_pos_ratio=neg_pos_ratio, alpha=alpha, n_neg_min=10)
    model.compile(optimizer=sgd, loss=loss_f.compute_loss)


    # get data generator
    train_gen = data_gen(config, args.trainset)
    val_gen = data_gen(config, args.valset)



    # create callbacks
    model_checkpoint = ModelCheckpoint(filepath=weightsDir+'/ssd300_pascal_07+12_epoch-{epoch:02d}_loss-{loss:.4f}_val_loss-{val_loss:.4f}.h5',
                                       monitor='val_loss',
                                       verbose=1,
                                       save_best_only=True,
                                       save_weights_only=False,
                                       mode='auto',
                                       period=1)
    csv_logger = CSVLogger(filename=csvPath,
                           separator=',',
                           append=True)
    learning_rate_scheduler = LearningRateScheduler(schedule=lr_schedule, verbose=1)
    terminate_on_nan = TerminateOnNaN()
    callbacks = [model_checkpoint,
                 csv_logger,
                 learning_rate_scheduler,
                 terminate_on_nan]


    # start training
    history = model.fit_generator(generator=train_gen,
                                  epochs=n_times,
                                  callbacks=callbacks,
                                  validation_data=val_gen,
                                  use_multiprocessing=True)