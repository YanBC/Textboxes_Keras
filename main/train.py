import os
import sys
sys.path.append(os.getcwd())

import yaml
import argparse
import cv2 as cv
import keras
from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TerminateOnNaN, CSVLogger, TensorBoard
from keras import backend as K
from keras.models import load_model
from keras.utils import multi_gpu_model

from models.loss_function import TextBoxes_Loss
from utils.data_master import Data_Master as data_gen
# from models.textboxes import textboxes, pixel
# from models.keras_layer_L2Normalization import L2Normalization

from main_config import get_model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='path to configuration file')
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f.read())

    image_h = config['img_height']
    image_w = config['img_width']
    nfeat = config['nfeat']
    offsets = config['offsets']
    aspect_ratios = config['aspect_ratios']

    neg_pos_ratio = config['loss_neg_pos_ratio']
    alpha = config['loss_alpha']

    weightsDir = config['weights_dir']
    csvPath = config['csv_path']
    tbDir = config['tbDir']
    dataDir = config['data_dir']
    trainset = os.path.join(dataDir, 'train.txt')
    valset = os.path.join(dataDir, 'val.txt')

    batch_size = config['batch_size']
    n_times = config['epochs']
    n_gpus = config['n_gpus']
    phase1 = config['phase1']
    phase2 = config['phase2']
    steps_per_epoch = config['steps_per_epoch']
    initial_epoch = config['initial_epoch']
    best_model = config['best_model']


    def lr_schedule(epoch):
        if epoch < phase1:
            return 0.001
        elif epoch < phase2:
            return 0.0001
        else:
            return 0.00001

    # build model
    sgd = SGD(lr=0.001, momentum=0.9, decay=5e-4, nesterov=False, clipvalue=0.3)
    adam = Adam(lr=0.001, decay=5e-4)
    loss_f = TextBoxes_Loss(neg_pos_ratio=neg_pos_ratio, alpha=alpha)

    if os.path.isfile(best_model):
        model = load_model(best_model, custom_objects={'compute_loss': loss_f.compute_loss})

        # try to determine the number of gpu used
        # and convert it to one-gpu model
        if isinstance(model.layers[1], keras.layers.Lambda):
            try:
                old_n_gpus = model.layers[1].arguments['parts']

                tmpDir = './_____tmp'
                weightPath = os.path.join(tmpDir, 'tmp_weights')
                # assert not os.path.isdir(tmpDir)
                os.mkdir(tmpDir)
                model.save_weights(weightPath)

                self.model = TextBoxes(config)
                parallel_model = multi_gpu_model(self.model, gpus=old_n_gpus)
                parallel_model.load_weights(weightPath, by_name=True)

                shutil.rmtree(tmpDir)
                del parallel_model
            except:
                pass
    else:
        model = get_model(image_h, image_w)

    if n_gpus > 1:
        model = multi_gpu_model(model, gpus=n_gpus)

    # if n_gpus == 1 and os.path.isfile(best_model):
    #     model = load_model(best_model, custom_objects={'L2Normalization': L2Normalization, 'compute_loss': loss_f.compute_loss})
    # else:
    #     model = textboxes(config)
    #     # model.load_weights(pretrain_weights, by_name=True)
    #     if n_gpus > 1:
    #         model = multi_gpu_model(model, gpus=n_gpus)

    model.compile(optimizer=adam, loss=loss_f.compute_loss)


    # get data generator
    train_gen = data_gen(trainset, batch_size, image_h, image_w, nfeat, offsets, aspect_ratios)
    val_gen = data_gen(valset, batch_size, image_h, image_w, nfeat, offsets, aspect_ratios)



    # create callbacks
    model_checkpoint = ModelCheckpoint(filepath=weightsDir+'/epoch-{epoch:02d}_loss-{loss:.4f}_val_loss-{val_loss:.4f}.h5', monitor='val_loss', verbose=1, save_best_only=False, save_weights_only=False, mode='auto', period=1)
    csv_logger = CSVLogger(filename=csvPath,
                           separator=',',
                           append=True)
    learning_rate_scheduler = LearningRateScheduler(schedule=lr_schedule, verbose=1)
    terminate_on_nan = TerminateOnNaN()
    tb_log = TensorBoard(log_dir=tbDir, write_graph=True, write_grads=True)
    callbacks = [model_checkpoint, csv_logger, learning_rate_scheduler, terminate_on_nan, tb_log]


    # start training
    if steps_per_epoch > 0:
        history = model.fit_generator(generator=train_gen,
                                      steps_per_epoch=steps_per_epoch,
                                      epochs=n_times,
                                      callbacks=callbacks,
                                      validation_data=val_gen,
                                      use_multiprocessing=True,
                                      initial_epoch=initial_epoch)
    else:
        history = model.fit_generator(generator=train_gen,
                                      epochs=n_times,
                                      callbacks=callbacks,
                                      validation_data=val_gen,
                                      use_multiprocessing=True,
                                      initial_epoch=initial_epoch)