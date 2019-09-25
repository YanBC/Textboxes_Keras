import tensorflow as tf
from tensorflow.python.framework import graph_util
from keras import backend as K

import yaml
import argparse
import time

import os
import sys
sys.path.append(os.getcwd())
# Using densenet_model by default
from models.densenet_model import densenet_model as get_model


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('config', help='path to configuration file')
    p.add_argument('src', help='path to source keras weights file')
    p.add_argument('des', help='where to save the fronzen pb file')
    args = p.parse_args()

    configPath = args.config
    srcFile = args.src
    desDir = args.des

    if not os.path.isdir(desDir):
        os.mkdir(desDir)

    with open(configPath) as f:
        config = yaml.safe_load(f.read())

    img_height = config['img_height']
    img_width = config['img_width']


    # create model
    K.set_learning_phase(0)       # very important


    model = get_model(img_height, img_width)
    model.load_weights(srcFile, by_name=True)

    input_names = ['input_1']
    output_names = ['concatenate_43/concat']

    sess = K.get_session()
    frozen_graph = graph_util.convert_variables_to_constants(sess, sess.graph.as_graph_def(), output_names)
    pruned_graph = tf.graph_util.remove_training_nodes(frozen_graph)


    # save files
    saveDirName = str(int(time.time())) + '_frozen_tf'
    saveDirPath = os.path.join(desDir, saveDirName)
    os.mkdir(saveDirPath)
    tblogPath = os.path.join(saveDirPath, 'logs')

    save_g = tf.Graph()
    with save_g.as_default():
        tf.import_graph_def(pruned_graph, name='')
        tf.summary.FileWriter(tblogPath, graph=save_g)

    tf.train.write_graph(pruned_graph, saveDirPath, 'model.pbtxt', as_text=True)
    tf.train.write_graph(pruned_graph, saveDirPath, 'model.pb', as_text=False)

    print('Files saved to {}'.format(saveDirPath))
