# -*- coding: utf-8 -*-

import os
import numpy as np
import cv2 as cv

import sys
sys.path.append('.')
from main.predict import Detector

imageExts = ['.jpg']
annoExts = ['.txt']


def centroid2minmax(box):
    '''
    box: [[conf, centre_x, centre_y, width, height], ...], dtype=float

    outBox: [[conf, left, top, right, bottom]], dtype=float
    '''
    inBox = np.array(box)

    outBox = np.empty(shape=inBox.shape)
    outBox[:, 0] = inBox[:, 0]
    outBox[:, 1] = inBox[:, 1] - inBox[:, 3] / 2
    outBox[:, 2] = inBox[:, 2] - inBox[:, 4] / 2
    outBox[:, 3] = inBox[:, 1] + inBox[:, 3] / 2
    outBox[:, 4] = inBox[:, 2] + inBox[:, 4] / 2

    return outBox


def _IoU(boxA, boxB):
    '''
    Calculate the intersection over union ratio between two boxes
    '''

    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou


def getPredictions(model, dirPath):
    '''
    make predictions on all images in <dirPath>

    INPUT:
        model <Detector>: the to-be-evaluated model 

        dirPath <string>: path to image and annotation directory


    OUTPUT:
        ret <dictionary>: Key <string>: 
                            image file name
                          
                          Value <list>: 
                            anno <np.array>:
                                [[confidence <float>, left <float>, top <float>, right <float>, bottom <float>], ...]


    NOTES:
        A typical yolo annotation direcotry should be like:
        .
        ├── image
        │   ├── 0.jpg
        │   ├── 1.jpg
        │   ├── 10.jpg
        │   ├── 11.jpg
        │   ├── 12.jpg
        │   ├── 13.jpg
        │   ├── 14.jpg
        │   ├── 15.jpg
        │   └── 9.jpg
        └── label
            ├── 0.txt
            ├── 1.txt
            ├── 10.txt
            ├── 11.txt
            ├── 12.txt
            ├── 13.txt
            ├── 14.txt
            ├── 15.txt
            └── 9.txt 
    '''
    imageDir = os.path.join(dirPath, 'image')

    files = os.listdir(imageDir)
    images = [f for f in files if (imageExts[0] in f)]

    ret = dict()
    for image in images:
        # read image
        imagePath = os.path.join(imageDir, image)
        img = cv.imread(imagePath)

        boxes = model.predict(img)

        if len(boxes) > 0:
            ret[image] = centroid2minmax(boxes)
        else:
            ret[image] = None

    return ret


def getAnnotations(dirPath):
    '''
    Read yolo annotations

    INPUT:
        dirPath <string>: path to image and annotation directory


    OUTPUT:
        ret <dictionary>: Key <string>: 
                            image file name
                          
                          Value <list>: 
                            anno <np.array>:
                                [[left <float>, top <float>, right <float>, bottom <float>], ...]


    NOTES:
        A typical yolo annotation direcotry should be like:
        .
        ├── image
        │   ├── 0.jpg
        │   ├── 1.jpg
        │   ├── 10.jpg
        │   ├── 11.jpg
        │   ├── 12.jpg
        │   ├── 13.jpg
        │   ├── 14.jpg
        │   ├── 15.jpg
        │   └── 9.jpg
        └── label
            ├── 0.txt
            ├── 1.txt
            ├── 10.txt
            ├── 11.txt
            ├── 12.txt
            ├── 13.txt
            ├── 14.txt
            ├── 15.txt
            └── 9.txt 
    '''
    imageDir = os.path.join(dirPath, 'image')
    annoDir = os.path.join(dirPath, 'label')
    images = [f for f in os.listdir(imageDir) if (imageExts[0] in f)]
    annoFiles = [f for f in os.listdir(annoDir) if (annoExts[0] in f)]

    ret = dict()
    for annoFile in annoFiles:

        # get image informations
        imageName = annoFile.split('.')[0] + '.jpg'
        imagePath = os.path.join(imageDir, imageName)
        image = cv.imread(imagePath)
        if image is None:
            print('Error in getAnnotations(%s): %s not exist' % (dirPath, imagePath))
            continue
        height, width, _ = image.shape

        # read annotation
        box_list = []
        with open(os.path.join(annoDir, annoFile), 'r') as tmp:
            line = tmp.readline().strip()
            while (line != ''):
                left, top, right, _, _, bottom, _, _ = line.split(', ')
                box_list.append(np.array([float(left), float(top), float(right), float(bottom)]))
                line = tmp.readline().strip()
        if len(box_list) > 0:
            ret[imageName] = np.stack(box_list)
        else:
            ret[imageName] = None

    return ret


def _maxIoUSuppression(iou_mat, iouThres=0.5):
    '''
    Assign each predicted box to ground true boxes based on IoU

    INPUT:
        iou_mat <2d array>: IoU matrix. Entry_i,j is the IoU ratio 
                            of the i_th ground true box and the 
                            j_th predicted box

        iouThres <float>: IoU threshold. Any predicted box that has
                            an IoU lower that this threshold will 
                            be considered false positive

    OUTPUT:
        gt_indices <1d array>: a 1d array of length iou_mat.shape[1],
                                gt_indices[j] is the index of ground
                                true box assigned to predicted box j;
                                gt_indices[j] = -1 if no ground true
                                box is assigned
    '''
    gt_indices = np.argmax(iou_mat, axis=0)

    for j in range(len(gt_indices)):
        gt_index = gt_indices[j]

        # if iou(gt, pd) is less than iouThres,
        # the corresponding pd is a false positive
        if iou_mat[gt_index, j] < iouThres:
            gt_indices[j] = -1
            continue

        # check if current prediction is matched with a 
        # ground-true box that has already been assigned
        if gt_index in set(gt_indices[:j]):
            prev_j = np.where(gt_indices[:j]==gt_index)[0][0]
            if iou_mat[gt_index, prev_j] > iou_mat[gt_index, j]:
                gt_indices[j] = -1
            else:
                gt_indices[prev_j] = -1

    return gt_indices


def getStats(y_pred, y_true, iouThres=0.5):
    '''
    Perform statistical analysis on bounding box predictions

    INPUT:
        y_pred <dictionary>: predictions; output from getPredictions()

        y_true <dictionary>: ground-true; output from getAnnotations()

        iouThres <float>: IoU threshold for positive predictions

    OUTPUT:
        tp_size <int>: number of true positives

        fp_size <int>: number of false positves

        fn_size <int>: number of false negatives

        ret_tp <list>: true positives information. each element is a true 
                        positive's confidence and its IoU with groud true:
                            [confidence, IoU]
    '''
    fn_size = 0
    fp_size = 0
    tp_size = 0
    ret_tp = []

    # make sure images in y_pred are the same with images in y_true
    assert y_pred.keys() == y_true.keys()
    images = sorted(y_pred.keys())

    # loop over each image
    for image in images:
        gt_list = y_true[image]
        pd_list = y_pred[image]

        n_gt = 0 if gt_list is None else len(gt_list)
        n_pd = 0 if pd_list is None else len(pd_list)

        # if there is no predicted boxes, all ground true
        # boxes would go to the false negative category
        if n_pd == 0:
            fn_size += n_gt
            continue

        # if there is no ground true boxes, all predicted 
        # boxes would be false negatives
        if n_gt == 0:
            fp_size += n_pd
            continue

        # get iou matrix
        iou = np.empty((n_gt, n_pd))
        for i in range(n_gt):
            for j in range(n_pd):
                iou[i,j] = _IoU(gt_list[i, :], pd_list[j, 1::])

        # decide whether a predicted box is true positive 
        # or false positive
        gt_assigned = _maxIoUSuppression(iou, iouThres)

        # record all true positives to ret_tp
        preds = np.where(gt_assigned != -1)[0]
        for pred in preds:
            tmp_confidence = y_pred[image][pred, 0]
            tmp_iou = iou[gt_assigned[pred], pred]

            ret_tp.append([tmp_confidence, tmp_iou])

        # record false positives and false negatives
        fp_size += n_pd - len(preds)
        fn_size += n_gt - len(preds)

    # yes, I am that obsessive-compulsive
    tp_size = len(ret_tp)
    fp_size = int(fp_size)
    fn_size = int(fn_size)
    return tp_size, fp_size, fn_size, ret_tp




if __name__ == '__main__':
    import argparse
    import yaml
    import pickle
    import shutil
    import sys
    import hashlib

    parser = argparse.ArgumentParser(description='Detection Evaluation')
    parser.add_argument('dirPath', help='Path to annotation directory')
    parser.add_argument('--configs', default='./detector.yml', help='Path to configuration file (yaml). Default: "./detector.yml"')
    parser.add_argument('--iouThres', type=float, default=0.5, help='IoU Threshold. Default: 0.5')
    args = parser.parse_args()

    # imageDir = os.path.join(args.dirPath, 'image')
    # annoDir = os.path.join(args.dirPath, 'label')
    imageDir = args.dirPath

    # load configurations
    configPath = args.configs
    with open(configPath, 'r') as f:
        config = yaml.safe_load(f)

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


    # try to load annos and preds from cach
    if sys.argv[0].rfind('/') == -1:
        codeLocale = './'
    else:
        codeLocale = sys.argv[0][:sys.argv[0].rfind('/')]
    cachDir = os.path.join(codeLocale, 'cach_eval')
    cachFile = 'cach.pkl'
    old_imageDir = None

    if os.path.isfile(os.path.join(cachDir, 'cach.pkl')):
        with open(os.path.join(cachDir, 'cach.pkl'), 'br') as f:
            old_configHash = pickle.load(f)
            old_imageDir = pickle.load(f)
            old_annos = pickle.load(f)
            old_preds = pickle.load(f)

    # get current config file's sha256 digest
    with open(configPath, 'rb') as f:
        sh = hashlib.sha256()
        sh.update(f.read())
        configHash = sh.hexdigest()

    # if image directory and config file are the same, there's no need
    # to calculate annos and preds; otherwise, do it all over again
    if old_imageDir == imageDir and old_configHash == configHash:
        annos = old_annos
        preds = old_preds
    else:
        # get annotations
        print('# READING ANNOTATIONS ...')
        annos = getAnnotations(imageDir)
        print('# DONE!\n')

        # get predictions
        print('# MAKING PREDICTIONS ...')
        d = Detector(modelPath, img_height, img_width, top_k, iou_thres, conf_thres, nfeat, offsets, aspect_ratios, n_gpus, neg_pos_ratio, alpha)
        preds = getPredictions(d, imageDir)
        print('# DONE!\n')

        # write to cach
        if os.path.isdir(cachDir):
            shutil.rmtree(cachDir)
        os.mkdir(cachDir)
        with open(os.path.join(cachDir, cachFile), 'bw') as f:
            pickle.dump(configHash, f)
            pickle.dump(imageDir, f)
            pickle.dump(annos, f)
            pickle.dump(preds, f)


    # performance analysis
    print('# ANALYSIS RESULTS')
    print('Using IoU threshold of %0.2f\n' % (args.iouThres))

    tps, fps, fns, tp_info = getStats(preds, annos, args.iouThres)
    tmp = np.array(tp_info)

    print('Recall: %0.3f' % (tps/(tps+fns)))
    print('Precision: %0.3f' % (tps/(tps+fps)))

    if len(tmp) > 0:
        tmp_avgs = tmp.mean(axis=0)
        print('True Positive:')
        print('    Average Confidence: %0.3f' % (tmp_avgs[0]))
        print('    Average IoU: %0.3f\n' % (tmp_avgs[1]))
    else:
        pass

