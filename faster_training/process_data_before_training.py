import os
import sys
sys.path.append(os.getcwd())
from utils.data_butler import Data_Butler

import argparse
import cv2 as cv
import multiprocessing
import pickle


####### Config ##########
img_h = 512
img_w = 512
nfeat = [32, 32, 16, 16]
offsets = [0, 0, 0, 0.5, 0.5, 0.5]
aspect_ratios = [3.0, 5.0, 7.0, 3.0, 5.0, 7.0]
dirName = 'faster_training'

# srcDir = '../make_dataset/generated/detect_for_textboxes_60000/'
srcDir = './samples/real_samples/images/'
####### Config ##########

imageDir = os.path.join(srcDir, 'image')
annoDir = os.path.join(srcDir, 'label')
desDir = os.path.join(srcDir, dirName)
if not os.path.isdir(desDir):
    os.mkdir(desDir)


def create_one(index):
    imagePath = os.path.join(imageDir, '{}.jpg'.format(index))
    annoPath = os.path.join(annoDir, '{}.txt'.format(index))
    desPath = os.path.join(desDir, '{}.pkl'.format(index))

    # continue after stopped
    # if file already exists, return True
    if os.path.isfile(desPath):
        return True

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

    processor = Data_Butler(img_h, img_w, nfeat, offsets, aspect_ratios)
    x, y_true = processor.encode(image, annos)

    with open(desPath, 'bw') as g:
        pickle.dump((x, y_true), g)

    return True


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--workers', type=int, default=20, help='number of processes to use')
    args = p.parse_args()

    num_images = len(os.listdir(imageDir))
    indices = [x for x in range(num_images)]

    with multiprocessing.Pool(processes=args.workers) as p:
        ret = p.map(create_one, indices)
    print('Sucessfully generated %d samples' % sum(ret))




