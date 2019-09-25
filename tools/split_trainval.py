import argparse
import os
import numpy as np

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('src', help='path to source data directory')
    p.add_argument('des', help='path to destination directory')
    p.add_argument('--ratio', help='a float equal to val/total. Default: 0.1')
    args = p.parse_args()

    imageDir = os.path.join(args.src, 'image')
    annoDir = os.path.join(args.src, 'label')

    imageNames = os.listdir(imageDir)
    annoNames = os.listdir(annoDir)

    assert len(imageNames) == len(annoNames)

    if args.ratio:
        ratio = args.ratio
    else:
        ratio = 0.1

    val_str = ''
    train_str = ''

    num_data = len(imageNames)
    for i in range(num_data):
        if np.random.rand() < ratio:
            # val set
            val_str = val_str + os.path.join(imageDir, str(i)+'.jpg') + ',' + os.path.join(annoDir, str(i)+'.txt') + '\n'
        else:
            # train set
            train_str = train_str + os.path.join(imageDir, str(i)+'.jpg') + ',' + os.path.join(annoDir, str(i)+'.txt') + '\n'

    with open(os.path.join(args.des, 'val.txt'), 'w') as f:
        f.write(val_str)
    with open(os.path.join(args.des, 'train.txt'), 'w') as f:
        f.write(train_str)

