import argparse
import os
import numpy as np

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('src', help='path to source data directory')
    p.add_argument('des', help='path to destination directory')
    p.add_argument('--ratio', type=float, default=0.1, help='a float equal to val/total. Default: 0.1')
    args = p.parse_args()

    srcDir = os.path.join(args.src, 'faster_training')
    annoNames = os.listdir(srcDir)

    val_str = ''
    train_str = ''

    num_data = len(annoNames)
    for i in range(num_data):
        if np.random.rand() < args.ratio:
            # val set
            val_str = val_str + os.path.join(srcDir, str(i)+'.pkl') + '\n'
        else:
            # train set
            train_str = train_str + os.path.join(srcDir, str(i)+'.pkl') + '\n'


    with open(os.path.join(args.des, 'val.txt'), 'w') as f:
        f.write(val_str)
    with open(os.path.join(args.des, 'train.txt'), 'w') as f:
        f.write(train_str)

