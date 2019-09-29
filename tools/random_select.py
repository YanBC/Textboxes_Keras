import argparse
import os
import shutil
import numpy as np


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('src', help='path to source directory')
    parser.add_argument('des', help='path to destination directory')
    args = parser.parse_args()

    srcDir = args.src
    desDir = args.des
    if not os.path.isdir(desDir):
        os.mkdir(desDir)
    # select_num = args.size
    # if args.ratio:
    #     select_ratio = args.ratio
    #     assert select_ratio > 0 and select_ratio < 1

    # list all target files
    targetExt = ['jpg', 'png']
    imagePaths = dict()
    for dirpath, dirnames, dirfiles in os.walk(srcDir):
        for file in dirfiles:
            filename, fileext = file.split('.')
            if fileext in targetExt:
                imagePaths[os.path.join(dirpath, file)] = file

    # prompt for selection size
    total_num_files = len(imagePaths.keys())
    print('Total number of files: {}'.format(total_num_files))
    size_str = input('How many files would you like to select?\nEnter an integer(number of files) or a float(ratio of files): ')
    try:
        num_files = int(size_str)
    except ValueError:
        num_files = int(float(size_str) * total_num_files)

    # select
    if num_files > total_num_files:
        num_files = total_num_files
    select_indices = np.random.choice(total_num_files, size=num_files, replace=False)

    for select_index in select_indices:
        srcPath = list(imagePaths.keys())[select_index]
        desPath = os.path.join(desDir, imagePaths[srcPath])
        shutil.copy(srcPath, desPath)
        