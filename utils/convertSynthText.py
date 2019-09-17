import scipy.io as sio
import numpy as np
import os
import shutil
import multiprocessing.dummy as fthreading


########## User Config ##############
SynthDir = '/workspace/yanbc/SynthText'
targetDir = '/workspace/yanbc/time_stamp_ocr/SynthText_transformed'
num_of_workers = 8
#####################################

targetImageDir = os.path.join(targetDir, 'image')
targetAnnoDir = os.path.join(targetDir, 'label')
if os.path.isdir(targetImageDir):
    shutil.rmtree(targetImageDir)
if os.path.isdir(targetAnnoDir):
    shutil.rmtree(targetAnnoDir)
os.mkdir(targetImageDir)
os.mkdir(targetAnnoDir)

annoName = 'gt.mat'
mat = sio.loadmat(os.path.join(SynthDir, annoName))
imagePaths = mat['imnames'][0]
annos = mat['wordBB'][0]

assert len(imagePaths) == len(annos)
n_data = len(imagePaths)



def create_one(index):
    imagePath = imagePaths[index][0]
    shutil.copy(os.path.join(SynthDir, imagePath), os.path.join(targetImageDir, '{}.jpg'.format(index)))

    anno = annos[index]
    string_anno = ''
    if anno.shape == (2,4):
        x_anno = anno[0,:]
        y_anno = anno[1,:]
        left = str(int(np.min(x_anno)))
        right = str(int(np.max(x_anno)))
        top = str(int(np.min(y_anno)))
        bottom = str(int(np.max(y_anno)))
        string_anno += left + ', ' + top + ', ' + right + ', ' + top + ', ' + right + ', ' + bottom + ', ' + left + ', ' + bottom + '\n'      
    else:
        n_boxes = anno.shape[2]
        for i in range(n_boxes):
            box_anno = anno[:,:,i]
            x_anno = box_anno[0,:]
            y_anno = box_anno[1,:]

            left = str(int(np.min(x_anno)))
            right = str(int(np.max(x_anno)))
            top = str(int(np.min(y_anno)))
            bottom = str(int(np.max(y_anno)))
            string_anno += left + ', ' + top + ', ' + right + ', ' + top + ', ' + right + ', ' + bottom + ', ' + left + ', ' + bottom + '\n'

    with open(os.path.join(targetAnnoDir, '{}.txt'.format(index)), 'w') as f:
        f.write(string_anno)

    return True



def transform_dataset():
    indices = [x for x in range(n_data)]
    with fthreading.Pool(processes=num_of_workers) as p:
        ret = p.map(create_one, indices)
    print('Sucessfully transformed %d samples' % sum(ret))


if __name__ == '__main__':
    # create_one(85350)
    transform_dataset()
