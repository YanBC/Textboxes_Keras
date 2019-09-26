import yaml
import time
import argparse
import cv2 as cv

import os
import sys
sys.path.append(os.getcwd())
from main.predict import Detector


def args_parser():
    p = argparse.ArgumentParser()
    p.add_argument('configPath', help='path to configuration file (*.yml)')
    p.add_argument('src', help='path to src files (image or video or image folder)')
    p.add_argument('--des', help='where to save the destination file(s)')
    p.add_argument('--show', action='store_true', help='show result')
    args = p.parse_args()
    return args


def draw_box(image, text, left, top, right, bottom):

    ff = image.copy()
    cv.rectangle(ff, (left, top), (right, bottom), (0, 255, 0), 3)

    if text:
        labelSize, baseLine = cv.getTextSize(text, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        top = max(top, labelSize[1])
        cv.rectangle(ff, (left, top - round(1.5*labelSize[1])), (left + round(1.5*labelSize[0]), top + baseLine), (0, 0, 255), cv.FILLED)
        cv.putText(ff, text, (left, top), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0), 2)   

    return ff


if __name__ == '__main__':
    args = args_parser()

    with open(args.configPath) as f:
        config = yaml.safe_load(f.read())

    # create detector
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

    d = Detector(modelPath, img_height, img_width, top_k, iou_thres, conf_thres, nfeat, offsets, aspect_ratios, n_gpus, neg_pos_ratio, alpha)


    # processing
    if os.path.isfile(args.src):
        srcFile = args.src
        cap = cv.VideoCapture(srcFile)
        length = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
        if args.show:
            cv.namedWindow(srcFile, cv.WINDOW_NORMAL)

        # if source is an image
        if length == 1:
            hasFrame, frame = cap.read()

            boxes = d.predict(frame)
            show = frame.copy()
            for i, box in enumerate(boxes):
                conf, cx, cy, w, h = box
                left = cx - w / 2
                right = cx + w / 2
                top = cy - h /2
                bottom = cy + h / 2
                cv.rectangle(show, (int(left), int(top)), (int(right), int(bottom)), (0, 255, 0), 2)
            if args.des:
                cv.imwrite(args.des, show)
            if args.show:
                cv.imshow(srcFile, show)
                cv.waitKey()

        # if source is a video
        elif length > 1:
            if args.des:
                vid_writer = cv.VideoWriter(args.des, cv.VideoWriter_fourcc(*'XVID'), cap.get(cv.CAP_PROP_FPS), (round(cap.get(cv.CAP_PROP_FRAME_WIDTH)),round(cap.get(cv.CAP_PROP_FRAME_HEIGHT))))

            while cv.waitKey(1) < 0:
                hasFrame, frame = cap.read()
                if not hasFrame:
                    print('Finish Processing')
                    print('Exiting ...')
                    break            

                boxes = d.predict(frame)
                show = frame.copy()
                for i, box in enumerate(boxes):
                    conf, cx, cy, w, h = box
                    left = cx - w / 2
                    right = cx + w / 2
                    top = cy - h /2
                    bottom = cy + h / 2
                    cv.rectangle(show, (int(left), int(top)), (int(right), int(bottom)), (0, 255, 0), 2)

                if args.des:
                    vid_writer.write(show.astype(np.uint8))
                if args.show:
                    cv.imshow(srcFile, show)


    elif os.path.isdir(args.src):
        imageNames = [f for f in os.listdir(args.src) if '.jpg' in f]

        for imageName in imageNames:
            imagePath = os.path.join(args.src, imageName)

            image = cv.imread(imagePath)
            boxes = d.predict(image)
            show = image.copy()
            for i, box in enumerate(boxes):
                conf, cx, cy, w, h = box
                left = cx - w / 2
                right = cx + w / 2
                top = cy - h /2
                bottom = cy + h / 2
                cv.rectangle(show, (int(left), int(top)), (int(right), int(bottom)), (0, 255, 0), 2)

            if args.des:
                cv.imwrite(os.path.join(args.des, imageName), show)
            if args.show:
                cv.namedWindow(imagePath, cv.WINDOW_NORMAL)
                cv.imshow(imagePath, show)
                cv.waitKey()
                cv.destroyAllWindows()