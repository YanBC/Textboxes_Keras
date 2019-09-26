import argparse
import os


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('src', help='path to opencv annotation file')
    p.add_argument('des', help='where to store the label files')
    args = p.parse_args()

    with open(args.src) as f:
        for line in f.readlines():
            tmp = line.strip()
            imagePath, _, left, top, width, height = tmp.split(' ')

            imageName = imagePath.split('/')[1]
            name = imageName.split('.')[0]
            annoName = name + '.txt'
            annoPath = os.path.join(args.des, annoName)

            left = int(left)
            top = int(top)
            right = left + int(width)
            bottom = top + int(height)
            annoString = '{}, {}, {}, {}, {}, {}, {}, {}\n'.format(left, top, right, top, right, bottom, left, bottom)

            with open(annoPath, 'w') as g:
                g.write(annoString)
