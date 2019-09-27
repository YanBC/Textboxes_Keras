import numpy as np
import cv2 as cv

class Data_Butler():
    def __init__(self, img_height, img_width, nfeat, offsets, aspect_ratios):
        assert len(aspect_ratios) == len(offsets)

        self.img_h = img_height
        self.img_w = img_width
        self.aspect_ratios = aspect_ratios
        self.nfeat = nfeat
        self.offsets = offsets


    def _anchor_box_per_layer(self, feat_h, feat_w, aspect_ratios, v_offsets):
        scale_h = 1 / feat_h
        scale_w = 1 / feat_w
        n_boxes = len(aspect_ratios)

        # width and height of anchor boxes
        wh_list = []
        for ar in aspect_ratios:
            box_h = scale_h * np.sqrt(ar)
            box_w = scale_w / np.sqrt(ar)
            wh_list.append((box_w, box_h))
        wh_list = np.array(wh_list)

        # vertical offsets
        offset_grid = np.zeros(shape=(feat_h, feat_w, n_boxes))
        offset_grid += np.array(v_offsets) * scale_h

        # centers of anchor boxes
        base_cy = np.linspace(scale_h * 0.5, 1 - scale_h * 0.5, feat_h)
        base_cx = np.linspace(scale_w * 0.5, 1 - scale_w * 0.5, feat_w)
        base_cx_grid, base_cy_grid = np.meshgrid(base_cx, base_cy)
        base_cx_grid = np.expand_dims(base_cx_grid, -1)
        base_cy_grid = np.expand_dims(base_cy_grid, -1)

        ## build anchors
        anchors = np.empty(shape=(feat_h, feat_w, n_boxes, 4))
        anchors[:, :, :, 0] = np.tile(base_cx_grid, (1,1,n_boxes))
        anchors[:, :, :, 1] = np.tile(base_cy_grid, (1,1,n_boxes)) + offset_grid
        anchors[:, :, :, 2] = wh_list[:, 0]
        anchors[:, :, :, 3] = wh_list[:, 1]


        ret = np.reshape(anchors, newshape=(-1, 4))

        return ret


    def _anchor_boxes(self):
        boxes_per_layer = []
        n_featuremap = len(self.nfeat) // 2

        for i in range(n_featuremap):
            f_h = self.nfeat[i * 2]
            f_w = self.nfeat[i * 2 + 1]
            ars = self.aspect_ratios
            offsets = self.offsets
            layer_boxes = self._anchor_box_per_layer(f_h, f_w, ars, offsets)
            boxes_per_layer.append(layer_boxes)

        return np.concatenate(boxes_per_layer)


    def image_preprocessing(self, image):
        img = cv.resize(image, (self.img_w, self.img_h))
        img = img / 255

        return img


    def _centroid2corners(self, box):
        tmp_x, tmp_y, tmp_w, tmp_h = box
        left = tmp_x - tmp_w / 2
        top = tmp_y - tmp_h / 2
        right = tmp_x + tmp_w / 2
        bottom = tmp_y + tmp_h / 2

        return [left, top, right, bottom]


    def _cal_iou(self, box1, box2):
        '''
        box1 & box2: [centre_x, centre_y, width, height]
        '''
        box1_area = box1[2] * box1[3]
        box2_area = box2[2] * box2[3]

        # get intersection area
        b1 = self._centroid2corners(box1)
        b2 = self._centroid2corners(box2)

        i_left = max(b1[0], b2[0])
        i_top = max(b1[1], b2[1])
        i_right = min(b1[2], b2[2])
        i_bottom = min(b1[3], b2[3])
        i_width = max(i_right-i_left, 0)
        i_height = max(i_bottom-i_top, 0)

        i_area = i_width * i_height

        return i_area / (box1_area + box2_area - i_area)


    def encode(self, image, gt_labels):
        # get anchor_boxes
        anchor_boxes = self._anchor_boxes()
        
        if len(gt_labels) > 0:
            # translate gt_labels to gt_boxes
            image_h, image_w, _ = image.shape
            gt_list = []
            for gt_label in gt_labels:
                cx, cy, width, height = gt_label
                cx /= image_w
                cy /= image_h
                width /= image_w
                height /= image_h
                gt_list.append(np.array([cx, cy, width, height]))
            gt_boxes = np.stack(gt_list)

            # iou map
            n_anchors = len(anchor_boxes)
            n_gts = len(gt_boxes)
            iou_map = np.empty(shape=(n_anchors, n_gts))        # shape of (n_anchors, n_gts)
            for i in range(n_anchors):
                for j in range(n_gts):
                    iou_map[i, j] = self._cal_iou(anchor_boxes[i, :], gt_boxes[j, :])

            match = np.argmax(iou_map, axis=0)
            if len(set(match)) != len(match):
                for this in range(1, len(match)):
                    cross = np.where(match[:this] == match[this])[0]
                    while len(cross) != 0:
                        that = cross[0]
                        assert match[this] == match[that]       # Just in case I fuck up

                        iou_that = iou_map[match[this], that]
                        iou_this = iou_map[match[this], this]
                        if iou_this < iou_that:
                            # find the next largest for this
                            iou_map[match[this], this] = 0
                            tmp = iou_map[:, this]
                            tmp_descending = np.argsort(tmp)[::-1]   # sort in descending order
                            match[this] = tmp_descending[0]
                            cross = np.where(match[:this] == match[this])[0]

            # y_true
            # background will be labeled 0, while text labeled 1
            y_true = np.zeros(shape=(len(anchor_boxes), 6))
            y_true[:, 0] = 1
            y_true[:, 2:] = anchor_boxes

            for gt_index, anchor_index in enumerate(match):
                x0, y0, w0, h0 = y_true[anchor_index, 2:]
                x, y, w, h = gt_boxes[gt_index]

                delta = np.empty(shape=(6,))
                delta[0] = 0
                delta[1] = 1
                delta[2] = (x - x0) / w0
                delta[3] = (y - y0) / h0
                delta[4] = np.log(w / w0)
                delta[5] = np.log(h / h0)
                
                y_true[anchor_index] = delta

        else:
            y_true = np.zeros(shape=(len(anchor_boxes), 6))
            y_true[:, 0] = 1
        
        # x (the processed image)
        img = self.image_preprocessing(image)

        return img, y_true


    def _nms(self, boxes, thres):
        boxes_descending_conf = boxes[boxes[:, 1].argsort()]
        to_be_deleted = []
        for this in range(1, len(boxes_descending_conf)):
            for that in range(this):
                if that in to_be_deleted:
                    continue
                if self._cal_iou(boxes[this, 1:], boxes[that, 1:]) > thres:
                    to_be_deleted.append(this)
                    break
        keep = np.delete(boxes_descending_conf, to_be_deleted, axis=0)
        return keep


    def decode(self, y_pred, conf_thres=0.7, iou_thres=0.45, top_k=200, image_width=None, image_height=None):
        text_pred = y_pred[:, 1]
        text_indices = np.where(text_pred > conf_thres)[0]

        anchor_boxes = self._anchor_boxes()
        text_boxes = []
        # confidence, centre_x, centre_y, width, height

        for text_index in text_indices:
            delta_x, delta_y, delta_w, delta_h = y_pred[text_index, 2:]
            x0, y0, w0, h0 = anchor_boxes[text_index, :]

            conf = y_pred[text_index, 1]
            x = x0 + w0 * delta_x
            y = y0 + h0 * delta_y
            w = w0 * np.exp(delta_w)
            h = h0 * np.exp(delta_h)
            text_boxes.append(np.array([conf, x, y, w, h]))
                
        if len(text_boxes) > 0:
            text_boxes = np.stack(text_boxes)
            text_boxes = self._nms(text_boxes, iou_thres)

            if len(text_boxes) > top_k:
                text_boxes = text_boxes[:top_k, :]

            if image_width and image_height:
                text_boxes[:, 1] = np.floor(text_boxes[:, 1] * image_width)
                text_boxes[:, 2] = np.floor(text_boxes[:, 2] * image_height)
                text_boxes[:, 3] = np.floor(text_boxes[:, 3] * image_width)
                text_boxes[:, 4] = np.floor(text_boxes[:, 4] * image_height)

        return text_boxes


    def input_shape(self):
        return (self.img_h, self.img_w, 3)


    def output_shape(self):
        n_featuremap = len(self.nfeat) // 2

        n_cell = 0
        for i in range(n_featuremap):
            n_cell += self.nfeat[2 * i] * self.nfeat[2 * i + 1]

        n_box_per_cell = len(self.aspect_ratios)

        return (n_cell * n_box_per_cell, 6)





# Parser.encode and Parser.decode test
if __name__ == '__main__':
    import yaml

    with open('./configs/conv_model.yml') as f:
        config = yaml.safe_load(f.read())
    p = Data_Butler(config['img_height'], \
                  config['img_width'], \
                  config['nfeat'], \
                  config['offsets'], \
                  config['aspect_ratios'])

    imagePath = './detect_for_textboxes_test/image/0.jpg'
    annoPath = './detect_for_textboxes_test/label/0.txt'
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

    # check if annos_prime is the same as annos
    x, y_true = p.encode(image, annos)

    image_height, image_width, _ = image.shape
    annos_prime = p.decode(y_true, image_width=image_width, image_height=image_height)


    # check by drawing annos
    text_boxes = []
    for i in range(len(annos_prime)):
        cx, cy, w, h = annos_prime[i, 1::]
        left = int(cx - w / 2)
        right = int(cx + w /2)
        top = int(cy - h /2)
        bottom = int(cy + h / 2)

        cv.rectangle(image, (int(left), int(top)), (int(right), int(bottom)), (0, 255, 0), 1)

    cv.namedWindow('show', cv.WINDOW_NORMAL)
    cv.imshow('show', image)
    cv.waitKey()


