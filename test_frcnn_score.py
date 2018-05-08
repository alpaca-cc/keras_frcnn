from __future__ import division
import os
import cv2
import numpy as np
import pickle
import time
from keras_frcnn import config
from keras import backend as K
from keras.layers import Input
from keras.models import Model
from keras_frcnn import roi_helpers
import argparse
import os
import keras_frcnn.resnet as nn
from keras_frcnn.visualize import draw_boxes_and_label_on_image_cv2
import re


def preprocess_img(img):
    img = cv2.bitwise_not(img)
    img = cv2.copyMakeBorder(img, 150, 150, 50, 50, cv2.BORDER_CONSTANT)
    img = cv2.resize(img, (200, 200))
    return img


def format_img_size(img, cfg):
    """ formats the image size based on config """
    img_min_side = float(cfg.im_size)
    (height, width, _) = img.shape

    if width <= height:
        ratio = img_min_side / width
        new_height = int(ratio * height)
        new_width = int(img_min_side)
    else:
        ratio = img_min_side / height
        new_width = int(ratio * width)
        new_height = int(img_min_side)
    img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    return img, ratio


def format_img_channels(img, cfg):
    """ formats the image channels based on config """
    img = img[:, :, (2, 1, 0)]
    img = img.astype(np.float32)
    img[:, :, 0] -= cfg.img_channel_mean[0]
    img[:, :, 1] -= cfg.img_channel_mean[1]
    img[:, :, 2] -= cfg.img_channel_mean[2]
    img /= cfg.img_scaling_factor
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    return img


def format_img(img, C):
    """ formats an image for model prediction based on config """
    img, ratio = format_img_size(img, C)
    img = format_img_channels(img, C)
    return img, ratio


# Method to transform the coordinates of the bounding box to its original size
def get_real_coordinates(ratio, x1, y1, x2, y2):
    real_x1 = int(round(x1 // ratio))
    real_y1 = int(round(y1 // ratio))
    real_x2 = int(round(x2 // ratio))
    real_y2 = int(round(y2 // ratio))

    return real_x1, real_y1, real_x2, real_y2


def predict_single_image(img_path, model_rpn, model_classifier_only, cfg, class_mapping, preprocess):
    st = time.time()
    img = cv2.imread(img_path)
    if img is None:
        print('reading image failed.')
        exit(0)

    if preprocess:
        img = preprocess_img(img)
    X, ratio = format_img(img, cfg)
    # print(ratio)
    if K.image_dim_ordering() == 'tf':
        X = np.transpose(X, (0, 2, 3, 1))
    # get the feature maps and output from the RPN
    [Y1, Y2, F] = model_rpn.predict(X)

    # this is result contains all boxes, which is [x1, y1, x2, y2]
    result = roi_helpers.rpn_to_roi(Y1, Y2, cfg, K.image_dim_ordering(), overlap_thresh=0.7)

    # convert from (x1,y1,x2,y2) to (x,y,w,h)
    result[:, 2] -= result[:, 0]
    result[:, 3] -= result[:, 1]
    # print(result)
    bbox_threshold = 0.3

    # apply the spatial pyramid pooling to the proposed regions
    boxes = dict()
    for jk in range(result.shape[0] // cfg.num_rois + 1):
        # print(jk)
        rois = np.expand_dims(result[cfg.num_rois * jk:cfg.num_rois * (jk + 1), :], axis=0)
        # print(rois)
        if rois.shape[1] == 0:
            break
        if jk == result.shape[0] // cfg.num_rois:
            # pad R
            curr_shape = rois.shape
            target_shape = (curr_shape[0], cfg.num_rois, curr_shape[2])
            rois_padded = np.zeros(target_shape).astype(rois.dtype)
            rois_padded[:, :curr_shape[1], :] = rois
            rois_padded[0, curr_shape[1]:, :] = rois[0, 0, :]
            rois = rois_padded
            # print (rois)

        [p_cls, p_regr] = model_classifier_only.predict([F, rois])

        for ii in range(p_cls.shape[1]):
            if np.max(p_cls[0, ii, :]) < bbox_threshold or np.argmax(p_cls[0, ii, :]) == (p_cls.shape[2] - 1):
                continue

            cls_num = np.argmax(p_cls[0, ii, :])
            if cls_num not in boxes.keys():
                boxes[cls_num] = []
            (x, y, w, h) = rois[0, ii, :]
            try:
                (tx, ty, tw, th) = p_regr[0, ii, 4 * cls_num:4 * (cls_num + 1)]
                tx /= cfg.classifier_regr_std[0]
                ty /= cfg.classifier_regr_std[1]
                tw /= cfg.classifier_regr_std[2]
                th /= cfg.classifier_regr_std[3]
                x, y, w, h = roi_helpers.apply_regr(x, y, w, h, tx, ty, tw, th)
            except Exception as e:
                print(e)
                pass
            boxes[cls_num].append(
                [cfg.rpn_stride * x, cfg.rpn_stride * y, cfg.rpn_stride * (x + w), cfg.rpn_stride * (y + h),
                 np.max(p_cls[0, ii, :])])
    print(boxes)
    concat_boxes = []
    # add some nms to reduce many boxes
    for cls_num, box in boxes.items():
        boxes_nms = roi_helpers.non_max_suppression_fast(box, overlap_thresh=0.2)
        # boxes[cls_num] = boxes_nums
        # reformat boxes into one array of form [x1, y1, x2, y2, prob, cls_num] for cross class non max suppresion
        for b in boxes_nms:
            b = np.append(b, cls_num)
            concat_boxes.append(b)
    # print (concat_boxes)
    # cross class non max suppresion
    boxes_nms_crs_cls = roi_helpers.non_max_suppression_fast(concat_boxes, overlap_thresh=0.5)
    for b in boxes_nms_crs_cls:
        cls_num = int(b[-1])
        boxes[cls_num].append(b[0:5])
        print(class_mapping[cls_num] + ":")
        b[0], b[1], b[2], b[3] = get_real_coordinates(ratio, b[0], b[1], b[2], b[3])
        print('{} prob: {}'.format(b[0: 4], b[4]))
    print(class_mapping)
    img = draw_boxes_and_label_on_image_cv2(img, class_mapping, boxes)
    print('Elapsed time = {}'.format(time.time() - st))
    # cv2.imshow('image', img)
    result_path = './results_images/{}.png'.format(os.path.basename(img_path).split('.')[0])
    print('result saved into ', result_path)
    cv2.imwrite(result_path, img)
    cv2.waitKey(30)
    if not preprocess:
        avg_overlap_rate, avg_accurate_rate = compute_accuracy(boxes_nms_crs_cls, class_mapping, re.findall("\d+", img_path)[0])
        print (avg_overlap_rate, avg_accurate_rate)
    return boxes_nms_crs_cls, class_mapping


def overlap(box1, box2):
    for i in range(len(box1)):
        box1[i] = int(box1[i])
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    area_int = (x2-x1) * (y2-y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    area_union = area1 + area2 - area_int
    return area_int / (area_union + 1e-6)


def compute_accuracy(boxes_nums_crs_cls, class_mapping, img_num):
    # path to write the accuracy
    accuracy_file = "./results_images/accuracy.txt"
    f_acc = open(accuracy_file, "a+")
    # path for labels
    label_path = "../dataset/label/test/medium_bar/"
    file_name = "label" + img_num + ".txt"
    sorted_boxes = sorted(boxes_nums_crs_cls, key=lambda v: v[0])
    rate_thres = 0.5
    rate_total = 0
    count = 0
    num_lines = 0
    line1 = ""
    with open(label_path + file_name, "r") as f_labels:
        for line in f_labels:
            num_lines += 1
            info = line.split(",")
            for box in sorted_boxes:
                # compute overlap of two boxes
                rate = overlap(info[1:5], box[0:4])
                info[-1] = info[-1].strip()
                if rate > rate_thres and class_mapping[int(box[-1])] == info[-1]:
                    rate_total += rate
                    count += 1
                    line1 += "label {} :".format(info[-1]) + "truth: {}".format(info[1:5]) + ", predict: {}".format(box[0:4]) + ", overlap = {}\n".format(rate)
                elif rate_thres:
                    line1 += "label truth{} :".format(info[-1]) + "truth: {}".format(info[1:5]) + "label predict{} :".format(class_mapping[int(box[-1])]) + ", predict: {}".format(box[0:4]) + ", overlap = {}\n".format(rate)
    avg_overlap_rate = rate_total / (count + 1e-6)
    avg_accurate_rate = count/num_lines
    line2 = "For img{}: ".format(img_num) + "avg_overlap = {}" .format(avg_overlap_rate) + ", avg_accurate = {}\n".format(avg_accurate_rate)
    f_acc.write(line1)
    f_acc.write(line2)
    return avg_overlap_rate, avg_accurate_rate


def predict_from_server(path):
    with open('config.pickle', 'rb') as f_in:
        cfg = pickle.load(f_in)
    cfg.use_horizontal_flips = False
    cfg.use_vertical_flips = False
    cfg.rot_90 = False

    class_mapping = cfg.class_mapping
    if 'bg' not in class_mapping:
        class_mapping['bg'] = len(class_mapping)

    class_mapping = {v: k for k, v in class_mapping.items()}
    input_shape_img = (None, None, 3)
    input_shape_features = (None, None, 1024)

    img_input = Input(shape=input_shape_img)
    roi_input = Input(shape=(cfg.num_rois, 4))
    feature_map_input = Input(shape=input_shape_features)

    shared_layers = nn.nn_base(img_input, trainable=True)

    # define the RPN, built on the base layers
    num_anchors = len(cfg.anchor_box_scales) * len(cfg.anchor_box_ratios)
    rpn_layers = nn.rpn(shared_layers, num_anchors)
    classifier = nn.classifier(feature_map_input, roi_input, cfg.num_rois, nb_classes=len(class_mapping),
                               trainable=True)
    model_rpn = Model(img_input, rpn_layers)
    model_classifier_only = Model([feature_map_input, roi_input], classifier)

    model_classifier = Model([feature_map_input, roi_input], classifier)

    print('Loading weights from {}'.format(cfg.model_path))
    model_rpn.load_weights(cfg.model_path, by_name=True)
    model_classifier.load_weights(cfg.model_path, by_name=True)

    model_rpn.compile(optimizer='sgd', loss='mse')
    model_classifier.compile(optimizer='sgd', loss='mse')

    print('predict image from {}'.format(path))
    boxes, class_mapping = predict_single_image(path, model_rpn, model_classifier_only, cfg, class_mapping, True)
    return boxes, class_mapping


def predict(args_):
    path = args_.path
    with open('config.pickle', 'rb') as f_in:
        cfg = pickle.load(f_in)
    cfg.use_horizontal_flips = False
    cfg.use_vertical_flips = False
    cfg.rot_90 = False

    class_mapping = cfg.class_mapping
    if 'bg' not in class_mapping:
        class_mapping['bg'] = len(class_mapping)

    class_mapping = {v: k for k, v in class_mapping.items()}
    input_shape_img = (None, None, 3)
    input_shape_features = (None, None, 1024)

    img_input = Input(shape=input_shape_img)
    roi_input = Input(shape=(cfg.num_rois, 4))
    feature_map_input = Input(shape=input_shape_features)

    shared_layers = nn.nn_base(img_input, trainable=True)

    # define the RPN, built on the base layers
    num_anchors = len(cfg.anchor_box_scales) * len(cfg.anchor_box_ratios)
    rpn_layers = nn.rpn(shared_layers, num_anchors)
    classifier = nn.classifier(feature_map_input, roi_input, cfg.num_rois, nb_classes=len(class_mapping),
                               trainable=True)
    model_rpn = Model(img_input, rpn_layers)
    model_classifier_only = Model([feature_map_input, roi_input], classifier)

    model_classifier = Model([feature_map_input, roi_input], classifier)

    print('Loading weights from {}'.format(cfg.model_path))
    model_rpn.load_weights(cfg.model_path, by_name=True)
    model_classifier.load_weights(cfg.model_path, by_name=True)

    model_rpn.compile(optimizer='sgd', loss='mse')
    model_classifier.compile(optimizer='sgd', loss='mse')

    preprocess = False
    if args_.ios == "True":
        preprocess = True

    if os.path.isdir(path):
        for idx, img_name in enumerate(sorted(os.listdir(path))):
            if not img_name.lower().endswith(('.bmp', '.jpeg', '.jpg', '.png', '.tif', '.tiff')):
                continue
            print(img_name)
            predict_single_image(os.path.join(path, img_name), model_rpn,
                                 model_classifier_only, cfg, class_mapping, preprocess)
    elif os.path.isfile(path):
        print('predict image from {}'.format(path))
        predict_single_image(path, model_rpn, model_classifier_only, cfg, class_mapping, preprocess)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', '-p', default='./images/', help='image path')
    parser.add_argument('--ios', '-i', default='False', help='whether data is from ios')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    # print(args)
    predict(args)
