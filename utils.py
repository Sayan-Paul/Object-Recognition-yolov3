#! /usr/bin/env python
"""Utilities" methods"""

import os
import copy
import hashlib
import tensorflow as tf
import numpy as np
import dataset_util
from PIL import Image

__author__ = "Sayan Paul"
__email__ = "sayanpau@usc.edu"


def get_file_list(directory, format='.png'):
    file_list = []
    for root, _, files in os.walk(directory, topdown=False):
        for name in files:
            if name.endswith(format):
                file_list.append(os.path.join(root, name))
    return file_list


def get_dir_list(directory, only_top=False):

    if only_top:
        return [os.path.join(directory, name) for name in os.listdir(directory)
                if os.path.isdir(os.path.join(directory, name))]

    dir_list = []
    for root, dirs, _ in os.walk(directory, topdown=False):
        for name in dirs:
            dir_list.append(os.path.join(root, name))
    return dir_list


def get_immediate_subdirectories(a_dir):
    return [os.path.join(a_dir, name) for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]


# Copyright Ferry Boender, released under the MIT license.
def deepupdate(target, src):
    """Deep update target dict with src
    For each k,v in src: if k doesn't exist in target, it is deep copied from
    src to target. Otherwise, if v is a list, target[k] is extended with
    src[k]. If v is a set, target[k] is updated with v, If v is a dict,
    recursively deep-update it.

    """
    for k, v in src.items():
        if type(v) == list:
            if not k in target:
                target[k] = copy.deepcopy(v)
            else:
                target[k].extend(v)
        elif type(v) == dict:
            if not k in target:
                target[k] = copy.deepcopy(v)
            else:
                deepupdate(target[k], v)
        elif type(v) == set:
            if not k in target:
                target[k] = v.copy()
            else:
                target[k].update(v.copy())
        else:
            target[k] = copy.copy(v)


def get_item(name, root, index=0):
    count = 0
    for item in root.iter(name):
        if count == index:
            return item.text
        count += 1
    # Failed to find "index" occurrence of item.
    return -1


def get_int(name, root, index=0):
    return int(get_item(name, root, index))


def get_bb_count(root):
    index = 0
    while True:
        if get_int('xmin', root, index) == -1:
            break
        index += 1
    return index


def group_to_tf_record(boxes, image_file, label_stoi):
    """

    :param boxes:
    :type boxes:
    :param image_file:
    :type image_file:
    :return:
    :rtype:
    """
    format = b'jpeg'
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    class_nums = []
    class_ids = []
    filename = image_file
    try:
        image = Image.open(filename)
        width, height = image.size
        with tf.gfile.GFile(filename, 'rb') as fid:
            encoded_jpg = bytes(fid.read())
    except:
        return None
    key = hashlib.sha256(encoded_jpg).hexdigest()
    for i, anno in enumerate(boxes):
        # Needs refactoring
        # xmins.append(float(anno[1]))
        # xmaxs.append(float(anno[3]))
        # ymins.append(float(anno[2]))
        # ymaxs.append(float(anno[4]))
        # class_nums.append(LABEL_STOI[anno[0]])
        # class_ids.append(bytes(anno[0], "utf-8"))
        boxes[i] = [anno[1], anno[2], anno[3], anno[4], label_stoi[anno[0]]]

    boxes = np.array(boxes, dtype=np.float32).tostring()

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
        'image/filename': dataset_util.bytes_feature(bytes(filename, "utf-8")),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(format),
        # 'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        # 'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        # 'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        # 'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        # 'image/object/class/text': dataset_util.bytes_list_feature(class_ids),
        # 'image/object/class/label': dataset_util.int64_list_feature(class_nums)
        'boxes': tf.train.Feature(bytes_list=tf.train.BytesList(value=[boxes]))
    }))
    return tf_example
