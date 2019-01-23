#! /usr/bin/env python
"""Preprocessing image datasets for model training"""

import tensorflow as tf
from PIL import Image
import os
import dataset_util
import json
import hashlib
import numpy as np
from tqdm import tqdm
from utils import get_immediate_subdirectories, get_file_list


__author__ = "Sayan Paul"
__email__ = "sayanpau@usc.edu"


OPEN_IMAGES_OBJECTS_LIST = set()
IMAGENET_OBJECTS_LIST = set()
LABEL_STOI = dict()


def save_label_map(data_dir):

    label_file = os.path.join(data_dir, 'oid.names')

    with open(label_file, 'w') as label_map_out:
        label_map_out.write("\n".join(LABEL_STOI.keys()))


def group_to_tf_record(boxes, image_file):
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
        boxes[i] = [anno[1], anno[2], anno[3], anno[4], LABEL_STOI[anno[0]]]

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


def process_openimages(data_dir):
    """
    Process open Images dataset
    :param data_dir:
    :type data_dir:
    :return:
    :rtype:
    """

    global OPEN_IMAGES_OBJECTS_LIST, LABEL_STOI

    dataset = dict()
    splits = [os.path.join(data_dir, split) for split in ['train', 'test', 'validation']]
    for split in splits:
        split_dir = os.path.basename(split)
        print("Current split:", split_dir)
        dataset[split_dir] = {'images': dict(), 'boxes': dict()}
        obj_list = get_immediate_subdirectories(split)
        for obj in tqdm(obj_list):
            obj_name = os.path.basename(obj).lower()

            img_file_list = get_file_list(obj, format=".jpg")

            if len(img_file_list) > 0:
                OPEN_IMAGES_OBJECTS_LIST.add(obj_name)

            label_dir = os.path.join(obj, 'Label')
            label_list = get_file_list(label_dir, format=".txt")
            for img in img_file_list:
                img_name, _ = os.path.splitext(os.path.basename(img))
                dataset[split_dir]['images'][img_name] = img
            for label in label_list:
                label_name, _ = os.path.splitext(os.path.basename(label))
                if label_name not in dataset[split_dir]['boxes']:
                    dataset[split_dir]['boxes'][label_name] = list()
                with open(label, 'r') as label_file:
                    annotations = label_file.readlines()
                for annotation in annotations:
                    dataset[split_dir]['boxes'][label_name].append(annotation.lower().split())

    OPEN_IMAGES_OBJECTS_LIST = sorted(OPEN_IMAGES_OBJECTS_LIST)
    update_label_map(OPEN_IMAGES_OBJECTS_LIST)

    return dataset


def update_label_map(obj_list):
    global LABEL_STOI

    cur_len = len(LABEL_STOI)
    for i, v in enumerate(obj_list):
        LABEL_STOI[v] = i + cur_len


def process_imagenet(data_dir):
    """
    Process imagenet dataset
    :param data_dir:
    :type data_dir:
    :return:
    :rtype:
    """
    pass


def write_tf_records(datasets, record_save_dir):

    if not os.path.exists(record_save_dir):
        os.mkdir(record_save_dir)

    for split in tqdm(datasets, desc="Splits completed"):
        record_save_path = os.path.join(record_save_dir, split)

        if not os.path.exists(record_save_path):
            os.mkdir(record_save_path)

        record_save_file = os.path.join(record_save_path, 'data.tfrecord')

        writer = tf.python_io.TFRecordWriter(record_save_file)
        for img_name in tqdm(datasets[split]['images'], desc="Writing to file"):
            record = group_to_tf_record(datasets[split]['boxes'][img_name], datasets[split]['images'][img_name])
            if record:
                serialized = record.SerializeToString()
                writer.write(serialized)
        writer.close()


def merge_datasets(dataset_list):
    """
    Merge dataset objects in the list
    :param dataset_list:
    :type dataset_list:
    :return:
    :rtype:
    """
    pass


def save_dataset(dataset_obj, data_dir, data_filename):
    """
    Json dump the dataset object
    :param dataset_obj: Parsed dataset object
    :type dataset_obj: Dictionary
    :param data_dir: Directory to save file
    :type data_dir: String
    :param data_filename: Filename for dump file
    :type data_filename: String
    :return: None
    :rtype: None
    """
    with open(os.path.join(data_dir, data_filename + ".json"), 'w') as data_file:
        json.dump(dataset_obj, data_file)


if __name__ == "__main__":

    main_data_dir = "data"
    openimages_data_dir = "data/OpenImages"
    imagenet_data_dir = "data/ImageNet"

    openimg_dataset = process_openimages(openimages_data_dir)
    # imagenet_dataset = process_imagenet(imagenet_data_dir)

    save_label_map(openimages_data_dir)
    save_dataset(openimg_dataset, openimages_data_dir, 'oid')

    write_tf_records(openimg_dataset, os.path.join(openimages_data_dir, 'tfrecords'))
