#! /usr/bin/env python
"""Pre-processing image datasets for model training"""


import os
import json
import tensorflow as tf
import csv
import xml.etree.ElementTree as ET
from tqdm import tqdm
from PIL import Image
from utils import get_immediate_subdirectories, get_file_list, get_bb_count, get_int, group_to_tf_record, get_dir_list


__author__ = "Sayan Paul"
__email__ = "sayanpau@usc.edu"


OPEN_IMAGES_OBJECTS_SET = set()
IMAGENET_OBJECTS_SET = set()
YOUCOLL_OBJECTS_SET = set()
IMAGENET_DATA_DIR = "data/IMGNET"
IMAGENET_DATASET_NAME = "imgnet"
OPENIMAGES_DATA_DIR = "data/OID"
OPENIMAGES_DATASET_NAME = "oid"
MERGED_DATA_DIR = "data/MERGED"
MERGED_DATASET_NAME = "merged"
RND_DATA_DIR = "data/RND"
RND_DATASET_NAME = "rnd"
YOUCOLL_DATA_DIR = "data/YOUCOLL"
YOUCOLL_DATASET_NAME = "youcoll"
LABEL_STOI = dict()


def save_label_map(data_dir, dataset_name="data"):
    """
    Save label map to disk
    :param data_dir: Path to save map
    :type data_dir: String
    :param dataset_name: dataset name to save file
    :type dataset_name: String
    """

    label_file = os.path.join(data_dir, dataset_name + ".names")

    with open(label_file, 'w') as label_map_out:
        label_map_out.write("\n".join(LABEL_STOI.keys()))


def process_youcoll(data_dir, class_filter=None, skip_frames=None):
    """
    Process Youcoll annotated datasets
    :param data_dir:
    :type data_dir:
    :param class_filter:
    :type class_filter:
    :return:
    :rtype:
    """
    global YOUCOLL_OBJECTS_SET

    object_track_dir = os.path.join(data_dir, "annotations")
    video_frames_dir = os.path.join(data_dir, "imgs")

    obj_file_list = get_file_list(object_track_dir, format='.txt')
    video_frames_dir_list = get_dir_list(video_frames_dir, only_top=True)

    dataset = dict()
    for split in ['train', 'test']:
        print("Current split:", split)

        dataset[split] = {'images': dict(), 'boxes': dict()}

        data_rows = []
        with open(os.path.join(data_dir, split + '_frames.csv')) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            for row in csv_reader:
                data_rows.append(row)

        for row in tqdm(data_rows):
            img_path = row[0]
            box = eval(row[1])
            cls_name = row[2]

            img_name = os.path.basename(img_path).split('.')[0]

            if class_filter:
                if cls_name not in class_filter:
                    continue

            # Store in label set
            YOUCOLL_OBJECTS_SET.add(cls_name)

            dataset[split]['images'][img_name] = img_path

            # Store annotation from xml to box format

            if img_name not in dataset[split]['boxes']:
                dataset[split]['boxes'][img_name] = list()

                annotation = []
                annotation.append(cls_name)
                annotation.append(box[0])
                annotation.append(box[1])
                annotation.append(box[2])
                annotation.append(box[3])
                dataset[split]['boxes'][img_name].append(annotation)

    YOUCOLL_OBJECTS_SET = sorted(YOUCOLL_OBJECTS_SET)
    update_label_map(YOUCOLL_OBJECTS_SET)

    return dataset


def process_openimages(data_dir, class_filter=None):
    """
    Process open Images dataset
    :param data_dir:
    :type data_dir:
    :return:
    :rtype:
    """

    global OPEN_IMAGES_OBJECTS_SET

    dataset = dict()
    splits = [os.path.join(data_dir, split) for split in ['train', 'test', 'validation']]
    for split in splits:
        split_dir = os.path.basename(split)
        print("Current split:", split_dir)
        dataset[split_dir] = {'images': dict(), 'boxes': dict()}
        obj_list = get_immediate_subdirectories(split)
        for obj in tqdm(obj_list):
            obj_name = os.path.basename(obj).lower()

            if class_filter:
                if obj_name not in class_filter:
                    continue

            img_file_list = get_file_list(obj, format=".jpg")

            if len(img_file_list) > 0:
                OPEN_IMAGES_OBJECTS_SET.add(obj_name)

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

    OPEN_IMAGES_OBJECTS_SET = sorted(OPEN_IMAGES_OBJECTS_SET)
    update_label_map(OPEN_IMAGES_OBJECTS_SET)

    return dataset


def update_label_map(obj_list):
    """
    Add new label ()
    :param obj_list:
    :type obj_list:
    :return:
    :rtype:
    """
    global LABEL_STOI

    cur_len = len(LABEL_STOI)
    for i, v in enumerate(obj_list):
        if v not in LABEL_STOI:
            LABEL_STOI[v] = cur_len
            cur_len += 1


def process_imagenet(data_dir, class_filter=None):
    """
    Process imagenet dataset
    :param data_dir: Directory with images folder, annotations folder and data_split.json
    :type data_dir: String
    :return:
    :rtype:
    """

    global IMAGENET_OBJECTS_SET

    images_dir = os.path.join(data_dir, 'images')
    anno_dir = os.path.join(data_dir, 'annotations')
    data_split_filename = os.path.join(data_dir, 'data_split.json')
    wnid_map_filename = os.path.join(data_dir, 'wnid_map.json')

    with open(data_split_filename, 'r') as data_split_file:
        data_split = json.load(data_split_file)

    with open(wnid_map_filename, 'r') as wnid_map_file:
        object_wnid_map = json.load(wnid_map_file)

    wnid_object_map = {v: k for k,v in object_wnid_map.items()}

    dataset = dict()
    splits = ['train', 'test', 'validation']
    for split in splits:

        print("Current split:", split)

        dataset[split] = {'images': dict(), 'boxes': dict()}

        for img_ind in tqdm(data_split[split]):
            img_name = data_split['data'][img_ind]
            cls_name = img_name.split('_')[0]
            cls_name = wnid_object_map[cls_name]

            if class_filter:
                if cls_name not in class_filter:
                    continue

            # Store in label set
            IMAGENET_OBJECTS_SET.add(cls_name)

            # Store image path
            img = os.path.join(images_dir, img_name+".jpg")
            if not os.path.exists(img):
                img = os.path.join(images_dir, img_name + ".JPEG")

            try:
                width, height = Image.open(img).size
            except:
                print('Error', img)
                continue

            dataset[split]['images'][img_name] = img

            # Store annotation from xml to box format

            if img_name not in dataset[split]['boxes']:
                dataset[split]['boxes'][img_name] = list()

            anno_filename = os.path.join(anno_dir, img_name + ".xml")

            root = ET.parse(anno_filename).getroot()

            for i in range(get_bb_count(root)):
                annotation = []
                annotation.append(cls_name)
                annotation.append(get_int('xmin', root, i) * (width / get_int('width', root)))
                annotation.append(get_int('ymin', root, i) * (height / get_int('height', root)))
                annotation.append(get_int('xmax', root, i) * (width / get_int('width', root)))
                annotation.append(get_int('ymax', root, i) * (height / get_int('height', root)))
                dataset[split]['boxes'][img_name].append(annotation)

    IMAGENET_OBJECTS_SET = sorted(IMAGENET_OBJECTS_SET)
    update_label_map(IMAGENET_OBJECTS_SET)

    return dataset


def write_tf_records(datasets, record_save_dir):
    """
    Write TF records to file
    :param datasets:
    :type datasets:
    :param record_save_dir:
    :type record_save_dir:
    :return:
    :rtype:
    """
    if not os.path.exists(record_save_dir):
        os.mkdir(record_save_dir)

    for split in tqdm(datasets, desc="Splits completed"):
        record_save_path = os.path.join(record_save_dir, split)

        if not os.path.exists(record_save_path):
            os.mkdir(record_save_path)

        record_save_file = os.path.join(record_save_path, 'data.tfrecord')

        writer = tf.python_io.TFRecordWriter(record_save_file)
        for img_name in tqdm(datasets[split]['images'], desc="Writing to file"):
            record = group_to_tf_record(datasets[split]['boxes'][img_name], datasets[split]['images'][img_name], LABEL_STOI)
            if record:
                serialized = record.SerializeToString()
                writer.write(serialized)
        writer.close()


def merge_datasets(dataset_list):
    """
    Merge dataset objects in the list
    :param dataset_list: datasets in list
    :type dataset_list: List[Dict{}]
    :return: Merged dataset
    :rtype: Dict{}
    """
    result_datset = {}
    for dataset in dataset_list:
        for split in dataset:
            if split not in result_datset:
                result_datset[split] = {'images': {}, 'boxes': {}}
            for img_name in dataset[split]['images']:
                result_datset[split]['boxes'][img_name] = dataset[split]['boxes'][img_name]
                result_datset[split]['images'][img_name] = dataset[split]['images'][img_name]

    return result_datset


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


def process_dataset(dataset_dir, dataset_names, class_filter=None):
    """
    Process dataset main method
    :param dataset_dir:
    :type dataset_dir:
    :param dataset_name:
    :type dataset_name:
    :return:
    :rtype:
    """

    datasets = []

    for i, dataset_name in enumerate(dataset_names):

        if dataset_name == OPENIMAGES_DATASET_NAME:
            dataset = process_openimages(dataset_dir[i], class_filter)
        elif dataset_name == IMAGENET_DATASET_NAME:
            dataset = process_imagenet(dataset_dir[i], class_filter)
        elif dataset_name == YOUCOLL_DATASET_NAME:
            dataset = process_youcoll(dataset_dir[i], class_filter)

        datasets.append(dataset)

    if len(datasets) == 1:
        dataset = datasets[0]
        dataset_dir = dataset_dir[0]
        dataset_name = dataset_names[0]
    else:
        dataset = merge_datasets(datasets)
        dataset_dir = RND_DATA_DIR # MERGED_DATA_DIR
        dataset_name = RND_DATASET_NAME # MERGED_DATASET_NAME

    save_label_map(dataset_dir, dataset_name)
    save_dataset(dataset, dataset_dir, dataset_name)
    write_tf_records(dataset, os.path.join(dataset_dir, 'tfrecords'))
    print("Processed", dataset_name, "dataset..")


if __name__ == "__main__":

    main_data_dir = "data"

    # process_dataset([OPENIMAGES_DATA_DIR], [OPENIMAGES_DATASET_NAME])
    # process_dataset([IMAGENET_DATA_DIR], [IMAGENET_DATASET_NAME])

    # process_dataset([OPENIMAGES_DATA_DIR, IMAGENET_DATA_DIR], [OPENIMAGES_DATASET_NAME, IMAGENET_DATASET_NAME],
    #                 class_filter=['bowl'])
    process_dataset([YOUCOLL_DATA_DIR], [YOUCOLL_DATASET_NAME])
