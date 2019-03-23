#! /usr/bin/env python
"""Preprocessing of image files/data to make it ready for model."""

import numpy as np
import os
import sklearn.model_selection as sk
import cv2
import csv
from tqdm import tqdm
from utils import get_file_list, get_dir_list


__author__ = "Sayan Paul"
__email__ = "sayanpau@usc.edu"


FRAME_STEP = 1
FRAME_MAX_X = 720
FRAME_MAX_Y = 404


def process_youcoll_dataset(directory_path, object_skip_list=None, frame_skip_list=None):
    """

    :param directory_path:
    :type directory_path:
    :return:
    :rtype:
    """
    object_track_dir = os.path.join(directory_path, "annotations")
    video_frames_dir = os.path.join(directory_path, "imgs")

    obj_file_list = get_file_list(object_track_dir, format='.txt')
    video_frames_dir_list = get_dir_list(video_frames_dir, only_top=True)

    frame_obj_map = dict()

    for obj_tr in obj_file_list:
        filename = os.path.basename(obj_tr)
        vid_name, _ = filename.split('.')

        if vid_name not in frame_obj_map:
            frame_obj_map[vid_name] = dict()

        locations = []
        labels = []
        frames = []

        with open(obj_tr, 'r') as anno_file:
            for line in anno_file.readlines():
                data = line.split()

                # Check if annotation not outside view screen
                if data[-4] == '0':
                    if object_skip_list and data[-1].replace('"', '') in object_skip_list:
                        continue
                    if frame_skip_list and frame_skip_list[vid_name] and int(data[-5]) in frame_skip_list[vid_name]:
                        continue
                    locations.append(data[:-1])
                    labels.append(data[-1].replace('"', ''))
                    frames.append(int(data[-5]))

        locations = np.array(locations)

        for i in range(len(frames)):

            obj_name = labels[i]

            box = (int(locations[i][1]), int(locations[i][2]), int(locations[i][3]), int(locations[i][4]))

            if frames[i] not in frame_obj_map[vid_name]:
                frame_obj_map[vid_name][frames[i]] = dict()

            if obj_name not in frame_obj_map[vid_name][frames[i]]:
                frame_obj_map[vid_name][frames[i]][obj_name] = list()

            frame_obj_map[vid_name][frames[i]][obj_name].append(box)

    train_csv = []

    for folder in tqdm(video_frames_dir_list):
        frame_list = get_file_list(folder, format=".jpg")

        for i, frame in enumerate(frame_list):

            img = cv2.imread(frame)
            video_id = os.path.basename(folder)
            frame_id = eval(os.path.basename(frame).split('.')[0])

            if frame_id not in frame_obj_map[video_id]:
                continue

            for obj in frame_obj_map[video_id][frame_id]:

                for box in frame_obj_map[video_id][frame_id][obj]:
                    x1, y1, x2, y2 = [int(n) for n in box]

                    if x1 == x2 or y1 == y2 or FRAME_MAX_X < x1 or FRAME_MAX_X < x2 or FRAME_MAX_Y < y1 \
                            or FRAME_MAX_Y < y2 or x1 < 0 or x2 < 0 or y1 < 0 or y2 < 0:
                        # Skip invalid frame
                        continue

                    crop_img = img[y1:y2, x1:x2]
                    if crop_img is None or len(crop_img) == 0:
                        continue

                    train_csv.append([frame, box, obj])

    with open(os.path.join(directory_path, 'train_frames.csv'), mode='w', newline='') as train_file:
        train_writer = csv.writer(train_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        train_writer.writerows(train_csv)

    train_valid_split(directory_path, test_split=True)


def train_valid_split(directory_path, test_split=False, split_size=0.1):
    """

    :param directory_path:
    :type directory_path:
    :return:
    :rtype:
    """
    with open(os.path.join(directory_path, 'train_frames.csv')) as train_file:
        csv_reader = csv.reader(train_file, delimiter=',')
        all_data = np.array(list(csv_reader))
        data = all_data[:, :-1]
        labels = all_data[:, -1]
        train_data, valid_data, train_label, valid_label = sk.train_test_split(data, labels, test_size=split_size,
                                                                             random_state=42,
                                                                             stratify=labels)
        valid_csv = np.hstack((valid_data, valid_label[:, np.newaxis]))
        train_csv = np.hstack((train_data, train_label[:, np.newaxis]))

        with open(os.path.join(directory_path, 'train_frames.csv'), mode='w', newline='') as new_train_file:
            train_writer = csv.writer(new_train_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            train_writer.writerows(train_csv)

        if test_split:
            out_csv = 'test_frames.csv'
        else:
            out_csv = 'valid_frames.csv'

        with open(os.path.join(directory_path, out_csv), mode='w', newline='') as valid_file:
            test_writer = csv.writer(valid_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            test_writer.writerows(valid_csv)


if __name__ == "__main__":

    data_dir = "data/YOUCOLL"

    process_youcoll_dataset(data_dir, object_skip_list=['fork', 'spatula'], frame_skip_list={'v_1': range(4867, 4920),
                                                                                             'v_2': None})
    # train_valid_split(data_dir)
