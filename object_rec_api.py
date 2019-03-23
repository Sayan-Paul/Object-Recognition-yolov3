#! /usr/bin/env python
"""API to use and import library for object recognition"""

import numpy as np
import tensorflow as tf
import os
from core import utils
from PIL import Image

__author__ = "Sayan Paul"
__email__ = "sayanpau@usc.edu"

IMAGE_H, IMAGE_W = 416, 416

OID_CLASS_NAMES_FILE = os.path.join(os.path.dirname(__file__), 'data', 'oid.names')
IMGNET_CLASS_NAMES_FILE = os.path.join(os.path.dirname(__file__), 'data', 'imgnet.names')
MERGED_CLASS_NAMES_FILE = os.path.join(os.path.dirname(__file__), 'data', 'merged.names')

OID_FROZEN_MODEL_DIR = os.path.join(os.path.dirname(__file__), 'frozen_models', 'OID')
IMGNET_FROZEN_MODEL_DIR = os.path.join(os.path.dirname(__file__), 'frozen_models', 'IMGNET')
MERGED_FROZEN_MODEL_DIR = os.path.join(os.path.dirname(__file__), 'frozen_models', 'MERGED')


class ObjectRecognition:
    """Object Recognition API"""

    def __init__(self, session=None, use_gpu=True, score_thresh=0.3, iou_thresh=0.5, dataset_name='OID'):
        """
        Initiate model for API
        :param session: Tensorflow session variable
        :type session: tensorflow session
        :param use_gpu: Use GPU, default True
        :type use_gpu: Boolean
        :param score_thresh: Box score threshold
        :type score_thresh: float
        :param iou_thresh: IOU threshold for non-max suppression
        :type iou_thresh: float
        :param dataset_name: Dataset name for the model (OID, IMGNET, MERGED)
        :type dataset_name: str
        """

        self.img_h = IMAGE_H
        self.img_w = IMAGE_W
        self.use_gpu = use_gpu
        self.score_thresh = score_thresh
        self.iou_thresh = iou_thresh
        self.dataset_name = dataset_name
        self.sess = None
        self.input_tensors = None
        self.output_tensors = None

        def get_tensors(frozen_model_dir):
            if session is not None:
                sess = session
            else:
                if not use_gpu:
                    cpu_nms_graph = tf.Graph()
                    sess = tf.Session(graph=cpu_nms_graph)
                else:
                    gpu_nms_graph = tf.Graph()
                    sess = tf.Session(graph=gpu_nms_graph)
            if not use_gpu:
                input_tensors, output_tensors = utils.read_pb_return_tensors(cpu_nms_graph,
                                                                             os.path.join(frozen_model_dir,
                                                                                          "yolov3_cpu_nms.pb"),
                                                                             ["Placeholder:0", "concat_9:0",
                                                                              "mul_6:0", "concat_8:0"])
            else:
                input_tensors, output_tensors = utils.read_pb_return_tensors(gpu_nms_graph,
                                                                             os.path.join(frozen_model_dir,
                                                                                          "yolov3_gpu_nms.pb"),
                                                                             ["Placeholder:0", "concat_10:0",
                                                                              "concat_11:0", "concat_8:0",
                                                                              "concat_13:0"])
            return sess, input_tensors, output_tensors

        if dataset_name == 'OID':
            self.classes = utils.read_class_names(OID_CLASS_NAMES_FILE)
            self.sess, self.input_tensors, self.output_tensors = get_tensors(OID_FROZEN_MODEL_DIR)
        elif dataset_name == 'IMGNET':
            self.classes = utils.read_class_names(IMGNET_CLASS_NAMES_FILE)
            self.sess, self.input_tensors, self.output_tensors = get_tensors(IMGNET_FROZEN_MODEL_DIR)
        elif dataset_name == 'MERGED':
            self.classes = utils.read_class_names(MERGED_CLASS_NAMES_FILE)
            self.sess, self.input_tensors, self.output_tensors = get_tensors(MERGED_FROZEN_MODEL_DIR)
        else:
            raise Exception('Unsupported dataset type')

        self.num_classes = len(self.classes)

    def predict_images(self, records):
        """
        Predict boxes, scores and labels for each image records
        :param records:
        :type records:
        :return:
        :rtype:
        """

        result = {'boxes': [], 'probs': [], 'labels': []}

        for record in records:
            preds = self.predict_image(np.expand_dims(record, axis=0))
            result['boxes'].append(preds['boxes'])
            result['probs'].append(preds['probs'])
            result['labels'].append(preds['labels'])

        return result

    def predict_image(self, record):
        """
        Predict boxes, scores and labels for a single image record
        :param record:
        :type record:
        :return:
        :rtype:
        """
        # Pre-process record
        record = self._process_image(record)

        if self.use_gpu:
            boxes, _, labels, probs = self.sess.run(self.output_tensors,
                                                    feed_dict={self.input_tensors: np.expand_dims(record, axis=0)})
        else:
            boxes, scores, probs = self.sess.run(self.output_tensors,
                                                 feed_dict={self.input_tensors: np.expand_dims(record, axis=0)})
            boxes, _, labels, probs = utils.cpu_nms(boxes, scores, probs, self.num_classes,
                                                    score_thresh=self.score_thresh,
                                                    iou_thresh=self.iou_thresh)

        return {'boxes': boxes, 'probs': probs, 'labels': labels}

    def _process_image(self, img):
        """
        Image pre-processing
        :param img: Single image records
        :type img: [HxWx3] numpy array
        :return: Pre-processed image
        :rtype: [self.img_w x self.img_h x 3] numpy array
        """

        img_resized = np.array(np.resize(img, [self.img_w, self.img_h, 3]), dtype=np.float32)
        img_resized = img_resized / 255.
        return img_resized


if __name__ == "__main__":
    model = ObjectRecognition(use_gpu=False, score_thresh=0.4, iou_thresh=0.5)
    image_path = "data/OID/test/Apple/0da61cd490c57814.jpg"
    img = Image.open(image_path)
    predictions = model.predict_image(img)
    image = utils.draw_boxes(img, predictions['boxes'], predictions['labels'], predictions['labels'],
                             model.classes, [IMAGE_H, IMAGE_W], show=True)
