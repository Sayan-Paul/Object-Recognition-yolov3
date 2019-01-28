#! /usr/bin/env python
"""API to use and import library for object recognition"""

import numpy as np
import tensorflow as tf
from core import utils
from PIL import Image


__author__ = "Sayan Paul"
__email__ = "sayanpau@usc.edu"


IMAGE_H, IMAGE_W = 416, 416
DEFAULT_CLASS_NAMES_FILE = './data/oid.names'


class ObjectRecognition:
    """Object Recognition API"""

    def __init__(self, session=None, use_gpu=True, score_thresh=0.3, iou_thresh=0.5):
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
        """
        self.classes = utils.read_class_names(DEFAULT_CLASS_NAMES_FILE)
        self.num_classes = len(self.classes)
        self.img_h = IMAGE_H
        self.img_w = IMAGE_W
        self.use_gpu = use_gpu
        self.score_thresh = score_thresh
        self.iou_thresh = iou_thresh

        if session is not None:
            self.sess = session
        else:
            if not use_gpu:
                self.cpu_nms_graph = tf.Graph()
                self.sess = tf.Session(graph=self.cpu_nms_graph)
            else:
                self.gpu_nms_graph = tf.Graph()
                self.sess = tf.Session(graph=self.gpu_nms_graph)

        if not use_gpu:
            self.input_tensor, self.output_tensors = utils.read_pb_return_tensors(self.cpu_nms_graph,
                                                                                  "frozen_models/OID/yolov3_cpu_nms.pb",
                                                                                  ["Placeholder:0", "concat_9:0",
                                                                                   "mul_6:0"])
        else:
            self.input_tensor, self.output_tensors = utils.read_pb_return_tensors(self.gpu_nms_graph,
                                                                                  "frozen_models/OID/yolov3_gpu_nms.pb",
                                                                                  ["Placeholder:0", "concat_10:0",
                                                                                   "concat_11:0", "concat_12:0"])

    def predict_image(self, image):
        """
        Predict boxes, scores and labels for a single image record
        :param image:
        :type image:
        :return:
        :rtype:
        """
        return self.predict_images(np.expand_dims(image, axis=0))

    def predict_images(self, records):
        """
        Predict boxes, scores and labels for each image records
        :param records:
        :type records:
        :return:
        :rtype:
        """

        # Pre-process records
        records = np.array(records)
        processed = []
        for i in range(records.shape[0]):
            processed.append(self._process_image(records[i]))
        records = np.array(processed)

        if self.use_gpu:
            return self.sess.run(self.output_tensors, feed_dict={self.input_tensor: records})
        else:
            boxes, scores = self.sess.run(self.output_tensors, feed_dict={self.input_tensor: records})
            return utils.cpu_nms(boxes, scores, self.num_classes, score_thresh=self.score_thresh,
                                 iou_thresh=self.iou_thresh)

    def _process_image(self, img):
        """
        Image pre-processing
        :param img: Single image records
        :type img: [HxWx3] numpy array
        :return: Pre-processed image
        :rtype: [self.img_h x self.img_w x 3] numpy array
        """

        img_resized = np.array(np.resize(img, [self.img_h, self.img_w, 3]), dtype=np.float32)
        img_resized = img_resized / 255.
        return img_resized


if __name__ == "__main__":

    model = ObjectRecognition(use_gpu=False)
    image_path = "data/OpenImages/test/Apple/0da61cd490c57814.jpg"
    img = Image.open(image_path)
    boxes, scores, labels = model.predict_image(img)
    image = utils.draw_boxes(img, boxes, scores, labels, model.classes, [model.img_h, model.img_w], show=True)
