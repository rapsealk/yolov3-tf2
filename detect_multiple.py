#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
import os
import time

import cv2
import numpy as np
import tensorflow as tf
from absl import app, flags, logging
from absl.flags import FLAGS

from yolov3_tf2.models import YoloV3, YoloV3Tiny
from yolov3_tf2.dataset import transform_images, load_tfrecord_dataset
from yolov3_tf2.utils import draw_outputs

# flags.DEFINE_string('classes', './data/coco.names', 'path to classes file')
flags.DEFINE_string('classes', './data/wows.names', 'path to classes file')
# flags.DEFINE_string('weights', './checkpoints/yolov3.tf', 'path to weights file')
flags.DEFINE_string('weights', './checkpoints/yolov3_train_15.tf', 'path to weights file')
flags.DEFINE_boolean('tiny', False, 'yolov3 or yolov3-tiny')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_string('image', './data/girl.png', 'path to input image')
flags.DEFINE_string('tfrecord', None, 'tfrecord instead of image')
flags.DEFINE_string('output', './output.png', 'path to output image')
flags.DEFINE_integer('num_classes', 80, 'number of classes in the model')
# flags.DEFINE_integer('num_classes', 1, 'number of classes in the model')


def main(_argv):
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    for physical_device in physical_devices:
        tf.config.experimental.set_memory_growth(physical_device, True)

    # if FLAGS.tiny:
    # yolo = YoloV3Tiny(classes=FLAGS.num_classes)
    # else:
    yolo = YoloV3(classes=FLAGS.num_classes)

    # weights_path = './checkpoints/yolov3-tiny_train_9.tf'
    weights_path = './checkpoints/yolov3_train_20.tf'

    yolo.load_weights(weights_path).expect_partial()
    logging.info('weights loaded')

    class_names = [c.strip() for c in open(FLAGS.classes).readlines()]
    logging.info('classes loaded')

    """
    if FLAGS.tfrecord:
        dataset = load_tfrecord_dataset(
            FLAGS.tfrecord, FLAGS.classes, FLAGS.size)
        dataset = dataset.shuffle(512)
        img_raw, _label = next(iter(dataset.take(1)))
    else:
        img_raw = tf.image.decode_image(
            open(FLAGS.image, 'rb').read(), channels=3)

    img = tf.expand_dims(img_raw, 0)
    img = transform_images(img, FLAGS.size)
    """

    images = [os.path.join(os.path.dirname(__file__), '..', 'workspace', 'data', '070.png'),
              os.path.join(os.path.dirname(__file__), '..', 'workspace', 'data', '075.png'),
              os.path.join(os.path.dirname(__file__), '..', 'workspace', 'data', '080.png'),
              os.path.join(os.path.dirname(__file__), '..', 'workspace', 'data', '085.png'),
              os.path.join(os.path.dirname(__file__), '..', 'workspace', 'data', '090.png'),
              os.path.join(os.path.dirname(__file__), '..', 'workspace', 'data', '095.png'),
              os.path.join(os.path.dirname(__file__), '..', 'workspace', 'data', '100.png'),
              os.path.join(os.path.dirname(__file__), '..', 'workspace', 'data', '105.png')]

    images = [tf.image.decode_image(open(path, 'rb').read(), channels=3)
              for path in images]

    for i, image in enumerate(images):
        img = tf.expand_dims(image, axis=0)
        img = transform_images(img, FLAGS.size)

        t1 = time.time()
        boxes, scores, classes, nums = yolo(img)
        t2 = time.time()
        logging.info('time: {}'.format(t2 - t1))

        logging.info('detections:')
        for i in range(nums[0]):
            logging.info('\t{}, {}, {}'.format(class_names[int(classes[0][i])],
                                               np.array(scores[0][i]),
                                               np.array(boxes[0][i])))

        img = cv2.cvtColor(image.numpy(), cv2.COLOR_RGB2BGR)
        img = draw_outputs(img, (boxes, scores, classes, nums), class_names)

        path = os.path.join(os.path.dirname(__file__), 'output%02d.png' % (i+1))
        cv2.imwrite(path, img)
        logging.info('output saved to: {}'.format(path))


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
