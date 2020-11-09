#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
import os
import io
import logging

import PIL
import tensorflow as tf

from yolov3_tf2.models import YoloV3
from yolov3_tf2.dataset import transform_images

physical_devices = tf.config.experimental.list_physical_devices('GPU')
for physical_device in physical_devices:
    tf.config.experimental.set_memory_growth(physical_device, True)


class ChesterDetector:

    def __init__(self):
        self._flags = {
            "classes": os.path.join(os.path.dirname(__file__), 'data', 'wows.names'),
            "weights": os.path.join(os.path.dirname(__file__), 'checkpoints', 'yolov3_train_20.tf'),
            "size": 416,
            "num_classes": 80
        }

        self._model = YoloV3(classes=self._flags["num_classes"])
        self._model.load_weights(self._flags["weights"]).expect_partial()
        logging.info('[%s] Weights loaded.' % self.__class__)

        self.class_names = [c.strip() for c in open(self._flags["classes"]).readlines()]
        logging.info('[%s] Classes loaded.' % self.class_names)

    def predict(self, image: PIL.Image.Image):
        bytesio = io.BytesIO()
        image.save(bytesio, format='PNG')
        image_raw = tf.image.decode_image(bytesio.getvalue(), channels=3)
        image = tf.expand_dims(image_raw, 0)
        image = transform_images(image, self._flags)

        boxes, scores, classes, nums = self._model(image)

        return boxes, scores, classes, nums


if __name__ == "__main__":
    pass
