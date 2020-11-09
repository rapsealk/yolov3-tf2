#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
import unittest
import os
import io
import time

import tensorflow as tf
import numpy as np
from PIL import ImageGrab

shape = (1080, 1920, 3)


class TensorFlowImageTestCase(unittest.TestCase):

    def test_pillow_image(self):
        ticks = []
        for _ in range(10):
            perf_time = time.perf_counter()
            bytesio = io.BytesIO()
            ImageGrab.grab().save(bytesio, format='PNG')
            image_raw = tf.image.decode_image(bytesio.getvalue(), channels=3)
            self.assertTrue(image_raw.shape == shape)
            self.assertTrue(image_raw.dtype == tf.uint8)
            tick = time.perf_counter() - perf_time
            ticks.append(tick)
        print('tick.mean:', np.mean(ticks))
        self.assertTrue(np.mean(ticks) < 0.2)

    def test_bytes_image(self):
        perf_time = time.perf_counter()
        path = os.path.join(os.path.dirname(__file__), '..', 'workspace', 'data', '000.png')
        image_raw = tf.image.decode_image(open(path, 'rb').read(), channels=3)
        self.assertTrue(image_raw.shape == shape)
        self.assertTrue(image_raw.dtype == tf.uint8)
        tick = time.perf_counter() - perf_time
        self.assertTrue(tick < 1)


if __name__ == "__main__":
    unittest.main()
