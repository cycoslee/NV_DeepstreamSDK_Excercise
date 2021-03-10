# calibrator.py

############################################################################
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
# 
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
############################################################################

import os
import numpy as np
import cv2
import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt


def _preprocess_yolo(img, input_shape):
    """Preprocess an image before TRT YOLO inferencing.
    # Args
        img: uint8 numpy array of shape either (img_h, img_w, 3)
             or (img_h, img_w)
        input_shape: a tuple of (H, W)
    # Returns
        preprocessed img: float32 numpy array of shape (3, H, W)
    """
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img = cv2.resize(img, (input_shape[1], input_shape[0]))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.transpose((2, 0, 1)).astype(np.float32)
    img /= 255.0
    return img


class YOLOEntropyCalibrator(trt.IInt8EntropyCalibrator2):
    """YOLOEntropyCalibrator
    This class implements TensorRT's IInt8EntropyCalibtrator2 interface.
    It reads all images from the specified directory and generates INT8
    calibration data for YOLO models accordingly.
    """

    def __init__(self, img_dir, net_hw, cache_file, batch_size=1):
        if not os.path.isdir(img_dir):
            raise FileNotFoundError('%s does not exist' % img_dir)
        if len(net_hw) != 2 or net_hw[0] % 32 or net_hw[1] % 32:
            raise ValueError('bad net shape: %s' % str(net_hw))

        super().__init__()  # trt.IInt8EntropyCalibrator2.__init__(self)

        self.img_dir = img_dir
        self.net_hw = net_hw
        self.cache_file = cache_file
        self.batch_size = batch_size
        self.blob_size = 3 * net_hw[0] * net_hw[1] * np.dtype('float32').itemsize * batch_size

        self.jpgs = [f for f in os.listdir(img_dir) if f.endswith('.jpg')]
        # The number "500" is NVIDIA's suggestion.  See here:
        # https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#optimizing_int8_c
        if len(self.jpgs) < 500:
            print('WARNING: found less than 500 images in %s!' % img_dir)
        self.current_index = 0

        # Allocate enough memory for a whole batch.
        self.device_input = cuda.mem_alloc(self.blob_size)

    def __del__(self):
        del self.device_input  # free CUDA memory

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, names):
        if self.current_index + self.batch_size > len(self.jpgs):
            return None
        current_batch = int(self.current_index / self.batch_size)

        batch = []
        for i in range(self.batch_size):
            img_path = os.path.join(
                self.img_dir, self.jpgs[self.current_index + i])
            img = cv2.imread(img_path)
            assert img is not None, 'failed to read %s' % img_path
            batch.append(_preprocess_yolo(img, self.net_hw))
        batch = np.stack(batch)
        assert batch.nbytes == self.blob_size

        cuda.memcpy_htod(self.device_input, np.ascontiguousarray(batch))
        self.current_index += self.batch_size
        return [self.device_input]

    def read_calibration_cache(self):
        # If there is a cache, use it instead of calibrating again.
        # Otherwise, implicitly return None.
        if os.path.exists(self.cache_file):
            with open(self.cache_file, 'rb') as f:
                return f.read()

    def write_calibration_cache(self, cache):
        with open(self.cache_file, 'wb') as f:
            f.write(cache)
