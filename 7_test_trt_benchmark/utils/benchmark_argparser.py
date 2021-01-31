#!/usr/bin/python

# Copyright (c) 2018-2021, NVIDIA CORPORATION. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import argparse

class benchmark_argparser():
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='')
        self.parser.add_argument('--csv_file_path', dest='csv_file_path', help='csv for model download and parameters', type=str)
        self.parser.add_argument('--model_dir', dest='model_dir', help='path to downloaded path', type=str)
        benchmark_group = self.parser.add_mutually_exclusive_group()
        benchmark_group.add_argument('--model_name', dest='model_name', help='only specified models will be executed', type=str)
        benchmark_group.add_argument('--all', dest='all', help='all models from DropBox will be downloaded',
                                      action='store_true')
        self.parser.add_argument('--precision', dest='precision', default='int8',
                                 help='precision for model int8 or fp16', type=str)
        # For Jetson Xavier: set to int8
        # For Jetson TX2: set to 3 fp16
        # For Jetson Nano: set to fp16
        self.parser.add_argument('--plot', dest='plot', help='Perf in Graph', action='store_true')
    def make_args(self):
        return self.parser.parse_args()
