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

from utils import load_store_engine, read_write_data, utilities, bcolors
import time
import subprocess

class run_benchmark_models():
    def __init__(self, csv_file_path, model_path, precision, benchmark_data):
        self.benchmark_data = benchmark_data
        self.model_path = model_path
        self.precision = precision
        self.wall_time = 0
        self.download_error_flag = False
    def execute(self, read_index):
        wall_start_t0 = time.time()
        self.model_name, _framework, num_devices, ws_gpu, ws_dla, model_input, model_output, self.batch_size_gpu, self.batch_size_dla = self.benchmark_data.benchmark_csv(read_index)
        print('------------' + bcolors.INFO  + 'Executing {}'.format(self.model_name) + bcolors.ENDC + '------------\n')
        framework = self.benchmark_data.framework2ext()
        # Save, Load and Delete Engine
        model_ext = str(self.model_name) + '.' + str(framework)
        self.trt_engine = load_store_engine(model_path=self.model_path, model_name=model_ext, num_devices=num_devices,
                                            batch_size_gpu=self.batch_size_gpu, batch_size_dla=self.batch_size_dla,
                                            precision=self.precision, ws_gpu=ws_gpu, ws_dla=ws_dla,
                                            model_input=model_input, model_output=model_output)

        self.download_error_flag = self.trt_engine.check_downloaded_models(model_name=model_ext, framework=framework)

        if not self.download_error_flag:
            commands, self.models = self.trt_engine.engine_gen()
            # Saving Engine
            self.trt_engine.save_all(commands=commands, models=self.models)
            # Loading Engine Concurrently
            self.trt_engine.load_all(commands=commands, models=self.models)
            wall_start_t1 = time.time()
            self.wall_time = wall_start_t1 - wall_start_t0
        return self.download_error_flag

    def report(self):
        latency_fps, error_log = self.benchmark_data.calculate_fps(models=self.models, batch_size_gpu=self.batch_size_gpu, batch_size_dla=self.batch_size_dla)
        print('--------------------------\n')
        print(bcolors.INFO + 'Model Name: {} \nFPS:{:.2f} \n'.format(self.model_name, latency_fps[3]) + bcolors.ENDC )
        print('--------------------------\n')
        latency_fps[len(latency_fps) - 1] = self.model_name
        return latency_fps, error_log

    def remove(self):
        self.trt_engine.remove_all(models=self.models)
        print('Wall Time for running model (secs): {}\n'.format(self.wall_time))

