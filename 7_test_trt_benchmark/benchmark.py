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

from utils import utilities, read_write_data, benchmark_argparser, run_benchmark_models
import sys
import pandas as pd
import gc
import warnings
warnings.simplefilter("ignore")

def main():
    # Set Parameters
    arg_parser = benchmark_argparser()
    args = arg_parser.make_args()
    csv_file_path = args.csv_file_path
    model_path = args.model_dir
    precision = args.precision

    dummy_check = utilities()
    if dummy_check.check_trt():
        print("Couldn't find TensorRT, please check trtexec in the path(\"/usr/src/tensorrt/bin/trtexec \")")
        sys.exit()
    dummy_check.close_all_warning()

    # Read CSV and Write Data
    benchmark_data = read_write_data(csv_file_path=csv_file_path, model_path=model_path)
    if args.all:
        latency_each_model =[]
        print("Running all benchmarks.. This would take long time if the trt engine is not prepared...")
        for read_index in range (0,len(benchmark_data)):
            gc.collect()
            model = run_benchmark_models(csv_file_path=csv_file_path, model_path=model_path, precision=precision, benchmark_data=benchmark_data)
            download_err = model.execute(read_index=read_index)
            if not download_err:
                # Reading Results
                latency_fps, error_log = model.report()
                latency_each_model.append(latency_fps)
            del gc.garbage[:]
        benchmark_table = pd.DataFrame(latency_each_model, columns=['GPU (ms)', 'DLA0 (ms)', 'DLA1 (ms)', 'FPS', 'Model Name'], dtype=float)
        # Note: GPU, DLA latencies are measured in miliseconds, FPS = Frames per Second
        print(benchmark_table[['Model Name', 'FPS']])
        if args.plot:
            benchmark_data.plot_perf(latency_each_model)

    elif args.model_name == 'inception_v4':
        model = run_benchmark_models(csv_file_path=csv_file_path, model_path=model_path, precision=precision, benchmark_data=benchmark_data)
        download_err = model.execute(read_index=0)
        if not download_err:
            _, error_log = model.report()

    elif args.model_name == 'super_resolution':
        model = run_benchmark_models(csv_file_path=csv_file_path, model_path=model_path, precision=precision, benchmark_data=benchmark_data)
        download_err = model.execute(read_index=2)
        if not download_err:
            _, error_log = model.report()

    elif args.model_name == 'unet':
        model = run_benchmark_models(csv_file_path=csv_file_path, model_path=model_path, precision=precision, benchmark_data=benchmark_data)
        download_err = model.execute(read_index=3)
        if not download_err:
            _, error_log = model.report()

    elif args.model_name == 'tiny-yolov3':
        model = run_benchmark_models(csv_file_path=csv_file_path, model_path=model_path, precision=precision, benchmark_data=benchmark_data)
        download_err = model.execute(read_index=5)
        if not download_err:
            _, error_log = model.report()

    elif args.model_name == 'resnet':
        model = run_benchmark_models(csv_file_path=csv_file_path, model_path=model_path, precision=precision, benchmark_data=benchmark_data)
        download_err = model.execute(read_index=6)
        if not download_err:
            _, error_log = model.report()

    elif args.model_name == 'ssd-mobilenet-v1':
        model = run_benchmark_models(csv_file_path=csv_file_path, model_path=model_path, precision=precision, benchmark_data=benchmark_data)
        download_err = model.execute(read_index=7)
        if not download_err:
            _, error_log = model.report()

if __name__ == "__main__":
    main()
