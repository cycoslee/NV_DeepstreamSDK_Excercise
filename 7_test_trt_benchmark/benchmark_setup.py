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

from utils import utilities, setup_argparser
import sys
import os

def main():
    # System Check
    arg_parser = setup_argparser()
    args = arg_parser.make_args()
    system_check = utilities(jetson_devkit=args.jetson_devkit, gpu_freq=args.gpu_freq, dla_freq=args.dla_freq)
    if args.benchmark_mode == 1 :
        print("Set Jetson Benchmark Mode : ON!\n")
        system_check.close_all_notice()
        if system_check.check_trt():
            sys.exit()
        system_check.set_power_mode(args.power_mode, args.jetson_devkit)
        system_check.clear_ram_space()
        if args.jetson_clocks:
            system_check.set_jetson_clocks()
        else:
            system_check.run_set_clocks_withDVFS()
            system_check.set_jetson_fan(255)
    else :
        system_check.clear_ram_space()
        system_check.set_jetson_fan(0)
        print("Set Jetson Benchmark Mode : OFF!\n")

if __name__ == "__main__":
    main()
