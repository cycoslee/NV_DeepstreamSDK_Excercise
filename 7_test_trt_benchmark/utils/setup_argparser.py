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

class setup_argparser():
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='')
        self.parser.add_argument('--benchmark_mode', dest='benchmark_mode', default=1, help='Set Benchmark model ( ON = 1 / OFF = 0 )', type=int)
        self.parser.add_argument('--jetson_devkit', dest='jetson_devkit', default='xavier-nx', help='Input Jetson Devkit name', type=str)
        # For Jetson Xavier: set to 'xavier'
        # For Jetson TX2: set to 'tx2'
        # For Jetson Nano: set to 'nano'
        self.parser.add_argument('--power_mode', dest='power_mode', help='Jetson Power Mode', default=0, type=int)
        # For Jetson Xavier: set to 0 (MAXN)
        # For Jetson TX2: set to 3 (MAXP)
        # For Jetson Nano: set to 0 (MAXN)
        self.parser.add_argument('--jetson_clocks', dest='jetson_clocks', help='Set Clock Frequency to Max (jetson_clocks)',
                                      action='store_true')
        self.parser.add_argument('--gpu_freq', dest='gpu_freq', default=1109250000,help='set GPU frequency', type=int)
        # Default values are for Xavier-NX
        # For Xavier set gpu_freq to 1377000000: Find using  $sudo cat /sys/devices/17000000.gv11b/devfreq/17000000.gv11b/available_frequencies
        # For TX2 set gpu freq to 1300500000: Find using $sudo cat /sys/devices/gpu.0/devfreq/17000000.gp10b/available_frequencies
        # For Nano set gpu freq to 921600000: Find using $sudo cat /sys/devices/gpu.0/devfreq/57000000.gpu/available_frequencies
        self.parser.add_argument('--dla_freq', dest='dla_freq', default=1100800000, help='set DLA frequency', type=int)
        # Default values are for Xavier-NX
        # For Xavier set dla_freq to 1395200000 : Find using $sudo cat /sys/kernel/debug/bpmp/debug/clk/nafll_dla/max_rate

    def make_args(self):
        return self.parser.parse_args()
