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

import os
import subprocess
import sys
import time

FNULL = open(os.devnull, 'w')

class bcolors:
    INFO = '\033[92m'
    WARN = '\033[93m'
    ENDC = '\033[0m'

# Class for Utilities (TRT check, Power mode switching)
# https://docs.nvidia.com/jetson/l4t/index.html#page/Tegra%2520Linux%2520Driver%2520Package%2520Development%2520Guide%2Fpower_management_jetson_xavier.html%23wwpID0E0KD0HA
class utilities():
    def __init__(self, jetson_devkit='xavier-nx', gpu_freq='1109250000', dla_freq='1100800000'):
        self.jetson_devkit = jetson_devkit
        self.gpu_freq = gpu_freq
        self.dla_freq = dla_freq
    def set_power_mode(self, power_mode, jetson_devkit):
        power_cmd0 = 'nvpmodel'
        power_cmd1 = str('-m'+str(power_mode))
        subprocess.call('{} {}'.format(power_cmd0, power_cmd1), shell=True,
                        stdout=FNULL)
        print('Setting Jetson {} in max performance mode'.format(jetson_devkit))

    def set_jetson_clocks(self):
        clocks_cmd = 'jetson_clocks'
        subprocess.call('{}'.format(clocks_cmd), shell=True,
                        stdout=FNULL)
        print("Jetson clocks are Set")

    def set_jetson_fan(self, switch_opt):
        fan_cmd = "sh" + " " + "-c" + " " + "'echo" + " " + str(
            switch_opt) + " " + ">" + " " + "/sys/devices/pwm-fan/target_pwm'"
        subprocess.call('{}'.format(fan_cmd), shell=True, stdout=FNULL)

    def run_set_clocks_withDVFS(self):
        if self.jetson_devkit == 'tx2':
            self.set_user_clock(device='gpu')
            self.set_clocks_withDVFS(frequency=self.gpu_freq, device='gpu')
        if self.jetson_devkit == 'nano':
            self.set_user_clock(device='gpu')
            self.set_clocks_withDVFS(frequency=self.gpu_freq, device='gpu')
        if self.jetson_devkit == 'xavier' or self.jetson_devkit == 'xavier-nx':
            self.set_user_clock(device='gpu')
            self.set_clocks_withDVFS(frequency=self.gpu_freq, device='gpu')
            self.set_user_clock(device='dla')
            self.set_clocks_withDVFS(frequency=self.dla_freq, device='dla')

    def set_user_clock(self, device):
        if self.jetson_devkit == 'tx2':
            self.enable_register = "/sys/devices/gpu.0/aelpg_enable"
            self.freq_register = "/sys/devices/gpu.0/devfreq/17000000.gp10b"
        if self.jetson_devkit == 'nano':
            self.enable_register = "/sys/devices/gpu.0/aelpg_enable"
            self.freq_register = "/sys/devices/gpu.0/devfreq/57000000.gpu"
        if self.jetson_devkit == 'xavier' or self.jetson_devkit == 'xavier-nx':
            if device == 'gpu':
                self.enable_register = "/sys/devices/gpu.0/aelpg_enable"
                self.freq_register = "/sys/devices/gpu.0/devfreq/17000000.gv11b"
            elif device == 'dla':
                base_register_dir = "/sys/kernel/debug/bpmp/debug/clk"
                self.enable_register = base_register_dir + "/nafll_dla/mrq_rate_locked"
                self.freq_register = base_register_dir + "/nafll_dla/rate"

    def set_clocks_withDVFS(self, frequency, device):
        from_freq = self.read_internal_register(register=self.freq_register, device=device)
        self.set_frequency(device=device, enable_register=self.enable_register, freq_register=self.freq_register, frequency=frequency, from_freq=from_freq)
        time.sleep(1)
        to_freq = self.read_internal_register(register=self.freq_register, device=device)
        print('{} frequency is set from {} Hz --> to {} Hz'.format(device, from_freq, to_freq))

    def set_frequency(self, device, enable_register, freq_register, frequency, from_freq):
        self.write_internal_register(enable_register, 1)
        if device == 'gpu':
            max_freq_reg = freq_register+"/max_freq"
            min_freq_reg = freq_register+"/min_freq"
            if int(frequency) > int(from_freq):
                self.write_internal_register(max_freq_reg, frequency)
                self.write_internal_register(min_freq_reg, frequency)
            else:
                self.write_internal_register(min_freq_reg, frequency)
                self.write_internal_register(max_freq_reg, frequency)
        elif device =='dla':
            self.write_internal_register(freq_register, frequency)

    def read_internal_register(self, register, device):
        if device == 'gpu':
            register = register+"/cur_freq"
        reg_read = open(register, "r")
        reg_value = reg_read.read().rstrip("\n")
        reg_read.close()
        return reg_value

    def write_internal_register(self, register, value):
        reg_write = open(register, "w")
        reg_write.write("%s" % value)
        reg_write.close()

    def clear_ram_space(self):
        cmd_0 = str("sh" + " " + "-c")
        cmd_1 = str("'echo") + " " + "2" + " " + " >" + " " + "/proc/sys/vm/drop_caches'"
        cmd = cmd_0 + " " + cmd_1
        subprocess.call('{}'.format(cmd), shell=True)

    def close_all_notice(self):
        print( bcolors.INFO + "\nPlease close all other applications before launching target application!!!\n"+ bcolors.ENDC )

    def close_all_warning(self):
        input( bcolors.WARN + "Please close all other applications and Press Enter to continue..." + bcolors.ENDC )

    def check_trt(self):
        if not os.path.isfile('/usr/src/tensorrt/bin/trtexec'):  # Check if TensorRT is installed
            print("Exiting. Check if TensorRT is installed \n Use ``dpkg -l | grep nvinfer`` ")
            return True
        return False
