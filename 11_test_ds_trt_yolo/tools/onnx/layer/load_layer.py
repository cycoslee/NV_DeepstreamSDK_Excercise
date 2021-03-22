# load_layer.py

#############################################################################
#  Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.             #
#                                                                           #
#  Licensed under the Apache License, Version 2.0 (the "License");          #
#  you may not use this file except in compliance with the License.         #
#  You may obtain a copy of the License at                                  #
#                                                                           #
#      http://www.apache.org/licenses/LICENSE-2.0                           #
#                                                                           #
#  Unless required by applicable law or agreed to in writing, software      #
#  distributed under the License is distributed on an "AS IS" BASIS,        #
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. #
#  See the License for the specific language governing permissions and      #
#  limitations under the License.                                           #
#############################################################################

from torch import from_numpy

def load_conv(buf, start, conv_model):
    num_w = conv_model.weight.numel()
    num_b = conv_model.bias.numel()
    conv_model.bias.data.copy_(from_numpy(buf[start:start + num_b]));
    start = start + num_b
    conv_model.weight.data.copy_(from_numpy(buf[start:start + num_w]).reshape(conv_model.weight.data.shape));
    start = start + num_w
    return start

def load_conv_bn(buf, start, conv_model, bn_model):
    num_w = conv_model.weight.numel()
    num_b = bn_model.bias.numel()
    bn_model.bias.data.copy_(from_numpy(buf[start:start + num_b]));
    start = start + num_b
    bn_model.weight.data.copy_(from_numpy(buf[start:start + num_b]));
    start = start + num_b
    bn_model.running_mean.copy_(from_numpy(buf[start:start + num_b]));
    start = start + num_b
    bn_model.running_var.copy_(from_numpy(buf[start:start + num_b]));
    start = start + num_b
    conv_model.weight.data.copy_(from_numpy(buf[start:start + num_w]).reshape(conv_model.weight.data.shape));
    start = start + num_w
    return start

def load_fc(buf, start, fc_model):
    num_w = fc_model.weight.numel()
    num_b = fc_model.bias.numel()
    fc_model.bias.data.copy_(from_numpy(buf[start:start + num_b]));
    start = start + num_b
    fc_model.weight.data.copy_(from_numpy(buf[start:start + num_w]));
    start = start + num_w
    return start

