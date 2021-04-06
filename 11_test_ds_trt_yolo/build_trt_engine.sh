#!/bin/bash

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


# (Notice)
#    - For DLA, Batch should be Static batch (MIN==OPT==MAX).


echo
echo
echo "[yolov3-608.onnx] Build & export TRT model."
python3 onnx_to_trt.py -i models/onnx/yolov3_8_3_608_608_static.onnx -p INT8 -c calib_database -g CUDA -b 8 8 8 -v 
python3 onnx_to_trt.py -i models/onnx/yolov3_4_3_608_608_static.onnx -p INT8 -c calib_database -g CUDA -b 4 4 4 -v 
python3 onnx_to_trt.py -i models/onnx/yolov3_8_3_608_608_static.onnx -p INT8 -c calib_database -g DLA0 -b 8 8 8 -v
python3 onnx_to_trt.py -i models/onnx/yolov3_8_3_608_608_static.onnx -p INT8 -c calib_database -g DLA1 -b 8 8 8 -v
echo "[yolov3-608.onnx] Build & export TRT model."
python3 onnx_to_trt.py -i models/onnx/yolov3_8_3_608_608_static.onnx -p INT8 -c calib_database -g CUDA -b 8 8 8 -v -j tools/onnx/graph/surgery/add_batchedNMSPlugin.json
python3 onnx_to_trt.py -i models/onnx/yolov3_4_3_608_608_static.onnx -p INT8 -c calib_database -g CUDA -b 4 4 4 -v -j tools/onnx/graph/surgery/add_batchedNMSPlugin.json
python3 onnx_to_trt.py -i models/onnx/yolov3_8_3_608_608_static.onnx -p INT8 -c calib_database -g DLA0 -b 8 8 8 -v -j tools/onnx/graph/surgery/add_batchedNMSPlugin.json
python3 onnx_to_trt.py -i models/onnx/yolov3_8_3_608_608_static.onnx -p INT8 -c calib_database -g DLA1 -b 8 8 8 -v -j tools/onnx/graph/surgery/add_batchedNMSPlugin.json
echo "[yolov3-608.onnx] Build & export TRT model."
python3 onnx_to_trt.py -i models/onnx/yolov3_-1_3_608_608_dynamic.onnx -p INT8 -c calib_database -g CUDA -b 4 6 8 -v 
echo "[yolov3-608.onnx] Build & export TRT-1model."
python3 onnx_to_trt.py -i models/onnx/yolov3_-1_3_608_608_dynamic.onnx -p INT8 -c calib_database -g CUDA -b 4 6 8 -v -j tools/onnx/graph/surgery/add_batchedNMSPlugin.json


echo
echo
echo "[yolov3s-608.onnx] Build & export TRT model."
python3 onnx_to_trt.py -i models/onnx/yolov3s_4_3_608_608_static.onnx -p INT8 -c calib_database -g CUDA -b 4 4 4 -v
python3 onnx_to_trt.py -i models/onnx/yolov3s_4_3_608_608_static.onnx -p INT8 -c calib_database -g DLA0 -b 4 4 4 -v
python3 onnx_to_trt.py -i models/onnx/yolov3s_4_3_608_608_static.onnx -p INT8 -c calib_database -g DLA1 -b 4 4 4 -v
echo "[yolov3s-608.gs.onnx] Build & export TRT model."
python3 onnx_to_trt.py -i models/onnx/yolov3s_4_3_608_608_static.onnx -p INT8 -c calib_database -g CUDA -b 4 4 4 -v -j tools/onnx/graph/surgery/add_batchedNMSPlugin.json
python3 onnx_to_trt.py -i models/onnx/yolov3s_4_3_608_608_static.onnx -p INT8 -c calib_database -g DLA0 -b 4 4 4 -v -j tools/onnx/graph/surgery/add_batchedNMSPlugin.json
python3 onnx_to_trt.py -i models/onnx/yolov3s_4_3_608_608_static.onnx -p INT8 -c calib_database -g DLA1 -b 4 4 4 -v -j tools/onnx/graph/surgery/add_batchedNMSPlugin.json


echo
echo
echo "[yolov3-tiny-608.onnx] Build & export TRT model."
python3 onnx_to_trt.py -i models/onnx/yolov3-tiny_8_3_608_608_static.onnx -p INT8 -c calib_database -g CUDA -b 8 8 8 -v 
python3 onnx_to_trt.py -i models/onnx/yolov3-tiny_4_3_608_608_static.onnx -p INT8 -c calib_database -g DLA0 -b 4 4 4 -v
python3 onnx_to_trt.py -i models/onnx/yolov3-tiny_4_3_608_608_static.onnx -p INT8 -c calib_database -g DLA1 -b 4 4 4 -v
echo "[yolov3-tiny-608.gs.onnx] Build & export TRT model."
ython3 onnx_to_trt.py -i models/onnx/yolov3-tiny_8_3_608_608_static.onnx -p INT8 -c calib_database -g CUDA -b 8 8 8 -v -j tools/onnx/graph/surgery/add_batchedNMSPlugin.json
python3 onnx_to_trt.py -i models/onnx/yolov3-tiny_4_3_608_608_static.onnx -p INT8 -c calib_database -g DLA0 -b 4 4 4 -v -j tools/onnx/graph/surgery/add_batchedNMSPlugin.json
python3 onnx_to_trt.py -i models/onnx/yolov3-tiny_4_3_608_608_static.onnx -p INT8 -c calib_database -g DLA1 -b 4 4 4 -v -j tools/onnx/graph/surgery/add_batchedNMSPlugin.json


echo
echo
echo "[yolov3s-tiny-608.onnx] Build & export TRT model."
python3 onnx_to_trt.py -i models/onnx/yolov3s-tiny_4_3_608_608_static.onnx -p INT8 -c calib_database -g CUDA -b 4 4 4 -v
python3 onnx_to_trt.py -i models/onnx/yolov3s-tiny_4_3_608_608_static.onnx -p INT8 -c calib_database -g DLA0 -b 4 4 4 -v
python3 onnx_to_trt.py -i models/onnx/yolov3s-tiny_4_3_608_608_static.onnx -p INT8 -c calib_database -g DLA1 -b 4 4 4 -v
echo "[yolov3s-tiny-608.gs.onnx] Build & export TRT model."
python3 onnx_to_trt.py -i models/onnx/yolov3s-tiny_4_3_608_608_static.onnx -p INT8 -c calib_database -g CUDA -b 4 4 4 -v -j tools/onnx/graph/surgery/add_batchedNMSPlugin.json
python3 onnx_to_trt.py -i models/onnx/yolov3s-tiny_4_3_608_608_static.onnx -p INT8 -c calib_database -g DLA0 -b 4 4 4 -v -j tools/onnx/graph/surgery/add_batchedNMSPlugin.json
python3 onnx_to_trt.py -i models/onnx/yolov3s-tiny_4_3_608_608_static.onnx -p INT8 -c calib_database -g DLA1 -b 4 4 4 -v -j tools/onnx/graph/surgery/add_batchedNMSPlugin.json


echo
echo
echo "[yolov3-608-dynamic.onnx] Build & export TRT model."
python3 onnx_to_trt.py -i models/onnx/yolov3_-1_3_608_608_dynamic.onnx -p INT8 -c calib_database -g CUDA -b 4 6 8 -v
echo "[yolov3-608-dynamic.gs.onnx] Build & export TRT model."
python3 onnx_to_trt.py -i models/onnx/yolov3_-1_3_608_608_dynamic.onnx -p INT8 -c calib_database -g CUDA -b 4 6 8 -v -j tools/onnx/graph/surgery/add_batchedNMSPlugin.json


echo
echo
echo "[yolov3-tiny-608-dynamic.onnx] Build & export TRT model."
python3 onnx_to_trt.py -i models/onnx/yolov3-tiny_-1_3_608_608_dynamic.onnx -p INT8 -c calib_database -g CUDA -b 4 6 8 -v
echo "[yolov3-tiny-608-dynamic.gs.onnx] Build & export TRT model."
python3 onnx_to_trt.py -i models/onnx/yolov3-tiny_-1_3_608_608_dynamic.onnx -p INT8 -c calib_database -g CUDA -b 4 6 8 -v -j tools/onnx/graph/surgery/add_batchedNMSPlugin.json


echo
echo
echo "[yolov4-608.onnx] Build & export TRT model."
python3 onnx_to_trt.py -i models/onnx/yolov4_4_3_608_608_static.onnx -p INT8 -c calib_database -g CUDA -b 4 4 4 -v
python3 onnx_to_trt.py -i models/onnx/yolov4_4_3_608_608_static.onnx -p INT8 -c calib_database -g DLA0 -b 4 4 4 -v
python3 onnx_to_trt.py -i models/onnx/yolov4_4_3_608_608_static.onnx -p INT8 -c calib_database -g DLA1 -b 4 4 4 -v
echo "[yolov4-608.gs.onnx] Build & export TRT model."
python3 onnx_to_trt.py -i models/onnx/yolov4_4_3_608_608_static.onnx -p INT8 -c calib_database -g CUDA -b 4 4 4 -v -j tools/onnx/graph/surgery/add_batchedNMSPlugin.json
python3 onnx_to_trt.py -i models/onnx/yolov4_4_3_608_608_static.onnx -p INT8 -c calib_database -g DLA0 -b 4 4 4 -v -j tools/onnx/graph/surgery/add_batchedNMSPlugin.json
python3 onnx_to_trt.py -i models/onnx/yolov4_4_3_608_608_static.onnx -p INT8 -c calib_database -g DLA1 -b 4 4 4 -v -j tools/onnx/graph/surgery/add_batchedNMSPlugin.json


echo
echo
echo "[yolov4s-608.onnx] Build & export TRT model."
python3 onnx_to_trt.py -i models/onnx/yolov4s_4_3_608_608_static.onnx -p INT8 -c calib_database -g CUDA -b 4 4 4 -v
python3 onnx_to_trt.py -i models/onnx/yolov4s_4_3_608_608_static.onnx -p INT8 -c calib_database -g DLA0 -b 4 4 4 -v
python3 onnx_to_trt.py -i models/onnx/yolov4s_4_3_608_608_static.onnx -p INT8 -c calib_database -g DLA1 -b 4 4 4 -v
echo "[yolov4s-608.gs.onnx] Build & export TRT model."
python3 onnx_to_trt.py -i models/onnx/yolov4s_4_3_608_608_static.onnx -p INT8 -c calib_database -g CUDA -b 4 4 4 -v -j tools/onnx/graph/surgery/add_batchedNMSPlugin.json
python3 onnx_to_trt.py -i models/onnx/yolov4s_4_3_608_608_static.onnx -p INT8 -c calib_database -g DLA0 -b 4 4 4 -v -j tools/onnx/graph/surgery/add_batchedNMSPlugin.json
python3 onnx_to_trt.py -i models/onnx/yolov4s_4_3_608_608_static.onnx -p INT8 -c calib_database -g DLA1 -b 4 4 4 -v -j tools/onnx/graph/surgery/add_batchedNMSPlugin.json


echo
echo
echo "[yolov4-tiny-608.onnx] Build & export TRT model."
python3 onnx_to_trt.py -i models/onnx/yolov4-tiny_4_3_608_608_static.onnx -p INT8 -c calib_database -g CUDA -b 4 4 4 -v
python3 onnx_to_trt.py -i models/onnx/yolov4-tiny_4_3_608_608_static.onnx -p INT8 -c calib_database -g DLA0 -b 4 4 4 -v
python3 onnx_to_trt.py -i models/onnx/yolov4-tiny_4_3_608_608_static.onnx -p INT8 -c calib_database -g DLA1 -b 4 4 4 -v
echo "[yolov4-tiny-608.gs.onnx] Build & export TRT model."
python3 onnx_to_trt.py -i models/onnx/yolov4-tiny_4_3_608_608_static.onnx -p INT8 -c calib_database -g CUDA -b 4 4 4 -v -j tools/onnx/graph/surgery/add_batchedNMSPlugin.json
python3 onnx_to_trt.py -i models/onnx/yolov4-tiny_4_3_608_608_static.onnx -p INT8 -c calib_database -g DLA0 -b 4 4 4 -v -j tools/onnx/graph/surgery/add_batchedNMSPlugin.json
python3 onnx_to_trt.py -i models/onnx/yolov4-tiny_4_3_608_608_static.onnx -p INT8 -c calib_database -g DLA1 -b 4 4 4 -v -j tools/onnx/graph/surgery/add_batchedNMSPlugin.json


echo
echo
echo "[yolov4s-tiny-608.onnx] Build & export TRT model."
python3 onnx_to_trt.py -i models/onnx/yolov4s-tiny_4_3_608_608_static.onnx -p INT8 -c calib_database -g CUDA -b 4 4 4 -v
python3 onnx_to_trt.py -i models/onnx/yolov4s-tiny_4_3_608_608_static.onnx -p INT8 -c calib_database -g DLA0 -b 4 4 4 -v
python3 onnx_to_trt.py -i models/onnx/yolov4s-tiny_4_3_608_608_static.onnx -p INT8 -c calib_database -g DLA1 -b 4 4 4 -v
echo "[yolov4s-tiny-608.gs.onnx] Build & export TRT model."
python3 onnx_to_trt.py -i models/onnx/yolov4s-tiny_4_3_608_608_static.onnx -p INT8 -c calib_database -g CUDA -b 4 4 4 -v -j tools/onnx/graph/surgery/add_batchedNMSPlugin.json
python3 onnx_to_trt.py -i models/onnx/yolov4s-tiny_4_3_608_608_static.onnx -p INT8 -c calib_database -g DLA0 -b 4 4 4 -v -j tools/onnx/graph/surgery/add_batchedNMSPlugin.json
python3 onnx_to_trt.py -i models/onnx/yolov4s-tiny_4_3_608_608_static.onnx -p INT8 -c calib_database -g DLA1 -b 4 4 4 -v -j tools/onnx/graph/surgery/add_batchedNMSPlugin.json


echo
echo
echo "[yolov4-608.onnx] Build & export TRT model."
python3 onnx_to_trt.py -i models/onnx/yolov4_-1_3_608_608_dynamic.onnx -p INT8 -c calib_database -g CUDA -b 4 6 8 -v
echo "[yolov4-608.gs.onnx] Build & export TRT model."
python3 onnx_to_trt.py -i models/onnx/yolov4_-1_3_608_608_dynamic.onnx -p INT8 -c calib_database -g CUDA -b 4 6 8 -v -j tools/onnx/graph/surgery/add_batchedNMSPlugin.json


echo
echo
echo "[yolov4-tiny-608.onnx] Build & export TRT model."
python3 onnx_to_trt.py -i models/onnx/yolov4-tiny_-1_3_608_608_dynamic.onnx -p INT8 -c calib_database -g CUDA -b 4 6 8 -v
echo "[yolov4-tiny-608.gs.onnx] Build & export TRT model."
python3 onnx_to_trt.py -i models/onnx/yolov4-tiny_-1_3_608_608_dynamic.onnx -p INT8 -c calib_database -g CUDA -b 4 6 8 -v -j tools/onnx/graph/surgery/add_batchedNMSPlugin.json
