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

#echo
#echo "[yolov2-608.onnx] Build & export TRT model."
#python3 onnx_to_trt.py --input_model models/onnx/yolov2-608.onnx --precision INT8 --calib_dataset calib_database --gpu_core CUDA --batch 4 4 4 -v
#python3 onnx_to_trt.py --input_model models/onnx/yolov2-608.onnx --precision INT8 --calib_dataset calib_database --gpu_core DLA0 --batch 6 6 6 -v
#python3 onnx_to_trt.py --input_model models/onnx/yolov2-608.onnx --precision INT8 --calib_dataset calib_database --gpu_core DLA1 --batch 6 6 6 -v
#
#echo
#echo "[yolov2-tiny-608.onnx] Build & export TRT model."
#python3 onnx_to_trt.py --input_model models/onnx/yolov2-tiny-608.onnx --precision INT8 --calib_dataset calib_database --gpu_core CUDA --batch 4 4 4 -v
#python3 onnx_to_trt.py --input_model models/onnx/yolov2-tiny-608.onnx --precision INT8 --calib_dataset calib_database --gpu_core DLA0 --batch 6 6 6 -v
#python3 onnx_to_trt.py --input_model models/onnx/yolov2-tiny-608.onnx --precision INT8 --calib_dataset calib_database --gpu_core DLA1 --batch 6 6 6 -v
#
echo
echo "[yolov3-608.onnx] Build & export TRT model."
python3 onnx_to_trt.py -i models/onnx/yolov3_8_3_608_608_static.onnx -p INT8 -c calib_database -g CUDA -b 8 8 8 -v -s -j tools/tensorrt/onnx/graph/surgery/add_batchedNMSPlugin.json
#python3 onnx_to_trt.py --input_model models/onnx/yolov3-608.onnx --precision INT8 --calib_dataset calib_database --gpu_core DLA0 --batch 6 6 6 -v
#python3 onnx_to_trt.py --input_model models/onnx/yolov3-608.onnx --precision INT8 --calib_dataset calib_database --gpu_core DLA1 --batch 6 6 6 -v

#echo
#echo "[yolov3-tiny-608.onnx] Build & export TRT model."
#python3 onnx_to_trt.py --input_model models/onnx/yolov3-tiny-608.onnx --precision INT8 --calib_dataset calib_database --gpu_core CUDA --batch 4 4 4 -v
#python3 onnx_to_trt.py --input_model models/onnx/yolov3-tiny-608.onnx --precision INT8 --calib_dataset calib_database --gpu_core DLA0 --batch 6 6 6 -v
#python3 onnx_to_trt.py --input_model models/onnx/yolov3-tiny-608.onnx --precision INT8 --calib_dataset calib_database --gpu_core DLA1 --batch 6 6 6 -v

#1echo
#1echo "[yolov3-lite-608.onnx] Build & export TRT model."
#1python3 onnx_to_trt.py --input_model models/onnx/yolov3-lite-608.onnx --precision INT8 --calib_dataset calib_database --gpu_core CUDA --batch 4 4 4 -v
#1python3 onnx_to_trt.py --input_model models/onnx/yolov3-lite-608.onnx --precision INT8 --calib_dataset calib_database --gpu_core DLA0 --batch 6 6 6 -v
#1python3 onnx_to_trt.py --input_model models/onnx/yolov3-lite-608.onnx --precision INT8 --calib_dataset calib_database --gpu_core DLA1 --batch 6 6 6 -v
#1
#1echo
#1echo "[yolov3-nano-608.onnx] Build & export TRT model."
#1python3 onnx_to_trt.py --input_model models/onnx/yolov3-nano-608.onnx --precision INT8 --calib_dataset calib_database --gpu_core CUDA --batch 4 4 4 -v
#1python3 onnx_to_trt.py --input_model models/onnx/yolov3-nano-608.onnx --precision INT8 --calib_dataset calib_database --gpu_core DLA0 --batch 6 6 6 -v
#1python3 onnx_to_trt.py --input_model models/onnx/yolov3-nano-608.onnx --precision INT8 --calib_dataset calib_database --gpu_core DLA1 --batch 6 6 6 -v
#1
#1echo
#1echo "[yolov3-spp-608.onnx] Build & export TRT model."
#1python3 onnx_to_trt.py --input_model models/onnx/yolov3-spp-608.onnx --precision INT8 --calib_dataset calib_database --gpu_core CUDA --batch 4 4 4 -v
#1python3 onnx_to_trt.py --input_model models/onnx/yolov3-spp-608.onnx --precision INT8 --calib_dataset calib_database --gpu_core DLA0 --batch 6 6 6 -v
#1python3 onnx_to_trt.py --input_model models/onnx/yolov3-spp-608.onnx --precision INT8 --calib_dataset calib_database --gpu_core DLA1 --batch 6 6 6 -v
#1

#echo
#echo "[yolov4-608.onnx] Build & export TRT model."
#python3 onnx_to_trt.py --input_model models/onnx/yolov4-608.onnx --precision INT8 --calib_dataset calib_database --gpu_core CUDA --batch 4 4 4 -v
#python3 onnx_to_trt.py --input_model models/onnx/yolov4-608.onnx --precision INT8 --calib_dataset calib_database --gpu_core DLA0 --batch 6 6 6 -v
#python3 onnx_to_trt.py --input_model models/onnx/yolov4-608.onnx --precision INT8 --calib_dataset calib_database --gpu_core DLA1 --batch 6 6 6 -v
#
#echo
#echo "[yolov4-tiny-608.onnx] Build & export TRT model."
#python3 onnx_to_trt.py --input_model models/onnx/yolov4-tiny-608.onnx --precision INT8 --calib_dataset calib_database --gpu_core CUDA --batch 6 6 6 -v
#python3 onnx_to_trt.py --input_model models/onnx/yolov4-tiny-608.onnx --precision INT8 --calib_dataset calib_database --gpu_core DLA0 --batch 6 6 6 -v
#python3 onnx_to_trt.py --input_model models/onnx/yolov4-tiny-608.onnx --precision INT8 --calib_dataset calib_database --gpu_core DLA1 --batch 6 6 6 -v

#echo
#echo "[yolov4-csp-608.onnx] Build & export TRT model."
#python3 onnx_to_trt.py --input_model models/onnx/yolov4-csp-608.onnx --precision INT8 --calib_dataset calib_database --gpu_core CUDA --batch 4 4 4 -v
#python3 onnx_to_trt.py --input_model models/onnx/yolov4-csp-608.onnx --precision INT8 --calib_dataset calib_database --gpu_core DLA0 --batch 6 6 6 -v
#python3 onnx_to_trt.py --input_model models/onnx/yolov4-csp-608.onnx --precision INT8 --calib_dataset calib_database --gpu_core DLA1 --batch 6 6 6 -v
#
#
#echo
#echo "[yolov4x-mish-608.onnx] Build & export TRT model."
#python3 onnx_to_trt.py --input_model models/onnx/yolov4x-mish-608.onnx --precision INT8 --calib_dataset calib_database --gpu_core CUDA --batch 4 4 4 -v
#python3 onnx_to_trt.py --input_model models/onnx/yolov4x-mish-608.onnx --precision INT8 --calib_dataset calib_database --gpu_core DLA0 --batch 6 6 6 -v
#python3 onnx_to_trt.py --input_model models/onnx/yolov4x-mish-608.onnx --precision INT8 --calib_dataset calib_database --gpu_core DLA1 --batch 6 6 6 -v




