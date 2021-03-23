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

echo
echo "Convert Darknet yolov3-608 to ONNX model"
python3 yolo_to_onnx.py -m models/yolov3-608.weights -c models/yolov3-608.cfg -b 8  # For static batch 8
python3 yolo_to_onnx.py -m models/yolov3-608.weights -c models/yolov3-608.cfg -b 4  # For static batch 4
python3 yolo_to_onnx.py -m models/yolov3-608.weights -c models/yolov3-608.cfg -b -1 # For Dynamic batch 

echo
echo "Convert Darknet yolov3-tiny-608 to ONNX model"
python3 yolo_to_onnx.py -m models/yolov3-tiny-608.weights -c models/yolov3-tiny-608.cfg -b 8  # For static batch 8
python3 yolo_to_onnx.py -m models/yolov3-tiny-608.weights -c models/yolov3-tiny-608.cfg -b 4  # For static batch 4
python3 yolo_to_onnx.py -m models/yolov3-tiny-608.weights -c models/yolov3-tiny-608.cfg -b -1 # For Dynamic batch 

echo
echo "Convert Darknet yolov4-608 to ONNX model"
python3 yolo_to_onnx.py -m models/yolov4-608.weights -c models/yolov4-608.cfg -b 8  # For static batch 8
python3 yolo_to_onnx.py -m models/yolov4-608.weights -c models/yolov4-608.cfg -b 4  # For static batch 4
python3 yolo_to_onnx.py -m models/yolov4-608.weights -c models/yolov4-608.cfg -b -1 # For Dynamic batch 

echo
echo "Convert Darknet yolov4-tiny-608 to ONNX model"
python3 yolo_to_onnx.py -m models/yolov4-tiny-608.weights -c models/yolov4-tiny-608.cfg -b 8  # For static batch 8  
python3 yolo_to_onnx.py -m models/yolov4-tiny-608.weights -c models/yolov4-tiny-608.cfg -b 4  # For static batch 4
python3 yolo_to_onnx.py -m models/yolov4-tiny-608.weights -c models/yolov4-tiny-608.cfg -b -1 # For Dynamic batch 
