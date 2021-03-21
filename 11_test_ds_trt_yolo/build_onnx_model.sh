#!/bin/bash


echo
echo "Convert Darknet yolov3-603 to ONNX model"
python3 yolo_to_onnx.py -m models/yolov3-608.weights -c models/yolov3-608.cfg -b 8

echo
echo "Convert Darknet yolov3-tiny-603 to ONNX model"
python3 yolo_to_onnx.py -m models/yolov3-tiny-608.weights -c models/yolov3-tiny-608.cfg -b 8

