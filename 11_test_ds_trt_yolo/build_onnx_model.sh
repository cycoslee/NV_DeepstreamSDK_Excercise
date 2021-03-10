#!/bin/bash


echo
echo "[yolov2-608.onnx] Build TRT_engine & Get calibration table"
python3 yolo_to_onnx.py --model_file models/yolov2-608.weights -b 8 -n 80
python3 yolo_to_onnx.py --model_file models/yolov2-608.weights -b 4 -n 80


echo
echo "[yolov2-tiny-608.onnx] Build TRT_engine & Get calibration table"
python3 yolo_to_onnx.py --model_file models/yolov2-tiny-608.weights -b 8 -n 80
python3 yolo_to_onnx.py --model_file models/yolov2-tiny-608.weights -b 4 -n 80


echo
echo "[yolov3-608.onnx] Build TRT_engine & Get calibration table"
python3 yolo_to_onnx.py --model_file models/yolov3-608.weights -b 8 -n 80
python3 yolo_to_onnx.py --model_file models/yolov3-608.weights -b 4 -n 80


echo
echo "[yolov3-lite-608.onnx] Build TRT_engine & Get calibration table"
python3 yolo_to_onnx.py --model_file models/yolov3-lite-608.weights -b 8 -n 80
python3 yolo_to_onnx.py --model_file models/yolov3-lite-608.weights -b 4 -n 80


echo
echo "[yolov3-nano-608.onnx] Build TRT_engine & Get calibration table"
python3 yolo_to_onnx.py --model_file models/yolov3-nano-608.weights -b -1 -n 80
python3 yolo_to_onnx.py --model_file models/yolov3-nano-608.weights -b -1 -n 80


echo
echo "[yolov3-spp-608.onnx] Build TRT_engine & Get calibration table"
python3 yolo_to_onnx.py --model_file models/yolov3-spp-608.weights -b -1 -n 80
python3 yolo_to_onnx.py --model_file models/yolov3-spp-608.weights -b -1 -n 80


echo
echo "[yolov3-tiny-608.onnx] Build TRT_engine & Get calibration table"
python3 yolo_to_onnx.py --model_file models/yolov3-tiny-608.weights -b 4 -n 80
python3 yolo_to_onnx.py --model_file models/yolov3-tiny-608.weights -b 4 -n 80


echo
echo "[yolov4-608.onnx] Build TRT_engine & Get calibration table"
python3 yolo_to_onnx.py --model_file models/yolov4-608.weights -b 4 -n 80
python3 yolo_to_onnx.py --model_file models/yolov4-608.weights -b 4 -n 80


echo
echo "[yolov4-csp-608.onnx] Build TRT_engine & Get calibration table"
python3 yolo_to_onnx.py --model_file models/yolov4-csp-608.weights -b 8 -n 80
python3 yolo_to_onnx.py --model_file models/yolov4-csp-608.weights -b 8 -n 80


echo
echo "[yolov4-tiny-608.onnx] Build TRT_engine & Get calibration table"
python3 yolo_to_onnx.py --model_file models/yolov4-tiny-608.weights -b 4 -n 80
python3 yolo_to_onnx.py --model_file models/yolov4-tiny-608.weights -b 4 -n 80


echo
echo "[yolov4x-mish-608.onnx] Build TRT_engine & Get calibration table"
python3 yolo_to_onnx.py --model_file models/yolov4x-mish-608.weights -b -1 -n 80
python3 yolo_to_onnx.py --model_file models/yolov4x-mish-608.weights -b -1 -n 80

