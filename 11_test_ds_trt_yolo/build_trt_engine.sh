#!/bin/bash

echo
echo "[yolov2-608.onnx] Build & export TRT model."
python3 onnx_to_trt.py --input_model models/onnx/yolov2-608.onnx --precision INT8 --calib_dataset calib_database --gpu_core CUDA --batch 4 4 4 -v
python3 onnx_to_trt.py --input_model models/onnx/yolov2-608.onnx --precision INT8 --calib_dataset calib_database --gpu_core DLA0 --batch 6 6 6 -v
python3 onnx_to_trt.py --input_model models/onnx/yolov2-608.onnx --precision INT8 --calib_dataset calib_database --gpu_core DLA1 --batch 6 6 6 -v

echo
echo "[yolov2-tiny-608.onnx] Build & export TRT model."
python3 onnx_to_trt.py --input_model models/onnx/yolov2-tiny-608.onnx --precision INT8 --calib_dataset calib_database --gpu_core CUDA --batch 4 4 4 -v
python3 onnx_to_trt.py --input_model models/onnx/yolov2-tiny-608.onnx --precision INT8 --calib_dataset calib_database --gpu_core DLA0 --batch 6 6 6 -v
python3 onnx_to_trt.py --input_model models/onnx/yolov2-tiny-608.onnx --precision INT8 --calib_dataset calib_database --gpu_core DLA1 --batch 6 6 6 -v

echo
echo "[yolov3-608.onnx] Build & export TRT model."
python3 onnx_to_trt.py --input_model models/onnx/yolov3-608.onnx --precision INT8 --calib_dataset calib_database --gpu_core CUDA --batch 6 6 6 -v
python3 onnx_to_trt.py --input_model models/onnx/yolov3-608.onnx --precision INT8 --calib_dataset calib_database --gpu_core DLA0 --batch 6 6 6 -v
python3 onnx_to_trt.py --input_model models/onnx/yolov3-608.onnx --precision INT8 --calib_dataset calib_database --gpu_core DLA1 --batch 6 6 6 -v

echo
echo "[yolov3-lite-608.onnx] Build & export TRT model."
python3 onnx_to_trt.py --input_model models/onnx/yolov3-lite-608.onnx --precision INT8 --calib_dataset calib_database --gpu_core CUDA --batch 4 4 4 -v
python3 onnx_to_trt.py --input_model models/onnx/yolov3-lite-608.onnx --precision INT8 --calib_dataset calib_database --gpu_core DLA0 --batch 6 6 6 -v
python3 onnx_to_trt.py --input_model models/onnx/yolov3-lite-608.onnx --precision INT8 --calib_dataset calib_database --gpu_core DLA1 --batch 6 6 6 -v

echo
echo "[yolov3-nano-608.onnx] Build & export TRT model."
python3 onnx_to_trt.py --input_model models/onnx/yolov3-nano-608.onnx --precision INT8 --calib_dataset calib_database --gpu_core CUDA --batch 4 4 4 -v
python3 onnx_to_trt.py --input_model models/onnx/yolov3-nano-608.onnx --precision INT8 --calib_dataset calib_database --gpu_core DLA0 --batch 6 6 6 -v
python3 onnx_to_trt.py --input_model models/onnx/yolov3-nano-608.onnx --precision INT8 --calib_dataset calib_database --gpu_core DLA1 --batch 6 6 6 -v

echo
echo "[yolov3-spp-608.onnx] Build & export TRT model."
python3 onnx_to_trt.py --input_model models/onnx/yolov3-spp-608.onnx --precision INT8 --calib_dataset calib_database --gpu_core CUDA --batch 4 4 4 -v
python3 onnx_to_trt.py --input_model models/onnx/yolov3-spp-608.onnx --precision INT8 --calib_dataset calib_database --gpu_core DLA0 --batch 6 6 6 -v
python3 onnx_to_trt.py --input_model models/onnx/yolov3-spp-608.onnx --precision INT8 --calib_dataset calib_database --gpu_core DLA1 --batch 6 6 6 -v

echo
echo "[yolov3-tiny-608.onnx] Build & export TRT model."
python3 onnx_to_trt.py --input_model models/onnx/yolov3-tiny-608.onnx --precision INT8 --calib_dataset calib_database --gpu_core CUDA --batch 6 6 6 -v
python3 onnx_to_trt.py --input_model models/onnx/yolov3-tiny-608.onnx --precision INT8 --calib_dataset calib_database --gpu_core DLA0 --batch 6 6 6 -v
python3 onnx_to_trt.py --input_model models/onnx/yolov3-tiny-608.onnx --precision INT8 --calib_dataset calib_database --gpu_core DLA1 --batch 6 6 6 -v

echo
echo "[yolov4-608.onnx] Build & export TRT model."
python3 onnx_to_trt.py --input_model models/onnx/yolov4-608.onnx --precision INT8 --calib_dataset calib_database --gpu_core CUDA --batch 4 4 4 -v
python3 onnx_to_trt.py --input_model models/onnx/yolov4-608.onnx --precision INT8 --calib_dataset calib_database --gpu_core DLA0 --batch 6 6 6 -v
python3 onnx_to_trt.py --input_model models/onnx/yolov4-608.onnx --precision INT8 --calib_dataset calib_database --gpu_core DLA1 --batch 6 6 6 -v

echo
echo "[yolov4-csp-608.onnx] Build & export TRT model."
python3 onnx_to_trt.py --input_model models/onnx/yolov4-csp-608.onnx --precision INT8 --calib_dataset calib_database --gpu_core CUDA --batch 4 4 4 -v
python3 onnx_to_trt.py --input_model models/onnx/yolov4-csp-608.onnx --precision INT8 --calib_dataset calib_database --gpu_core DLA0 --batch 6 6 6 -v
python3 onnx_to_trt.py --input_model models/onnx/yolov4-csp-608.onnx --precision INT8 --calib_dataset calib_database --gpu_core DLA1 --batch 6 6 6 -v

echo
echo "[yolov4-tiny-608.onnx] Build & export TRT model."
python3 onnx_to_trt.py --input_model models/onnx/yolov4-tiny-608.onnx --precision INT8 --calib_dataset calib_database --gpu_core CUDA --batch 6 6 6 -v
python3 onnx_to_trt.py --input_model models/onnx/yolov4-tiny-608.onnx --precision INT8 --calib_dataset calib_database --gpu_core DLA0 --batch 6 6 6 -v
python3 onnx_to_trt.py --input_model models/onnx/yolov4-tiny-608.onnx --precision INT8 --calib_dataset calib_database --gpu_core DLA1 --batch 6 6 6 -v

echo
echo "[yolov4x-mish-608.onnx] Build & export TRT model."
python3 onnx_to_trt.py --input_model models/onnx/yolov4x-mish-608.onnx --precision INT8 --calib_dataset calib_database --gpu_core CUDA --batch 4 4 4 -v
python3 onnx_to_trt.py --input_model models/onnx/yolov4x-mish-608.onnx --precision INT8 --calib_dataset calib_database --gpu_core DLA0 --batch 6 6 6 -v
python3 onnx_to_trt.py --input_model models/onnx/yolov4x-mish-608.onnx --precision INT8 --calib_dataset calib_database --gpu_core DLA1 --batch 6 6 6 -v




