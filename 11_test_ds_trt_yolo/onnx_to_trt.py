# onnx_to_trt.py

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

import os
import time
import sys
import argparse

from tools.tensorrt.trt_utils import *
from tools.onnx.onnx_utils import *
from tools.utils import get_input_dim

# Args option flag maps
core_flage = {
    'CUDA' : -1,
    'DLA0' : 0,
    'DLA1' : 1,
}

precision_flags = {
    'FP32'  : 0b00,
    'FP16'  : 0b01,
    'INT8'  : 0b11,
}

# Main function
def main():
    """Create a TensorRT engine for ONNX-based Model."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i', '--input_model', type=str, required=True, help=('Put the ONNX model that you want to convert to TRT engine.'
              'models/onnx/yolov3_b_ch_h_w.onnx'))
    parser.add_argument(
        '-p', '--precision', type=str, required=True,
        help=('[ FP32 | FP16 | INT8 ]'
              'For INT8 quantization, you need to prepare calibration dataset in the \"calib_database/\" folder'))
    parser.add_argument(
        '-c', '--calib_dataset', type=str, default='calib_dataset',
        help=('calib_dataset'
              'which will be used to calculate entropy between value saturation and do not.'))
    parser.add_argument(
        '-g', '--gpu_core', type=str, default='CUDA',
        help=('[ CUDA | DLA0 | DLA1 ]'
              ' Strictly set which core execute the TRT_engine between CUDA core and DLAs.'
              ' You need to check the target specification such as DLA'))
    parser.add_argument(
        '-b', '--batch', nargs='+', required=True,
        help=('dynamic = [ MIN OPT MAX ], static = [MAX MAX MAX]'
               ' Set Dynamic batch sizes for MIN OPT MAX'))
    parser.add_argument(
        '-n', '--num_category', type=int, default=80,
        help='The number of object categories [80]')
    parser.add_argument(
        '-v', '--verbose', action='store_true',
        help='Enable verbose output logs (for debug)')
    parser.add_argument(
        '-j', '--json_for_surgery', nargs='+', type=str,
        help=('Put the json file of ONNX graph surgery requestion format to Add|Remove|Change'
              'node to the target graph before generate TensorRT engine'))
    args = parser.parse_args()

    model_name = None
    onnx_model = None
    dynamic_batch = False

    # Exceptions
    if (args.precision == "INT8") and (args.calib_dataset is None):
        parser.error("--INT8 needs --calib_dataset.")
    if len(args.batch) != 3:
        parser.error("batch needs to have 3 element [MIN OPT MAX] for dynamic and [MAX MAX MAX] for static")
    if not (args.batch[0] == args.batch[2]):
        if "DLA" in args.gpu_core:
            parser.error("For DLA, batch should be ( MIN == OPT == MAX)")
        dynamic_batch = True
        print("Dynamic Batch On  !!!")
    else:
        print("Dynamic batch Off !!!")

    # Version Checking
    print("ONNX version: %s" % onnx_version())
    print("TensorRT version: %s" % tensorrt_version())

    model_file = args.input_model

    # If ONNX graph surgeon option        
    # Graph Surgeon
    if args.json_for_surgery != "":
        print("Graph surgery has been requested by JSON files: \n", args.json_for_surgery)
    gs = GraphSurgery(model_file, args.json_for_surgery, dynamic_batch)
    gs.do_graph_surgeon()
    onnx_model = gs.get_onnx_model()

    model_name = model_file.split('.onnx')[0].split('/')[-1]
    input_dim = get_input_dim(model_name)
    print("Input_dim: ", input_dim)

    engine_path = "trt_output/engine/" 
    calib_path  = "trt_output/calib/"
    engine_name = model_name + "_" + args.gpu_core + "_" + args.precision

    if args.json_for_surgery != "":
        engine_name = engine_name + ".gs"

    calib_name  = engine_name + ".cache"
    engine_name = engine_name + ".trt"

    os.makedirs(engine_path, exist_ok=True)
    os.makedirs(calib_path, exist_ok=True)
    trt_engine_path = engine_path + engine_name
    calib_file_path = calib_path + calib_name
    print("TensorRT engine : %s" % (trt_engine_path))
    print("TensorRT calib  : %s" %(calib_file_path))

    # Build TensorRT engine and get its context
    engine, context = build_TRTengine(onnx_model, (input_dim[2], input_dim[3]), \
            args.batch, precision_flags[args.precision], args.calib_dataset, calib_file_path, \
            core_flage[args.gpu_core], args.num_category, args.verbose)
    if engine is None:
        raise SystemExit('ERROR: failed to build the TensorRT engine!')

    # Save the TensorRT engine
    save_TRTengine(engine, trt_engine_path)
    print("TensorRT engine has benn built and exported to: %s" % (trt_engine_path))

if __name__ == "__main__":
    main()
