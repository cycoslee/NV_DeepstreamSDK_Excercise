# onnx_to_trt.py

############################################################################
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
# 
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
############################################################################

import os
import time
import sys
import numpy as np
#np.set_printoptions(threshold=sys.maxsize)
import onnx
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import argparse

from tools.calibrator import YOLOEntropyCalibrator
from tools.plugins import get_input_wh, add_yolo_plugins

# TensorRT Config Flag maps 
network_flags = {
    'explicit_batch'    : 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH),
    'explicit_precision': 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_PRECISION),
}
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
builder_flags = {
    'gpu_fallback': trt.BuilderFlag.GPU_FALLBACK,
    'refittable'  : trt.BuilderFlag.REFIT,
    'debug'       : trt.BuilderFlag.DEBUG,
    'strict_types': trt.BuilderFlag.STRICT_TYPES,
    'fp16'        : trt.BuilderFlag.FP16,
    'int8'        : trt.BuilderFlag.INT8,
}

def load_onnx(model_name):
    """Read the ONNX file."""
    if not os.path.isfile(model_name):
        print('ERROR: file (%s) not found!  You might want to run yolo_to_onnx.py first to generate it.' % model_name)
        return None
    else:
        with open(model_name, 'rb') as f:
            return f.read()

# Build TensorRT engine from ONNX Graph function
def build_TRTengine(onnx_file, input_hw, batch, precision, calib_dataset, calib_file_path, core_id, num_category, verbose=False):
    print('calib_dataset: ' + calib_dataset)
    print('input_hw: ', input_hw)
    print('calib_out: ', calib_file_path)

    print("ONNX version: %s" % onnx.__version__)
    print('Loading the ONNX file...')
    onnx_model = load_onnx( onnx_file )
    model_name = onnx_file.split("/")[-1].split(".onnx")[0]
    

    # TensorRT LOGGER 
    TRT_LOGGER=trt.Logger(trt.Logger.VERBOSE) if verbose else trt.Logger()
    # Load TRT plugins from the libnvinfer_plugin library
    trt.init_libnvinfer_plugins(TRT_LOGGER, '')
    network_flag = network_flags['explicit_batch']
    with trt.Builder(TRT_LOGGER) as builder, \
                                    builder.create_network(network_flag) as network, \
                                    builder.create_builder_config() as config, \
                                    trt.OnnxParser(network, TRT_LOGGER) as parser:
        print('Parsing ONNX graph to build TRT graph')
        # Parsing from loaded onnx graph
        if not parser.parse(onnx_model):
            print('ERROR: Failed to parse the ONNX file.')
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None
        
        #print('Adding yolo_layer plugins...')
        #network = add_yolo_plugins( network, model_name, num_category, TRT_LOGGER )

        print('Doing TRT optimizer configuration...')
        # Set bulider config
        builder.max_batch_size = int(batch[2])
        config.max_workspace_size = 1 << 31  # 1GB:30
        config.set_flag(builder_flags['gpu_fallback'])
        # Set Profile for Dynamic Shape and Batch
        profile = builder.create_optimization_profile()
        profile.set_shape(input='000_net', \
                          min=(int(batch[0]), 3, input_hw[0], input_hw[1]), \
                          opt=(int(batch[1]), 3, input_hw[0], input_hw[1]), \
                          max=(int(batch[2]), 3, input_hw[0], input_hw[1])) 
        config.add_optimization_profile(profile)

        if precision & 0b01:
            print('FP16_Configuration')
            config.set_flag(builder_flags['fp16'])
        if precision & 0b10:
            print("INT8 Configuration & EntropyClibration")
            config.set_flag(builder_flags['int8']) 
            config.int8_calibrator = YOLOEntropyCalibrator(
                calib_dataset, input_hw,
                calib_file_path, int(batch[1]))
            print("Set YOLOEntropyCalibrator!!!")
            config.set_calibration_profile(profile)

        if core_id >= 0:
            config.default_device_type = trt.DeviceType.DLA
            config.DLA_core = core_id
            #config.set_flag(trt.BuilderFlag.STRICT_TYPES)
            print('Using DLA%d !!!' % core_id)

        # Check Input/output Tensors
        for j in range(network.num_inputs):
            print("input[\t", j, "] : ", network.get_input(j).name)
        for k in range(network.num_outputs):
            print("output[\t", k, "] : ", network.get_output(k).name)

        # Check TensorRT Network definition
        for i in range(len(network)):
            print(network[i].type)

        # Build TensorRT engine and get its Execution context 
        print('Building a TRT engine.  This would take a while...')
        engine = builder.build_engine(network, config)
        context = engine.create_execution_context()
    return engine, context

# Simple helper data class that's a little nicer to use than a 2-tuple.
class HostDeviceMem(object):
    def __init__(self, name, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem
        self.name = name
    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()

def allocate_buffers(engine, context, inputs):
    """Allocates host and device buffer for TensorRT engine inference.
    Args:
        engine (trt.ICudaEngine): TensorRT engine
        inputs [numpy.array]:  Input Data array
    
    Returns:
        inputs [HostDeviceMem]: Engine input memory
        outputs [HostDeviceMem]: Engine output memory
        bindings [int]: Buffer to device bindings
        stream (cuda.Stream): Cuda stream for engine inference synchronization
    """
    _inputs = []
    _outputs = []
    _bindings = []
    _stream = cuda.Stream()
    
    # Set input shapes to let the TensorRT engine knows
    for i in range(len(inputs)):
        print("Inputs [",i, "] : ", inputs[i].shape)
        context.set_binding_shape(i, inputs[i].shape)

    # Bind Host and Device Memory
    idx_binding = 0 
    for binding in engine:
        print("[Engine] binding Tensor : ", binding)
        print("[Engine] binding_shape( ", engine.get_binding_shape(idx_binding), ")")
        print("[Context] binding_shape( ", context.get_binding_shape(idx_binding), ")\n")

        size = trt.volume(context.get_binding_shape(idx_binding)) * engine.max_batch_size
        dtype = np.float32

        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)

        # Append the device buffer to device bindings.
        _bindings.append(int(device_mem))

        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            _inputs.append(HostDeviceMem(binding, host_mem, device_mem))
        else:
            _outputs.append(HostDeviceMem(binding, host_mem, device_mem))
        idx_binding+=1
        
    return _inputs, _outputs, _bindings, _stream

# TensorRT inference function
# This function is generalized for multiple inputs/outputs.
# inputs and outputs are expected to be lists of HostDeviceMem objects.
def do_inference(context, bindings, inputs, outputs, stream):
    print("Start to run TRT inference...")
    # Transfer input data to the GPU.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]

    # Run inference.
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)

    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]

    # Synchronize the stream
    stream.synchronize()

    # Return only the host outputs.
    print("TRT inference done.")
    return [out.host for out in outputs]

# Main function
def main():
    """Create a TensorRT engine for ONNX-based Model."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i', '--input_model', type=str, required=True,
        help=('Put the ONNX model that you want to convert to TRT engine.'
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
    args = parser.parse_args()

    if (args.precision == "INT8") and (args.calib_dataset is None):
        parser.error("--INT8 needs --calib_dataset.")

    if len(args.batch) != 3:
        parser.error("batch needs to have 3 element [MIN OPT MAX] for dynamic and [MAX MAX MAX] for static")

    if "DLA" in args.gpu_core:
        if not (args.batch[0] == args.batch[2]):
            parser.error("For DLA, batch should be ( MIN == OPT == MAX)")

    inputs    = []
    input_buffers  = []
    output_buffers = []
    bindings = []
    stream   = None

    model_name = args.input_model.split('.onnx')[0].split('/')[-1]

    input_dim = get_input_wh(model_name)
    print("Input_dim: ", input_dim)

    engine_path = "outputs/trt_engine/" 
    calib_path  = "outputs/calib/"
    engine_name = args.gpu_core + "_" + model_name
    calib_name = engine_name + ".calib"
    engine_name = engine_name + ".trt"

    os.makedirs(engine_path, exist_ok=True)
    os.makedirs(calib_path, exist_ok=True)
    trt_engine_path = engine_path + engine_name
    calib_file_path = calib_path + calib_name
    print("engine : %s" % (trt_engine_path))
    print("calib : %s" %(calib_file_path))

    # Build TensorRT engine and get its context
    engine, context = build_TRTengine(args.input_model, (input_dim[1], input_dim[0]), \
            args.batch, precision_flags[args.precision], args.calib_dataset, calib_file_path, \
            core_flage[args.gpu_core], args.num_category, args.verbose)
    if engine is None:
        raise SystemExit('ERROR: failed to build the TensorRT engine!')

    with open(trt_engine_path, 'wb') as f:
        f.write(engine.serialize())
    print('Serialized the TensorRT engine to file: %s' % trt_engine_path)

if __name__ == "__main__":
    main()
