# trt_utils.py

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

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np

from tools.tensorrt.calibrator import YOLOEntropyCalibrator

# TensorRT Config Flag maps 
network_flags = {
    'explicit_batch'    : 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH),
    'explicit_precision': 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_PRECISION),
}
builder_flags = {
    'gpu_fallback': trt.BuilderFlag.GPU_FALLBACK,
    'refittable'  : trt.BuilderFlag.REFIT,
    'debug'       : trt.BuilderFlag.DEBUG,
    'strict_types': trt.BuilderFlag.STRICT_TYPES,
    'fp16'        : trt.BuilderFlag.FP16,
    'int8'        : trt.BuilderFlag.INT8,
}

def tensorrt_version():
    return trt.__version__

def save_TRTengine(trt_engine, trt_engine_path):
    with open(trt_engine_path, 'wb') as f:
        f.write(trt_engine.serialize())
    f.close()
    print('Serialized the TensorRT engine to file: %s' % trt_engine_path)

# Build TensorRT engine from ONNX Graph function
def build_TRTengine(onnx_model, input_hw, batch, precision, calib_dataset, calib_file_path, core_id, num_category, verbose=False):
    print('calib_dataset: ' + calib_dataset)
    print('input_hw: ', input_hw)
    print('calib_out: ', calib_file_path)

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
        
        print('Doing TRT optimizer configuration...')
        # Set bulider config
        builder.max_batch_size = int(batch[2])
        config.max_workspace_size = 1 << 31  # 1GB:30
        config.set_flag(builder_flags['gpu_fallback'])
        # Set Profile for Dynamic Shape and Batch
        profile = builder.create_optimization_profile()
        profile.set_shape(input='input', \
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
