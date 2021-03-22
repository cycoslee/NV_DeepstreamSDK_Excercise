# yolo_to_onnx.py

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
import sys
import cv2
import onnx
import torch
import argparse
import onnxruntime
import numpy as np

from tools.onnx.net.darknet import Darknet

# Convert Darknet to Onnx
def transform_to_onnx(model_file, config_file, batch_size, onnx_path):
    # Building Darknet by torch
    model = Darknet(config_file)
    print('Buliding Darknet from %s... Done!' % (model_file))

    # Check the parsed model
    model.print_network()

    # Loading weights to Darknet
    model.load_weights(model_file)
    print('Loading weights from %s... Done!' % (model_file))

    # Input and Output define
    input_names = ["input"]
    output_names = ['boxes', 'confs']

    onnx_path = "models/onnx/"
    os.makedirs(onnx_path, exist_ok=True)
    model_name_parts = model_file.split('.weights')[0].split('/')[-1].split('-')
    model_name = model_name_parts[0]
    for _str in model_name_parts:
        if _str.isalpha():
            model_name += "-" + _str

    # Exporting ONNX model from Darknet
    if batch_size <= 0:
        x = torch.randn((batch_size, 3, model.height, model.width), requires_grad=True)
        onnx_file_name = "{}_-1_3_{}_{}_dynamic.onnx".format(model_name, model.height, model.width)
        dynamic_axes = {"input": {0: "batch_size"}, "boxes": {0: "batch_size"}, "confs": {0: "batch_size"}}
        # Export the model
        print('Export the onnx model ...')
        torch.onnx.export(model,
                          x,
                          onnx_path + onnx_file_name,
                          export_params=True,
                          opset_version=12,
                          do_constant_folding=True,
                          verbose=True,
                          input_names=input_names, output_names=output_names,
                          dynamic_axes=dynamic_axes)

        print('Onnx model exporting done')
        print('Please check >>> \"%s\"'%(onnx_path + onnx_file_name))
    else:
        x = torch.randn((batch_size, 3, model.height, model.width), requires_grad=True)
        onnx_file_name = "{}_{}_3_{}_{}_static.onnx".format(model_name, batch_size, model.height, model.width)
        torch.onnx.export(model,
                          x,
                          onnx_path + onnx_file_name,
                          export_params=True,
                          opset_version=12,
                          do_constant_folding=True,
                          verbose=True,
                          input_names=input_names, output_names=output_names,
                          dynamic_axes=None)

        print('Onnx model exporting done')
        print('Please check >>> \"%s\"'%(onnx_path + onnx_file_name))
    return onnx_file_name

# Test function for detection with exported onnx model by onnxruntime
def detect(session, image_file, label_file):
    IN_IMAGE_H = session.get_inputs()[0].shape[2]
    IN_IMAGE_W = session.get_inputs()[0].shape[3]

    # Input
    image_src = cv2.imread(input_file)
    resized = cv2.resize(image_src, (IN_IMAGE_W, IN_IMAGE_H), interpolation=cv2.INTER_LINEAR)
    img_in = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    img_in = np.transpose(img_in, (2, 0, 1)).astype(np.float32)
    img_in = np.expand_dims(img_in, axis=0)
    img_in /= 255.0
    print("Shape of the network input: ", img_in.shape)

    # Compute
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: img_in})
    boxes = post_processing(img_in, 0.4, 0.6, outputs)

    class_names = load_class_names(label_file)
    plot_boxes_cv2(image_src, boxes[0], savename='predictions_onnx.jpg', class_names=class_names)

# Main function
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-m', '--model_file', type=str, required=True,
        help=('Put YOLO model file path'
              'Samples are like models/yolovX[-spp|]-[288|416|608].weights'))
    parser.add_argument(
        '-c', '--config_file', type=str, required=True,
        help=('Put YOLO model configuration file path'
              'Samples are like models/yolovX-[288|416|608].cfg'))
    parser.add_argument(
        '-b', '--batch_size', type=int, default=-1,
        help=('Set specific batchsize'
              'For dyamic batch, set 0/-1 (Default)'
              'For static batch  set [1, N]'))
    parser.add_argument(
        '-o', '--output_file', type=str, default="models/onnx/",
        help=('Put Output ONNX file path'
              'Samples are like models/onnx/yolovX_{batch}_{channel}_{height}_{width}_[static|dynamic].onnx'))
    parser.add_argument(
        '-t','--test', default=False, action="store_true",
        help = "Flag to do test with converted onnx model by onnxruntime")
    parser.add_argument(
        '-i', '--input_test', type=str,
        help='Put input file path for test')
    parser.add_argument(
        '-l', '--label_file', type=str,
        help='Put label of classes file path')
    args = parser.parse_args()

    if not os.path.isfile(args.model_file):
        raise SystemExit('ERROR: Model file (%s) not found!' % weights_file_path)
    if not os.path.isfile(args.config_file):
        raise SystemExit('ERROR: Model config file (%s) not found!' % weights_file_path)

    if args.test == False:
        if args.batch_size <= 0:
            # Transform to onnx as dynamicspecified batch size
            transform_to_onnx(args.model_file, args.config_file, args.batch_size, args.output_file)
        else:
            # Transform to onnx as specified batch size
            transform_to_onnx(args.model_file, args.config_file, args.batch_size, args.output_file)
    else:
        if args.input_test == None and args.label_file==None:
            raise SystemExit('ERROR: You need to put options --input_test and --label_file for testing onnx file.')
        # Transform to onnx as demo
        onnx_path_demo = transform_to_onnx(args.model_file, args.config_file, 1, args.output_file)

        # Set ONNXRUNTIME session for inference
        session = onnxruntime.InferenceSession(onnx_path_demo)
        print("The model expects input shape: ", session.get_inputs()[0].shape)

        # Do test
        detect(session, args.input_test, args.label_file)

if __name__ == '__main__':
    main()
