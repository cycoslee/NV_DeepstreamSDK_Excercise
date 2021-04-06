# onnx_utils.py

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
import onnx
import json
import numpy as np
import onnx_graphsurgeon as gs

from onnx import helper
from jsonschema import validate
from tools.onnx.graph.node.BatchedNMS_TRT import BatchedNMS_TRT
from tools.onnx.graph.node.SimplePAD_TRT import SimplePAD_TRT

# ONNX version check function
def onnx_version():
    return onnx.__version__

# ONNX model loading function for TensorRT parser
def load_onnx(model_name):
    print('Loading the ONNX file...')

    if not os.path.isfile(model_name):
        raise SystemExit('ERROR: Could not find ONNX file in %s'%(model_name))
    else:
        with open(model_name, 'rb') as f:
            return f.read()

# Showing ONNX graph function
def show_onnx_graph(graph):
    print(onnx.helper.printable_graph(graph))

# ONNX graph Validation function
def check_onnx_model(onnx_model):
    onnx.checker.check_model(onnx_model)

# A Class for the ONNX graph surgery 
class GraphSurgery():
    def __init__(self, onnx_model_file, req_jsons, dynamic_batch):
        self.onnx_model_file = onnx_model_file
        self.onnx_model_fixed_file = self.onnx_model_file.split('.onnx')[0] + '_tuned.onnx'
        self.onnx_model = onnx.load(onnx_model_file)
        self.onnx_model_fixed = None
        self.req_jsons  = req_jsons
        self.req_json_dicts = []
        self.graph = gs.import_onnx(self.onnx_model) 
        self.dynamic_batch = dynamic_batch
        self._validate_requests()

    def _simplify_node(self, i, node):
        node_op = node.op
        switcher = {
                'Pad': SimplePAD_TRT
        }
        create_plugin = switcher.get(node_op)
        if create_plugin == None:
            print("This node does not need to be simplified!!!")
        else:
            plugin = create_plugin(self.graph, i)
            self.graph = plugin.change_node()
            print("%s op node has been simplified!!!"%node_op)

    def do_graph_surgeon(self):
        print("Getting new node params")

        for i, node in enumerate( self.graph.nodes ):
            self._simplify_node(i, node)

        for json_dict in self.req_json_dicts:
            print(json_dict)
            if 'add_node_req' in json_dict:
                print("Got add_node_request")
                self._add_node(json_dict['add_node_req'])

            elif "delete_node_req" in json_dict:
                print("Got delete_node_request")
                self._delete_node(json_dict['delete_node_req'])

            elif "change_node_req" in json_dict:
                print("Got change_node_request")
                self._change_node(json_dict['change_node_req'])
            else:
                raise SystemExit('ERROR: It does not support the requested surgeon.')

        self.onnx_model_fixed = gs.export_onnx(self.graph)
        self.onnx_model = onnx.save(self.onnx_model_fixed, self.onnx_model_fixed_file)
        print('Fixed ONNX Graph model has been saved to : ', self.onnx_model_fixed_file)

    def get_onnx_model(self):
        return load_onnx(self.onnx_model_fixed_file)

    def _validate_requests(self):
        json_schema_file = 'tools/onnx/graph/surgery/schema/onnx_graph_surgeon_req.schema.json'
        with open(json_schema_file, "r") as sc_json:
            schema = json.loads(sc_json.read())
            sc_json.close()

        if self.req_jsons is None:
            self.req_jsons=[]

        for _json in self.req_jsons:
            with open(_json, "r") as req_json:
                surgeon_job = json.loads(req_json.read())
                self.req_json_dicts.append(surgeon_job)
                validate(instance=surgeon_job, schema=schema)
                req_json.close()

    def _add_node(self, add_dict):
        print("add_node")
        node_op = add_dict['node_optype']
        input_name = add_dict['input_name']
        op_param = add_dict['node_param']
        node_name = None
        output_name = None

        if 'node_name' in add_dict:
            node_name = add_dict['node_name']
            print(node_name)
        if 'output_name' in add_dict:
            output_name = add_dict['output_name']
            print(output_name)

        switcher = {
                'BatchedNMS_TRT': BatchedNMS_TRT
        }
        create_plugin = switcher.get(node_op, lambda: "Invalid Plugin name for adding node")
        plugin = create_plugin(op_param, self.dynamic_batch)
        self.graph = plugin.add_node(self.graph)

    def _delete_node(self, del_dict):
        #TODO : Need to implement here
        print("delete_node")


    def _change_node(self, chan_dict):
        #TODO : Need to implement here
        print("change_node")




