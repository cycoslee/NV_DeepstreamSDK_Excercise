# SimplePAD_TRT.py

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


import onnx_graphsurgeon as gs
import numpy as np

# SimplePAD_TRT Plugin node class
class SimplePAD_TRT():
    def __init__(self, graph, idx_node):
        self.plugin_name='Pad'
        self.rdd_nodes = 13
        self.graph = graph
        self.idx_node = idx_node
        self.trt_pad_values = [0]*4*2 #4d tensor and 2 stride padding for each dimension
        self.attrs = self._create_attributes(graph, idx_node)


    # Prepare attributes for this SimplePAD_TRT plugin
    def _create_attributes(self, graph, i):
        print("graph.nodes[%d] = "%i, graph.nodes[i].attrs)
        self.rdd_nodes += 1

        _attrs= {}
        _attrs['mode']  = graph.nodes[i].attrs['mode']
        _attrs['value'] = 0
        onnx_pad_values = graph.nodes[i - self.rdd_nodes+1].inputs[0].values
        j = 3
        for i in range(0, len(onnx_pad_values), 2):
            self.trt_pad_values[j] = onnx_pad_values[i]
            self.trt_pad_values[j+4] = onnx_pad_values[i+1]
            j -= 1

        _attrs['pads']  = self.trt_pad_values
        print("Attrs :", _attrs)
        return _attrs

    # Graph Surgery - Add
    def add_node(self, graph):
        print("Start adding node(op = %s)"%(self.plugin_name))
        print("ADD graph node surgery complete")
        return True

    # Graph Surgery - Delete
    def delete_node(self, graph):
        print("Start deleting node(layer) ... ")
        #TODO : Need to implement here
        print("ADD graph node surgery complete")
        return True

    # Graph Surgery - Change
    def change_node(self):
        print("change_node")
        for i in range(1, len(self.graph.nodes[self.idx_node].inputs)-1):
            del self.graph.nodes[self.idx_node].inputs[i]
        for removed_node in self.graph.nodes[self.idx_node-self.rdd_nodes:self.rdd_nodes]:
            removed_node.outputs.clear()
        pads_folded_tensor = gs.Constant(name = self.graph.nodes[self.idx_node].name, values= np.array(self.trt_pad_values))
        self.graph.nodes[self.idx_node].inputs[1] = pads_folded_tensor
        #self.graph.nodes[self.idx_node].attrs = self.attrs


        return self.graph 

