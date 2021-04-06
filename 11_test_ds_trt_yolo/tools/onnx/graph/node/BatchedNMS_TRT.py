# BatchedNMS_TRT.py

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

# BatchedNMS_TRT Plugin node class
class BatchedNMS_TRT():
    def __init__(self, params, dynamic_batch):
        self.plugin_name='BatchedNMS_TRT'
        self.shareLocation=1
        self.backgroundLabelId=-1
        self.numClasses=80
        self.numAnchors=3
        self.topK=2000
        self.keepTopK=200
        self.scoreThreshold=0.4
        self.iouThreshold=0.6
        self.isNormalized=True
        self.clipBoxes=True
        self.plugin_version=1
        self.dynamicBatch=dynamic_batch
        self.attrs = self._create_attributes(params)

    # Prepare attributes for this BatchedNMS_TRT plugin
    def _create_attributes(self, params):
        self.shareLocation     = params['shareLocation']
        self.backgroundLabelId = params['backgroundLabelId']
        self.numClasses        = params['numClasses']
        self.numAnchors        = params['numAnchors']
        self.topK              = params['topK']
        self.keepTopK          = params['keepTopK']
        self.scoreThreshold    = params['scoreThreshold']
        self.iouThreshold      = params['iouThreshold']
        self.isNormalized      = params['isNormalized']
        self.clipBoxes         = params['clipBoxes']
        #if self.dynamicBatch is True:
        #    self.plugin_name = 'BatchedNMSDynamic_TRT'
        self.plugin_name = 'BatchedNMSDynamic_TRT'

        attrs= {}
        attrs["shareLocation"]     = self.shareLocation
        attrs["backgroundLabelId"] = self.backgroundLabelId
        attrs["numClasses"]        = self.numClasses
        attrs["topK"]              = self.topK
        attrs["keepTopK"]          = self.keepTopK
        attrs["scoreThreshold"]    = self.scoreThreshold
        attrs["iouThreshold"]      = self.iouThreshold
        attrs["isNormalized"]      = self.isNormalized
        attrs["clipBoxes"]         = self.clipBoxes
        attrs["plugin_version"]    = "1"
        print("Attrs :", attrs)
        return attrs

    # Graph Surgery - Add
    def add_node(self, graph):
        print("Start adding node(op = %s)"%(self.plugin_name))
        batch_size = graph.inputs[0].shape[0]
        input_h = graph.inputs[0].shape[2]
        input_w = graph.inputs[0].shape[3]
        print("width %d, height %d"%(input_h, input_w))
        print("target Plugin: %s"%(self.plugin_name))

        tensors = graph.tensors()
        boxes_tensor = tensors['boxes']
        confs_tensor = tensors['confs']

        num_detections = gs.Variable(name="num_detections").to_variable(dtype=np.int32, shape=[batch_size, 1])
        nms_boxes = gs.Variable(name="nms_boxes").to_variable(dtype=np.float32, shape=[batch_size, self.keepTopK, 4])
        nms_scores = gs.Variable(name="nms_scores").to_variable(dtype=np.float32, shape=[batch_size, self.keepTopK])
        nms_classes = gs.Variable(name="nms_classes").to_variable(dtype=np.float32, shape=[batch_size, self.keepTopK])

        outputs = [num_detections, nms_boxes, nms_scores, nms_classes]

        nms_node = gs.Node(
            op=self.plugin_name,
            attrs=self.attrs,
            inputs=[boxes_tensor, confs_tensor],
            outputs=outputs)

        graph.nodes.append(nms_node)
        graph.outputs = outputs
        print("ADD graph node surgery complete")
        return graph.cleanup().toposort()

    # Graph Surgery - Delete
    def delete_node(self, graph):
        print("Start deleting node(layer) ... ")
        #TODO : Need to implement here
        print("ADD graph node surgery complete")
        return True

    # Graph Surgery - Change
    def change_node(self, graph):
        print("change_node")
        #TODO : Need to implement here
        return True

