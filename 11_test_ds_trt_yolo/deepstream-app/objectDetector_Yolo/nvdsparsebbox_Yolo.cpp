/*
 * Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <unordered_map>
#include "nvdsinfer_custom_impl.h"

static const int NUM_CLASSES_YOLO = 80;

float clamp(const float val, const float minVal, const float maxVal)
{
    assert(minVal <= maxVal);
    return std::min(maxVal, std::max(minVal, val));
}

extern "C" bool NvDsInferParseCustomYoloJonghwanl(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferParseObjectInfo>& objectList);

extern "C" bool NvDsInferParseCustomYoloJonghwanl_withNMS(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferParseObjectInfo>& objectList);

static void addBBoxProposalYoloJonghwanl(const float bx1, const float by1, const float bx2, const float by2,
                     const uint& netW, const uint& netH, const int maxIndex,
                     const float maxProb, std::vector<NvDsInferParseObjectInfo>& binfo){

    NvDsInferParseObjectInfo box;
    
    // Restore coordinates to network input resolution
    float x1 = bx1 * netW;
    float y1 = by1 * netH;
    float x2 = bx2 * netW;
    float y2 = by2 * netH;

    x1 = clamp(x1, 0, netW);
    y1 = clamp(y1, 0, netH);
    x2 = clamp(x2, 0, netW);
    y2 = clamp(y2, 0, netH);

    box.left = x1;
    box.width = clamp(x2 - x1, 0, netW);
    box.top = y1;
    box.height = clamp(y2 - y1, 0, netH);

    if (box.width < 1 || box.height < 1) return;

    box.detectionConfidence = maxProb;
    box.classId = maxIndex;
    binfo.push_back(box);
}


static std::vector<NvDsInferParseObjectInfo>
decodeYoloJonghwanlTensor(
    const float* boxes, const float* scores,
    const uint num_bboxes, NvDsInferParseDetectionParams const& detectionParams,
    const uint& netW, const uint& netH) {

    std::vector<NvDsInferParseObjectInfo> binfo;

    uint bbox_location = 0;
    uint score_location = 0;
    for (uint b = 0; b < num_bboxes; ++b) {
        float bx1 = boxes[bbox_location];
        float by1 = boxes[bbox_location + 1];
        float bx2 = boxes[bbox_location + 2];
        float by2 = boxes[bbox_location + 3];

        float maxProb = 0.0f;
        int maxIndex = -1;

        for (uint c = 0; c < detectionParams.numClassesConfigured; ++c) {
            float prob = scores[score_location + c];
            if (prob > maxProb) {
                maxProb = prob;
                maxIndex = c;
            }
        }

        if (maxProb > detectionParams.perClassPreclusterThreshold[maxIndex]) {
            addBBoxProposalYoloJonghwanl(bx1, by1, bx2, by2, netW, netH, maxIndex, maxProb, binfo);
        }

        bbox_location += 4;
        score_location += detectionParams.numClassesConfigured;
    }

    return binfo;
}


// Use this function when you generate TensorRT engine without NMS plugin
// !! Assume that yolo layers are already included and the outputs are concatenated with its anchor information
extern "C" bool NvDsInferParseCustomYoloJonghwanl(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferParseObjectInfo>& objectList) {
    if (NUM_CLASSES_YOLO != detectionParams.numClassesConfigured) {
        std::cerr << "WARNING: Num classes mismatch."<< std::endl
		  << "Deepstream Configured( DeepstreamAppConfig  ): " << detectionParams.numClassesConfigured << std::endl
                  << "Library configured( NUM_CLASSES_YOLO define ): " << NUM_CLASSES_YOLO << std::endl;
    }

    std::vector<NvDsInferParseObjectInfo> objects;

    const NvDsInferLayerInfo &boxes = outputLayersInfo[0];  // num_boxes x 4
    const NvDsInferLayerInfo &scores = outputLayersInfo[1]; // num_boxes x num_classes

    assert(boxes.inferDims.numDims == 3);  // 3 dimensional: [num_boxes, 1, 4]
    assert(scores.inferDims.numDims == 2); // 2 dimensional: [num_boxes, num_classes]
    assert(detectionParams.numClassesConfigured == scores.inferDims.d[1]);  // The second dimension should be num_classes

    uint num_bboxes = boxes.inferDims.d[0];

    std::vector<NvDsInferParseObjectInfo> outObjs =
        decodeYoloJonghwanlTensor((const float*)(boxes.buffer), (const float*)(scores.buffer),\
		       	num_bboxes, detectionParams, networkInfo.width, networkInfo.height);

    objects.insert(objects.end(), outObjs.begin(), outObjs.end());

    objectList = objects;

    return true;

}

// Use this function when you generate TensorRT engine already with NMS plugins
extern "C" bool NvDsInferParseCustomYoloJonghwanl_withNMS(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferParseObjectInfo>& objectList){

    const int* num_detections = static_cast <const int*>(outputLayersInfo.at(0).buffer);   // [batch_size, 1]
    const float* nms_boxes    = static_cast <const float*>(outputLayersInfo.at(1).buffer); // [batch_size, keepTopK, 4]
    const float* nms_scores   = static_cast <const float*>(outputLayersInfo.at(2).buffer); // [batch_size, keepTopK]
    //const float* nms_classes  = static_cast <const float*>(outputLayersInfo.at(3).buffer); // [batch_size, keepTopK]

    uint netW = networkInfo.width;
    uint netH = networkInfo.height;
    uint score_location = 0;

    for(int i = 0; i < num_detections[0]; ++i) {
        NvDsInferParseObjectInfo box;
	const float* _rect  = &nms_boxes[0]   + (i * 4);
	const float* _score = &nms_scores[0]  + i;
	//const float* _class = &nms_classes[0] + i;
	// Restore coordinates to network input resolution
        float x1 = _rect[0] * netW; 
        float y1 = _rect[1] * netH;
        float x2 = _rect[2] * netW;
        float y2 = _rect[3] * netH;
        float maxProb = 0.0f;
        int maxIndex = -1;

        box.left = x1;
        box.width = clamp(x2 - x1, 0, netW);
        box.top = y1;
        box.height = clamp(y2 - y1, 0, netH);

        for (uint c = 0; c < detectionParams.numClassesConfigured; ++c) {
            float prob = _score[score_location + c];
            if (prob > maxProb) {
                maxProb = prob;
                maxIndex = c;
            }
        }
        score_location += detectionParams.numClassesConfigured;
        box.detectionConfidence = maxProb;
        box.classId = maxIndex;

   	objectList.push_back(box);
    }

    return true;
}

/* Check that the custom function has been defined correctly */
CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseCustomYoloJonghwanl);
CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseCustomYoloJonghwanl_withNMS);
