/*
 * Copyright (c) 2019-2020, NVIDIA CORPORATION. All rights reserved.
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
#include <iostream>
#include <fstream>
#include <experimental/filesystem>
#include "nvdsinfer_custom_impl.h"
#include "nvdsinfer_context.h"

struct NetworkInfo{
    std::string networkType;
    std::string configFilePath;
    std::string wtsFilePath;
    std::string deviceType;
    std::string inputBlobName;
};

bool fileExists(const std::string fileName, bool verbose) {
    if (!std::experimental::filesystem::exists(std::experimental::filesystem::path(fileName)))
    {
        if (verbose) std::cout << "File does not exist : " << fileName << std::endl;
        return false;
    }
    return true;
}

static bool getYoloNetworkInfo (NetworkInfo &networkInfo, const NvDsInferContextInitParams* initParams) {
    std::string yoloCfg = initParams->customNetworkConfigFilePath;
    std::string yoloType;

    std::transform (yoloCfg.begin(), yoloCfg.end(), yoloCfg.begin(), [] (uint8_t c) {
        return std::tolower (c);});

    if (yoloCfg.find("yolov2") != std::string::npos) {
        if (yoloCfg.find("yolov2-tiny") != std::string::npos)
            yoloType = "yolov2-tiny";
        else
            yoloType = "yolov2";
    } else if (yoloCfg.find("yolov3") != std::string::npos) {
        if (yoloCfg.find("yolov3-tiny") != std::string::npos)
            yoloType = "yolov3-tiny";
        else
            yoloType = "yolov3";
    } else if (yoloCfg.find("yolov4") != std::string::npos) {
        if (yoloCfg.find("yolov4-tiny") != std::string::npos)
            yoloType = "yolov4-tiny";
        else
            yoloType = "yolov4";
    } else if (yoloCfg.find("yolov5") != std::string::npos) {
        if (yoloCfg.find("yolov5-tiny") != std::string::npos)
            yoloType = "yolov5-tiny";
        else
            yoloType = "yolov5";
    } else {
        std::cerr << "Yolo type is not defined from config file name:"
                  << yoloCfg << std::endl;
        return false;
    }

    networkInfo.networkType     = yoloType;
    networkInfo.configFilePath  = initParams->customNetworkConfigFilePath;
    networkInfo.wtsFilePath     = initParams->modelFilePath;
    networkInfo.deviceType      = (initParams->useDLA ? "kDLA" : "kGPU");
    networkInfo.inputBlobName   = "input";

    if (networkInfo.configFilePath.empty() ||
        networkInfo.wtsFilePath.empty()) {
        std::cerr << "Yolo config file or weights file is NOT specified."
                  << std::endl;
        return false;
    }

    if (!fileExists(networkInfo.configFilePath, false) ||
        !fileExists(networkInfo.wtsFilePath, false)) {
        std::cerr << "Yolo config file or weights file is NOT exist."
                  << std::endl;
        return false;
    }

    return true;
}

extern "C"
bool NvDsInferYoloCudaEngineGet(nvinfer1::IBuilder * const builder,
        const NvDsInferContextInitParams * const initParams,
        nvinfer1::DataType dataType,
        nvinfer1::ICudaEngine *& cudaEngine);

extern "C"
bool NvDsInferYoloCudaEngineGet(nvinfer1::IBuilder * const builder,
        const NvDsInferContextInitParams * const initParams,
        nvinfer1::DataType dataType,
        nvinfer1::ICudaEngine *& cudaEngine)
{
    NetworkInfo networkInfo;
    std::cerr << "Not supported!!!, Please follow Darknet -> Onnx -> TRT -> DS pipeline!!!"<< std::endl;
    return false;
}
