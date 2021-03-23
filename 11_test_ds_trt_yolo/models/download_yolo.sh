#!/bin/bash

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

set -e
## Download yolo weights
## [yolov2]
#echo "Downloading yolov2 config and weights files ... "
#wget https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov2.cfg -q --show-progress
#wget https://pjreddie.com/media/files/yolov2.weights -q --show-progress
#
## [yolov2-tiny]
#echo "Downloading yolov2-tiny config and weights files ... "
#wget https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov2-tiny.cfg -q --show-progress
#wget https://pjreddie.com/media/files/yolov2-tiny.weights -q --show-progress

# [yolov3]
echo "Downloading yolov3 config and weights files ... "
wget https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg -q --show-progress
wget https://pjreddie.com/media/files/yolov3.weights -q --show-progress
echo "Creating yolov3-288.cfg and yolov3-288.weights"
cat yolov3.cfg | sed -e '6s/batch=64/batch=1/' | sed -e '8s/width=608/width=288/' | sed -e '9s/height=608/height=288/' > yolov3-288.cfg
echo >> yolov3-288.cfg
ln -sf yolov3.weights yolov3-288.weights
echo "Creating yolov3-416.cfg and yolov3-416.weights"
cat yolov3.cfg | sed -e '6s/batch=64/batch=1/' | sed -e '8s/width=608/width=416/' | sed -e '9s/height=608/height=416/' > yolov3-416.cfg
echo >> yolov3-416.cfg
ln -sf yolov3.weights yolov3-416.weights
echo "Creating yolov3-608.cfg and yolov3-608.weights"
cat yolov3.cfg | sed -e '6s/batch=64/batch=1/' | sed -e '8s/width=608/width=608/' | sed -e '9s/height=608/height=608/' > yolov3-608.cfg
echo >> yolov3-608.cfg
ln -sf yolov3.weights yolov3-608.weights

# [yolov3-tiny]
echo "Downloading yolov3-tiny config and weights files ... "
wget https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3-tiny.cfg -q --show-progress
wget https://pjreddie.com/media/files/yolov3-tiny.weights -q --show-progress
echo "Creating yolov3-tiny-288.cfg and yolov3-tiny-288.weights"
cat yolov3-tiny.cfg | sed -e '3s/batch=1/batch=1/' | sed -e '8s/width=416/width=288/' | sed -e '9s/height=416/height=288/' > yolov3-tiny-288.cfg
echo >> yolov3-tiny-288.cfg
ln -sf yolov3-tiny.weights yolov3-tiny-288.weights
echo "Creating yolov3-tiny-416.cfg and yolov3-tiny-416.weights"
cat yolov3-tiny.cfg | sed -e '3s/batch=1/batch=1/' | sed -e '8s/width=416/width=416/' | sed -e '9s/height=416/height=416/' > yolov3-tiny-416.cfg
echo >> yolov3-tiny-416.cfg
ln -sf yolov3-tiny.weights yolov3-tiny-416.weights
echo "Creating yolov3-tiny-608.cfg and yolov3-tiny-608.weights"
cat yolov3-tiny.cfg | sed -e '3s/batch=1/batch=1/' | sed -e '8s/width=416/width=608/' | sed -e '9s/height=416/height=608/' > yolov3-tiny-608.cfg
echo >> yolov3-tiny-608.cfg
ln -sf yolov3-tiny.weights yolov3-tiny-608.weights

# [yolov4]
echo
echo "Download and set for yolov4 models"
wget -O yolov4.cfg https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4.cfg -q --show-progress --no-clobber
wget -O yolov4.weights https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights -q --show-progress --no-clobber
echo "Creating yolov4-288.cfg and yolov4-288.weights"
cat yolov4.cfg | sed -e '2s/batch=64/batch=1/' | sed -e '5s/width=608/width=288/' | sed -e '6s/height=608/height=288/' > yolov4-288.cfg
echo >> yolov4-288.cfg
ln -sf yolov4.weights yolov4-288.weights
echo "Creating yolov4-416.cfg and yolov4-416.weights"
cat yolov4.cfg | sed -e '2s/batch=64/batch=1/' | sed -e '5s/width=608/width=416/' | sed -e '6s/height=608/height=416/' > yolov4-416.cfg
echo >> yolov4-416.cfg
ln -sf yolov4.weights yolov4-416.weights
echo "Creating yolov4-608.cfg and yolov4-608.weights"
cat yolov4.cfg | sed -e '2s/batch=64/batch=1/' | sed -e '5s/width=608/width=608/' | sed -e '6s/height=608/height=608/' > yolov4-608.cfg
echo >> yolov4-608.cfg
ln -sf yolov4.weights yolov4-608.weights

# [yolov4-tiny]
echo
echo "Download and set for yolov4-tiny models"
wget -O yolov4-tiny.cfg https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny.cfg -q --show-progress --no-clobber
wget -O yolov4-tiny.weights https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights -q --show-progress --no-clobber
echo "Creating yolov4-tiny-288.cfg and yolov4-tiny-288.weights"
cat yolov4-tiny.cfg | sed -e '6s/batch=64/batch=1/' | sed -e '8s/width=416/width=288/' | sed -e '9s/height=416/height=288/' > yolov4-tiny-288.cfg
echo >> yolov4-tiny-288.cfg
ln -sf yolov4-tiny.weights yolov4-tiny-288.weights
echo "Creating yolov4-tiny-416.cfg and yolov4-tiny-416.weights"
cat yolov4-tiny.cfg | sed -e '6s/batch=64/batch=1/' | sed -e '8s/width=416/width=416/' | sed -e '9s/height=416/height=416/' > yolov4-tiny-416.cfg
echo >> yolov4-tiny-416.cfg
ln -sf yolov4-tiny.weights yolov4-tiny-416.weights
echo "Creating yolov4-tiny-608.cfg and yolov4-tiny-608.weights"
cat yolov4-tiny.cfg | sed -e '6s/batch=64/batch=1/' | sed -e '8s/width=416/width=608/' | sed -e '9s/height=416/height=608/' > yolov4-tiny-608.cfg
echo >> yolov4-tiny-608.cfg
ln -sf yolov4-tiny.weights yolov4-tiny-608.weights

## [yolov4x-Mish]
#echo
#echo "Download and set for yolov4x-mish models"
#wget -O yolov4x-mish.cfg https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4x-mish.cfg -q --show-progress --no-clobber
#wget -O yolov4x-mish.weights https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4x-mish.weights -q --show-progress --no-clobber
#echo "Creating yolov4x-mish-288.cfg and yolov4x-mish-288.weights"
#cat yolov4x-mish.cfg | sed -e '6s/batch=64/batch=1/' | sed -e '8s/width=640/width=288/' | sed -e '9s/height=640/height=288/' > yolov4x-mish-288.cfg
#echo >> yolov4x-mish-288.cfg
#ln -sf yolov4x-mish.weights yolov4x-mish-288.weights
#echo "Creating yolov4x-mish-416.cfg and yolov4x-mish-416.weights"
#cat yolov4x-mish.cfg | sed -e '6s/batch=64/batch=1/' | sed -e '8s/width=640/width=416/' | sed -e '9s/height=640/height=416/' > yolov4x-mish-416.cfg
#echo >> yolov4x-mish-416.cfg
#ln -sf yolov4x-mish.weights yolov4x-mish-416.weights
#echo "Creating yolov4x-mish-608.cfg and yolov4x-mish-608.weights"
#cat yolov4x-mish.cfg | sed -e '6s/batch=64/batch=1/' | sed -e '8s/width=640/width=608/' | sed -e '9s/height=640/height=608/' > yolov4x-mish-608.cfg
#echo >> yolov4x-mish-608.cfg
#ln -sf yolov4x-mish.weights yolov4x-mish-608.weights
#
##[yolov4-csp]
#echo
#echo "Download and set for yolov4-csp models"
#wget -O yolov4-csp.cfg https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-csp.cfg -q --show-progress --no-clobber
#wget -O yolov4-csp.weights https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-csp.weights -q --show-progress --no-clobber
#echo "Creating yolov4-csp-288.cfg and yolov4-csp-288.weights"
#cat yolov4-csp.cfg | sed -e '6s/batch=64/batch=1/' | sed -e '8s/width=512/width=288/' | sed -e '9s/height=512/height=288/' > yolov4-csp-288.cfg
#echo >> yolov4-csp-288.cfg
#ln -sf yolov4-csp.weights yolov4-csp-288.weights
#echo "Creating yolov4-csp-416.cfg and yolov4-csp-416.weights"
#cat yolov4-csp.cfg | sed -e '6s/batch=64/batch=1/' | sed -e '8s/width=512/width=416/' | sed -e '9s/height=512/height=416/' > yolov4-csp-416.cfg
#echo >> yolov4-csp-416.cfg
#ln -sf yolov4-csp.weights yolov4-csp-416.weights
#echo "Creating yolov4-csp-608.cfg and yolov4-csp-608.weights"
#cat yolov4-csp.cfg | sed -e '6s/batch=64/batch=1/' | sed -e '8s/width=512/width=608/' | sed -e '9s/height=512/height=608/' > yolov4-csp-608.cfg
#echo >> yolov4-csp-608.cfg
#ln -sf yolov4-csp.weights yolov4-csp-608.weights
#
##[yolov4]
#echo
#echo "Download and set for yolov4 models"
#wget -O yolov4.cfg https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4.cfg -q --show-progress --no-clobber
#wget -O yolov4.weights https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights -q --show-progress --no-clobber
#echo "Creating yolov4-288.cfg and yolov4-288.weights"
#cat yolov4.cfg | sed -e '2s/batch=64/batch=1/' | sed -e '5s/width=608/width=288/' | sed -e '6s/height=608/height=288/' > yolov4-288.cfg
#echo >> yolov4-288.cfg
#ln -sf yolov4.weights yolov4-288.weights
#echo "Creating yolov4-416.cfg and yolov4-416.weights"
#cat yolov4.cfg | sed -e '2s/batch=64/batch=1/' | sed -e '5s/width=608/width=416/' | sed -e '6s/height=608/height=416/' > yolov4-416.cfg
#echo >> yolov4-416.cfg
#ln -sf yolov4.weights yolov4-416.weights
#echo "Creating yolov4-608.cfg and yolov4-608.weights"
#cat yolov4.cfg | sed -e '2s/batch=64/batch=1/' | sed -e '5s/width=608/width=608/' | sed -e '6s/height=608/height=608/' > yolov4-608.cfg
#echo >> yolov4-608.cfg
#ln -sf yolov4.weights yolov4-608.weights
#
##[yolov4-tiny]
#echo
#echo "Download and set for yolov4-tiny models"
#wget -O yolov4-tiny.cfg https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny.cfg -q --show-progress --no-clobber
#wget -O yolov4-tiny.weights https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights -q --show-progress --no-clobber
#echo "Creating yolov4-tiny-288.cfg and yolov4-tiny-288.weights"
#cat yolov4-tiny.cfg | sed -e '6s/batch=64/batch=1/' | sed -e '8s/width=416/width=288/' | sed -e '9s/height=416/height=288/' > yolov4-tiny-288.cfg
#echo >> yolov4-tiny-288.cfg
#ln -sf yolov4-tiny.weights yolov4-tiny-288.weights
#echo "Creating yolov4-tiny-416.cfg and yolov4-tiny-416.weights"
#cat yolov4-tiny.cfg | sed -e '6s/batch=64/batch=1/' | sed -e '8s/width=416/width=416/' | sed -e '9s/height=416/height=416/' > yolov4-tiny-416.cfg
#echo >> yolov4-tiny-416.cfg
#ln -sf yolov4-tiny.weights yolov4-tiny-416.weights
#echo "Creating yolov4-tiny-608.cfg and yolov4-tiny-608.weights"
#cat yolov4-tiny.cfg | sed -e '6s/batch=64/batch=1/' | sed -e '8s/width=416/width=608/' | sed -e '9s/height=416/height=608/' > yolov4-tiny-608.cfg
#echo >> yolov4-tiny-608.cfg
#ln -sf yolov4-tiny.weights yolov4-tiny-608.weights
#
##[yolov3-spp]
#echo
#echo "Download and set for yolov3-spp models"
#wget -O yolov3-spp.cfg https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3-spp.cfg -q --show-progress --no-clobber
#wget -O yolov3-spp.weights https://pjreddie.com/media/files/yolov3-spp.weights -q --show-progress --no-clobber
#echo "Creating yolov3-spp-288.cfg and yolov3-spp-288.weights"
#cat yolov3-spp.cfg | sed -e '3s/batch=1/batch=1/' | sed -e '8s/width=608/width=288/' | sed -e '9s/height=608/height=288/' > yolov3-spp-288.cfg
#echo >> yolov3-spp-288.cfg
#ln -sf yolov3-spp.weights yolov3-spp-288.weights
#echo "Creating yolov3-spp-416.cfg and yolov3-spp-416.weights"
#cat yolov3-spp.cfg | sed -e '3s/batch=1/batch=1/' | sed -e '8s/width=608/width=416/' | sed -e '9s/height=608/height=416/' > yolov3-spp-416.cfg
#echo >> yolov3-spp-416.cfg
#ln -sf yolov3-spp.weights yolov3-spp-416.weights
#echo "Creating yolov3-spp-608.cfg and yolov3-spp-608.weights"
#cat yolov3-spp.cfg | sed -e '3s/batch=1/batch=1/' | sed -e '8s/width=608/width=608/' | sed -e '9s/height=608/height=608/' > yolov3-spp-608.cfg
#echo >> yolov3-spp-608.cfg
#ln -sf yolov3-spp.weights yolov3-spp-608.weights
#
##[yolov3]
#echo
#echo "Download and set for yolov3 models"
#wget -O yolov3.cfg https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg -q --show-progress --no-clobber
#wget -O yolov3.weights https://pjreddie.com/media/files/yolov3.weights -q --show-progress --no-clobber
#echo "Creating yolov3-288.cfg and yolov3-288.weights"
#cat yolov3.cfg | sed -e '6s/batch=64/batch=1/' | sed -e '8s/width=608/width=288/' | sed -e '9s/height=608/height=288/' > yolov3-288.cfg
#echo >> yolov3-288.cfg
#ln -sf yolov3.weights yolov3-288.weights
#echo "Creating yolov3-416.cfg and yolov3-416.weights"
#cat yolov3.cfg | sed -e '6s/batch=64/batch=1/' | sed -e '8s/width=608/width=416/' | sed -e '9s/height=608/height=416/' > yolov3-416.cfg
#echo >> yolov3-416.cfg
#ln -sf yolov3.weights yolov3-416.weights
#echo "Creating yolov3-608.cfg and yolov3-608.weights"
#cat yolov3.cfg | sed -e '6s/batch=64/batch=1/' | sed -e '8s/width=608/width=608/' | sed -e '9s/height=608/height=608/' > yolov3-608.cfg
#echo >> yolov3-608.cfg
#ln -sf yolov3.weights yolov3-608.weights
#
##[yolov3-tiny]
#echo
#echo "Download and set for yolov3-tiny models"
#wget -O yolov3-tiny.cfg https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3-tiny.cfg -q --show-progress --no-clobber
#wget -O yolov3-tiny.weights https://pjreddie.com/media/files/yolov3-tiny.weights -q --show-progress --no-clobber
#echo "Creating yolov3-tiny-288.cfg and yolov3-tiny-288.weights"
#cat yolov3-tiny.cfg | sed -e '3s/batch=1/batch=1/' | sed -e '8s/width=416/width=288/' | sed -e '9s/height=416/height=288/' > yolov3-tiny-288.cfg
#echo >> yolov3-tiny-288.cfg
#ln -sf yolov3-tiny.weights yolov3-tiny-288.weights
#echo "Creating yolov3-tiny-416.cfg and yolov3-tiny-416.weights"
#cat yolov3-tiny.cfg | sed -e '3s/batch=1/batch=1/' | sed -e '8s/width=416/width=416/' | sed -e '9s/height=416/height=416/' > yolov3-tiny-416.cfg
#echo >> yolov3-tiny-416.cfg
#ln -sf yolov3-tiny.weights yolov3-tiny-416.weights
#echo "Creating yolov3-tiny-608.cfg and yolov3-tiny-608.weights"
#cat yolov3-tiny.cfg | sed -e '3s/batch=1/batch=1/' | sed -e '8s/width=416/width=608/' | sed -e '9s/height=416/height=608/' > yolov3-tiny-608.cfg
#echo >> yolov3-tiny-608.cfg
#ln -sf yolov3-tiny.weights yolov3-tiny-608.weights
#
##[yolov3-lite]
#echo
#echo "Download and set for yolov3-lite models"
#wget -O yolov3-lite.cfg https://raw.githubusercontent.com/dog-qiuqiu/MobileNet-Yolo/master/MobileNetV2-YOLOv3-Lite/COCO/MobileNetV2-YOLOv3-Lite-coco.cfg -q --show-progress --no-clobber
#wget -O yolov3-lite.weights https://github.com/dog-qiuqiu/MobileNet-Yolo/raw/master/MobileNetV2-YOLOv3-Lite/COCO/MobileNetV2-YOLOv3-Lite-coco.weights -q --show-progress --no-clobber
#echo "Creating yolov3-lite-288.cfg and yolov3-lite-288.weights"
#cat yolov3-lite.cfg | sed -e '2s/batch=128/batch=1/' | sed -e '4s/width=320/width=288/' | sed -e '5s/height=320/height=288/' > yolov3-lite-288.cfg
#echo >> yolov3-lite-288.cfg
#ln -sf yolov3-lite.weights yolov3-lite-288.weights
#echo "Creating yolov3-lite-416.cfg and yolov3-lite-416.weights"
#cat yolov3-lite.cfg | sed -e '2s/batch=128/batch=1/' | sed -e '4s/width=320/width=416/' | sed -e '5s/height=320/height=416/' > yolov3-lite-416.cfg
#echo >> yolov3-lite-416.cfg
#ln -sf yolov3-lite.weights yolov3-lite-416.weights
#echo "Creating yolov3-lite-608.cfg and yolov3-lite-608.weights"
#cat yolov3-lite.cfg | sed -e '2s/batch=128/batch=1/' | sed -e '4s/width=320/width=608/' | sed -e '5s/height=320/height=608/' > yolov3-lite-608.cfg
#echo >> yolov3-lite-608.cfg
#ln -sf yolov3-lite.weights yolov3-lite-608.weights
#
##[yolov3-nano]
#echo
#echo "Download and set for yolov3-nano models"
#wget -O yolov3-nano.cfg https://raw.githubusercontent.com/dog-qiuqiu/MobileNet-Yolo/master/MobileNetV2-YOLOv3-Nano/COCO/MobileNetV2-YOLOv3-Nano-coco.cfg -q --show-progress --no-clobber
#wget -O yolov3-nano.weights https://github.com/dog-qiuqiu/MobileNet-Yolo/raw/master/MobileNetV2-YOLOv3-Nano/COCO/MobileNetV2-YOLOv3-Nano-coco.weights -q --show-progress --no-clobber
#echo "Creating yolov3-nano-288.cfg and yolov3-nano-288.weights"
#cat yolov3-nano.cfg | sed -e '2s/batch=128/batch=1/' | sed -e '4s/width=320/width=288/' | sed -e '5s/height=320/height=288/' > yolov3-nano-288.cfg
#echo >> yolov3-nano-288.cfg
#ln -sf yolov3-nano.weights yolov3-nano-288.weights
#echo "Creating yolov3-nano-416.cfg and yolov3-nano-416.weights"
#cat yolov3-nano.cfg | sed -e '2s/batch=128/batch=1/' | sed -e '4s/width=320/width=416/' | sed -e '5s/height=320/height=416/' > yolov3-nano-416.cfg
#echo >> yolov3-nano-416.cfg
#ln -sf yolov3-nano.weights yolov3-nano-416.weights
#echo "Creating yolov3-nano-608.cfg and yolov3-nano-608.weights"
#cat yolov3-nano.cfg | sed -e '2s/batch=128/batch=1/' | sed -e '4s/width=320/width=608/' | sed -e '5s/height=320/height=608/' > yolov3-nano-608.cfg
#echo >> yolov3-nano-608.cfg
#ln -sf yolov3-nano.weights yolov3-nano-608.weights
#
##[yolov2]
#echo
#echo "Download and set for yolov2 models"
#wget -O yolov2.cfg https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov2.cfg -q --show-progress --no-clobber
#wget -O yolov2.weights https://pjreddie.com/media/files/yolov2.weights -q --show-progress --no-clobber
#echo "Creating yolov2-288.cfg and yolov2-288.weights"
#cat yolov2.cfg | sed -e '3s/batch=1/batch=1/' | sed -e '8s/width=608/width=288/' | sed -e '9s/height=608/height=288/' > yolov2-288.cfg
#echo >> yolov2-288.cfg
#ln -sf yolov2.weights yolov2-288.weights
#echo "Creating yolov2-416.cfg and yolov2-416.weights"
#cat yolov2.cfg | sed -e '3s/batch=1/batch=1/' | sed -e '8s/width=608/width=416/' | sed -e '9s/height=608/height=416/' > yolov2-416.cfg
#echo >> yolov2-416.cfg
#ln -sf yolov2.weights yolov2-416.weights
#echo "Creating yolov2-608.cfg and yolov2-608.weights"
#cat yolov2.cfg | sed -e '3s/batch=1/batch=1/' | sed -e '8s/width=608/width=608/' | sed -e '9s/height=608/height=608/' > yolov2-608.cfg
#echo >> yolov2-608.cfg
#ln -sf yolov2.weights yolov2-608.weights
#
##[yolov2-tiny]
#echo
#echo "Download and set for yolov2-tiny models"
#wget -O yolov2-tiny.cfg https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov2-tiny.cfg -q --show-progress --no-clobber
#wget -O yolov2-tiny.weights https://pjreddie.com/media/files/yolov2-tiny.weights -q --show-progress --no-clobber
#echo "Creating yolov2-tiny-288.cfg and yolov2-tiny-288.weights"
#cat yolov2-tiny.cfg | sed -e '3s/batch=1/batch=1/' | sed -e '8s/width=416/width=288/' | sed -e '9s/height=416/height=288/' > yolov2-tiny-288.cfg
#echo >> yolov2-tiny-288.cfg
#ln -sf yolov2-tiny.weights yolov2-tiny-288.weights
#echo "Creating yolov2-tiny-416.cfg and yolov2-tiny-416.weights"
#cat yolov2-tiny.cfg | sed -e '3s/batch=1/batch=1/' | sed -e '8s/width=416/width=416/' | sed -e '9s/height=416/height=416/' > yolov2-tiny-416.cfg
#echo >> yolov2-tiny-416.cfg
#ln -sf yolov2-tiny.weights yolov2-tiny-416.weights
#echo "Creating yolov2-tiny-608.cfg and yolov2-tiny-608.weights"
#cat yolov2-tiny.cfg | sed -e '3s/batch=1/batch=1/' | sed -e '8s/width=416/width=608/' | sed -e '9s/height=416/height=608/' > yolov2-tiny-608.cfg
#echo >> yolov2-tiny-608.cfg
#ln -sf yolov2-tiny.weights yolov2-tiny-608.weights

echo
echo "Done."
