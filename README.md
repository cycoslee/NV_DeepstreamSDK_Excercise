# NV_DeepstreamSDK_Excercise
-----
This repository includes DeepstreamSDK hands-on material to understand how can develop IVA application with Gstreamer with plugins that DeepstreamSDK serves.
Specially, this hands-on is working with Jetson Series but prepared docker container is for the Jetson NX. (TODO) It will be updated to have Dockerfile that could build
with custom configurations.

## Prerequisites
### 1. Host machine 
  - SSH terminal
  - VLC player
  - (For TLT exercise) GPU machine is required & docker runtime environment
    - $ docker run --gpus all -it -v $(pwd):/workspace –w /usr/local/src -p 8888:8888 cycoslee/nv-deepstreamsdk-handson:tlt_host_210127
### 2. Jetson NX (based on Jetpack 4.4.1)
  - docker container pull
    - $ sudo docker run -ti --runtime=nvidia --rm --net=host -e DISPLAY=$DISPLAY -w /opt/nvidia/deepstream/deepstream-5.0 --device /dev/video0 -v /tmp/.X11-unix/:/tmp/.X11-unix -v $(pwd):/workspace cycoslee/nv-deepstreamsdk-handson:ds_nx_210127
  - Assets
    - IP camera(supports RTSP)
    - USB camera(webcam)
    
## Exmaples
(Please check README file in the every each examples.)
  - 1_test_ds_input_video
  - 2_test_ds_input_webcam
  - 3_test_ds_input_uri
  - 4_test_ds_output_video
  - 5_test_ds_output_rtsp
  - 6_test_ds_output_rtsp_multi
  - 7_test_trt_benchmark
  - 8_test_ds_pose_estimation
  - 9_test_ds_yolov4
  - 10_test_ds_tlt_facemask
  
