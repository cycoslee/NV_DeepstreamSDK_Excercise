/*
 * Copyright (c) 2018-2021, NVIDIA CORPORATION. All rights reserved.
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

#include <gst/gst.h>
#include <glib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <unistd.h>
#include <iostream>
#include "gstnvdsmeta.h"

/* The muxer output resolution must be set if the input streams will be of
 * different resolution. The muxer will scale all the input frames to this
 * resolution. */
#define MUXER_OUTPUT_WIDTH 1920
#define MUXER_OUTPUT_HEIGHT 1080

/* Muxer batch formation timeout, for e.g. 40 millisec. Should ideally be set
 * based on the fastest source's framerate. */
#define MUXER_BATCH_TIMEOUT_USEC 4000000

#define TILED_OUTPUT_WIDTH 1920
#define TILED_OUTPUT_HEIGHT 1080

std::string log_file;

static gboolean
bus_call (GstBus * bus, GstMessage * msg, gpointer data) {
    GMainLoop *loop = (GMainLoop *) data;
    switch (GST_MESSAGE_TYPE (msg)) {
    case GST_MESSAGE_EOS:
        g_print ("End of stream\n");
        g_main_loop_quit (loop);
        break;
    case GST_MESSAGE_ERROR: {
        gchar *debug;
        GError *error;
        gst_message_parse_error (msg, &error, &debug);
        g_printerr ("ERROR from element %s: %s\n",
                    GST_OBJECT_NAME (msg->src), error->message);
        if (debug)
            g_printerr ("Error details: %s\n", debug);
        g_free (debug);
        g_error_free (error);
        g_main_loop_quit (loop);
        break;
    }
    default:
        break;
    }
    return TRUE;
}

static GstPadProbeReturn osd_sink_pad_buffer_probe (GstPad * pad, GstPadProbeInfo * info, gpointer u_data) {

    GstBuffer *buf = (GstBuffer *) info->data;
    NvDsObjectMeta *obj_meta = NULL;
    NvDsMetaList * l_frame = NULL;
    NvDsMetaList * l_obj = NULL;
    NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta (buf);
    FILE *bbox_params_dump_file = NULL;
    gchar bbox_file[1024] = { 0 };
    time_t now;

    for (l_frame = batch_meta->frame_meta_list; l_frame != NULL; l_frame = l_frame->next) {
        NvDsFrameMeta *frame_meta = (NvDsFrameMeta *) (l_frame->data);
        
        if (frame_meta == NULL) {
            g_print ("NvDS Meta contained NULL meta \n");
            return GST_PAD_PROBE_OK;
        }

        if (!log_file.empty()){
            g_print("1\n");
            g_snprintf (bbox_file, sizeof (bbox_file) - 1, "%s", log_file.c_str());
            g_print("%s\n", bbox_file);
            bbox_params_dump_file = fopen (bbox_file, "a");
        }
  
        for (l_obj = frame_meta->obj_meta_list; l_obj != NULL; l_obj = l_obj->next) {
            obj_meta = (NvDsObjectMeta *) (l_obj->data);
  
            NvOSD_RectParams * rect_params = &(obj_meta->rect_params);
            NvOSD_TextParams * text_params = &(obj_meta->text_params);
  
            if (text_params->display_text) {
                text_params->font_params.font_size = 24;
                text_params->set_bg_clr = 1;
                text_params->text_bg_clr.red = 0.0;
                text_params->text_bg_clr.green = 0.0;
                text_params->text_bg_clr.blue = 0.0;
                text_params->text_bg_clr.alpha = 0.5;
                text_params->x_offset = rect_params->left;
                text_params->y_offset = rect_params->top;
            }
  
            /* Draw black patch to cover license plates (class_id = 1) */
            if (obj_meta->class_id == 1) {
                rect_params->border_width = 3;
                rect_params->border_color.red = 1.0;
                rect_params->border_color.green = 0.0;
                rect_params->border_color.blue = 0.0;
                rect_params->border_color.alpha = 0.8;
                text_params->text_bg_clr.red = 1.0;

                if(bbox_params_dump_file){
                    g_print("2-1\n");
                    time(&now);
                    struct tm *local = localtime(&now);
                    g_print("2-2\n");
                    fprintf ( bbox_params_dump_file, "%02d/%02d/%d, %02d:%02d:%02d WARN: Someone No Wearing face mask!!!\n",
                              local->tm_mday, local->tm_mon+1, local->tm_year+1900,
                              local->tm_hour, local->tm_min, local->tm_sec );
                }
            }
            /* Draw skin-color patch to cover faces (class_id = 0) */
            if (obj_meta->class_id == 0) {
                rect_params->border_width = 3;
                rect_params->border_color.red = 0.0;
                rect_params->border_color.green = 1.0;
                rect_params->border_color.blue = 0.0;
                rect_params->border_color.alpha = 0.8;
                text_params->text_bg_clr.green = 1.0;
            }
        }  
        if (bbox_params_dump_file) {
            g_print("3\n");
            fclose (bbox_params_dump_file);
            bbox_params_dump_file = NULL;
        } 
    }
    return GST_PAD_PROBE_OK;
}

static void printUsage(const char* cmd) {
    g_printerr ("\tUsage: %s -c pgie_config_file -i <H264 or JPEG filename> [-b BATCH] [-d]\n", cmd);
    g_printerr ("-h: \n\tprint help info \n");
    g_printerr ("-c: \n\tpgie config file, e.g. pgie_frcnn_tlt_config.txt  \n");
    g_printerr ("-i: \n\tH264 or JPEG input file  \n");
    g_printerr ("-b: \n\tbatch size, this will override the value of \"baitch-size\" in pgie config file  \n");
    g_printerr ("-d: \n\tenable display, otherwise dump to output H264 or JPEG file  \n");
    g_printerr ("-w: \n\tuse webcam device\n");
}

int main (int argc, char *argv[]) {
    GMainLoop *loop = NULL;
    GstElement *pipeline = NULL, *source = NULL, *parser = NULL,
               *caps_v4l2src = NULL, *vidconv_src = NULL, *caps_vidconv_src = NULL,
               *decoder = NULL, *streammux = NULL, *sink = NULL,
               *pgie = NULL, *nvvidconv = NULL, *nvdsosd = NULL,
               *parser1 = NULL, *nvvidconv1 = NULL, *enc = NULL,
               *tiler = NULL, *tee = NULL;

#ifdef PLATFORM_TEGRA
    GstElement *transform = NULL;
#endif
    GstBus *bus = NULL;
    guint bus_watch_id;
    GstPad *osd_sink_pad = NULL;
    GstCaps *caps = NULL;

    gboolean isH264 = FALSE;
    gboolean isCAM = FALSE;
    gboolean useDisplay = FALSE;
    guint tiler_rows, tiler_cols;
    guint batchSize = 1;
    guint pgie_batch_size;
    guint c;
    const char* optStr = "b:c:dhiw:l:";
    std::string pgie_config;
    std::string input_file;

    while ((c = getopt(argc, argv, optStr)) != -1) {
        switch (c) {
            case 'b':
                batchSize = std::atoi(optarg);
                batchSize = batchSize == 0 ? 1:batchSize;
                break;
            case 'c':
                pgie_config.assign(optarg);
                break;
            case 'd':
                useDisplay = TRUE;
                break;
            case 'i':
                input_file.assign(optarg);
                break;
            case 'w':
                input_file.assign(optarg);
                g_print("CAM device : %s\n",input_file.c_str());
                break;
            case 'l':
                log_file.assign(optarg);
                g_print("Logging : %s\n",log_file.c_str());
                break;
            case 'h':
            default:
                printUsage(argv[0]);
                return -1;
          }
     }

    /* Check input arguments */
    if (argc == 1) {
        printUsage(argv[0]);
        return -1;
    }

    const gchar *p_end = input_file.c_str() + strlen(input_file.c_str());
    const gchar *p_start = input_file.c_str();
    if(!strncmp(p_end - strlen("h264"), "h264", strlen("h264"))) {
        isH264 = TRUE;
    } else if(!strncmp(p_end - strlen("jpg"), "jpg", strlen("jpg")) || !strncmp(p_end - strlen("jpeg"), "jpeg", strlen("jpeg"))) {
        isH264 = FALSE;
    } else if(!strncmp(p_start, "/dev/", strlen("/dev/"))){
        isCAM = TRUE;
    } else {
        g_printerr("input file only support H264 and JPEG\n");
        return -1;
    }

    const char* use_display = std::getenv("USE_DISPLAY");
    if(use_display != NULL && std::stoi(use_display) == 1) {
        useDisplay = true;
    }

    const char* batch_size = std::getenv("BATCH_SIZE");
    if(batch_size != NULL ) {
        batchSize = std::stoi(batch_size);
        g_printerr("batch size is %d \n", batchSize);
    }

    /* Standard GStreamer initialization */
    gst_init (&argc, &argv);
    loop = g_main_loop_new (NULL, FALSE);

    /* Create gstreamer elements */
    /* Create Pipeline element that will form a connection of other elements */
    pipeline = gst_pipeline_new ("ds-custom-pipeline");

    if( isCAM == FALSE ){
        /* Source element for reading from the file */
        source = gst_element_factory_make ("filesrc", "file-source");

        /* Since the data format in the input file is elementary h264 stream,
         * we need a h264parser */
        if(isH264 == TRUE) {
            parser = gst_element_factory_make ("h264parse", "h264-parser");
        } else {
            parser = gst_element_factory_make ("jpegparse", "jpeg-parser");
        }

        /* Use nvdec_h264 for hardware accelerated decode on GPU */
        decoder = gst_element_factory_make ("nvv4l2decoder", "nvv4l2-decoder");
        if (!source || !parser || !decoder) {
            g_printerr ("One element could not be created. Exiting.\n");
            return -1;
        }
    } else {
        /* Source element for capturing from usb-camera */
        source = gst_element_factory_make ("v4l2src", "usb-camera");
        /* capsfilter for v4l2src */
        caps_v4l2src = gst_element_factory_make("capsfilter", "v4l2src_caps");
        
        /* nvvideoconvert element to convert incoming raw buffers to NVMM Mem (NvBufSurface API) */
        vidconv_src = gst_element_factory_make ("nvvideoconvert", "vidconv_src");
        /* capsfilter for nvvidconv_src */
        caps_vidconv_src = gst_element_factory_make ("capsfilter", "nvmm_caps");
        if (!source || !caps_v4l2src || !vidconv_src || !caps_vidconv_src) {
            g_printerr ("One element could not be created. Exiting.\n");
            return -1;
        }
    }


    /* Create nvstreammux instance to form batches from one or more sources. */
    streammux = gst_element_factory_make ("nvstreammux", "stream-muxer");

    if (!pipeline || !streammux) {
        g_printerr ("One element could not be created. Exiting.\n");
        return -1;
    }

    /* Use nvinfer to run inferencing on decoder's output,
     * behaviour of inferencing is set through config file */
    pgie = gst_element_factory_make ("nvinfer", "primary-nvinference-engine");

    /* Use convertor to convert from NV12 to RGBA as required by nvdsosd */
    nvvidconv = gst_element_factory_make ("nvvideoconvert", "nvvideo-converter");

    /* Create OSD to draw on the converted RGBA buffer */
    nvdsosd = gst_element_factory_make ("nvdsosd", "nv-onscreendisplay");

    tee = gst_element_factory_make("tee", "tee");
    tiler = gst_element_factory_make ("nvmultistreamtiler", "nvtiler");

    /* Finally render the osd output */
#ifdef PLATFORM_TEGRA
    transform = gst_element_factory_make ("nvegltransform", "nvegl-transform");
#endif
    if(useDisplay == FALSE) {
        if(isH264 == TRUE){
            parser1 = gst_element_factory_make ("h264parse", "h264-parser1");
            enc = gst_element_factory_make ("nvv4l2h264enc", "h264-enc");
        } else{
            parser1 = gst_element_factory_make ("jpegparse", "jpeg-parser1");
            enc = gst_element_factory_make ("jpegenc", "jpeg-enc");
        }
        nvvidconv1 = gst_element_factory_make ("nvvideoconvert", "nvvideo-converter1");
        sink = gst_element_factory_make ("filesink", "file-sink");
        if ( !tee || !pgie || !tiler || !nvvidconv || !nvvidconv1 || !nvdsosd || !enc || !sink) {
            g_printerr ("One element could not be created. Exiting.\n");
            return -1;
        }
    } else {
        sink = gst_element_factory_make ("nveglglessink", "nvvideo-renderer");
        if (!tee || !pgie || !tiler || !nvvidconv || !nvdsosd || !sink) {
            g_printerr ("One element could not be created. Exiting.\n");
            return -1;
        }
    }

#ifdef PLATFORM_TEGRA
    if(!transform) {
        g_printerr ("One tegra element could not be created. Exiting.\n");
        return -1;
    }
#endif

    if(isCAM == FALSE){
        /* we set the input filename to the source element */
        g_object_set (G_OBJECT (source), "location", input_file.c_str(), NULL);
    } else {
        /* Source setting */
        g_object_set (G_OBJECT (source), "device", input_file.c_str(), NULL);

        /* V4L2 source capsfilter setting */
        caps = gst_caps_from_string ("video/x-raw, width=640, height=480, framerate=30/1");
        g_object_set (G_OBJECT (caps_v4l2src), "caps", caps, NULL);

        /* nvvideo converter source capsfilter setting */
        caps = gst_caps_from_string ("video/x-raw(memory:NVMM), format=NV12, width=640, height=480, framerate=30/1");
        g_object_set (G_OBJECT (caps_vidconv_src), "caps", caps, NULL);

    }

    //save the file to local dir
    if(useDisplay == FALSE) {
        if(isH264 == TRUE)
            g_object_set (G_OBJECT (sink), "location", "./out.h264", NULL);
        else
            g_object_set (G_OBJECT (sink), "location", "./out.jpg", NULL);
    } else {
        g_object_set (G_OBJECT (sink), "sync", 0, NULL);
    }

    g_object_set (G_OBJECT (streammux), "width", MUXER_OUTPUT_WIDTH, "height",
                  MUXER_OUTPUT_HEIGHT, "batch-size", batchSize,
                  "batched-push-timeout", MUXER_BATCH_TIMEOUT_USEC, NULL);

    /* Set all the necessary properties of the nvinfer element,
     * the necessary ones are : */
    g_object_set (G_OBJECT (pgie),
                  "config-file-path", pgie_config.c_str(), NULL);

    /* Override the batch-size set in the config file with the number of sources. */
    g_object_get (G_OBJECT (pgie), "batch-size", &pgie_batch_size, NULL);
    if (pgie_batch_size != batchSize) {
        g_printerr("WARNING: Overriding infer-config batch-size (%d) with number of sources (%d)\n",
            pgie_batch_size, batchSize);
        g_object_set (G_OBJECT (pgie), "batch-size", batchSize, NULL);
    }

    tiler_rows = (guint) sqrt (batchSize);
    tiler_cols = (guint) ceil (1.0 * batchSize / tiler_rows);
    /* we set the tiler properties here */
    g_object_set (G_OBJECT (tiler), "rows", tiler_rows, "columns", tiler_cols,
      "width", TILED_OUTPUT_WIDTH, "height", TILED_OUTPUT_HEIGHT, NULL);

    /* we add a message handler */
    bus = gst_pipeline_get_bus (GST_PIPELINE (pipeline));
    bus_watch_id = gst_bus_add_watch (bus, bus_call, loop);
    gst_object_unref (bus);

    /* Set up the pipeline */
    /* we add all elements into the pipeline */
    if(isCAM == FALSE){
        gst_bin_add_many (GST_BIN (pipeline),
                          source, parser, decoder, tee, streammux, NULL);
    } else {
        gst_bin_add_many (GST_BIN (pipeline),
            source, caps_v4l2src, vidconv_src, caps_vidconv_src, streammux,  NULL);

    }

    if(useDisplay == FALSE) {
        gst_bin_add_many (GST_BIN (pipeline),
                          pgie, tiler, nvvidconv, nvdsosd, nvvidconv1, enc, parser1, sink, NULL);
    } else {
#ifdef PLATFORM_TEGRA
        gst_bin_add_many (GST_BIN (pipeline),
                          pgie, tiler, nvvidconv, nvdsosd, transform, sink, NULL);
#else
        gst_bin_add_many (GST_BIN (pipeline),
                          pgie, tiler, nvvidconv, nvdsosd, sink, NULL);
#endif
    }

    /* We link the elements together */
    if( isCAM == FALSE ){
        /* file-source -> h264-parser -> nvv4l2decoder ->
         * nvinfer -> nvvideoconvert -> nvdsosd -> video-renderer */

        if (!gst_element_link_many (source, parser, decoder, tee, NULL)) {
            g_printerr ("Elements could not be linked: 1. Exiting.\n");
            return -1;
        }
        for(guint i = 0; i < batchSize; i++) {
            GstPad *sinkpad, *srcpad;
            gchar pad_name_sink[16] = {};
            gchar pad_name_src[16] = {};
        
            g_snprintf (pad_name_sink, 15, "sink_%u", i);
            g_snprintf (pad_name_src, 15, "src_%u", i);
            sinkpad = gst_element_get_request_pad (streammux, pad_name_sink);
            if (!sinkpad) {
                g_printerr ("Streammux request sink pad failed. Exiting.\n");
                return -1;
            }
        
            srcpad = gst_element_get_request_pad(tee, pad_name_src);
            if (!srcpad) {
                g_printerr ("tee request src pad failed. Exiting.\n");
                return -1;
            }
        
            if (gst_pad_link (srcpad, sinkpad) != GST_PAD_LINK_OK) {
                g_printerr ("Failed to link tee to stream muxer. Exiting.\n");
                return -1;
            }
        
            gst_object_unref (sinkpad);
            gst_object_unref (srcpad);
        }
    } else {
        /* Link v4l2src -> vidconv_src -> nvvidconv_src */
        if (!gst_element_link_many (source, caps_v4l2src, vidconv_src, caps_vidconv_src, NULL)) {
            g_printerr ("Elements could not be linked: 1. Exiting.\n");
            return -1;
        }

        /* Link the srcpad of videoconv_src to the sinkpad of streammux */
        GstPad *sinkpad, *srcpad;
        gchar pad_name_sink[16] = "sink_0";
        gchar pad_name_src[16] = "src";
        sinkpad = gst_element_get_request_pad (streammux, pad_name_sink);
         
        if (!sinkpad) {
            g_printerr ("Streammux request sink pad failed. Exiting.\n");
            return -1;
        }
        
        srcpad = gst_element_get_static_pad (caps_vidconv_src, pad_name_src);
        if (!srcpad) {
            g_printerr ("Decoder request src pad failed. Exiting.\n");
            return -1;
        }
        
        if (gst_pad_link (srcpad, sinkpad) != GST_PAD_LINK_OK) {
            g_printerr ("Failed to link decoder to stream muxer. Exiting.\n");
            return -1;
        }
        gst_object_unref (sinkpad);
        gst_object_unref (srcpad);

    }
    if (useDisplay == FALSE) {
        if (!gst_element_link_many (streammux, pgie, tiler,
                                    nvvidconv, nvdsosd, nvvidconv1, enc, parser1, sink, NULL)) {
            g_printerr ("Elements could not be linked: 2. Exiting.\n");
            return -1;
        }
    } else {
#ifdef PLATFORM_TEGRA
        if (!gst_element_link_many (streammux, pgie, tiler,
                                    nvvidconv, nvdsosd, transform, sink, NULL)) {
            g_printerr ("Elements could not be linked: 2. Exiting.\n");
            return -1;
        }
#else
        if (!gst_element_link_many (streammux, pgie, tiler,
                                    nvvidconv, nvdsosd, sink, NULL)) {
            g_printerr ("Elements could not be linked: 2. Exiting.\n");
            return -1;
        }
#endif
    }

    /* add probe to get informed of the meta data generated, we add probe to
     * the sink pad of the osd element, since by that time, the buffer would have
     * had got all the metadata. */
    osd_sink_pad = gst_element_get_static_pad (nvdsosd, "sink");
    if (!osd_sink_pad)
        g_print ("Unable to get sink pad\n");
    else
        gst_pad_add_probe (osd_sink_pad, GST_PAD_PROBE_TYPE_BUFFER,
          osd_sink_pad_buffer_probe, NULL, NULL);
    gst_object_unref (osd_sink_pad);


    /* Set the pipeline to "playing" state */
    g_print ("Now playing: %s\n", pgie_config.c_str());
    gst_element_set_state (pipeline, GST_STATE_PLAYING);

    /* Wait till pipeline encounters an error or EOS */
    g_print ("Running...\n");
    g_main_loop_run (loop);

    /* Out of the main loop, clean up nicely */
    g_print ("Returned, stopping playback\n");
    gst_element_set_state (pipeline, GST_STATE_NULL);
    g_print ("Deleting pipeline\n");
    gst_object_unref (GST_OBJECT (pipeline));
    g_source_remove (bus_watch_id);
    g_main_loop_unref (loop);
    return 0;
}
