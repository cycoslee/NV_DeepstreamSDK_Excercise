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

/* The muxer output resolution must be set if the input streams will be of
 * different resolution. The muxer will scale all the input frames to this
 * resolution. */
#define MUXER_OUTPUT_WIDTH 1920
#define MUXER_OUTPUT_HEIGHT 1080

/* Muxer batch formation timeout, for e.g. 40 millisec. Should ideally be set
 * based on the fastest source's framerate. */
#define MUXER_BATCH_TIMEOUT_USEC 40000

/* Callback function to receive messages from components
 * in the pipeline. */
static gboolean
bus_call (GstBus * bus, GstMessage * msg, gpointer data)
{
    GMainLoop *loop = (GMainLoop *) data;
    switch (GST_MESSAGE_TYPE (msg)) {
        case GST_MESSAGE_EOS:
            g_print ("End of stream\n");
            g_main_loop_quit (loop);
            break;
        case GST_MESSAGE_ERROR:{
            gchar *debug;
            GError *error;
            gst_message_parse_error (msg, &error, &debug);
            g_printerr ("ERROR from element %s: %s\n", \
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

/* Main function to execute pipeline */
int main (int argc, char *argv[])
{
    GMainLoop *loop = NULL;
    GstElement *pipeline = NULL, *source = NULL, *caps_v4l2src = NULL, *vidconv_src = NULL,
        *caps_vidconv_src = NULL, *streammux = NULL, *sink = NULL;
#ifdef PLATFORM_TEGRA
    GstElement *transform = NULL;
#endif
    GstCaps *caps = NULL;
    GstBus *bus = NULL;
    guint bus_watch_id;
    
    /* Check input arguments */
    if (argc != 2) {
        g_printerr ("Usage: %s <v4l2-device-path>\n", argv[0]);
        return -1;
    }

    /* 1. Standard GStreamer initialization */
    gst_init (&argc, &argv);
    loop = g_main_loop_new (NULL, FALSE);
    
    /* 2. Create gstreamer elements */
    /* Create Pipeline element that will form a connection of other elements */
    pipeline = gst_pipeline_new ("dstest-input-webcam");
    
    /* Source element for capturing from usb-camera */
    source = gst_element_factory_make ("v4l2src", "usb-camera");
    /* capsfilter for v4l2src */
    caps_v4l2src = gst_element_factory_make("capsfilter", "v4l2src_caps");
    
    /* nvvideoconvert element to convert incoming raw buffers to NVMM Mem (NvBufSurface API) */
    vidconv_src = gst_element_factory_make ("nvvideoconvert", "vidconv_src");
    /* capsfilter for nvvidconv_src */
    caps_vidconv_src = gst_element_factory_make ("capsfilter", "nvmm_caps");
    
    /* Create nvstreammux instance to form batches from one or more sources */
    streammux = gst_element_factory_make ("nvstreammux", "stream-muxer");
    
    /* Check elements are created properly */
    if (!pipeline || !source || !caps_v4l2src || !vidconv_src || !caps_vidconv_src) {
        g_printerr ("One element could not be created. Exiting.\n");
        return -1;
    }

    /* Finally render the mux output */
#ifdef PLATFORM_TEGRA
    transform = gst_element_factory_make ("nvegltransform", "nvegl-transform");
    if(!transform) {
        g_printerr ("One tegra element could not be created. Exiting.\n");
        return -1;
    }
#endif
    sink = gst_element_factory_make ("nveglglessink", "nvvideo-renderer");
    if (!sink) {
      g_printerr ("One element could not be created. Exiting.\n");
      return -1;
    }

    /* 3. Set up objects */
    /* Source setting */
    g_object_set (G_OBJECT (source), "device", argv[1], NULL);

    /* V4L2 source capsfilter setting */
    caps = gst_caps_from_string ("video/x-raw, width=640, height=480, framerate=30/1");
    g_object_set (G_OBJECT (caps_v4l2src), "caps", caps, NULL);

    /* nvvideo converter source capsfilter setting */
    caps = gst_caps_from_string ("video/x-raw(memory:NVMM), format=NV12, width=640, height=480, framerate=30/1");
    g_object_set (G_OBJECT (caps_vidconv_src), "caps", caps, NULL);

    /* Streammux setting */
    g_object_set (G_OBJECT (streammux), "width", MUXER_OUTPUT_WIDTH, "height", MUXER_OUTPUT_HEIGHT,
        "batch-size", 1, "batched-push-timeout", MUXER_BATCH_TIMEOUT_USEC, NULL);
    /* Renderer setting */
    g_object_set (G_OBJECT (sink), "sync", 0, NULL);
    
    /* 4. Set up the pipeline */
    /* Add a message handler */
    bus = gst_pipeline_get_bus (GST_PIPELINE (pipeline));
    bus_watch_id = gst_bus_add_watch (bus, bus_call, loop);
    gst_object_unref (bus);
    
    /* Add all elements into the pipeline */
#ifdef PLATFORM_TEGRA
    gst_bin_add_many (GST_BIN (pipeline),
      source, caps_v4l2src, vidconv_src, caps_vidconv_src, streammux, transform, sink, NULL);
#else
    gst_bin_add_many (GST_BIN (pipeline),
      source, caps_v4l2src, vidconv_src, caps_vidconv_src, streammux, sink, NULL);
#endif

    /* 5.Link the elements together */
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

  
    /* Link streammux -> video-render */
#ifdef PLATFORM_TEGRA
    if (!gst_element_link_many (streammux, transform, sink, NULL)) {
        g_printerr ("Elements could not be linked: 2. Exiting.\n");
        return -1;
    }
#else
    if (!gst_element_link_many (streammux, sink, NULL)) {
        g_printerr ("Elements could not be linked: 2. Exiting.\n");
        return -1;
    }
#endif

    /* Set the pipeline to "playing" state */
    g_print ("Now playing: %s\n", argv[1]);
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
