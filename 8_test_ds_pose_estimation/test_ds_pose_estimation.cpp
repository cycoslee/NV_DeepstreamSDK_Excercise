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

#include "gstnvdsmeta.h"
#include "nvdsgstutils.h"
#include "nvbufsurface.h"

#include <vector>
#include <array>
#include <queue>
#include <cmath>
#include <string>

#include "util/post_process.hpp"

#define EPS 1e-6

#define MAX_DISPLAY_LEN 64

/* The muxer output resolution must be set if the input streams will be of
 * different resolution. The muxer will scale all the input frames to this
 * resolution. */
#define MUXER_OUTPUT_WIDTH 1920
#define MUXER_OUTPUT_HEIGHT 1080

/* Muxer batch formation timeout, for e.g. 40 millisec. Should ideally be set
 * based on the fastest source's framerate. */
#define MUXER_BATCH_TIMEOUT_USEC 4000000

template <class T>
using Vec1D = std::vector<T>;
template <class T>
using Vec2D = std::vector<Vec1D<T>>;
template <class T>
using Vec3D = std::vector<Vec2D<T>>;

/* FPS measurement */
gint frame_number = 0;
gdouble t_prev=0.0;
gdouble fps_val = 0.0;
static void perf_fps(gint frame_interval){
    struct timeval tv;
    glong t_curr=0.0;

    gettimeofday(&tv,NULL);
    t_curr = tv.tv_sec *1000LL + tv.tv_usec / 1000;
    fps_val = (gdouble)(1000 * frame_interval)/(t_curr - t_prev);
    t_prev = t_curr;
}

/* Method to parse information returned from the model */
std::tuple<Vec2D<int>, Vec3D<float>>
parse_objects_from_tensor_meta(NvDsInferTensorMeta *tensor_meta)
{
    Vec1D<int> counts;
    Vec3D<int> peaks;
    
    float threshold = 0.1;
    int window_size = 5;
    int max_num_parts = 2;
    int num_integral_samples = 7;
    float link_threshold = 0.1;
    int max_num_objects = 100;
    
    void *cmap_data = tensor_meta->out_buf_ptrs_host[0];
    NvDsInferDims &cmap_dims = tensor_meta->output_layers_info[0].inferDims;
    void *paf_data = tensor_meta->out_buf_ptrs_host[1];
    NvDsInferDims &paf_dims = tensor_meta->output_layers_info[1].inferDims;
    
    /* Finding peaks within a given window */
    find_peaks(counts, peaks, cmap_data, cmap_dims, threshold, window_size, max_num_parts);
    /* Non-Maximum Suppression */
    Vec3D<float> refined_peaks = refine_peaks(counts, peaks, cmap_data, cmap_dims, window_size);
    /* Create a Bipartite graph to assign detected body-parts to a unique person in the frame */
    Vec3D<float> score_graph = paf_score_graph(paf_data, paf_dims, topology, counts, refined_peaks, num_integral_samples);
    /* Assign weights to all edges in the bipartite graph generated */
    Vec3D<int> connections = assignment(score_graph, topology, counts, link_threshold, max_num_parts);
    /* Connecting all the Body Parts and Forming a Human Skeleton */
    Vec2D<int> objects = connect_parts(connections, topology, counts, max_num_objects);
    return {objects, refined_peaks};
}

/* MetaData to handle drawing onto the on-screen-display */
static void
create_display_meta(Vec2D<int> &objects, Vec3D<float> &normalized_peaks, NvDsFrameMeta *frame_meta, int frame_width, int frame_height)
{
    int K = topology.size();
    int count = objects.size();
    NvDsBatchMeta *bmeta = frame_meta->base_meta.batch_meta;
    NvDsDisplayMeta *dmeta = nvds_acquire_display_meta_from_pool(bmeta);
    nvds_add_display_meta_to_frame(frame_meta, dmeta);
    
    for (auto &object : objects)
    {
        int C = object.size();
        for (int j = 0; j < C; j++)
        {
            int k = object[j];
            if (k >= 0)
            {
                auto &peak = normalized_peaks[j][k];
                int x = peak[1] * MUXER_OUTPUT_WIDTH;
                int y = peak[0] * MUXER_OUTPUT_HEIGHT;
                if (dmeta->num_circles == MAX_ELEMENTS_IN_DISPLAY_META)
                {
                    dmeta = nvds_acquire_display_meta_from_pool(bmeta);
                    nvds_add_display_meta_to_frame(frame_meta, dmeta);
                }
                NvOSD_CircleParams &cparams = dmeta->circle_params[dmeta->num_circles];
                cparams.xc = x;
                cparams.yc = y;
                cparams.radius = 8;
                cparams.circle_color = NvOSD_ColorParams{244, 67, 54, 1};
                cparams.has_bg_color = 1;
                cparams.bg_color = NvOSD_ColorParams{0, 255, 0, 1};
                dmeta->num_circles++;
            }
        }
      
        for (int k = 0; k < K; k++)
        {
            int c_a = topology[k][2];
            int c_b = topology[k][3];
            if (object[c_a] >= 0 && object[c_b] >= 0)
            {
                auto &peak0 = normalized_peaks[c_a][object[c_a]];
                auto &peak1 = normalized_peaks[c_b][object[c_b]];
                int x0 = peak0[1] * MUXER_OUTPUT_WIDTH;
                int y0 = peak0[0] * MUXER_OUTPUT_HEIGHT;
                int x1 = peak1[1] * MUXER_OUTPUT_WIDTH;
                int y1 = peak1[0] * MUXER_OUTPUT_HEIGHT;
                if (dmeta->num_lines == MAX_ELEMENTS_IN_DISPLAY_META)
                {
                    dmeta = nvds_acquire_display_meta_from_pool(bmeta);
                    nvds_add_display_meta_to_frame(frame_meta, dmeta);
                }
                NvOSD_LineParams &lparams = dmeta->line_params[dmeta->num_lines];
                lparams.x1 = x0;
                lparams.x2 = x1;
                lparams.y1 = y0;
                lparams.y2 = y1;
                lparams.line_width = 3;
                lparams.line_color = NvOSD_ColorParams{0, 255, 0, 1};
                dmeta->num_lines++;
            }
        }
    }
}

/* pgie_src_pad_buffer_probe  will extract metadata received from pgie
 * and update params for drawing rectangle, object information etc. */
static GstPadProbeReturn
pgie_src_pad_buffer_probe(GstPad *pad, GstPadProbeInfo *info, gpointer u_data)
{
    gchar *msg = NULL;
    GstBuffer *buf = (GstBuffer *)info->data;
    NvDsMetaList *l_frame = NULL;
    NvDsMetaList *l_obj = NULL;
    NvDsMetaList *l_user = NULL;
    NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta(buf);
  
    for (l_frame = batch_meta->frame_meta_list; l_frame != NULL;
         l_frame = l_frame->next)
    {
        NvDsFrameMeta *frame_meta = (NvDsFrameMeta *)(l_frame->data);
  
        for (l_user = frame_meta->frame_user_meta_list; l_user != NULL;
             l_user = l_user->next)
        {
            NvDsUserMeta *user_meta = (NvDsUserMeta *)l_user->data;
            if (user_meta->base_meta.meta_type == NVDSINFER_TENSOR_OUTPUT_META)
            {
                NvDsInferTensorMeta *tensor_meta =
                    (NvDsInferTensorMeta *)user_meta->user_meta_data;
                Vec2D<int> objects;
                Vec3D<float> normalized_peaks;
                tie(objects, normalized_peaks) = parse_objects_from_tensor_meta(tensor_meta);
                create_display_meta(objects, normalized_peaks, frame_meta, frame_meta->source_frame_width, frame_meta->source_frame_height);
            }
        }
  
        for (l_obj = frame_meta->obj_meta_list; l_obj != NULL;
             l_obj = l_obj->next)
        {
            NvDsObjectMeta *obj_meta = (NvDsObjectMeta *)l_obj->data;
            for (l_user = obj_meta->obj_user_meta_list; l_user != NULL;
                 l_user = l_user->next)
            {
                NvDsUserMeta *user_meta = (NvDsUserMeta *)l_user->data;
                if (user_meta->base_meta.meta_type == NVDSINFER_TENSOR_OUTPUT_META)
                {
                    NvDsInferTensorMeta *tensor_meta =
                        (NvDsInferTensorMeta *)user_meta->user_meta_data;
                    Vec2D<int> objects;
                    Vec3D<float> normalized_peaks;
                    tie(objects, normalized_peaks) = parse_objects_from_tensor_meta(tensor_meta);
                    create_display_meta(objects, normalized_peaks, frame_meta, frame_meta->source_frame_width, frame_meta->source_frame_height);
                }
            }
        }
    }
    return GST_PAD_PROBE_OK;
}

/* osd_sink_pad_buffer_probe  will extract metadata received from OSD
 * and update params for drawing rectangle, object information etc. */
static GstPadProbeReturn
osd_sink_pad_buffer_probe(GstPad *pad, GstPadProbeInfo *info, gpointer u_data)
{
    GstBuffer *buf = (GstBuffer *)info->data;
    guint num_rects = 0;
    NvDsObjectMeta *obj_meta = NULL;
    NvDsMetaList *l_frame = NULL;
    NvDsMetaList *l_obj = NULL;
    NvDsDisplayMeta *display_meta = NULL;
  
    NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta(buf);
  
    gint perf_interval = 100;
    if( frame_number % perf_interval == 0 ){
        perf_fps(perf_interval);
    }
  
    for (l_frame = batch_meta->frame_meta_list; l_frame != NULL;
         l_frame = l_frame->next)
    {
        NvDsFrameMeta *frame_meta = (NvDsFrameMeta *)(l_frame->data);
        int offset = 0;
        for (l_obj = frame_meta->obj_meta_list; l_obj != NULL; l_obj = l_obj->next)
        {
            obj_meta = (NvDsObjectMeta *)(l_obj->data);
        }
        display_meta = nvds_acquire_display_meta_from_pool(batch_meta);
  
        /* Parameters to draw text onto the On-Screen-Display */
        NvOSD_TextParams *txt_params = &display_meta->text_params[0];
        display_meta->num_labels = 1;
        txt_params->display_text = (char *)g_malloc0(MAX_DISPLAY_LEN);
        offset = snprintf(txt_params->display_text, MAX_DISPLAY_LEN, "FPS =  %0.1f", fps_val);
  
        txt_params->x_offset = 10;
        txt_params->y_offset = 12;
  
        txt_params->font_params.font_name = "Mono";
        txt_params->font_params.font_size = 10;
        txt_params->font_params.font_color.red = 1.0;
        txt_params->font_params.font_color.green = 1.0;
        txt_params->font_params.font_color.blue = 1.0;
        txt_params->font_params.font_color.alpha = 1.0;
  
        txt_params->set_bg_clr = 1;
        txt_params->text_bg_clr.red = 0.0;
        txt_params->text_bg_clr.green = 0.0;
        txt_params->text_bg_clr.blue = 0.0;
        txt_params->text_bg_clr.alpha = 1.0;
  
        nvds_add_display_meta_to_frame(frame_meta, display_meta);
    }
    frame_number++;
    return GST_PAD_PROBE_OK;
}

static gboolean
bus_call(GstBus *bus, GstMessage *msg, gpointer data)
{
    GMainLoop *loop = (GMainLoop *)data;
    switch (GST_MESSAGE_TYPE(msg)) {
    case GST_MESSAGE_EOS:
        g_print("End of Stream\n");
        g_main_loop_quit(loop);
        break;
  
    case GST_MESSAGE_ERROR:
        gchar *debug;
        GError *error;
        gst_message_parse_error(msg, &error, &debug);
        g_printerr("ERROR from element %s: %s\n",
                   GST_OBJECT_NAME(msg->src), error->message);
        if (debug)
          g_printerr("Error details: %s\n", debug);
        g_free(debug);
        g_error_free(error);
        g_main_loop_quit(loop);
        break;
    default:
        break;
    }
    return TRUE;
}

/* Main function to execute pipeline */
int main(int argc, char *argv[])
{
    GMainLoop *loop = NULL;
    GstElement *pipeline = NULL, *source = NULL, *caps_v4l2src = NULL, *vidconv_src = NULL,
               *caps_vidconv_src = NULL, *streammux = NULL, *sink = NULL, *pgie = NULL,
               *nvvidconv = NULL, *nvosd = NULL, *nvvideoconvert = NULL, *nvsink = NULL;
#ifdef PLATFORM_TEGRA
    GstElement *transform = NULL;
#endif
    GstCaps *caps = NULL;
    GstBus *bus = NULL;
    guint bus_watch_id;
    GstPad *osd_sink_pad = NULL;

    /* Check input arguments */
    if (argc != 2)
    {
      g_printerr("Usage: %s <webcam device>\n", argv[0]);
      return -1;
    }
    
    /* 1. Standard GStreamer initialization */
    gst_init(&argc, &argv);
    loop = g_main_loop_new(NULL, FALSE);
    
    /* 2. Create gstreamer elements */
    /* Create Pipeline element that will form a connection of other elements */
    pipeline = gst_pipeline_new("deepstream-tensorrt-openpose-pipeline");

    /* Source element for capturing from usb-camera */
    source = gst_element_factory_make ("v4l2src", "usb-camera");
    /* capsfilter for v4l2src */
    caps_v4l2src = gst_element_factory_make("capsfilter", "v4l2src_caps");
    
    /* nvvideoconvert element to convert incoming raw buffers to NVMM Mem (NvBufSurface API) */
    vidconv_src = gst_element_factory_make ("nvvideoconvert", "vidconv_src");
    /* capsfilter for nvvidconv_src */
    caps_vidconv_src = gst_element_factory_make ("capsfilter", "nvmm_caps");
    
    /* Create nvstreammux instance to form batches from one or more sources. */
    streammux = gst_element_factory_make("nvstreammux", "stream-muxer");
    
    /* Check elements are created properly */
    if (!pipeline || !source || !caps_v4l2src || !vidconv_src || !caps_vidconv_src) {
        g_printerr ("One element could not be created. Exiting.\n");
        return -1;
    }
    
    /* Use nvinfer to run inferencing on decoder's output,
     * behaviour of inferencing is set through config file */
    pgie = gst_element_factory_make("nvinfer", "primary-nvinference-engine");
    
    /* Use convertor to convert from NV12 to RGBA as required by nvosd */
    nvvidconv = gst_element_factory_make("nvvideoconvert", "nvvideo-converter");
    
    /* Create OSD to draw on the converted RGBA buffer */
    nvosd = gst_element_factory_make("nvdsosd", "nv-onscreendisplay");
    
    /* Finally render the osd output */
#ifdef PLATFORM_TEGRA
    transform = gst_element_factory_make("nvegltransform", "nvegl-transform");
#endif
    sink = gst_element_factory_make("nveglglessink", "nvvideo-renderer");

    if (!source || !pgie || !nvvidconv || !nvosd || !sink )
    {
      g_printerr("One element could not be created. Exiting.\n");
      return -1;
    }
#ifdef PLATFORM_TEGRA
    if (!transform)
    {
      g_printerr("One tegra element could not be created. Exiting.\n");
      return -1;
    }
#endif

    /* 3. Set up objects */
    /* Source setting */
    g_object_set (G_OBJECT (source), "device", argv[1], NULL);

    /* V4L2 source capsfilter setting */
    caps = gst_caps_from_string ("video/x-raw, width=640, height=480, framerate=30/1");
    g_object_set (G_OBJECT (caps_v4l2src), "caps", caps, NULL);

    /* nvvideo converter source capsfilter setting */
    caps = gst_caps_from_string ("video/x-raw(memory:NVMM), format=NV12");
    g_object_set (G_OBJECT (caps_vidconv_src), "caps", caps, NULL);

    /* Streammux setting */
    g_object_set (G_OBJECT (streammux), "width", MUXER_OUTPUT_WIDTH, "height", MUXER_OUTPUT_HEIGHT,
        "batch-size", 1, "batched-push-timeout", MUXER_BATCH_TIMEOUT_USEC, NULL);

    /* Set all the necessary properties of the nvinfer element,
     * the necessary ones are : */
    g_object_set(G_OBJECT(pgie), "output-tensor-meta", TRUE,
                 "config-file-path", "config/deepstream_pose_estimation_config.txt", NULL);
    g_object_set(G_OBJECT(sink), "sync", 0, NULL);
  
    /* 4. Set up the pipeline */
    /* Add a message handler */
    bus = gst_pipeline_get_bus(GST_PIPELINE(pipeline));
    bus_watch_id = gst_bus_add_watch(bus, bus_call, loop);
    gst_object_unref(bus);
  
    /* Add all elements into the pipeline */
#ifdef PLATFORM_TEGRA
    gst_bin_add_many(GST_BIN(pipeline),
                   source, caps_v4l2src, vidconv_src, caps_vidconv_src, streammux, pgie,
                   nvvidconv, nvosd, transform, sink, NULL );
#else
    gst_bin_add_many(GST_BIN(pipeline),
                   source, caps_v4l2src, vidconv_src, caps_vidconv_src, streammux, pgie,
                   nvvidconv, nvosd, sink, NULL );
#endif
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

    /* Link streammux -> pgie -> nvvidconv -> nvosd -> video-render */
#ifdef PLATFORM_TEGRA
    if (!gst_element_link_many (streammux, pgie, nvvidconv, nvosd, transform, sink, NULL)) {
        g_printerr ("Elements could not be linked: 2. Exiting.\n");
        return -1;
    }
#else
    if (!gst_element_link_many (streammux, pgie, nvvidconv, nvosd, sink, NULL)) {
        g_printerr ("Elements could not be linked: 2. Exiting.\n");
        return -1;
    }
#endif

    /* Lets add probe to get informed of the meta data generated, we add probe to
     * the sink pad of the osd element, since by that time, the buffer would have
     * had got all the metadata. */
    GstPad *pgie_src_pad = gst_element_get_static_pad(pgie, "src");
    if (!pgie_src_pad)
        g_print("Unable to get pgie src pad\n");
    else
        gst_pad_add_probe(pgie_src_pad, GST_PAD_PROBE_TYPE_BUFFER,
                        pgie_src_pad_buffer_probe, (gpointer)sink, NULL);
  
    /* Lets add probe to get informed of the meta data generated, we add probe to
     * the sink pad of the osd element, since by that time, the buffer would have
     * had got all the metadata. */
    osd_sink_pad = gst_element_get_static_pad(nvosd, "sink");
    if (!osd_sink_pad)
        g_print("Unable to get sink pad\n");
    else
        gst_pad_add_probe(osd_sink_pad, GST_PAD_PROBE_TYPE_BUFFER,
                        osd_sink_pad_buffer_probe, (gpointer)sink, NULL);
  
    /* Set the pipeline to "playing" state */
    g_print("Now playing: %s\n", argv[1]);
    gst_element_set_state(pipeline, GST_STATE_PLAYING);
  
    /* Wait till pipeline encounters an error or EOS */
    g_print("Running...\n");
    g_main_loop_run(loop);
  
    /* Out of the main loop, clean up nicely */
    g_print("Returned, stopping playback\n");
    gst_element_set_state(pipeline, GST_STATE_NULL);
    g_print("Deleting pipeline\n");
    gst_object_unref(GST_OBJECT(pipeline));
    g_source_remove(bus_watch_id);
    g_main_loop_unref(loop);
    return 0;
}
