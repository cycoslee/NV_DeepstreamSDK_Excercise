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

#include "pair_graph.hpp"
#include "cover_table.hpp"
#include "munkres_algorithm.hpp"

#include <gst/gst.h>
#include <glib.h>
#include <stdio.h>

#include "gstnvdsmeta.h"
#include "gstnvdsinfer.h"
#include "nvdsgstutils.h"
#include "nvbufsurface.h"

#include <stdio.h>
#include <vector>
#include <array>
#include <queue>
#include <cmath>

#define EPS 1e-6

template <class T>
using Vec1D = std::vector<T>;
template <class T>
using Vec2D = std::vector<Vec1D<T>>;
template <class T>
using Vec3D = std::vector<Vec2D<T>>;

static const int M = 2;

static Vec2D<int> topology{
    {0, 1, 15, 13},
    {2, 3, 13, 11},
    {4, 5, 16, 14},
    {6, 7, 14, 12},
    {8, 9, 11, 12},
    {10, 11, 5, 7},
    {12, 13, 6, 8},
    {14, 15, 7, 9},
    {16, 17, 8, 10},
    {18, 19, 1, 2},
    {20, 21, 0, 1},
    {22, 23, 0, 2},
    {24, 25, 1, 3},
    {26, 27, 2, 4},
    {28, 29, 3, 5},
    {30, 31, 4, 6},
    {32, 33, 17, 0},
    {34, 35, 17, 5},
    {36, 37, 17, 6},
    {38, 39, 17, 11},
    {40, 41, 17, 12}};

/* Method to find peaks in the output tensor. 'window_size' represents how many pixels we are considering at once to find a maximum value, or a peak. 
   Once we find a peak, we mark it using the is_peak boolean in the inner loop and assign this maximum value to the center pixel of our window. 
   This is then repeated until we cover the entire frame. */
void find_peaks(Vec1D<int> &counts_out, Vec3D<int> &peaks_out, void *cmap_data,
                NvDsInferDims &cmap_dims, float threshold, int window_size, int max_count);

/* Normalize the peaks found in 'find_peaks' and apply non-maximal suppression*/
Vec3D<float>
refine_peaks(Vec1D<int> &counts,
             Vec3D<int> &peaks, void *cmap_data, NvDsInferDims &cmap_dims,
             int window_size);

/* Create a bipartite graph to assign detected body-parts to a unique person in the frame. This method also takes care of finding the line integral to assign scores
   to these points */
Vec3D<float>
paf_score_graph(void *paf_data, NvDsInferDims &paf_dims,
                Vec2D<int> &topology, Vec1D<int> &counts,
                Vec3D<float> &peaks, int num_integral_samples);

/*
 This method takes care of solving the graph assignment problem using Munkres algorithm. Munkres algorithm is defind in 'munkres_algorithm.cpp'
 */
Vec3D<int>
assignment(Vec3D<float> &score_graph,
           Vec2D<int> &topology, Vec1D<int> &counts, float score_threshold, int max_count);

/* This method takes care of connecting all the body parts detected to each other 
   after finding the relationships between them in the 'assignment' method */
Vec2D<int>
connect_parts(
    Vec3D<int> &connections, Vec2D<int> &topology, Vec1D<int> &counts,
    int max_count);
