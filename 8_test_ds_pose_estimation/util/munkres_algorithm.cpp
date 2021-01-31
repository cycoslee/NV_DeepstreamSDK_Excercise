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

#include "munkres_algorithm.hpp"

// Helper method to subtract the minimum row from cost_graph
void subtract_minimum_row(Vec2D<float> &cost_graph, int nrows, int ncols)
{
    for (int i = 0; i < nrows; i++)
    {
        // Iterate the find the minimum
        float min = cost_graph[i][0];
        for (int j = 0; j < ncols; j++)
        {
            float val = cost_graph[i][j];
            if (val < min)
            {
                min = val;
            }
        }
  
        // Subtract the Minimum
        for (int j = 0; j < ncols; j++)
        {
            cost_graph[i][j] -= min;
        }
    }
}

// Helper method to subtract the minimum col from cost_graph
void subtract_minimum_column(Vec2D<float> &cost_graph, int nrows, int ncols)
{
    for (int j = 0; j < ncols; j++)
    {
        // Iterate and find the minimum
        float min = cost_graph[0][j];
        for (int i = 0; i < nrows; i++)
        {
            float val = cost_graph[i][j];
            if (val < min)
            {
                min = val;
            }
        }
  
        // Subtract the minimum
        for (int i = 0; i < nrows; i++)
        {
            cost_graph[i][j] -= min;
        }
    }
}

void munkresStep1(Vec2D<float> &cost_graph, PairGraph &star_graph, int nrows,
                  int ncols)
{
    for (int i = 0; i < nrows; i++)
    {
        for (int j = 0; j < ncols; j++)
        {
            if (!star_graph.isRowSet(i) && !star_graph.isColSet(j) && (cost_graph[i][j] == 0))
            {
                star_graph.set(i, j);
            }
        }
    }
}

// Exits if '1' is returned
bool munkresStep2(const PairGraph &star_graph, CoverTable &cover_table)
{
    int k = star_graph.nrows < star_graph.ncols ? star_graph.nrows : star_graph.ncols;
    int count = 0;
    for (int j = 0; j < star_graph.ncols; j++)
    {
        if (star_graph.isColSet(j))
        {
            cover_table.coverCol(j);
            count++;
        }
    }
    return count >= k;
}

bool munkresStep3(Vec2D<float> &cost_graph, const PairGraph &star_graph,
                  PairGraph &prime_graph, CoverTable &cover_table, std::pair<int, int> &p,
                  int nrows, int ncols)
{
    for (int i = 0; i < nrows; i++)
    {
        for (int j = 0; j < ncols; j++)
        {
            if (cost_graph[i][j] == 0 && !cover_table.isCovered(i, j))
            {
                prime_graph.set(i, j);
                if (star_graph.isRowSet(i))
                {
                    cover_table.coverRow(i);
                    cover_table.uncoverCol(star_graph.colForRow(i));
                }
                else
                {
                    p.first = i;
                    p.second = j;
                    return 1;
                }
            }
        }
    }
    return 0;
};

void munkresStep4(PairGraph &star_graph, PairGraph &prime_graph,
                  CoverTable &cover_table, std::pair<int, int> &p)
{
    // This process should be repeated until no star is found in prime's column
    while (star_graph.isColSet(p.second))
    {
        // First find and reset any star found in the prime's columns
        std::pair<int, int> s = {star_graph.rowForCol(p.second), p.second};
        star_graph.reset(s.first, s.second);
  
        // Set this prime to a star
        star_graph.set(p.first, p.second);
  
        // Repeat the same process for prime in cleared star's row
        p = {s.first, prime_graph.colForRow(s.first)};
    }
    star_graph.set(p.first, p.second);
    cover_table.clear();
    prime_graph.clear();
}

void munkresStep5(Vec2D<float> &cost_graph, const CoverTable &cover_table,
                  int nrows, int ncols)
{
    bool valid = false;
    float min;
    for (int i = 0; i < nrows; i++)
    {
        for (int j = 0; j < ncols; j++)
        {
            if (!cover_table.isCovered(i, j))
            {
                if (!valid)
                {
                    min = cost_graph[i][j];
                    valid = true;
                }
                else if (cost_graph[i][j] < min)
                {
                    min = cost_graph[i][j];
                }
            }
        }
    }
  
    for (int i = 0; i < nrows; i++)
    {
        if (cover_table.isRowCovered(i))
        {
            for (int j = 0; j < ncols; j++)
            {
                cost_graph[i][j] += min;
            }
        }
    }
    for (int j = 0; j < ncols; j++)
    {
        if (!cover_table.isColCovered(j))
        {
            for (int i = 0; i < nrows; i++)
            {
                cost_graph[i][j] -= min;
            }
        }
    }
}

void munkres_algorithm(Vec2D<float> &cost_graph, PairGraph &star_graph, int nrows,
              int ncols)
{
    PairGraph prime_graph(nrows, ncols);
    CoverTable cover_table(nrows, ncols);
    prime_graph.clear();
    cover_table.clear();
    star_graph.clear();
  
    int step = 0;
    if (ncols >= nrows)
    {
        subtract_minimum_row(cost_graph, nrows, ncols);
    }
    if (ncols > nrows)
    {
        step = 1;
    }
  
    std::pair<int, int> p;
    bool done = false;
    while (!done)
    {
        switch (step)
        {
        case 0:
            subtract_minimum_column(cost_graph, nrows, ncols);
        case 1:
            munkresStep1(cost_graph, star_graph, nrows, ncols);
        case 2:
            if (munkresStep2(star_graph, cover_table))
            {
              done = true;
              break;
            }
        case 3:
            if (!munkresStep3(cost_graph, star_graph, prime_graph, cover_table, p,
                              nrows, ncols))
            {
              step = 5;
              break;
            }
        case 4:
            munkresStep4(star_graph, prime_graph, cover_table, p);
            step = 2;
            break;
        case 5:
            munkresStep5(cost_graph, cover_table, nrows, ncols);
            step = 3;
            break;
        }
    }
}
