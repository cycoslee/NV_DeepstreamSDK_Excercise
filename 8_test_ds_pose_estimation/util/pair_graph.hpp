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
#pragma once

#include <memory>
#include <vector>

class PairGraph
{
public:
    PairGraph(int nrows, int ncols) : nrows(nrows), ncols(ncols)
    {
        this->rows.resize(nrows);
        this->cols.resize(ncols);
    }
  
    /**
     * Returns the column index of the pair matching this row
     */
    inline int colForRow(int row) const
    {
        return this->rows[row];
    }
  
    /**
     * Returns the row index of the pair matching this column
     */
    inline int rowForCol(int col) const
    {
        return this->cols[col];
    }
  
    /**
     * Creates a pair between row and col
     */
    inline void set(int row, int col)
    {
        this->rows[row] = col;
        this->cols[col] = row;
    }
  
    inline bool isRowSet(int row) const
    {
        return rows[row] >= 0;
    }
  
    inline bool isColSet(int col) const
    {
        return cols[col] >= 0;
    }
  
    inline bool isPair(int row, int col)
    {
        return rows[row] == col;
    }
  
    /**
     * Clears pair between row and col
     */
    inline void reset(int row, int col)
    {
        this->rows[row] = -1;
        this->cols[col] = -1;
    }
  
    /**
     * Clears all pairs in graph
     */
    void clear()
    {
        for (int i = 0; i < this->nrows; i++)
        {
            this->rows[i] = -1;
        }
        for (int j = 0; j < this->ncols; j++)
        {
            this->cols[j] = -1;
        }
    }
  
    int numPairs()
    {
        int count = 0;
        for (int i = 0; i < nrows; i++)
        {
            if (rows[i] >= 0) { count++; }
        }
        return count;
    }
  
    std::vector<std::pair<int, int>> pairs()
    {
        std::vector<std::pair<int, int>> p(numPairs());
        int count = 0;
        for (int i = 0; i < nrows; i++)
        {
            if (isRowSet(i)) { p[count++] = {i, colForRow(i)}; }
        }
        return p;
    }
  
    const int nrows;
    const int ncols;
  
private:
    std::vector<int> rows;
    std::vector<int> cols;
};
