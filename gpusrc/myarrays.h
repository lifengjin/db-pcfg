#ifndef myarrays_hpp
#define myarrays_hpp

#include <iostream>
#include "cusp/array2d.h"
#include "cusp/array1d.h"
#include "cusp/csr_matrix.h"
using namespace cusp;
typedef cusp::array1d<int,cusp::device_memory> IndexArray;
typedef cusp::array1d<float,cusp::device_memory> ValueArray;
typedef typename IndexArray::view IndexArrayView;
typedef typename ValueArray::view ValueArrayView;

class Array : public array1d<float, device_memory>{
public:
    Array(float* vals, int num_vals);
    Array(int num_vals, int val);
    Array(int num_vals);
};

class Sparse : public csr_matrix<int, float, device_memory>{
public:
    Sparse(int num_rows, int num_cols, int num_vals, int* row_offsets, int* col_ptrs, float* vals);
    Sparse(int num_rows, int num_cols, int num_vals);
};

class Dense : public array2d<float, device_memory>{
public:
    Dense(int num_rows, int num_cols, int num_vals, float* vals);
    Dense(int num_rows, int num_cols, float value);
};

class SparseView {
public:
    SparseView(int pi_num_rows, int pi_num_cols, int pi_vals_size, int* pi_row_offsets, int* pi_col_indices, float* pi_vals, int pi_row_offsets_size, int pi_col_indices_size);
    csr_matrix_view<IndexArrayView, IndexArrayView, ValueArrayView>* get_view();
    ~SparseView();
private:
    Array device_vals;
    array1d<int, device_memory> device_row_offsets;
    array1d<int, device_memory> device_col_ptrs;
    csr_matrix_view<IndexArrayView, IndexArrayView, ValueArrayView> sparse_view;
};

class DenseView {
public:
    DenseView(int num_rows, int num_cols, int num_vals, float* vals);
    array2d_view<ValueArrayView, row_major>* get_view();
    ~DenseView();
private:
    Array device_vals;
    array2d_view<ValueArrayView, row_major> dense_view;
};

#endif /* myarrays_hpp */
