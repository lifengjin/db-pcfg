#include "myarrays.h"
//#include "cusp/array2d.h"
//#include "cusp/array1d.h"
//#include "cusp/csr_matrix.h"

using namespace cusp;

//typedef cusp::array1d<int,cusp::device_memory> IndexArray;
//typedef cusp::array1d<float,cusp::device_memory> ValueArray;
//typedef typename IndexArray::view IndexArrayView;
//typedef typename ValueArray::view ValueArrayView;

//class Array : public array1d<float, device_memory>{
//public:
Array::Array(float* vals, int num_vals): array1d<float, device_memory>(vals, vals+num_vals){}
Array::Array(int num_vals, int val): array1d<float, device_memory>(num_vals, val){}
Array::Array(int num_vals) :array1d<float, device_memory>(num_vals){}
//};

//class Sparse : public csr_matrix<int, float, device_memory>{
//public:
Sparse::Sparse(int num_rows, int num_cols, int num_vals, int* row_offsets, int* col_ptrs, float* vals) : csr_matrix<int, float, device_memory>(num_rows, num_cols, num_vals){
        for (int i = 0; i <= num_rows; i++){
            this -> row_offsets[i] = row_offsets[i];
        }
        for (int i = 0; i < num_vals; i++){
            this -> column_indices[i] = col_ptrs[i];
            this -> values[i] = vals[i];
        }
    }
Sparse::Sparse(int num_rows, int num_cols, int num_vals): csr_matrix<int, float, device_memory>(num_rows, num_cols, num_vals){}
//};

//class SparseView {
//public:
SparseView::SparseView(int pi_num_rows, int pi_num_cols, int pi_vals_size, int* pi_row_offsets, int* pi_col_indices, float* pi_vals, int pi_row_offsets_size, int pi_col_indices_size) : device_vals(pi_vals, pi_vals_size), device_row_offsets(pi_row_offsets, pi_row_offsets+pi_row_offsets_size), device_col_ptrs(pi_col_indices, pi_col_indices+pi_col_indices_size), sparse_view(pi_num_rows, pi_num_cols, pi_vals_size, make_array1d_view(device_row_offsets), make_array1d_view(device_col_ptrs), make_array1d_view(device_vals)){
    }
csr_matrix_view<IndexArrayView, IndexArrayView, ValueArrayView>* SparseView::get_view(){
        
        return &sparse_view;
    }
SparseView::~SparseView(){
       // delete sparse_view;
       // delete device_col_ptrs;
       // delete device_row_offsets;
       // delete device_vals;
    }
//private:
//    Array device_vals;
//    array1d<int, device_memory> device_row_offsets;
//    array1d<int, device_memory> device_col_ptrs;
//    csr_matrix_view<IndexArrayView, IndexArrayView, ValueArrayView> sparse_view;
//};

//class Dense : public array2d<float, device_memory>{
//public:
Dense::Dense(int num_rows, int num_cols, int num_vals, float* vals) : array2d<float, device_memory>(num_rows, num_cols){
        for (int i = 0; i < num_rows; i++){
            for (int j = 0; j < num_cols; j ++) {
                this -> operator()(i, j) = vals[i * num_cols + j];
            }
        }
    }
Dense::Dense(int num_rows, int num_cols, float value) : array2d<float, device_memory>(num_rows, num_cols, value){}
//};

//class DenseView {
//public:
DenseView::DenseView(int num_rows, int num_cols, int num_vals, float* vals): device_vals(vals, num_vals), dense_view(num_rows, num_cols, num_cols, make_array1d_view(device_vals)){}
array2d_view<ValueArrayView, row_major>* DenseView::get_view(){
        return &dense_view;
    }
DenseView::~DenseView(){
       // delete dense_view;
       // delete device_vals;
    }
//private:
//    Array device_vals;
//    array2d_view<ValueArrayView, row_major> dense_view;
//};
//
//class DenseView: array2d_view<ValueArrayView, row_major> {
//public:
//    DenseView(int num_rows, int num_cols, int num_vals, float* vals):array2d_view<ValueArrayView, row_major>(num_rows, num_cols, num_cols, make_array1d_view(Array(vals, num_vals))){}
//};
