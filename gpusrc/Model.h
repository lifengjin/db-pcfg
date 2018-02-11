#ifndef Model_hpp
#define Model_hpp

//#include "myarrays.h"

class SparseView;
class DenseView;
class Dense;
class Array;
class Sparse;

class Model{
public:
    Model(int pi_num_rows,int pi_num_cols,float* pi_vals, int pi_vals_size, int* pi_row_offsets, int pi_row_offsets_size, int* pi_col_indices, int pi_col_indices_size, float* lex_vals, int lex_vals_size, int lex_num_rows, int lex_num_cols, int a_max, int b_max, int g_max, int depth);
    ~Model();
    int get_depth();
    int pi_num_rows;
    int pi_num_cols;
    float* pi_vals;
    int pi_vals_size;
    int* pi_row_offsets;
    int pi_row_offsets_size;
    int* pi_col_indices;
    int pi_col_indices_size;
    // these are precalculated for indexer
    int a_max;
    int b_max;
    int g_max;
    float* lex_vals;
    int lex_vals_size;
    int lex_num_rows;
    int lex_num_cols;
    const unsigned int depth = 2;
    DenseView * lex;
    SparseView * pi;
};

#endif