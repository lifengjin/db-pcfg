
#include "HmmSampler.h"
#include <stdio.h>
#include <thrust/transform.h>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>
#include <thrust/scatter.h>
#include <tuple>
#include <ctime>
#include <utility>
#include <cmath>
#include <algorithm>
#include "cusp/array1d.h"
#include "cusp/array2d.h"
#include "cusp/csr_matrix.h"
#include "cusp/blas/blas.h"
#include "cusp/multiply.h"
#include "cusp/elementwise.h"
#include "cusp/functional.h"
#include "cusp/print.h"
#include "myarrays.cu"
#include "Indexer.cu"
#include "State.cu"
#include <chrono> // for testing
#include <random>
#include <limits>
#include <iostream>
#include <iomanip>
using namespace cusp;
using namespace std;
typedef std::chrono::high_resolution_clock Clock;
float nano_to_sec = 1.0e-9f;
__device__ int G_SIZE;
typedef std::numeric_limits<float> float_limit;
typedef cusp::array1d<float,cusp::device_memory> cusparray;
typedef cusparray::view ArrayView;

Model::Model(int pi_num_rows, int pi_num_cols, float* pi_vals, int pi_vals_size, int* pi_row_offsets,
int pi_row_offsets_size, int* pi_col_indices, int pi_col_indices_size, float* lex_vals, int lex_vals_size,
int lex_num_rows, int lex_num_cols, int a_max, int b_max, int g_max, int depth, float* pos_vals, int pos_vals_size,
int EOS_index):
pi_num_rows(pi_num_rows), pi_num_cols(pi_num_cols), pi_vals(pi_vals),
pi_vals_size(pi_vals_size), pi_row_offsets(pi_row_offsets), pi_row_offsets_size(pi_row_offsets_size),
pi_col_indices(pi_col_indices), pi_col_indices_size(pi_col_indices_size), lex_vals(lex_vals),
lex_vals_size(lex_vals_size), lex_num_rows(lex_num_rows), lex_num_cols(lex_num_cols), a_max(a_max),
b_max(b_max), g_max(g_max), depth(depth), pos_vals(pos_vals), pos_vals_size(pos_vals_size), EOS_index(EOS_index){
    pi = new SparseView(pi_num_rows, pi_num_cols, pi_vals_size, pi_row_offsets, pi_col_indices, pi_vals, pi_row_offsets_size, pi_col_indices_size);
    lex = new DenseView(lex_num_rows, lex_num_cols, lex_vals_size, lex_vals);
    pos = new Array(pos_vals, pos_vals_size);
}

Model::~Model(){
    delete lex;
    delete pi;
    delete pos;
}

int Model::get_depth(){
    return depth;
}

// taken away the new tensor
template <class AView>
int HmmSampler::get_sample(AView &v){
    float dart;
//    cusp::print(v);
//     cout << "get_sample()" << endl;
    // array1d<float, device_memory> sum_dict(v.size()); // building a new array, maybe not needed
    // this is the equivalent of np.cumsum() or partial_sum in the stl:
    thrust::inclusive_scan(thrust::device, v.begin(), v.end(), sum_dict->begin());
    // cout << "v[0] = " << v[0] << "v[-1] = " << v[v.size()-1] << " with length " << v.size() << endl;
//     cout << "sum_dict[0] = " << (*sum_dict)[0] << "sum_dict[-1] = " << (*sum_dict)[sum_dict->size()-1] << " with length: " << sum_dict->size() <<  endl;
    //cusp::print(*sum_dict);
//     cout << "sum done" << endl;
    // dart =  static_cast <float> (rand()) / static_cast <float> (RAND_MAX);// / RAND_MAX;
    int dart_target;
    int condition = 1;
    while (condition) {
        dart = dist(mt);
        if (dart != 0.0f and dart != 1.0f) {
//            cout << "dart: "<< scientific << dart << endl;
           dart_target = thrust::upper_bound(thrust::device, sum_dict->begin(), sum_dict->end(), dart) - sum_dict->begin();
           if (dart_target == 0) {
           print(*sum_dict);
           }
//           cout << "dart target (summed): " << dart_target << " " << scientific << (*sum_dict)[dart_target - 1] << endl; //<< " " << scientific << (*sum_dict)[dart_target] << " " << scientific  << (*sum_dict)[dart_target+ 1] << endl;
//           cout << "dart target (summed): " << dart_target << " " << scientific << v[dart_target - 1] << " " << scientific << v[dart_target] << " " << scientific  << v[dart_target+ 1] << endl;
//            cout << "dart target (summed): " << dart_target << " with v size=" << v.size() << endl;
//            float minus_one = (*sum_dict)[dart_target - 1];
//            printf( "%A" , minus_one);
//            cout  << " ";
//            float on_target = (*sum_dict)[dart_target];
//            printf("%A", on_target);
//            cout << " ";
//            float plus_one = (*sum_dict)[dart_target+ 1];
//            printf("%A\n", plus_one);
           if (dart_target != v.size() && v[dart_target] != 0.0f ){
                condition = 0;
//                cout << "out of loop" << endl;
           }
        }
     }
//      cout << "get_sample() done." << endl;
//      cout << "dart_target: " << dart_target << endl;
     return dart_target;
}

class cuda_exp : public thrust::unary_function<float, float>{
public:
    __device__
   float operator()(float x){
       return pow(10,x);
   }
};
class calc_sparse_column : public thrust::unary_function<int, int>{
public:
   __device__
   int operator()(int x){
       return x % G_SIZE;
   }
};

class g_integer_division : public thrust::unary_function<int, int>{
public:
    __device__
   int operator()(int x){
       return x / G_SIZE;
   }
};

template <class AView>
void exp_array(AView & v){
   cuda_exp x;
   thrust::transform(v.begin(), v.end(), v.begin(), x);
}

// this gives a TRANSPOSED version of a tile matrix for computation convenience
Sparse * tile(int g_len, int state_size){

//     cout << "5.1" << endl;
//    int copy_times = state_size / g_len;
    // cout << "5.2" << endl;
    int nnz = state_size;
//     cout << "5.3" << endl;
    // cout << g_len << " " << state_size << " " << nnz << endl;
    Sparse * temp_mat = new Sparse(state_size, g_len, nnz); //TRANSPOSED
    blas::fill(temp_mat->values, 1.0f);
//     cout << "5.4" << endl;
    calc_sparse_column temp_func;
    thrust::tabulate(thrust::device, temp_mat->column_indices.begin(), temp_mat->column_indices.end(), temp_func);
    thrust::sequence(thrust::device, temp_mat->row_offsets.begin(), temp_mat->row_offsets.end());
    // for (int i = 0; i <= state_size; i++){
    //     if (i < state_size){
    //         temp_mat->row_offsets[i] = i;
    //     } else {
    //         temp_mat->row_offsets[i] = nnz;
    //     }
    // }
//     cout << "5.5" << endl;

    return temp_mat;
}

// this gives a TRANSPOSED version of matrix repetition for computation convenience
Sparse * expand(int g_len, int state_size){

//     cout << "expand5.1" << endl;
    int col_size = state_size / g_len;
//     cout << "expand5.2" << endl;
    int nnz = state_size;
//     cout << "expand5.3" << endl;
    // cout << g_len << " " << state_size << " " << nnz << endl;
    Sparse * temp_mat = new Sparse(state_size, col_size, nnz); //TRANSPOSED
    blas::fill(temp_mat->values, 1.0f);
//     cout << "expand5.4" << endl;
    g_integer_division temp_func;
    thrust::tabulate(thrust::device, temp_mat->column_indices.begin(), temp_mat->column_indices.end(), temp_func);
    thrust::sequence(thrust::device, temp_mat->row_offsets.begin(), temp_mat->row_offsets.end());
//     cout << "expand5.5" << endl;
    return temp_mat;
}

void get_row(csr_matrix_view<IndexArrayView, IndexArrayView, ValueArrayView>* s, int i, Array& result,
Array* pos_full_array, int g_max, int b_max){
    blas::fill(result, 0.0f);
    int pi_row_index = i / g_max;
    int b_index = pi_row_index % b_max;
    int g_index = i % g_max;
//    cout <<"get row in"<< endl;
//    cout << pi_row_index << "pi row index" << endl;
    if (s->row_offsets[pi_row_index] - s->row_offsets[pi_row_index+1] != 0){
        int num_of_entry = s->row_offsets[pi_row_index+1] - s->row_offsets[pi_row_index];
        int start_column_index = s->row_offsets[pi_row_index];
        IndexArrayView column_indices = s -> column_indices.subarray(start_column_index, num_of_entry);
        ValueArrayView values = s -> values.subarray(start_column_index, num_of_entry);
        thrust::scatter(thrust::device, values.begin(), values.end(), column_indices.begin(), result.begin());
//        cusp::print(result);
//        cout << "g is " << g_index << "and the prob is " <<(*pos_matrix)[pos_matrix_dim] << endl;
//        blas::scal(result, (*pos_full_array)[i]);
    }
//    cout << "get row out" << endl;
}

int get_max_len(std::vector<std::vector<int> > sents){
//    cout <<"get_max_len in"<< endl;
    int max_len = 0;
    for(std::vector<int> sent : sents){
        if(sent.size() > max_len){
            max_len = sent.size();
        }
    }
//    cout << "get_max_len out" << endl;

    return max_len;
}

Dense* get_sentence_array(std::vector<std::vector<int> > sents, int max_len){
    Dense* array = new Dense( sents.size(), max_len, 0 );
//    cout <<"get_sentence_array in"<< endl;
    for(int i = 0; i < sents.size(); i++){
      std::vector<int> sent = sents[i];
      for(int token_ind = 0; token_ind < sent.size(); token_ind++){
        array -> operator()(i, token_ind) = sent[token_ind];
      }
    }
//    cout <<"get_sentence_array out"<< endl;
    return array;

}

void HmmSampler::set_models(Model * models){

//     cout << "set_models 1" << endl;
    p_model = models;
//     cout << "set_models 2" << endl;
    if (p_indexer != NULL){
        delete p_indexer;
	p_indexer = NULL;
    }
    p_indexer = new Indexer(models);
//     cout << "set_models 3" << endl;
    int g_len = p_model-> g_max;
    int state_size = p_indexer -> get_state_size();
    cudaMemset(&G_SIZE,0,sizeof(int));
    cudaMemcpyToSymbol(G_SIZE, &g_len, sizeof(int), 0, cudaMemcpyHostToDevice);
//     cout << "set_models 4" << endl;
    //if (lexMatrix != NULL){
    //    delete lexMatrix;
    //    lexMatrix = NULL;
    //}

    lexMatrix = p_model -> lex;
    // print(*(lexMatrix->get_view()));
    // exp_array(lexMatrix->get_view()->values); // exp the lex dist // the gpu models are not logged, should not need this
//     cout << "set_models 5" << endl;
    pos_matrix = p_model -> pos;
//    cusp::print(*pos_matrix);
    if (lexMultiplier != NULL){
        delete lexMultiplier;
        lexMultiplier = NULL;
    }
    lexMultiplier = tile(g_len, p_indexer->get_state_size());
//    cout << "set_models 6" << endl;
    expand_mat = expand(g_len, p_indexer->get_state_size());
//    cout << "set_models 7" << endl;
    // print(*lexMultiplier);
    pi = p_model -> pi;
    // print( *(pi->get_view()) );
    if (trans_slice != NULL){
        delete trans_slice;
        trans_slice = NULL;
    }
    trans_slice = new Array(p_indexer->get_state_size(), 0.0f);
//    cout << "set_models 8" << endl;
    expanded_lex = trans_slice;
    sum_dict = trans_slice;
    int b_len = p_model -> b_max;
//    cout << "pos full array" << endl;
    int depth = p_model -> get_depth();
    pos_full_array = make_pos_full_array(pos_matrix, g_len, b_len, depth, state_size);
//    cout << "set_models 9" << endl;
//    print(*pos_full_array);
    // cout.precision(float_limit::max_digits10);
}

Array* HmmSampler::make_pos_full_array(Array* pos_matrix ,int g_max, int b_max, int depth, int state_size){
//    cout << "make_pos_full_array in" << endl;
    int pos_matrix_size = pow(b_max, depth) * g_max * 2;
    int copy_times = state_size / pos_matrix_size;
    Array* temp_array = new Array(state_size, 0.0f);
    for (int i = 0; i < copy_times; i ++){
//        cout << "calc full pos " << i  << " from " << i*pos_matrix_size << " to " << (i+1)*pos_matrix_size <<
//        " total " << copy_times << endl;
        ArrayView one_section_of_array(temp_array->subarray(i*pos_matrix_size, pos_matrix_size));
        copy(*pos_matrix, one_section_of_array);
    }
//    cout << "make_pos_full_array out" << endl;
//    cout << " THE VALUE " << (*temp_array)[26426] << endl;
    return temp_array;
}

void HmmSampler::initialize_dynprog(int batch_size, int max_len){
    try{
//    cout << "initialize_dynprog in" << endl;
    sampler_batch_size = batch_size;
    max_sent_len = max_len;
    dyn_prog = new Dense*[max_len];
    for(int i = 0; i < max_len; i++){
        dyn_prog[i] = new Dense(p_indexer->get_state_size(), batch_size, 0.0f);
    }
//    cout << "initialize_dynprog 2" << endl;
    start_state = new Dense(p_indexer->get_state_size(), batch_size, 0.0f);
    for(int i = 0; i < batch_size; i++){
        start_state->operator()(0, i) = 1;
    }
    int a_max, b_max, g_max;
//    cout << "initialize_dynprog 3" << endl;
    std::tie(a_max, b_max, g_max) = p_indexer -> getVariableMaxes();
    int state_size_no_g = p_indexer->get_state_size() / g_max;
    dyn_prog_part = new Dense(state_size_no_g, batch_size, 0.0f);
//    cout << "initialize_dynprog out" << endl;
    } catch (...) {
        cout << "init dynprog error!" << endl;
        throw;
    }
}

void HmmSampler::g_factored_multiply(Dense* prev_dyn_prog_slice, Dense* this_dyn_prog_slice){
//    cout << "factor multiply 1" << endl;
    csr_matrix_view<IndexArrayView,IndexArrayView,ValueArrayView>* curtailed_transition = pi -> get_view();
    multiply(*curtailed_transition, *prev_dyn_prog_slice, *dyn_prog_part);
//    cout << "dyn prog part" << endl;
//    print(*dyn_prog_part);
//    cout << "factor multiply 2" << endl;
    int a_max, b_max, g_max;
    std::tie(a_max, b_max, g_max) = p_indexer -> getVariableMaxes();
    int state_size_no_g = p_indexer->get_state_size() / g_max;
    int state_size = p_indexer->get_state_size();
    multiply(*expand_mat, *dyn_prog_part, *this_dyn_prog_slice);
//    cout << "factor multiply 3" << endl;
//    cout << "this dyn prog slice" << endl;
//    print(*this_dyn_prog_slice);
    for(int i = 0; i < sampler_batch_size; i++ ){ // this is not efficient. maybe there is a better way
//        cout << "insde g_multipy i " << i << endl;
        array2d<float, device_memory>::column_view this_dyn_prog_slice_column = this_dyn_prog_slice->column(i);
//        blas::xmy(this_dyn_prog_slice_column, *pos_matrix, this_dyn_prog_slice_column);
//        print(this_dyn_prog_slice_column);
//        print(*pos_full_array);
        blas::xmy(this_dyn_prog_slice_column, *pos_full_array ,this_dyn_prog_slice_column);
//        cout << "this dyn prog slice column" << endl;
//        print(this_dyn_prog_slice_column);

    }
//    cout << "factor multiply 4" << endl;
}

std::vector<float> HmmSampler::forward_pass(std::vector<std::vector<int> > sents, int sent_index){
    // auto t1 = Clock::now();
//     cout << "Forward in " << endl;
    float normalizer;
    int a_max, b_max, g_max; // index, token, g_len;
    std::tie(a_max, b_max, g_max) = p_indexer -> getVariableMaxes();
    std::vector<int> sent = sents[0];
    std::vector<float> log_probs;
    int batch_size = sents.size();
    int batch_max_len = get_max_len(sents);
    // np_sents is |batches| x max_len
    Dense* np_sents = get_sentence_array(sents, batch_max_len);

    array2d_view<ValueArrayView, row_major>* lex_view = lexMatrix -> get_view();
//    cout << "Forward in 2 " << endl;
    // initialize likelihood vector:
    for(int sent_ind = 0; sent_ind < sents.size(); sent_ind++){
        log_probs.push_back(0);
    }


    for(int ind = 0; ind < batch_max_len; ind++){
//        cout << "Processing token index " << ind << " for " << batch_size << " sentences." << endl;
        Dense *cur_mat = dyn_prog[ind];
        Dense *prev_mat;

        if(ind == 0){
            prev_mat = start_state;
        }else{
            // Grab the ind-1th row of dyn_prog and multiply it by the transition matrix and put it in
            // the ind^th row.
            prev_mat = dyn_prog[ind-1];
        }

        // prev_mat is |states| x |batches| at time ind-1
        // pi_view is |states| x |states| transition matrix with time t on rows and t-1 on columns (i.e. transposed)
        // so after this multiply cur_mat is |states| x |batches| incorporating transition probabilities
        // but not evidence
//        cout << "Performing transition multiplication" << endl;
        g_factored_multiply(prev_mat, cur_mat);
//        print(cur_mat[0]);
//        cout << "Done with transition" << endl;
//        cout << "performing observation multiplications" << endl;

        auto trans_done = Clock::now();

        // for now incorporate the evidence sentence-by-sentence:
        for(int sent_ind = 0; sent_ind < sents.size(); sent_ind++){
            // not every sentence in the batch will need the full batch size
            if(sents[sent_ind].size() <= ind){
                continue;
            }
//            cout << "Processing sentence index " << sent_ind << endl;
            int token = sents[sent_ind][ind];

            // lex_column is |g| x 1
            array2d<float, device_memory>::column_view lex_column = lex_view -> column(token);
//            if (sents[sent_ind].size() > 3){
//                cout << "lex"<< endl;
//                print(lex_column);
//            }
//             cout << "6" << endl;
            // lexMultiplier is state_size x |g|, expanded_lex is state_size x 1
//             cout << "Multiplying lex multiplier by lex column" << endl;
//            cout << "lex column" << endl;
//            print(lex_column);
            multiply(* lexMultiplier, lex_column, * expanded_lex);
//            if (sents[sent_ind].size() > 3){
//                cout << "expanded lex" << endl;
//                print(*expanded_lex);
//            }

//             cout << "lex finished" << endl;
            // dyn_prog_row is 1 x state_size
            // dyn_prog_column is state_size x 1
            array2d<float, device_memory>::column_view dyn_prog_col = cur_mat->column(sent_ind);
//             cout << "Multiplying expanded_lex by dyn prog row" << endl;
//            vector<int> v = {8, 19406, 24314, 26426};
//            if (sents[sent_ind].size() > 3 && ind < 4){
//                cout << "column view" << endl;
//                cout << "ind" << ind << " " << dyn_prog_col[v[ind]] << endl;
//                cout << "lex" << ind << " " << (*expanded_lex)[v[ind]] << endl;
//                cout << "pos" << ind << " " << (*pos_full_array)[v[ind]] << endl;
//            }
            blas::xmy(*expanded_lex, dyn_prog_col, dyn_prog_col);
//             cout << "Computing normalizer" << endl;
            normalizer = thrust::reduce(thrust::device, dyn_prog_col.begin(), dyn_prog_col.end());
//             cout << "Normalizing over col with result: " << normalizer << endl;
//            cout << "Scaling by normalizer" << endl;
            blas::scal(dyn_prog_col, 1.0f/normalizer);
//            cout << "normed column" << endl;
//            print(dyn_prog_col);
//                cout << "Adding logged normalizer to sentence logprobs" << endl;
            log_probs[sent_ind] += log10f(normalizer);
//            cout << " ind "<< ind << " sent_ind " << sent_ind << " token " << token << " log prob " << log10f(normalizer) << " " << normalizer << endl;
            if (sents[sent_ind].size() - 1 == ind){
                int EOS = p_indexer -> get_EOS_full();
//                cout << EOS << endl;
                array2d<float, device_memory>::column_view final_dyn_col = cur_mat->column(sent_ind);
//                cout << cusp::blas::asum(final_dyn_col) << endl;
                get_row(pi->get_view(), EOS, *trans_slice, pos_full_array, g_max, b_max);
//                cout << cusp::blas::asum(*trans_slice) << endl;
                float final_normalizer = cusp::blas::dot(*trans_slice, final_dyn_col);
                log_probs[sent_ind] += log10f(final_normalizer);
//                cout << " ind "<< ind << " sent_ind " << sent_ind << "end log prob " << log10f(final_normalizer) << endl;
            }
        }

        auto norm_done = Clock::now();
    }
//    for (int i =0; i < sents[0].size(); i++){
//    print(*dyn_prog[i]);
//    }
//    cout << "end of dyn prog" << endl;
//    cout << "Finished forward pass (cuda) and returning vector with " << log_probs.size() << " elements." << endl;
    return log_probs;
}

std::vector<std::vector<State> > HmmSampler::reverse_sample(std::vector<std::vector<int>> sents, int sent_index){
//     cout << "Reverse sampling batch with starting sent index of " << sent_index << endl;
    // auto t2 = Clock::now();
    std::vector<std::vector<State>> sample_seqs;
    std::vector<State> sample_seq;
    std::vector<int> sample_t_seq;
    int last_index, sample_t, sample_depth; // , t, ind; totalK, depth,
    int batch_size = sents.size();
    int batch_max_len = get_max_len(sents);
    // int prev_depth, next_f_depth, next_awa_depth;
    //float sample_log_prob;//trans_prob,
    // double t0, t1;
//    array1d<float, device_memory> trans_slice;
    State sample_state;
    //sample_log_prob = 0.0f;
//    std::vector<int> fake_ts = {25,218,21,213,25,230,42,22,38,229,233,41,25,154,2};
//    reverse(fake_ts.begin(), fake_ts.end());
//    for(std::vector<int> sent : sents){
    for(int sent_ind = 0; sent_ind < batch_size; sent_ind++){
        sample_seq = std::vector<State>();
//        cout << "Processing sentence " << sent_ind << " of the batch" << endl;
        std::vector<int> sent = sents[sent_ind];
//        for(int token_ind = 0; token_ind < sent.size(); token_ind++){
//          cout << sent[token_ind] << " ";
//        }
//        cout << endl;

        // Start with EOS
//        if (sent.size() == 1) {
//            sample_t = p_indexer->get_EOS_1wrd_full();
//        } else {
            sample_t = p_indexer->get_EOS_full();
//        }

        for (int t = sent.size() - 1; t > -1; t --){
//            cout << "t" << t << " prev sample t is " << sample_t <<endl;
            // auto t11 = Clock::now();
            std::tie(sample_state, sample_t) = _reverse_sample_inner(sample_t, t, sent_ind);
//             cout << "Sample t is " << sample_t << endl;
//             cout << sample_state.f << " " << sample_state.j << " " << sample_state.a[0] << " " << sample_state.a[1] << " " << sample_state.b[0] << " " << sample_state.b[1] << " " << sample_state.g << endl;
            if(!sample_state.depth_check()){
              cout << "Depth error in state assigned at index" << t << endl;
            }
            sample_seq.push_back(sample_state);
            sample_t_seq.push_back(sample_t);
            // auto t12 = Clock::now();
//             cout << "backpass2inside: " << (float)std::chrono::duration_cast<std::chrono::nanoseconds>(t12 - t11).count() * nano_to_sec << " s" << endl;

        }
        // auto t4 = Clock::now();
        std::reverse(sample_seq.begin(), sample_seq.end());
//        cout << "finished backward sampling of a sentence" << endl;
        //cout << sample_seq->size() << endl;
        // auto t5 = Clock::now();
        // cout << "backpass1: " << (float)std::chrono::duration_cast<std::chrono::nanoseconds>(t3 - t2).count() * nano_to_sec << " s" << endl;
        // cout << "backpass2: " << (float)std::chrono::duration_cast<std::chrono::nanoseconds>(t4 - t3).count() * nano_to_sec << " s" << endl;
        // cout << "backpass: " << (float)std::chrono::duration_cast<std::chrono::nanoseconds>(t5 - t2).count() * nano_to_sec << " s" << endl;
        //for (int k : sample_t_seq){
        //    cout << sent_index << " : " << k  << endl;
        //}
        sample_seqs.push_back(sample_seq);
    }
//    for(auto i : sample_t_seq){
//        cout << i << endl;
//    }
//    for (int i =0; i < sents[0].size(); i++){
//    print(*dyn_prog[i]);
//    }
//    cout << "end of dyn prog" << endl;
//    cout << "Done with reverse()" << endl;
    return sample_seqs;
}


std::tuple<State, int> HmmSampler::_reverse_sample_inner(int& sample_t, int& t, int sent_ind){
    //int ind;
    float normalizer;
    // auto t11 = Clock::now();
//    cout << "before get row" << endl;
//    int prev_sample_t = sample_t;
    int g_max =  p_model -> g_max;
    int b_max = p_model -> b_max;
    get_row(pi->get_view(), sample_t, *trans_slice, pos_full_array, g_max, b_max);
    // print(*trans_slice);
    // auto t12 = Clock::now();
    array2d<float, device_memory>::column_view dyn_prog_col = dyn_prog[t]->column(sent_ind);
    float dyn_prog_col_sum = thrust::reduce(thrust::device, dyn_prog_col.begin(), dyn_prog_col.end());
//     cout << "Dyn_prog_row " << dyn_prog_col_sum << endl;
    // print(dyn_prog_row);
    float trans_slice_sum = thrust::reduce(thrust::device, (*trans_slice).begin(), (*trans_slice).end());
//    cout << "trans_slice " << trans_slice_sum << endl;
    //if (trans_slice_sum != 0.0f){
        cusp::blas::xmy(*trans_slice, dyn_prog_col, dyn_prog_col);
    //}
    // auto t13 = Clock::now();
//    cusp::array1d<float, host_memory> un_normalized_sums(dyn_prog_col);
    normalizer = thrust::reduce(thrust::device, dyn_prog_col.begin(), dyn_prog_col.end());
//     cout << "normalizer " << normalizer <<endl;
    blas::scal(dyn_prog_col, 1.0f/normalizer);
    // thrust::transform(dyn_prog_row.begin(), dyn_prog_row.end(), dyn_prog_row.begin(), multiplies_value<float>(1 / normalizer));
    // auto t14 = Clock::now();
    sample_t = get_sample(dyn_prog_col);
//    cout << "sample_t "<< sample_t << endl;
//    sample_t = fake_ts.back();
//    fake_ts.pop_back();
    //if (sample_t == 0){
    //    print(dyn_prog_row);
    //}
    // auto t15 = Clock::now();
    State sample_state = p_indexer -> extractState(sample_t);
//    cout << "done with reverse sampler inner" << endl;
    // if ((sample_state.a[1] == 0 && sample_state.b[1] != 0)|| (sample_state.a[1] != 0 && sample_state.b[1] == 0) ){
//    get_row(pi->get_view(), prev_sample_t, *trans_slice); // why is this used?
        // cout << "Dyn Prog(normalized): " << scientific << dyn_prog_row[sample_t -1] << " " << scientific << dyn_prog_row[sample_t] <<" "<< scientific <<dyn_prog_row[sample_t + 1] <<endl;
        // cout << "Trans Slice (Pi Row): " <<" " << scientific << (*trans_slice)[sample_t-1]<< " "<< scientific << (*trans_slice)[sample_t] <<" " << scientific << (*trans_slice)[sample_t+1]<< endl;
        // cout << "Prefix Sum(unnormalized, before sum): " << scientific << un_normalized_sums[sample_t-1] << " " << scientific << un_normalized_sums[sample_t] << " " << scientific << un_normalized_sums[sample_t + 1] << endl;
        // cout << "Prefix Sum(unnormalozed, before sum): ";
        // float temp_1 = un_normalized_sums[sample_t-1];
        // printf( "%A", temp_1);
        // cout << " ";
        // float temp_2 = un_normalized_sums[sample_t];
        // printf("%A", temp_2);
        // cout << " ";
        // float temp_3 = un_normalized_sums[sample_t+1];
        // printf("%A\n", temp_3);
        // cout << "sample_t: " << sample_t << endl;

    // }
    // cout << "backpass1reverseinner: " << (float)std::chrono::duration_cast<std::chrono::nanoseconds>(t12 - t11).count() * nano_to_sec << " s" << endl;
    // cout << "backpass1reverseinner: " << (float)std::chrono::duration_cast<std::chrono::nanoseconds>(t13 - t12).count() * nano_to_sec << " s" << endl;
    // cout << "backpass1reverseinner: " << (float)std::chrono::duration_cast<std::chrono::nanoseconds>(t14 - t13).count() * nano_to_sec << " s" << endl;
    // cout << "backpass1reverseinner: " << (float)std::chrono::duration_cast<std::chrono::nanoseconds>(t15 - t14).count() * nano_to_sec << " s" << endl;

    return std::make_tuple(sample_state, sample_t);
}

std::tuple<std::vector<std::vector<State> >, std::vector<float>> HmmSampler::sample(std::vector<std::vector<int>> sents, int sent_index) {
    std::vector<float> log_probs;
    std::vector<std::vector<State> > states;

    try{
        log_probs = forward_pass(sents, sent_index);
    }catch(thrust::system_error &e){
        cerr << "Error in forward pass: " << e.what() << endl;
        throw e;
    }
    try{
        states = reverse_sample(sents, sent_index);
    }catch(thrust::system_error &e){
        cerr << "Error in reverse sample: " << e.what() << endl;
        throw e;
    }


    return std::make_tuple(states, log_probs);
}

HmmSampler::HmmSampler() : seed(std::time(0)){
    mt.seed(seed);
}

HmmSampler::HmmSampler(int seed) : seed(seed){
    if (seed == 0){
        mt.seed(std::time(0));
    } else{
        mt.seed(seed);
    }
}
HmmSampler::~HmmSampler(){
    if(dyn_prog != NULL){
        for(int i = 0; i < max_sent_len; i++){
            delete dyn_prog[i];
        }
        delete[] dyn_prog;
    }
    delete start_state;
    //delete[] dyn_prog;
    //delete p_model;
    delete p_indexer;
    //delete lexMatrix;
    //delete dyn_prog;
    delete lexMultiplier;
    //delete pi;
    delete trans_slice;
    //delete expanded_lex;
    //delete sum_dict;
    delete expand_mat;
    delete dyn_prog_part;
    delete pos_full_array;

}
