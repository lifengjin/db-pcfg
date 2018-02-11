
#include "Indexer.h"
#include <tuple>
#include <cmath>
//#include "State.h"
#include <vector>

using namespace std;

vector<int> unravel_index(int index, vector<int> & v){
    vector<int> result(v.size());
    int v_size = (int) v.size();
    for (int i = v_size - 1; i >= 0 ; --i) {
        int last_element = v[i];
        result[i] = index % last_element;
        index = (index - result[i]) / last_element;
    }
    return result;
}


Indexer::Indexer(){};
Indexer::Indexer(Model * model): fj_size(4){
    depth = model -> get_depth();
    a_max = model -> a_max;
    b_max = model -> b_max;
    g_max = model -> g_max;
    state_size = fj_size * pow((a_max * b_max), depth) * g_max;
    a_size = pow(a_max, depth);
    b_size = pow(b_max, depth);
//    stack_dims = {fj_size, a_size, b_size, g_max};
    stack_dims = {fj_size/2, a_size, b_size,fj_size/2, g_max};
    EOS_index_full = model -> EOS_index;
}
std::tuple<int, int, int> Indexer::getVariableMaxes(){
    return std::make_tuple(a_max, b_max, g_max);
}
int Indexer::get_state_size(){
    return state_size;
}
int Indexer::get_EOS(){
    return EOS_index_full / g_max;
}
int Indexer::get_EOS_full(){
    return EOS_index_full;
}
//int Indexer::get_EOS_1wrd(){
//    return 3*state_size / (4 * g_max);
//}
//int Indexer::get_EOS_1wrd_full(){
//    return 3*state_size / 4;
//}
tuple<int, vector<int>, vector<int>, int, int> Indexer::extractStacks(int index){
    int fj_ind, a_ind, b_ind, g, f, j; //max_d, f_val,j_val, 
    auto result = unravel_index(index, stack_dims);
    j = result[0];
    a_ind = result[1];
    b_ind = result[2];
    f = result[3];
    g = result[4];
//    f = fj_ind / 2 == 0 ? 0 : 1;
//    j = fj_ind % 2 == 0 ? 0 : 1;
    vector<int> a_dims(depth);
    std::fill(a_dims.begin(), a_dims.end(), a_max);
    vector<int> b_dims(depth);
    std::fill(b_dims.begin(), b_dims.end(), b_max);
    vector<int> a = unravel_index(a_ind, a_dims);
    vector<int> b = unravel_index(b_ind, b_dims);
    return std::make_tuple(j, a, b, f, g);
}

State Indexer::extractState(int index){
    int f, j, g;
    vector<int> a, b;
    std::tie(j, a, b, f, g) = extractStacks(index);
    State state = State(depth);
    state.f = f;
    state.j = j;
    state.a = a;
    state.b = b;
    state.g = g;
    return state;
}

