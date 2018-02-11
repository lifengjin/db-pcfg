// include the csr_matrix header file
// nvcc gpusrc/multiply_test.cu -o test_multiply
#include <stdio.h>
#include <iostream>

#include <cusp/csr_matrix.h>
#include <cusp/coo_matrix.h>
#include <cusp/print.h>
#include <cusp/functional.h>
#include <cusp/multiply.h>
#include <cusp/array2d.h>
#include <cusp/array1d.h>
#include <cusp/coo_matrix.h>
#include <cusp/print.h>
#include <cusp/gallery/poisson.h>

using namespace std;

int main()
{
  int state_size = 15;

  cusp::array2d<float, cusp::device_memory> pi(state_size, state_size);
  pi(1,0) = 0.19; pi(2,0) = 0.2; pi(3,0) = 0.21; pi(4,0) = 0.22; pi(5,0) = 0.18;
//  pi(1,1) = 0.191; pi(2,2) = 0.201; pi(3,3) = 0.211; pi(4,4) = 0.221; pi(5,5) = 0.181;
//  pi(1,6) = 0.192; pi(2,7) = 0.202; pi(3,8) = 0.212; pi(4,9) = 0.222; pi(5,10) = 0.182;
//  pi(5,1) = 0.29; pi(6,2) = 0.3; pi(7,3) = 0.31; pi(8,4) = 0.32; pi(9,5) = 0.28;
//  pi(5,6) = 0.29; pi(6,7) = 0.3; pi(7,8) = 0.31; pi(8,9) = 0.32; pi(9,10) = 0.28;
  
  cout << "Dense version of pi:" << endl;
  cusp::print(pi);
  cusp::csr_matrix<int,float,cusp::device_memory> pi_sparse(pi);
  cout << "pi row offsets: " << endl;
  cusp::print(pi_sparse.row_offsets);
  cout << "pi column indices: " << endl;
  cusp::print(pi_sparse.column_indices);
  cout << "pi values: " << endl;
  cusp::print(pi_sparse.values);
  
  for(int batch_size = 2; batch_size <= 16; batch_size*=2){
      cusp::array2d<float, cusp::device_memory> prev_mat(state_size, batch_size, 0.0f);
      cusp::array2d<float, cusp::device_memory> next_mat(state_size, batch_size, 0.0f);

      for(int i = 0; i < batch_size; i++){
          prev_mat(0, i) = 1;
      }
  
      cusp::multiply(pi_sparse, prev_mat, next_mat);
  
      cout << "Previous matrix:" << endl;
      cusp::print(prev_mat);
      cout << "Next matrix non-zeros:" << endl;
  
      for(int i = 0; i < batch_size; i++){
        for(int j = 0; j < state_size; j++){
             if(next_mat(j,i) > 0){
                 cout << "Value of next_mat[ " << j << ", " << i << "] = " << next_mat(j,i) << endl;
             }
          }
      }
  }
}

