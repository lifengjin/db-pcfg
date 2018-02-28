#DB PCFG

*UNDER CONSTRUCTION*

## Introduction
This is the repo for the paper [Unsupervised Grammar Induction with Depth-bounded PCFG](https://arxiv.org/abs/1802.08545) that will appear in Transcations of Association for 
Computational Linguistics.

Dependencies include:

- Python 3.4+
- CUDA 8.0+
- PyZMQ
- Numpy & Scipy
- Cython
- NLTK
- CUSP

## Steps to set up
There are a few steps you need to do before the system is runnable. The first step is to compile 
to GPU source files. The second step is to compile the Cython files. Finally you need to set up a
 config file.
### 1. Compile the CUDA C scripts
There are a couple of compile scripts in the `./gpusrc/` folder for reference. Since it is most 
likely that you want to run the system on a super computer with many GPUs, the `compile_osc.sh`
file best shows how to compile the CUDA scripts. You need to replace the paths in the file for 
CUDA libraries, Python and Numpy with your paths. After you have setup the paths, you only need 
to do `source ./gpusrc/compile_osc.sh` for `nvcc` to compile all the scripts.

### 2. Compile Cython scripts
After compiling the CUDA C scripts, you should be able to compile to Cython scripts by doing 
`python3 setup.py build_ext`.

### 3. Set up a config file
There is a sample config file in the `./config/` folder. The config file has two parts, `io` and 
`params`. The settings are explained below. Please see the sample file for the format in which 
the parameters should be written.  
`io.input_file`: the path to the input `ints` file.  
`io.output_dir`: the folder where all the outputs will be write into.  
`io.dict_file`: the path to the input `dict` file.  
`params.random_restarts`: the number of random restarts the sampler will do and evaluate before 
doing a chain.  
`params.num_samples`: the number of iterations the sampler will run.
`params.startabp`: the number of A/B/P categories given to the sampler, which is equivalent to 
*K* in the paper.  
`params.init_alpha`: the value for the hyperparameter for the symmetric Dirichlet prior, which is
 equivalent to *beta* in the paper.  
 `params.cpu_workers`: the number of workers on CPUs. The CPU workers only do model compilation, 
 not sampling.  
 `params.gpu_workers`: the number of workers on GPUs. The GPU workers do both model compilation 
 and sampling.  
 `params.depth`: the maximum depth limit to the sampler.  
 `gpu`: the flag to use GPU or not.  
 `gpu_batch_size`: the size of a batch used on the GPU.  
 
 #### Notes:
 1. You can do `make xxx.ints.txt` to convert a space-delimited one-line-per-sentence file into 
 an `ints` and a `dict` file used by the system.
 2. `start_abp` and `depth` control the size of the compiled model. The largest value one can 
 reasonably try is 15 and 2 respectively, which is what's used in the paper. Larger than this, 
 you may risk running out of memory on the GPU.
 3. `cpu_workers` and `gpu_workers` can both to set to 0, which is usually what you want in order
  to run on a super computer or a cluster. In this case, the master process will write out a 
  `masterConfig.txt` file into the root directory of the package, and you can start arbitrary 
  number of workers by doing `python scripts/workers.py .`.  