# Depth-Bounded PCFG Induction

## Introduction
This is the repo for the paper [Unsupervised Grammar Induction with Depth-bounded PCFG](https://arxiv.org/abs/1802.08545) 
that appears in Transcations of Association for 
Computational Linguistics. A large part of the code is based on another system called [UHHMM](https://github.com/tmills/uhhmm)
so some scripts may still have older names.

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

## Run the system
Now the hard work is over, and the system is runnable. By running the system, you should just do 
`python scripts/uhhmm-trainer.py config/config.file`. This launches the master process. The outputs 
will be written into the specified output directory. If you have specified number of GPU workers 
in the config file, then the system will start running now. If you have 0 specified GPU workers, 
please follow Note 3 to start some GPU workers for sampling.

Because most of the runs take longer than a couple of days, you may want to continue a stopped 
run. In order to do that, you just pass in the output directory instead of the config file to the
 trainer: `python scripts/uhhmm-trainer.py outputs/output_dir` and the master process will be 
 started and pick up where it is stopped at. For the workers it is the same as starting a new run.
 
The important output files written to the output directory include a `pcfg_hypparams.txt` which 
records log-likelihoods and right-branching tendency scores at every iteration. The linetrees 
files are the sampled trees for the training corpus in the brackets format at every iteration. 
They are ready for normal unlabeled bracketing evaluation when punctuation is removed from these 
trees.
