UHHMM - Unsupervised Hierarchical Hidden Markov Model

Usage:

./train.sh &lt;config file&gt;  (For a brand new run)

OR

./train.sh &lt;output directory&gt; (To resume a run)

OR

make (Start a new run using config/myconfig.ini as the configuration, creating a default config file if it doesn't yet exist)

The train.sh script is new. Since this project uses cython extensively for optimization, the first step in every run is to recompile any changed cython code. This is a change from previous versions of the code which used pyximport to do "Just in time" compiling of cython code. The reason for the change is that the pyximport method built the code in the user's home directory, and so there could be collisions between versions if a user had two copies of the repository checked out running different experiments. The new style compiles the .pyx code into .c and .so files in the scripts/ directory so that every clone of the repo has its own build.

After compiling the code ./train.sh calls uhhmm-trainer.py, which replaces d1trainer.py. They are very similar but now the script accepts a depth parameter greater than one.

There is a sample config file in config/d1train.ini. If you run using make, the first time it runs it will copy that file into myconfig.ini. You can then modify that config file and subsequent runs of make will use it.

The config has an 'io' (input/output) section and a 'params' (machine learning parameters) section. 'io' requires an input file, output directory, a dictionary file, and a working directory (filesystem location -- defaults to output directory) that both the workers and the job distributer have access to.

The config contains a few important optional parameters. 'gpu' is one of them. If you do not write it in the config, the default is to run the CPU version. If you write 'gpu=1' then the GPU version will be run. 'num_procs' is the number of processes, however if you do not include it in the config, the program will automatically switch to OSC cluster mode.

The input file should contain one sentence per line, with space-separated ints representing tokens (if evaluating for POS tagging, the system accepts space-separated tag/word tokens).

The output directory is where all output will be stored. This is so multiple runs will be preserved if desired. The first thing the uhhmm-trainer.py script will do is copy the config file into the output directory, to promote reproducibility.
The dictionary file should contain a mapping between words and their index. Each line should be &lt;word&gt; &lt;index&gt;.

There are scripts in the scripts/ sub-directory to convert token files into int files, creating a dictionary as well.

## Resuming jobs
If a run terminates early or needs to be stopped, they can often be resumed. After every iteration the model files will be pickled into a file named sample.obj. To resume a job rather than running a new one, simply run:
./train.sh &lt;output dir&gt;

where &lt;output dir&gt; is the directory that the run was writing its output to. The system will read in the sample.obj file as well as the config file that was written to that directory to resume inference. This means you can change parameters if you edit that config file. This can be useful (for increasing the number of workers) but could also be abused in ways that would probably break the system (e.g., changing the number of POS tags in the finite system).

## Cluster usage
This software has been written to be highly parallelizable, either on a single node with multiple cores or in a cluster environment with multiple nodes. We use Pyzmq inter-process messaging to pass sentences to workers and receive parses from workers. The main job collects the sampled sentences, recomputes UHHMM model parameters, and then iterates.

### Cluster submission scripts:
When running the code on the cluster, set the num_procs parameter to 0. This will disable the main thread from creating workers outside the cluster job submission framework. There is an additional parameter to use instead (cluster_cmd), which tells the software how to submit job arrays for its workers. This command needs to have a parameter %%c where the executable name goes which our code will replace with the call to start the workers (it needs two % because the .ini script will remove the first one). We have tested this code on three different cluster platforms -- LSF (bsub), SLURM (sbatch), and Sun GridEngine (qsub) -- see the examples below.

#### LSF:
cluster_cmd=bsub -e %%J.err -J workers[1-70] -q &lt;queue name&gt; %%c

This command writes its output to a file that uses the job id (%%J), creates a job array with 70 workers (workers[1-70]), and is submitted to the job queue you give it. 

#### SLURM:
cluster_cmd=sbatch --array=1-30 --mem-per-cpu 3000 -p &lt;partition name&gt; -t 1-00 --wrap=%%c

This command uses 30 workers, requests 3 GB of RAM per worker, is submitted to the partition given (similar to the queues above) and will be killed after one day. 

#### GridEngine:
cluster_cmd=qsub -V -b y -j y -t 1-100 -cwd %%c

This command creates 100 workers, indicates that the command given is a binary to be run directly with -b y (rather than a job submission script), and -j y indicates that the stdout and stderr should be merged into one output file.


### Cluster debugging
Since clusters vary so much in configuration, policy, and availability, the above may not work out of the box. In particular, we've encountered the problem where the master starts immediately, but then the job array it attempts to start queues for hours or days, wasting the time of the originally submitted job. To work around this, we allow the workers to start first. If they are not given the hostname and ports of the job distributer at startup, they will periodically check the working directory for a file containing that information. Once all the jobs in the array are started, the single threaded master can start, and once it binds to all its ports it will write the file that the workers are looking for.
