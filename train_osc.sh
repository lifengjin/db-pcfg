#!/bin/bash -l
#eval `$LMOD_CMD bash "module load cuda/7.0.28"`
#eval `$LMOD_CMD bash "module load python/3.4.2"`
#eval `$LMOD_CMD bash "module load gnu/4.9.1"`
module load cuda/7.0.28
module load python/3.4.2
module load gnu/4.9.1
./gpusrc/compile_osc.sh
python3 setup.py build_ext || exit
#if [ $# -ne 1 ]; 
#    then echo "Running the script requires a config argument; exiting."
#    exit
#fi
qsub gpusrc/master.scr
qsub gpusrc/worker.scr

