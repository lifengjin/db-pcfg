#!/bin/bash
#./gpusrc/compile_linux.sh
#./gpusrc/compile.sh
#./gpusrc/compile_osc.sh
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

make scripts/CHmmSampler.so || echo "WARNING: Building gpu-enabled version failed; may fail if gpu flag is set in config."

python3 setup.py build_ext || exit
if [ $# -ne 1 ]; 
    then echo "Running the script requires a config argument; exiting."
    exit
fi
python3 scripts/uhhmm-trainer.py $*

