#!/bin/bash -l


nvcc -rdc=true -c -o gpusrc/temp.o gpusrc/HmmSampler.cu -I/home/jin.544/include -std=c++11 --shared -Xcompiler -fPIC -m64
nvcc -dlink -o gpusrc/hmmsampler.o gpusrc/temp.o -lcudart --shared -Xcompiler -fPIC -m64 -L/usr/local/cuda/8.0.61/lib64 -Xlinker -rpath -Xlinker /usr/local/cuda/8.0.61/lib64
ar cru gpusrc/libhmm.a gpusrc/hmmsampler.o gpusrc/temp.o
ranlib gpusrc/libhmm.a
cython --cplus gpusrc/CHmmSampler.pyx
g++ -pthread -DNDEBUG -g -fwrapv -O2 -Wall -g -Wformat -Werror=format-security -D_FORTIFY_SOURCE=2 -fPIC -I/home/jin.544/envs/nn3.5/dist-packages/numpy/core/include -I/home/jin.544/envs/nn3.5/include/python3.6m -I/usr/local/include/ -c gpusrc/CHmmSampler.cpp -o gpusrc/CHmmSampler.o -w -std=c++11  -L/usr/local/cuda/8.0.61/lib64 -lcudart -L/usr/lib/x86_64-linux-gnu -lpython3.6m -Lgpusrc/ -lhmm
# clang -bundle -undefined dynamic_lookup -arch x86_64 -g -std=c++11 build/CHmmSampler.o libhmm.a -o build/CHmmSampler.so -L. -lhmm -L/usr/local/cuda/lib -lcudart
g++ -pthread -shared -Wl,-O1 -Wl,-Bsymbolic-functions -Wl,-z,relro -Wl,-z,relro -g -Wformat -Werror=format-security -D_FORTIFY_SOURCE=2 -std=c++11 gpusrc/CHmmSampler.o gpusrc/libhmm.a -o scripts/CHmmSampler.so -Lgpusrc/ -lhmm -L/usr/local/cuda/8.0.61/lib64 -lcudart
rm gpusrc/temp.o
rm gpusrc/hmmsampler.o
rm gpusrc/libhmm.a
rm gpusrc/CHmmSampler.o
rm gpusrc/CHmmSampler.cpp
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
# python3 test_suite.py
