#!/bin/bash

/usr/local/cuda/bin/nvcc -rdc=true -c -o gpusrc/temp.o gpusrc/HmmSampler.cu -std=c++11
/usr/local/cuda/bin/nvcc -dlink -o gpusrc/hmmsampler.o gpusrc/temp.o -lcudart
ar cru gpusrc/libhmm.a gpusrc/hmmsampler.o gpusrc/temp.o
ranlib gpusrc/libhmm.a
cython --cplus gpusrc/CHmmSampler.pyx
# clang -DNDEBUG -g -fwrapv -O3 -Wall -Wstrict-prototypes -std=c++11 -I/Library/Python/2.7/site-packages/numpy/core/include -Isrc -I/usr/local/include/ -I/Library/Frameworks/Python.framework/Versions/2.7/include/python2.7 -c CHmmSampler.cpp -o build/CHmmSampler.o -std=c++11 -L. -lhmm -L/usr/local/cuda/lib -lcudart
 clang -DNDEBUG -g -fwrapv -O3 -Wall -Wstrict-prototypes -std=c++11 -I/Library/Frameworks/Python.framework/Versions/3.4/lib/python3.4/site-packages/numpy/core/include -I/Library/Frameworks/Python.framework/Versions/3.4/include/python3.4m -Isrc -I/usr/local/include/  -c gpusrc/CHmmSampler.cpp -o gpusrc/CHmmSampler.o -std=c++11 -L. -Lgpusrc/ -lhmm -L/usr/local/cuda/lib -lcudart 
#x86_64-linux-gnu-g++ -pthread -DNDEBUG -g -fwrapv -O2 -Wall -g -fstack-protector-strong -Wformat -Werror=format-security -D_FORTIFY_SOURCE=2 -fPIC -I/usr/local/lib/python3.4/dist-packages/numpy/core/include -I/usr/include/python3.4m -I/usr/local/include/ -c CHmmSampler.cpp -o build/CHmmSampler.o -w -std=c++11 -L. -lhmm -L/usr/local/cuda/lib -lcudart -fPIC
# /usr/bin/clang -fno-strict-aliasing -fno-common -dynamic -DNDEBUG -g -fwrapv -O3 -Wall -Wstrict-prototypes -arch i386 -arch x86_64 -g -DCYTHON_TRACE=0 -I/Library/Frameworks/Python.framework/Versions/3.4/lib/python3.4/site-packages/numpy/core/include -I/Library/Frameworks/Python.framework/Versions/3.4/include/python3.4m -c scripts/HmmSampler.c -o build/temp.macosx-10.6-intel-3.4/scripts/HmmSampler.o -w
 clang -bundle -undefined dynamic_lookup -arch x86_64 -g -std=c++11 gpusrc/CHmmSampler.o gpusrc/libhmm.a -o scripts/CHmmSampler.so -L. -Lgpusrc/ -lhmm -L/usr/local/cuda/lib -lcudart
#x86_64-linux-gnu-g++ -pthread -shared -Wl,-O1 -Wl,-Bsymbolic-functions -Wl,-z,relro -Wl,-z,relro -g -fstack-protector-strong -Wformat -Werror=format-security -D_FORTIFY_SOURCE=2 -std=c++11 build/CHmmSampler.o libhmm.a -o build/CHmmSampler.so -L. -lhmm -L/usr/local/cuda/lib -lcudart
rm gpusrc/temp.o
rm gpusrc/hmmsampler.o
rm gpusrc/libhmm.a
rm gpusrc/CHmmSampler.o
rm gpusrc/CHmmSampler.cpp
export LD_LIBRARY_PATH=/usr/local/cuda/lib:$LD_LIBRARY_PATH
# python3 test_suite.py