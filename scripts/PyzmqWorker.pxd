
cimport Indexer
from cpython cimport bool

cdef class PyzmqWorker:
    cdef int jobs_port, results_port, models_port, maxLen, out_freq, tid, seed, debug_level, batch_size
    cdef bool quit, gpu
    cdef Indexer.Indexer indexer
    cdef str host, my_ip
    cdef tuple model_file_sig
