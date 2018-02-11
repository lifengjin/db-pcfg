
cimport numpy as np
import numpy as np

cdef class Indexer:
    cdef int depth, f_size, j_size, fj_size, a_max, b_max, g_max, a_size, b_size, EOS_index
    
    cdef public int state_size
    
    cpdef public int extractPos(self, int index)
    
    cpdef public extractStacks(self, int index)
    
    cpdef int getStateIndex(self, int j, np.ndarray a, np.ndarray b, int f, int g)
    
#    cpdef getStateTuple(self, int f, int j, np.ndarray a, np.ndarray b)
