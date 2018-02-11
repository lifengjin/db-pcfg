
import numpy as np
cimport numpy as np

cdef class State:
    cdef public  int f, j, g, depth
    cdef public np.ndarray a, b
    