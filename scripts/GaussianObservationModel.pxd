# cython: profile=False
# cython: linetrace=False
# cython: binding=False
# distutils: define_macros=CYTHON_TRACE=0
import numpy as np
cimport Indexer
cimport numpy as np
cimport PosDependentObservationModel

cdef class GaussianObservationModel(PosDependentObservationModel.PosDependentObservationModel):
    cdef lex
