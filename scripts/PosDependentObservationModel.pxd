# cython: profile=False
# cython: linetrace=False
# cython: binding=False
# distutils: define_macros=CYTHON_TRACE=0

cimport models
cimport Indexer
import numpy as np
cimport numpy as np
cimport ObservationModel

cdef class PosDependentObservationModel(ObservationModel.ObservationModel):
    cdef Indexer.Indexer indexer
    cdef np.ndarray lexMatrix, data, indices, indptr

    cdef set_models(self, models)
    cdef get_probability_vector(self, token)
    cdef get_pos_probability_vector(self, token)
