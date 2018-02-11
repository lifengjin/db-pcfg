# cython: profile=False
# cython: linetrace=False
# cython: binding=False
# distutils: define_macros=CYTHON_TRACE=0

cimport models
cimport Indexer
import numpy as np
cimport numpy as np
import scipy.sparse
cimport ObservationModel
import sys

cdef class PosDependentObservationModel(ObservationModel.ObservationModel):
    def __init__(self):
        self.indexer = None

    cdef set_models(self, models):
        self.indexer = Indexer.Indexer(models)
        g_len = models.pos.dist.shape[1]
        lexMultiplier = scipy.sparse.csc_matrix(np.tile(np.identity(g_len), (1, self.indexer.get_state_size() // g_len)))
        self.data = lexMultiplier.data
        self.indices = lexMultiplier.indices
        self.indptr = lexMultiplier.indptr

    cdef get_probability_vector(self, token):
        maxes = self.indexer.getVariableMaxes()
        (a_max,b_max,g_max) = maxes
        lexMultiplier = scipy.sparse.csc_matrix((self.data, self.indices, self.indptr), shape=(g_max, self.indexer.get_state_size() ) )
        pos_probs = self.get_pos_probability_vector(token)
        retVal = pos_probs.transpose() * lexMultiplier
        return retVal

    cdef get_pos_probability_vector(self, token):
        raise NotImplementedError
