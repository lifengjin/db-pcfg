
import numpy as np
cimport numpy as np

cdef class Model:
    cdef public tuple corpus_shape
    cdef public str name

cdef class Models:
    cdef list models
    cdef public list F, J, A, B_J0, B_J1
    cdef public Model pos
    cdef public lex

cdef class CategoricalModel(Model):
    cdef public tuple shape
    cdef public np.ndarray pairCounts, globalPairCounts, dist, u, beta
    cdef public float alpha
    cdef public trans_prob

cdef class GaussianModel(Model):
    cdef public tuple shape
    cdef public np.ndarray globalPairCounts, embeddings
    cdef public dist, pairCounts
    cdef public int embed_dims
