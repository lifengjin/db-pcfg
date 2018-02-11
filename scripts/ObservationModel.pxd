cimport cython
import models
cimport models

cdef class ObservationModel:
    cdef set_models(self, models)
    cdef get_probability_vector(self, token)
