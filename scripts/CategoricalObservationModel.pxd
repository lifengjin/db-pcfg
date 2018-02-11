import numpy as np
cimport Indexer
cimport numpy as np
cimport PosDependentObservationModel

cdef class CategoricalObservationModel(PosDependentObservationModel.PosDependentObservationModel):
    cdef get_pos_probability_vector(self, token)
