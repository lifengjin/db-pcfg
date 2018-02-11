# cython: profile=False
# cython: linetrace=False
# cython: binding=False
# distutils: define_macros=CYTHON_TRACE=0

import models
cimport models

## TODO: have the observation model take care of multiplying values of g
## across all states. i.e. most observation models only use the value of g
## to compute observation probability and then have to multiply that vector
## across all states -- observation model should do that for them.
## Or an abstract subclass called POSDependentObservationModel could define a new
## method called get_probability_for_g()
cdef class ObservationModel:

    cdef public set_models(self, models):
        raise NotImplementedError()

    cdef get_probability_vector(self, token):
        raise NotImplementedError()
