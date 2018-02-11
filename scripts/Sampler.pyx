#!/usr/bin/env python3

import logging
import numpy as np
from uhhmm_io import printException


cdef class Sampler:
    def __cinit__(self):
        self.ff_time = 0
        self.bs_time = 0
        
    def __init__(self, seed=-1):
        if seed > 0:
            np.random.seed(seed)

    cpdef sample(self, pi, list sents, int sent_index):
        
        try:
            logging.debug("Starting forward pass of sentence %d with batch size %d" % (sent_index, len(sents)))
            log_probs = self.forward_pass(pi, sents, sent_index)
            logging.debug("Starting backwards pass of sentence %d" % (sent_index) )
            sent_samples = self.reverse_sample(pi, sents, sent_index)
            logging.debug("Finished parsing sentence %d" % (sent_index) )
        except Exception as e:
            printException()
            raise e
                
        return (sent_samples, log_probs)
