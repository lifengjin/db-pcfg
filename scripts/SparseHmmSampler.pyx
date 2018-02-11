# cython: profile=True
# cython: linetrace=True
# cython: binding=True
# distutils: define_macros=CYTHON_TRACE=1

cimport cython
import logging
import pickle
import time
import numpy as np
import sys
import zmq
import scipy.sparse
from Indexer import *
import Sampler
from State import *
import subprocess

def boolean_depth(l):
    for index,val in enumerate(l):
        if val >= 0:
            return index

class HmmSampler(Sampler.Sampler):
    def __init__(self, seed):
        Sampler.Sampler.__init__(self, seed)
        self.ff_time = 0
        self.bs_time = 0
        self.indexer = None
        
    def set_models(self, models):
        (self.models, self.pi) = models
        unlog_models(self.models)
        self.lexMatrix = np.matrix(self.models.lex.dist, copy=False)
        self.depth = len(self.models.fork)
        self.indexer = Indexer(self.models)
        
        g_len = self.models.pos.dist.shape[-1]
        w_len = self.models.lex.dist.shape[-1]
        scipy = True
        try:
            import scipy.sparse
            self.lexMultiplier = scipy.sparse.csc_matrix(np.tile(np.identity(g_len), (1, self.indexer.get_state_size() / g_len)))
        except:
            logging.warn("Could not find scipy! Using numpy will be much less memory efficient!")
            self.lexMultipler = np.tile(np.identity(g_len), (1, self.indexer.get_state_size() / g_len))
            scipy = False
        
    def initialize_dynprog(self, batch_size, maxLen):
        self.dyn_prog = np.zeros((self.indexer.get_state_size(), maxLen))

#    @profile
    def forward_pass(self,sent,sent_index):
        t0 = time.time()
        try:
            ## keep track of forward probs for this sentence:
            maxes = self.indexer.getVariableMaxes()
            (a_max,b_max,g_max) = maxes

            #self.dyn_prog[:] = 0
            sentence_log_prob = 0

            ## Copy=False indicates that this matrix object is just a _view_ of the
            ## array -- we don't have to copy it into a matrix and recopy back to array
            ## to get the return value
#            forward = np.matrix(self.dyn_prog, copy=False)
            forward = []
            for index,token in enumerate(sent):
#                logging.info("Index %d" % index)
                ## Still use special case for 0
                if index == 0:
#                    forward[1:g_max-1,0] = self.lexMatrix[1:-1,token]
                    start = np.zeros((self.indexer.get_state_size(), 1))
                    start[1:g_max-1] = self.lexMatrix[1:-1,token]
                    forward.append( scipy.sparse.csc_matrix( start ) )
                else:
#                    forward[:,index] = self.pi.transpose() * forward[:,index-1]
                    intermediate = self.pi.transpose() * forward[-1]
                    
                    expanded_lex = self.lexMatrix[:,token].transpose() * self.lexMultiplier
                    
#                    forward[:,index] = np.multiply(forward[:,index], expanded_lex.transpose())
                    probs = intermediate.multiply( expanded_lex.transpose() )
#                    logging.info("Probability has %d non-zero elements before pruning" % (len(probs > 0)))
                   
#                    logging.info("probs has type %s and shape %s and datatype %s, about to compare to %f" % (type(probs), probs.shape, probs.dtype, probs.max()))
                    
                    is_too_small = np.less(probs, probs.max(0) / 100)
                    small_inds = np.where( is_too_small )
                    
                    probs[small_inds] = 0
                    sparse_probs = scipy.sparse.csc_matrix(probs) 
#                    logging.info("Probability has %d non-zero elements after pruning" % sparse_probs.nnz)
                    
                    forward.append( sparse_probs )
                    
                normalizer = forward[index].sum()
                forward[index] /= normalizer
                
                ## Normalizer is p(y_t)
                sentence_log_prob += np.log10(normalizer)

            last_index = len(sent)-1
            if forward[-1].max() == 0 or np.isnan(forward[-1].max()):
                logging.error("Error; There is a word with no positive probabilities for its generation in the forward filter.")
                raise Exception("There is a word with no positive probabilities for its generation in the forward filter.")

            ## FIXME - Do we need to multiply by -/+ at last time step for d > 0 system?
            ## TODO - should we make probability of transitioning to d > 0 state at 
            ## last time step = 0.0? Or just sample it with prob = 0.0?
        except Exception as e:
            printException()
            raise e
        
        self.dyn_prog = forward
        t1 = time.time()
        self.ff_time += (t1-t0)
        return sentence_log_prob

    #@profile
    def reverse_sample(self, sent, sent_index):
        cdef int totalK, depth, last_index, sample_t, sample_depth, t, ind
        cdef int prev_depth, next_f_depth, next_awa_depth
        cdef float trans_prob, normalizer
        
        t0 = time.time()
        try:      
            sample_seq = []
            sample_log_prob = 0
            maxes = self.indexer.getVariableMaxes()
            totalK = self.indexer.get_state_size()
            depth = len(self.models.fork)
        
            ## Normalize and grab the sample from the forward probs at the end of the sentence
            last_index = len(sent)-1
#            if self.dyn_prog[:,last_index].max() == 0:
#                logging.error("Error: The last word of the sentence has no positive probabilities in the reverse sample method.")
                
#            if np.isnan(self.dyn_prog[:,last_index].max()):
#                logging.error("Error: The last word of the sentence is nan in the reverse sample method.")
            
            
            ## normalize after multiplying in the transition out probabilities
            self.dyn_prog[last_index] /= self.dyn_prog[last_index].sum()
            
            sample_t = -1
            sample_depth = -1
            ## We require that the sample comes from the set of states that are at depth
            ## 0 (i.e. the sentence must fully reduce to be a valid parse)
            #print(dyn_prog[:,last_index])
            while sample_t < 0 or sample_depth > 0:
                sample_t = get_sample(self.dyn_prog[last_index].toarray())
                sample_state = self.indexer.extractState(sample_t)
                sample_depth = sample_state.max_awa_depth()
                #logging.debug("Sampled final state %s with depth %d" % (sample_state.str(), sample_depth))
    
            sample_seq.append(self.indexer.extractState(sample_t))
            #logging.debug("Sampled state %s at time %d" % (sample_seq[-1].str(), last_index))
            
            if last_index > 0 and (sample_seq[-1].a[0] == 0 or sample_seq[-1].b[0] == 0 or sample_seq[-1].g == 0):
                logging.error("Error: First sample for sentence %d has index %d and has a|b|g = 0" % (sent_index, sample_t))
                sys.exit(-1)
            
            for t in range(len(sent)-2,-1,-1):
                trans_slice = self.pi[:, sample_t].toarray()
                for ind in np.where(self.dyn_prog[t] != 0.0)[0]:
                     
                    self.dyn_prog[t][ind] *= trans_slice[ind]

                normalizer = self.dyn_prog[t].sum()
                if normalizer == 0.0:
                    logging.warning("No positive probability states at this time step %d." % (t))
                      
                self.dyn_prog[t] /= normalizer
                sample_t = get_sample(self.dyn_prog[t].toarray())
                sample_state = self.indexer.extractState(sample_t)
                logging.debug("Sampled state %s with index %d at time %d" % (sample_state.str(), sample_t, t))
            
                if t > 0 and sample_state.g == 0:
                    logging.error("Error: Sampled a g=0 state in backwards pass! {0}".format(sample_state.str()))
                
                sample_seq.append(sample_state)
                
            sample_seq.reverse()
            logging.log(logging.DEBUG, "Sample sentence %d: %s" % (sent_index, list(map(lambda x: x.str(), sample_seq))))
        except Exception as e:
            printException()
            raise e
        
        t1 = time.time()
        self.bs_time += (t1-t0)
        return sample_seq

cdef max_awa_depth(b):
    cdef int d = 0
    for d in range(0, len(b)):
        if b[d] == 0:
            return d-1
            
    return 0

cdef int get_sample(dist):
    cdef int i = 0
    sum_dist = np.cumsum(dist)
    dart = np.random.random()
    for i in range(0, len(dist)):
        if dart < sum_dist[i]:
            return i
    
def unlog_models(models):
    depth = len(models.fork)
    for d in range(0, depth):
        models.fork[d].dist = 10**models.fork[d].dist
        
        models.reduce[d].dist = 10**models.reduce[d].dist
        models.trans[d].dist = 10**models.trans[d].dist
        
        models.act[d].dist = 10**models.act[d].dist
        models.root[d].dist = 10**models.root[d].dist
        
        models.cont[d].dist = 10**models.cont[d].dist
        models.exp[d].dist = 10**models.exp[d].dist
        models.next[d].dist = 10**models.next[d].dist
        models.start[d].dist = 10**models.start[d].dist
        
    models.pos.dist = 10**models.pos.dist
    models.lex.dist = 10**models.lex.dist
