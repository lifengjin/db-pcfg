# cython: profile=False
# cython: linetrace=False
# cython: binding=False
# distutils: define_macros=CYTHON_TRACE=0

cimport cython
import logging
import pickle
import time
import numpy as np
cimport numpy as np
import zmq
import scipy.sparse
cimport Indexer
import Indexer
import Sampler
cimport Sampler
import State
cimport State
import subprocess
from uhhmm_io import printException
import models
cimport models
import scipy.sparse
from CategoricalObservationModel import CategoricalObservationModel

def boolean_depth(l):
    for index,val in enumerate(l):
        if val >= 0:
            return index

cdef class HmmSampler(Sampler.Sampler):

    def __init__(self, obs_model=None, seed=0):
        Sampler.Sampler.__init__(self, seed)
        self.indexer = None
        self.models = None
        if obs_model is None:
            self.obs_model = CategoricalObservationModel(self.indexer)
        else:
            self.obs_model = obs_model
#        self.pi = None

    def get_factor_expand_mat(self):
        state_size = self.indexer.get_state_size()
        g_max = self.indexer.getVariableMaxes()[2]
        factored_state_size = state_size // g_max;
        temp_mat = np.zeros((factored_state_size, state_size))

#        for col_ind in range(state_size):
#            row_ind = col_ind // g_len
#            temp_mat[row_ind, col_ind] = 1
        col_ind = 0
        for row_ind in range(factored_state_size):
            expanded_ind = row_ind * g_max
            state = self.indexer.extractState(expanded_ind)
            b_val = self.indexer.extractAwa(expanded_ind)
            #logging.info("Factored row index=%d expanded to full index=%d with b=%d, g=%d, col_ind=%d" % (row_ind, expanded_ind, b_val, state.g, col_ind))
            for g in range(g_max-1):
                prob = self.models.pos.dist[b_val, g]
                if prob < 0.0 or prob > 1.0:
                    logging.warn("POS model has invalid probability %f for b=%d, g=%d" % (prob, b_val, g))

                if g == g_max-1 and prob > 0:
                    logging.warn("POS model has non-zero probability %f for b=%d, g=%d" % (prob, b_val, g))

                temp_mat[row_ind, col_ind + g] = prob
            col_ind += g_max

        return temp_mat

    def set_models(self, models):
        self.models = models[0]
        unlog_models(self.models)
        #self.lexMatrix = np.matrix(self.models.lex.dist, copy=False)
        self.depth = len(self.models.F)
        self.indexer = Indexer.Indexer(self.models)

        #g_len = self.models.pos.dist.shape[1]
        #w_len = self.models.lex.dist.shape[1]
        #lexMultiplier = scipy.sparse.csc_matrix(np.tile(np.identity(g_len), (1, self.indexer.get_state_size() // g_len)))
        #self.data = lexMultiplier.data
        #self.indices = lexMultiplier.indices
        #self.indptr = lexMultiplier.indptr

        self.factor_expand_mat = self.get_factor_expand_mat()

        self.obs_model.set_models(self.models)

        self.factor_expand_mat = self.get_factor_expand_mat()

    def initialize_dynprog(self, batch_size, maxLen):
        ## We ignore batch size since python only processes one at a time
        #self.dyn_prog = np.zeros((self.indexer.get_state_size(), maxLen))
        self.dyn_prog = np.zeros((maxLen, self.indexer.get_state_size()))
        self.start_state = np.zeros(self.indexer.get_state_size())
        self.start_state[0] = 1

    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef forward_pass(self, pi, list sents, int sent_index):
        cdef float sentence_log_prob = 0, normalizer
        cdef double t0, t1
        cdef int a_max, b_max, g_max, index, token, g_len
        cdef tuple maxes
        cdef list sent

        #logging.warning("Forward pass for sentence %d" % (sent_index))
        if len(sents) > 1:
            raise Exception("Error: Python version only accepts batch size 1")

        sent = sents[0]
        t0 = time.time()
        try:
            ## keep track of forward probs for this sentence:
            maxes = self.indexer.getVariableMaxes()
            (a_max,b_max,g_max) = maxes

            ## Copy=False indicates that this matrix object is just a _view_ of the
            ## array -- we don't have to copy it into a matrix and recopy back to array
            ## to get the return value
            self.dyn_prog[:] = 0
            #lexMultiplier = scipy.sparse.csc_matrix((self.data, self.indices, self.indptr), shape=(g_max, self.indexer.get_state_size() ) )
            sparse_factored_expand_mat = scipy.sparse.csc_matrix(self.factor_expand_mat)
            ## Make forward be transposed up front so we don't have to do all the transposing inside the loop
            forward = np.matrix(self.dyn_prog, copy=False)

            for index,token in enumerate(sent):
                if index == 0:
                    prev_state = self.start_state
                else:
                    prev_state = forward[index-1,:]

                # Goes from 1 x N state to 1 x K for total state size N and factored state size K
                factored_transition = prev_state * pi

                # Expand 1 x K to 1 x N with p(g | b) replicated values
                forward[index,:] = factored_transition * sparse_factored_expand_mat

                lex_prob = self.obs_model.get_probability_vector(token)

                if lex_prob.max() < 0:
                    ## The observation model passed back a log prob
                    log_lex_prob = lex_prob
                    #print("lex_prob:", lex_prob)
                    #print("Received log probs from obs model: ", log_lex_prob)

                    ## This is the "exp-normalize trick" for normalizing probability
                    ## distributions coming out of log space, seen at:
                    ## timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick
                    forward[index,:] = np.log(forward[index,:]) + log_lex_prob
                    b = forward[index,:].max()
                    exponentiated = np.exp(forward[index,:] - b)
                    forward[index,:] =  exponentiated / exponentiated.sum()
                    normalizer = exponentiated.sum()
                else:
                    print("Received regular probs from obs model: ", lex_prob)
                    ## Original way of incorporating lex probs, if observation
                    ## model passed back a standard probability:
                    forward[index,:] = np.multiply(forward[index,:], lex_prob)

                    normalizer = forward[index,:].sum()
                    print("normalizer:", normalizer)
                    forward[index,:] /= normalizer

                ## Normalizer is p(y_t)
                sentence_log_prob += np.log10(normalizer)
            last_index = len(sent)-1
            if np.argwhere(forward.max(1)[0:last_index+1,:] == 0).size > 0 or np.argwhere(np.isnan(forward.max(1)[0:last_index+1,:])).size > 0:
                logging.error("There is a word with no positive probabilities for its generation in the forward filter: %s" % forward.max(1)[0:last_index+1,:])
                raise Exception("There is a word with no positive probabilities for its generation in the forward filter.")

            ## FIXME - Do we need to multiply by -/+ at last time step for d > 0 system?
            ## TODO - should we make probability of transitioning to d > 0 state at
            ## last time step = 0.0? Or just sample it with prob = 0.0?
        except Exception as e:
            printException()
            raise e
        # np.set_printoptions(threshold=np.nan)
        # print(self.dyn_prog)
        t1 = time.time()
        # print('forward time', t1-t0)
        self.ff_time += (t1-t0)
        return [sentence_log_prob]

    cpdef reverse_sample(self, pi, list sents, int sent_index):
        cdef int totalK, depth, last_index, sample_t, sample_depth, t, ind
        cdef int prev_depth, next_f_depth, next_awa_depth
        cdef float trans_prob, sample_log_prob
        cdef double t0, t1
        cdef list sample_seq, sent
        cdef tuple maxes
        cdef np.ndarray trans_slice
        cdef State.State sample_state

        sent = sents[0]
        t0 = time.time()
        try:
            sample_seq = []
            sample_log_prob = 0
            maxes = self.indexer.getVariableMaxes()
            totalK = self.indexer.get_state_size()
            depth = len(self.models.F)

            ## Normalize and grab the sample from the forward probs at the end of the sentence
            last_index = len(sent)-1

            ## normalize after multiplying in the transition out probabilities
            self.dyn_prog[last_index,:] /= self.dyn_prog[last_index,:].sum()

            ## EOS state is f=0,j=1,a=[0]*d,b=[0]*d,g=0. Located 1/4 way through state list.
            if len(sent) == 1:
                sample_t = self.indexer.get_EOS_1wrd_full()
            else:
                sample_t = self.indexer.get_EOS_full()

            for t in range(len(sent)-1,-1,-1):
                sample_state, sample_t = self._reverse_sample_inner(pi, sample_t, t)
                sample_seq.append(sample_state)

            sample_seq.reverse()
        except Exception as e:
            printException()
            raise e

        t1 = time.time()
        # print('backward time', t1-t0)
        self.bs_time += (t1-t0)
        # np.set_printoptions(threshold=np.nan)
        # print(self.dyn_prog)
        return [sample_seq]

    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef _reverse_sample_inner(self, pi, int sample_t, int t):
        cdef np.ndarray trans_slice
        cdef int ind
        cdef float normalizer
        cdef tuple maxes
        maxes = self.indexer.getVariableMaxes()
        g_max = maxes[2]

        #logging.info("G max is %d"  % (g_max))

        reduced_state = sample_t // g_max
        trans_slice = pi[:,reduced_state].toarray()

        for ind in np.where(self.dyn_prog[t,:] != 0.0)[0]:
            self.dyn_prog[t,ind] *= trans_slice[ind]

        normalizer = self.dyn_prog[t,:].sum()
        if normalizer == 0.0:
            logging.warning("No positive probability states at this time step %d." % (t))

        self.dyn_prog[t,:] /= normalizer

        sample_t = get_sample(self.dyn_prog[t,:])
        sample_state = self.indexer.extractState(sample_t)
        #logging.log(logging.DEBUG-1, "Sampled state %s with index %d at time %d" % (sample_state.str(), sample_t, t))

        if t > 0 and sample_state.g == 0:
            logging.error("Error: Sampled a g=0 state with state index %d in backwards pass: %s" % (sample_t, sample_state.str()) )
            raise Exception

        return sample_state, sample_t

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef int max_awa_depth(np.ndarray b):
    cdef int d = 0
    for d in range(0, len(b)):
        if b[d] == 0:
            return d-1

    return 0

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.cdivision(True)    # faster division without checks
cdef int get_sample(np.ndarray[np.float64_t] dist):
    cdef float dart
    cdef np.ndarray[np.float64_t] sum_dist
    cdef int start, end, cur, ret

    sum_dist = np.cumsum(dist)
    dart = np.random.random()
    start = 0
    end = len(dist) - 1
    cur = (end + start) / 2
    while True:
        if dart > sum_dist[cur]:
            if cur == end - 1:
                ret = cur+1
                break

            start = cur
        else:
            ## dart < sum_dist but need to check 3 cases:
            if cur == 0:
                ret = cur
                break
            elif sum_dist[cur-1] < dart:
                ret = cur
                break
            else:
                end = cur

        cur = (end+start) / 2

    return ret

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef int old_get_sample(np.ndarray dist):
    cdef np.ndarray sum_dist
    cdef int i
    cdef float dart

    sum_dist = np.cumsum(dist)
    dart = np.random.random()

    for i in range(0, len(dist)):
        if dart < sum_dist[i]:
            return i

def unlog_models(models):
    depth = len(models.F)
    for d in range(0, depth):
        models.F[d].dist = 10**models.F[d].dist
        models.J[d].dist = 10**models.J[d].dist

        models.A[d].dist = 10**models.A[d].dist

        models.B_J0[d].dist = 10**models.B_J0[d].dist
        models.B_J1[d].dist = 10**models.B_J1[d].dist

    models.pos.dist = 10**models.pos.dist
    if type(models.lex.dist).__name__ == 'ndarray':
        models.lex.dist = 10**models.lex.dist
