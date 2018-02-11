# cython: profile=False
# cython: linetrace=False
# cython: binding=False
# distutils: define_macros=CYTHON_TRACE=0

cimport cython
import cython
import logging
import time
import numpy as np
cimport numpy as np
import sys
import State
from Sampler import *
from HmmSampler import *
from PyzmqMessage import ModelWrapper
#from Indexer import Indexer
import Indexer
import scipy.sparse

#@profile
def compile_one_line(int depth, int prev_index, models, indexer, full_pi = False):

    cdef int totalK, a_max, b_max, g_max, start_depth, above_act, prev_b_above, state_index
    cdef int f,j,a,b,g, prevF, prevJ
    cdef float range_probs
    cdef np.ndarray range_probs_full

    prev_state = indexer.extractState(prev_index)
    logging.debug(' '.join(map(str, ['index',prev_index,'state',prev_state.unfiltered_str()])))
    start_depth = get_cur_awa_depth(prev_state.b)
    nominal_depth = start_depth + prev_state.f
    (a_max, b_max, g_max) = indexer.getVariableMaxes()
    totalK = indexer.get_state_size()

    ## Get EOS indices
    EOS_full = EOS_1wrd_full = indexer.get_EOS_full()
    EOS =  EOS_1wrd = indexer.get_EOS()
    # EOS_1wrd_full = indexer.get_EOS_1wrd_full()
    # EOS_1wrd = indexer.get_EOS_1wrd()

    ## Initialize sparse value lists
    indices =  []
    data = []
    indices_full = []
    data_full = []

    ## If previous was EOS, no outgoing transitions allowed
    if prev_index/g_max == EOS or prev_index/g_max == EOS_1wrd:
        return indices, data, indices_full, data_full

    ## Skip invalid start states
    ## never have b_max or a_max or p_max in the state for the time being
    if any([x == a_max -1 for x in prev_state.a]) or any([x == b_max -1 for x in prev_state.b]) or prev_state.g == g_max - 1:
        return indices, data, indices_full, data_full

    ## if fork and join are both 0 or 1 and a and b stacks are empty
    ## then it is impossible. F or J should be 1 if a and b stacks are empty
    if prev_state.j == 1 and not any(prev_state.a) and not any(prev_state.b) and prev_index != 0:
        return indices, data, indices_full, data_full

    ## g must be always of some valid value (first init state) except when it is at the depth -1
    if prev_state.g == 0 and prev_index != 0:
        return indices, data, indices_full, data_full

    ## if no fork, prev.b must agree with prev.g
    if prev_state.f == 0:
        for b in prev_state.b[::-1]:
            if b != 0 and prev_state.g != b:
                return indices, data, indices_full, data_full
            elif b != 0 and prev_state.g == b:
                break

    if depth >= 1:
        ## Others that are allowed but not really possible:
        ## 1) where lower depths have real values but higher depths have
        ## placeholder 0 values
        ## 2) where one of A and B has a placeholder at the same depth level
        lowest_depth = -1
        for d in range(depth-1, -1, -1):
            if(prev_state.a[d] * prev_state.b[d] == 0 and prev_state.a[d] + prev_state.b[d] != 0):
                ## Exactly one of the values at this depth is zero -- not allowed
                return indices, data, indices_full, data_full
            if lowest_depth == -1:
                if prev_state.a[d] > 0:
                    ## Until we find a non-zero value move the depth up
                    lowest_depth = d
            else:
                ## we have seen a non-zero value deeper than us, so we can't see any zeros now
                if prev_state.a[d] == 0 or prev_state.b[d] == 0:
                    return indices, data, indices_full, data_full

    cum_probs = np.zeros(3)
    next_state = State.State(depth)

    # Populate previous state conditional dependencies
    prev_g = prev_state.g
    if start_depth == -1:
        prev_a = 0
        prev_b = 0
        prev_b_above = 0
    else:
        prev_a = prev_state.a[start_depth]
        prev_b = prev_state.b[start_depth]
        if start_depth == 0:
            prev_b_above = 0
            prev_a_above = 0
        else:
            prev_b_above = prev_state.b[start_depth-1]
            prev_a_above = prev_state.a[start_depth - 1]

    t00 = time.time()

    ## special case for start state:
    if prev_index == 0:

        state_f = 1
        state_j = 0
        state_a = np.zeros(depth, dtype=np.int64)
        state_b = np.zeros(depth, dtype=np.int64)
        state_g = 0
        state_index_full = indexer.getStateIndex(state_j, state_a, state_b, state_f, state_g)
        state_index = int(state_index_full / g_max)
        if full_pi:
            range_probs_full = models.pos.dist[0, :-1]
            for g in range(1,len(range_probs_full)):
                indices_full.append(state_index + g)
                data_full.append(range_probs_full[g])
        indices.append(state_index)
        data.append(1)
        logging.debug(' '.join(map(str, [prev_state.unfiltered_str(), '->', indexer.extractState(state_index_full).unfiltered_str(), 1])))
        return indices, data, indices_full, data_full

    # for f in (0,1):
    #     next_state.j = j
    #
    #     ## when t=0, start_depth will be -1, which in the d> 1 case will wraparound.
    #     ## we want in the t=0 case for f_{t=1} to be [-/-]*d
    #     if start_depth > 0:
    #         cum_probs[0] = models.fork[start_depth].dist[ prev_b, prev_g, f ]
    #     else:
    #         ## if start depth is -1 we're only allowed to fork:
    #         if next_state.j == 1:
    #             cum_probs[0] = 1.0
    #         else:
    #             continue

    for j in (0,1):
        if j == 0 and nominal_depth == depth:
            continue
        next_state.j = j
        ## See note above where we set next_state.f
        if prev_state.f == 0:
            cum_probs[0] = models.J[start_depth].dist[ prev_a, prev_b_above, j ]
        else:
            if nominal_depth == depth:
                next_state.j = 1
                cum_probs[0] = 1
            else:
                cum_probs[0] = models.J[start_depth+1].dist[ prev_g, prev_b, j ]

        ## Add probs for transition to EOS
        if (prev_state.f==0 and j==1 and start_depth == 0) or (prev_state.f==1 and j==1 and start_depth == -1
                                                               and prev_state.max_awa_depth() == -1):
            # FJ decision into EOS is observed, don't model. Just extract prob from awaited transition
            if start_depth == -1:
                start_depth_tmp = 0
            else:
                start_depth_tmp = start_depth
            if prev_state.f == 0:
                EOS_prob = models.J[start_depth_tmp].dist[ prev_a, prev_b_above, 1 ]
            else:
                EOS_prob = models.J[start_depth_tmp].dist[ prev_g, prev_b, 1]
            if full_pi:
                indices_full.append(EOS_full)
                data_full.append(EOS_prob)
            indices.append(EOS)
            data.append(EOS_prob)
            logging.debug(' '.join(map(str, [prev_state.unfiltered_str(), '->', indexer.extractState(EOS_full).unfiltered_str(), EOS_prob])))
            continue

        for a in range(1, a_max-1):
            next_state.a[:] = 0
            for b in range(1, b_max-1):
                next_state.b[:] = 0

                if prev_state.f == 1 and j == 1:
                    if a == prev_state.a[start_depth]:
                        next_state.a[0:start_depth] = prev_state.a[0:start_depth]
                        next_state.b[0:start_depth] = prev_state.b[0:start_depth]
                        next_state.b[start_depth] = b
                        next_state.a[start_depth] = prev_state.a[start_depth]
                        cum_probs[1] = cum_probs[0] * models.B_J1[start_depth].dist[ prev_b, prev_g, b ]
                    else:
                        continue

                elif prev_state.f == 0 and j == 1:
                    ## -/+ case, reducing a level unless already at minimum depth
                    if start_depth <= 0:
                        continue
                    next_state.a[0:start_depth-1] = prev_state.a[0:start_depth-1]
                    next_state.b[0:start_depth-1] = prev_state.b[0:start_depth-1]

                    if a == prev_state.a[start_depth-1]:
                        next_state.a[start_depth-1] = a
                        next_state.b[start_depth-1] = b
                        cum_probs[1] = cum_probs[0] * models.B_J0[start_depth-1].dist[ prev_b_above, prev_a, b ]
                    else:
                        continue

                elif prev_state.f == 0 and j == 0:
                    ## -/-, reduce in place
                    next_state.a[0:start_depth] = prev_state.a[0:start_depth]
                    next_state.b[0:start_depth] = prev_state.b[0:start_depth]
                    next_state.a[start_depth] = a
                    cum_probs[1] = cum_probs[0] * models.A[start_depth].dist[prev_b_above, prev_a, a ] * \
                                   models.B_J0[start_depth].dist[ a, prev_a, b ]

                    next_state.b[start_depth] = b

                elif prev_state.f == 1 and j == 0:
                    ## +/-, create a new stack level unless we're at the limit
                    if start_depth+1 == depth:
                        continue
                    next_state.a[0:start_depth+1] = prev_state.a[0:start_depth+1]
                    next_state.b[0:start_depth+1] = prev_state.b[0:start_depth+1]
                    next_state.a[start_depth+1] = a
                    cum_probs[1] = cum_probs[0] * models.A[start_depth+1].dist[ prev_b, prev_g, a ] * \
                                   models.B_J0[start_depth+1].dist[ a, prev_g, b ]

                    next_state.b[start_depth+1] = b
                for f in (0, 1):
                    next_state.f = f

                    ## when t=0, start_depth will be -1, which in the d> 1 case will wraparound.
                    ## we want in the t=0 case for f_{t=1} to be [-/-]*d
                    if nominal_depth - j >= 0:
                        cum_probs[2] = cum_probs[1] * models.F[nominal_depth - j].dist[b, f]
                    else:
                        continue
                        # ## if start depth is -1 we're only allowed to fork:
                        # if next_state.f == 1:
                        #     cum_probs[2] = cum_probs[1]
                        # else:
                        #     continue

                    ## Now multiply in the pos tag probability:
                    state_index_full = indexer.getStateIndex(next_state.j, next_state.a, next_state.b, next_state.f, 0)
                    state_index = state_index_full / g_max

                    logging.debug(' '.join(map(str,[prev_state.unfiltered_str(), '->', next_state.unfiltered_str(), cum_probs[2]])))
                    # the g is factored out
                    range_probs = cum_probs[2] #* (models.pos.dist[b,:-1])
                    if full_pi:
                        if next_state.f == 0:
                            range_probs_full = cum_probs[2] * (np.ones_like(models.pos.dist[b, :-1]))
                        else:
                            range_probs_full = cum_probs[2] * (models.pos.dist[b,:-1])
                        #logging.info("Building model with %s => %s" % (prev_state.str(), next_state.str() ) )
                        for g in range(1,len(range_probs_full)):
                            indices_full.append(state_index_full + g)
                            data_full.append(range_probs_full[g])
                    indices.append(state_index)
                    data.append(range_probs)

    return indices, data, indices_full, data_full

cdef class FullDepthCompiler:
    cdef int depth

    def __init__(self, depth):
        self.depth = depth

    #@profile
    def compile_and_store_models(self, models, working_dir):
        indexer = Indexer.Indexer(models)
        logging.info("Compiling component models into mega-HMM transition and observation matrices")
        maxes = indexer.getVariableMaxes()
        (a_max, b_max, g_max) = maxes
        totalK = indexer.get_state_size()

        cache_hits = 0
        t0 = time.time()
        indptr = np.zeros(totalK+1)
        indices =  []
        data = []

        ## Take exponent out of inner loop:
        unlog_models(models, self.depth)

        for prev_index in range(0,totalK):
            indptr[prev_index+1] = indptr[prev_index]
            (local_indices, local_data) = compile_one_line(self.depth, prev_index, models, indexer)
            indptr[prev_index+1] += len(local_indices)
            indices.append(local_indices)
            data.append(local_data)

        logging.info("Flattening sublists into main list")
        flat_indices = [item for sublist in indices for item in sublist]
        flat_data = [item for sublist in data for item in sublist]

        relog_models(models, self.depth)
        logging.info("Creating csr transition matrix from sparse indices")
        pi = scipy.sparse.csr_matrix((flat_data,flat_indices,indptr), (totalK, totalK), dtype=np.float64)
        fn = working_dir+'/models.bin'
        out_file = open(fn, 'wb')
        logging.info("Transforming and writing csc model")
        model = ModelWrapper(ModelWrapper.HMM, (models,pi.tocsc()), self.depth)
        pickle.dump(model, out_file)
        out_file.close()
        nnz = pi.nnz
        pi = None

        time_spent = time.time() - t0
        logging.info("Done in %d s with %d cache hits and %d non-zero values" % (time_spent, cache_hits, nnz))


def unlog_models(models, depth):

    for d in range(0, depth):
        models.F[d].dist = 10**models.F[d].dist

        models.J[d].dist = 10**models.J[d].dist

        models.A[d].dist = 10**models.A[d].dist

        models.B_J0[d].dist = 10**models.B_J0[d].dist
        models.B_J1[d].dist = 10**models.B_J1[d].dist

    models.pos.dist = 10**models.pos.dist

def relog_models(models, depth):
    for d in range(0, depth):
        models.F[d].dist = np.log10(models.F[d].dist)

        models.J[d].dist = np.log10(models.J[d].dist)

        models.A[d].dist = np.log10(models.A[d].dist)

        models.B_J0[d].dist = np.log10(models.B_J0[d].dist)
        models.B_J1[d].dist = np.log10(models.B_J1[d].dist)

    models.pos.dist = np.log10(models.pos.dist)

def get_cur_awa_depth(stack):
    ## Essentially empty -- used for first time step
    if stack[0] <= 0:
        return -1

    ## If we encounter a zero at position 1, then the depth is 0
    for d in range(1, len(stack)):
        if stack[d] == 0:
            return d-1

    ## Stack is full -- if d=4 then max depth index is 3
    return len(stack)-1
