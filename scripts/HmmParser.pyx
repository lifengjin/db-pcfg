#!python3.4
# cython: profile=True
# cython: linetrace=True
# cython: binding=True
# distutils: define_macros=CYTHON_TRACE=1

cimport cython
import numpy as np
cimport numpy as np
import sys, os, linecache
import Sampler
from State import State
import time
import logging
import scipy.sparse
from Indexer import Indexer

class ImportanceSet:
    """ A class that represents a finite number of elements according to an importance 
        function. It is initialized with a size and a function that can be called on the
        elements that it will contain. When add() is called, it will check whether the
        set is full and if so, whether the new element is greater than the existing minimum
        element."""
    def __init__(self, size, importance_function, min_value = 0):
        self.size = size
        self.imp_fun = importance_function
        self.minimum_value = min_value
        self.minimum_ind = -1
        self.elements = []
        self.current = 0
        
    def add(self, element):
        if len(self.elements) < self.size:
            self.elements.append(element)
            if len(self.elements) == 1 or self.imp_fun(element) < self.minimum_value:
                self.minimum_value = self.imp_fun(element)
                self.minimum_ind = len(self.elements) - 1
        elif self.imp_fun(element) > self.minimum_value:
            ## Swap out element with minimum value for new element
            self.elements[self.minimum_ind] = element            
            
            ## Set mins to new element (index is already set via above)
            self.minimum_value = self.imp_fun(element)
            
            ## Check for element with minimum value
            for i in range(self.size):
                if self.imp_fun(self.elements[i]) < self.minimum_value:
                    self.minimum_value = self.imp_fun(self.elements[i])
                    self.minimum_ind = i        
        
        ## Otherwise ignore it
        
    def __iter__(self):
        return iter(self.elements)
    
    def __len__(self):
        return len(self.elements)
    
class ParserCell:
    def __init__(self):
        self.prob = 0
        self.state_index = -1
        self.bp = None

class HmmParser:    
    def __init__(self, models):
        (self.models, self.pi) = models.model
        print("Type of pi matrix is: %s" % (type(self.pi)) )
        ## lil seems to be fastest of sparse for indexing, also faster than using
        ## csc and getting a slice
        #self.pi = self.pi.tolil()
        unlog_models(self.models)
        self.indexer = Indexer(self.models)
        self.totalK = self.indexer.get_state_size()
        self.maxes = self.indexer.getVariableMaxes()
        g_len = self.models.pos.dist.shape[1]
        w_len = self.models.lex.dist.shape[1]
        self.lexMatrix = np.matrix(self.models.lex.dist, copy=False)
        self.lexMultiplier = scipy.sparse.csc_matrix(np.tile(np.identity(g_len), (1, self.indexer.get_state_size() / g_len)))
#        self.lexMultiplier = scipy.sparse.csc_matrix((self.data, self.indices, self.indptr), shape=(g_max, self.indexer.get_state_size() ) )
    
    @cython.boundscheck(False)
    def matrix_parse(self, list sent):
        #cdef np.ndarray prob_mat, full_prob_mat, expanded_lex
        cdef np.ndarray[np.int_t, ndim=2] bp_mat
        cdef int index, token, a_max, b_max, g_max, token_index, state_index
        cdef list sent_states
        
        bp_mat = np.zeros( (len(sent), self.indexer.state_size), dtype=np.int)
        prob_mat = [] #np.zeros( (len(sent), self.indexer.state_size) )
        
        (a_max, b_max, g_max) = self.maxes
        try:
            for index,token in enumerate(sent):
                next_prob = np.zeros( (1, self.indexer.state_size) )
                t0 = time.time()
                ## Still use special case for 0
                if index == 0:
                    next_prob[0,1:g_max-1] = self.lexMatrix[1:g_max-1,token].transpose()
                    prob_mat.append( scipy.sparse.csc_matrix( next_prob ) )
                else:
                    #print("Trans prob type = %s, shape=%s, prevState index=%d, curState=%d" % (type(trans_prob), trans_prob.shape, prevState.state_index, curState))
                    ## Since we're grabbing a slice, it is a row by default and will multiply across
                    ## rows. So we need to transpose pi to get multiplication across the right dimension
                    ## and then transpose the result at the end.
                    next_prob = prob_mat[0] * self.pi
                    # self.pi.transpose().multiply( prob_mat[index-1,:] ).transpose()
                    ### Now we have a matrix of joint probabilites:
                    ### p( x_t, x_{t-1} ). Multiply in observations:

                    ## For this multiplication expanded_lex is not a slice but it is the wrong shape so we transpose after creating.
                    expanded_lex = (self.lexMatrix[:,token].transpose() * self.lexMultiplier)
                    #assert expanded_lex.shape[1] == 1

                    next_prob = np.multiply( expanded_lex, next_prob)
                    
                    ## And take the max for each x_t:
                    cutoff = next_prob.max() / 10
                    
                    maxes = next_prob.max(0)
                    prob_mat.append( next_prob.max(0) ) #  np.multiply(maxes, (maxes > cutoff))
                    bp_mat[index, :] = next_prob.argmax(0)
                    
                    ## Remove the old thing that was t-1, moving t back to t-1 for the next iteration
                    prob_mat.pop(0)
                    
#                    cutoff = prob_mat[index,:].max() / 10
                    
#                    bad_indices = np.where( prob_mat[index,:] < cutoff)
#                    nz_before = len(np.where( prob_mat[index,:] != 0)[0])
#                    prob_mat[index, bad_indices] = 0
#                    nz_after = len(np.where( prob_mat[index,:] != 0)[0])
                    #print("Reduced from %d nnz to %d after filtering." % (nz_before, nz_after) )
                    
            ## Get the max probability at the end:
            token_index = len(sent) - 1
            state_index = prob_mat[-1].argmax()
            sent_states = []
            while token_index >= 0:
                sent_states.append( self.indexer.extractState(state_index) )
                state_index = bp_mat[token_index, state_index]
                token_index -= 1
            
            sent_states.reverse()
        except Exception as e:
            print("Encountered exception while parsing sentence %s" % (sent) )
            raise Exception
        
        assert len(sent_states) == len(sent)
        
        return sent_states
    
    @cython.boundscheck(False)
    def parse(self, sent):
        cdef int index, token, a_max, b_max, g_max, curState, g
        cdef list forward
        cdef float normalizer, t0, t1
        
        cdef np.ndarray trans_slice
        
        (a_max, b_max, g_max) = self.maxes
        
        forward = []
        try:
            for index,token in enumerate(sent):                    
                t0 = time.time()
                ## Still use special case for 0
                if index == 0:
                    cur_probs = ImportanceSet(100, lambda x: x.prob)
                    for curState in range(0,g_max):
                        cell = ParserCell()
                        if curState >= 1:
                            cell.prob = self.lexMatrix[curState, token]
                            cell.bp = None
                            cell.state_index = curState                            
                            cur_probs.add(cell)
                    forward.append(cur_probs)
                else:
                    cur_probs = ImportanceSet(50, lambda x: x.prob)
                    normalizer = 0
                    for curState in range(self.totalK):
                        ## Create the parser cell for this state:
                        max_trans = 0
                        max_bp = None
#                        trans_slice = self.pi[:,curState].toarray()
                        
                        ## Find the input with the highest existing * transition prob:
                        for prevState in forward[-1]:
                            trans_prob = prevState.prob * float( self.pi[prevState.state_index, curState] )
#                            trans_prob = prevState.prob * float(trans_slice[prevState.state_index])
                            #print("Trans prob type = %s, shape=%s, prevState index=%d, curState=%d" % (type(trans_prob), trans_prob.shape, prevState.state_index, curState))
                            ## See if the transition with highest probability
                            if trans_prob > max_trans:
                                max_trans = trans_prob
                                max_bp = prevState
                                
                        if max_trans > 0:
                            ## Extract the pos index for this state so we can get the output prob:
                            (f,j,a,b,g) = self.indexer.extractStacks(curState)
                            max_trans *= self.models.lex.dist[g][token]
                            cur_cell = ParserCell()
                            cur_cell.state_index = curState
                            cur_cell.prob = max_trans
                            cur_cell.bp = max_bp
                            cur_probs.add(cur_cell)                            
                            normalizer += cur_cell.prob
                        
                    forward.append(cur_probs)
                
                t1 = time.time()
                        
            ## Find the maximum value that is at depth 0:
            max_prob_cell = None
            max_prob_state = None
            print("Finished sentence with last time step containing %d cells" % ( len(forward[-1]) ) )
            for cell in forward[-1]:
                state = self.indexer.extractState(int(cell.state_index))
                if (max_prob_cell == None or cell.prob > max_prob_cell.prob):
                    max_prob_cell = cell
                    max_prob_state = state
            
            if max_prob_state == None or max_prob_cell == None:
                logging.warning("No parse for this sentence -- size of forward is %d" % ( len(forward[-1]) ) )
                return []

            reverse_cell_list = [max_prob_cell]
            reverse_sent_list = [max_prob_state]
            
            ## Backtrack through the sentence
            while reverse_cell_list[-1].bp != None:
                cell = reverse_cell_list[-1].bp
                reverse_cell_list.append(cell)
                reverse_sent_list.append( self.indexer.extractState(cell.state_index) )

            reverse_sent_list.reverse()
                
        except Exception as e:
            printException()
            raise e
        
        return reverse_sent_list    

def printException():
    exc_type, exc_obj, tb = sys.exc_info()
    f = tb.tb_frame
    lineno = tb.tb_lineno
    filename = f.f_code.co_filename
    linecache.checkcache(filename)
    line = linecache.getline(filename, lineno, f.f_globals)
    print 'EXCEPTION IN ({}, LINE {} "{}"): {}'.format(filename, lineno, line.strip(), exc_obj)

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
