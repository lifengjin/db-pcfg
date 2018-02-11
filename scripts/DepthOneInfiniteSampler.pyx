#!/usr/bin/env python3
import logging
import pickle
import time
import numpy as np
import sys
import subprocess
from PyzmqWorker import *
from Sampler import *
#cimport cython
#cimport State
#import pyximport; pyximport.install()
import log_math as lm
from Indexer import *

class InfiniteSampler(Sampler):
    def __init__(self, seed):
        Sampler.__init__(self, seed)
        self.indexer = None
        
    def set_models(self, models):
        self.models = models
        self.indexer = Indexer(self.models)

    def initialize_dynprog(self, batch_size, maxLen):
        self.dyn_prog = np.zeros((2,2,self.models.act[0].dist.shape[-1], self.models.cont[0].dist.shape[-1], self.models.pos.dist.shape[-1], maxLen))

    def forward_pass(self, sent, sent_index):
        t0 = time.time()
        maxes = self.indexer.getVariableMaxes()
        (a_max,b_max,g_max) = maxes
        self.dyn_prog[:] = -np.inf
        
        ## keep track of forward probs for this sentence:
        for index,token in enumerate(sent):
            if index == 0:
                self.dyn_prog[0,0,0,0,1:g_max-1,0] = self.models.lex.dist[1:-1,token]
            else:
                for (ind,val) in np.ndenumerate(self.dyn_prog[...,index-1]):
                    if val == -np.inf:
                        continue

                    (prevF, prevJ, prevA, prevB, prevG) = ind

                    assert index == 1 or (prevA != 0 and prevB != 0 and prevG != 0), 'Unexpected values in sentence {0} with non-zero probabilities: {1}, {2}, {3} at index {4}, and f={5} and j={6}'.format(sent_index,prevA, prevB, prevG, index, prevF, prevJ)
                
                    cumProbs = np.zeros(5)
                
                    ## Sample f & j:
                    for f in (0,1):
                        if index == 1 and f == 0:
                            continue

                        cumProbs[0] = self.dyn_prog[prevF,prevJ,prevA,prevB,prevG,index-1] + self.models.fork[0].dist[prevB,prevG,f]
                        if index == 1:
                            j = 0
                        else:
                            j = f
                        
                        ## At depth 1 -- no probability model for j 
                        cumProbs[1] = cumProbs[0]
                        
                        for a in range(1,a_max-1):
                            if f == 0 and j == 0:
                                ## active transition: -- technically depends on above awa but
                                ## in depth one that doesn't exist so just set it to 0
                                cumProbs[2] = cumProbs[1] + lm.log_boolean(self.models.act[0].dist[prevA,0,a] > self.models.act[0].u[sent_index,index])
                            elif f == 1 and j == 0:
                                ## root -- technically depends on prevA and prevG
                                ## but in depth 1 this case only comes up at start
                                ## of sentence and prevA will always be 0
                                cumProbs[2] = cumProbs[1] + lm.log_boolean(self.models.root[0].dist[0,prevG,a] > self.models.root[0].u[sent_index, index])
                            elif f == 1 and j == 1 and prevA == a:
                                cumProbs[2] = cumProbs[1]
                            else:
                                ## zero probability here
                                continue
                        
                            if cumProbs[2] == -np.inf:
                                continue
                      
                            for b in range(1,b_max-1):
                                if f == 1 and j == 1:
                                    cumProbs[3] = cumProbs[2] + self.models.cont[0].dist[prevB,prevG,b]
                                elif f == 1 and j == 0:
                                    ## First token:
                                    cumProbs[3] = cumProbs[2] + self.models.exp[0].dist[prevG, a, b]
                                elif f == 0 and j == 0:
                                    cumProbs[3] = cumProbs[2] + self.models.start[0].dist[prevA, a, b]
                                else:
                                    ## Probably should never get here
                                    continue
                                    
                                if cumProbs[3] == -np.inf:
                                    continue
                                                                
                                # Multiply all the g's in one pass:
                                range_probs = cumProbs[3] + lm.log_boolean(self.models.pos.dist[b,:-1] > self.models.pos.u[sent_index,index]) + self.models.lex.dist[:-1,token]
                                self.dyn_prog[f,j,a,b,:-1,index] = lm.log_vector_add(self.dyn_prog[f,j,a,b,:-1,index], range_probs)
        
            if self.dyn_prog[...,index].max() == -np.inf:
                logging.error("Error: Every value in the forward probability of sentence %d, index %d is inf!" % (sent_index, index))
                raise Exception("Every value in the forward probability of sentence %d, index %d is inf!" % (sent_index, index))
        
        ## For the last token, multiply in the probability
        ## of transitioning to the end state. also can add up
        ## total probability of data given model here.
        sentence_log_prob = -np.inf
        last_index = len(sent)-1
#        for (ind,val) in np.ndenumerate(dyn_prog[...,0]):
#            (f,j,a,b,g) = ind
#            dyn_prog[ind][last_index] += ((self.models.fork.dist[b,g,0] + self.models.reduce.dist[a,1]))
#            sentence_log_prob = lm.log_add(sentence_log_prob, dyn_prog[f,j,a,b,g, last_index])
#            logging.debug(dyn_prog[state,last_index])
                       
#            if last_index > 0 and (a == 0 or b == 0 or g == 0) and dyn_prog[f,j,a,b,g, last_index] != -np.inf:
#                logging.error("Error: Non-zero probability at g=0 in forward pass!")
#                sys.exit(-1)

        if self.dyn_prog[...,last_index].max() == -np.inf:
            logging.error("Error; In sentence %d there is a word with no positive probabilities for its generation" % (sent_index))
            raise Exception("Error; In sentence %d there is a word with no positive probabilities for its generation" % (sent_index))

        t1 = time.time()
        self.ff_time += (t1-t0)
        return sentence_log_prob

    def reverse_sample(self, sent, sent_index):            
        t0 = time.time()
        sample_seq = []
        sample_log_prob = 0
        
        ## Normalize and grab the sample from the forward probs at the end of the sentence
        last_index = len(sent)-1
        
        self.dyn_prog[...,last_index] = lm.normalize_from_log(self.dyn_prog[...,last_index])
        sample_t = sample_from_ndarray(self.dyn_prog[...,last_index])
        sample_state = State.State(1)
        sample_state.f[0], sample_state.j[0], sample_state.a[0], sample_state.b[0], sample_state.g = sample_t
        
        
        sample_seq.append(sample_state)
        if (last_index > 0 and (sample_seq[-1].a == 0 or sample_seq[-1].b == 0)) or sample_seq[-1].g == 0:
            logging.error("Error: First sample has a|b|g = 0")
            raise Exception("Error: First sample has a|b|g = 0")
  
        for t in range(len(sent)-2,-1,-1):
            for (ind,val) in np.ndenumerate(self.dyn_prog[...,t]):
                if val == -np.inf:
                    continue

                (pf,pj,pa,pb,pg) = ind
                (nf,nj,na,nb,ng) = sample_seq[-1].to_list()
                
                trans_prob = 1.0
                if t > 0:
                    trans_prob += self.models.fork[0].dist[pb,pg,nf]
                    
#                if nf == 0:
#                    trans_prob += self.models.reduce.dist[pa,nj]
      
                if nf == 0 and nj == 0:
                    trans_prob += self.models.act[0].dist[pa,0,na]
                    trans_prob += self.models.start[0].dist[pa,na,nb]
                elif nf == 1 and nj == 0:
                    trans_prob += self.models.root[0].dist[0,pg,na]
                    trans_prob += self.models.exp[0].dist[pg,na,nb]
                elif nf == 1 and nj == 1:
                    if na != pa:
                        trans_prob = -np.inf
                    trans_prob += self.models.cont[0].dist[pb,pg,nb]
      
                trans_prob += self.models.pos.dist[nb,ng]
                if np.isnan(trans_prob):
                    logging.error("pos model is nan!")
                    
                self.dyn_prog[ind][t] += trans_prob

            ### TODO step-through this to verify expected behavior (copy not reference)
            normalized_dist = lm.normalize_from_log(self.dyn_prog[...,t])
            sample_t = sample_from_ndarray(normalized_dist)            
            sample_state = State.State(1)
            sample_state.f[0], sample_state.j[0], sample_state.a[0], sample_state.b[0], sample_state.g = sample_t

            if t > 0 and sample_state.g == 0:
                logging.error("Error: Sampled a g=0 state in backwards pass! {0}".format(sample_state.str()))
                
            sample_seq.append(sample_state)
            self.dyn_prog[...,t] = normalized_dist
            
        sample_seq.reverse()
        logging.log(logging.DEBUG-1, "Sample sentence %s", list(map(lambda x: x.str(), sample_seq)))
        t1 = time.time()
        self.bs_time += (t1-t0)
        return sample_seq

def sample_from_ndarray(a):
    dart = np.random.random()
    sum = 0
    for ind,val in np.ndenumerate(a):
        sum += val
        if dart < sum:
            return ind

