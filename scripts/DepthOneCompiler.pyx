#!/usr/bin/env python3

#import pyximport; pyximport.install()
from Sampler import *
from State import *
from HmmSampler import *

class DepthOneCompiler():
    def compile_models(models):
        logging.info("Compiling component models into mega-HMM transition and observation matrices")
        (a_max, b_max, g_max) = getVariableMaxes(models)
        totalK = get_state_size(models)
    
        t0 = time.time()
        pi = np.zeros((totalK, totalK))
        phi = np.zeros((totalK, models.lex.dist.shape[1]))
        cache = np.zeros((a_max, b_max, g_max, totalK)) - 1
        
        ## Take exponent out of inner loop:
        word_dist = 10**models.lex.dist
    
        (fork,act,root,cont,start,pos) = unlog_models(models)

        ## Build the observation matrix first:
        for state in range(0,totalK):
            (f,j,a,b,g) = extractStates(state, totalK, getVariableMaxes(models))
            phi[state] = word_dist[g,:]

        ## For previous state i:
        for prevState in range(0,totalK):
            (prevF, prevJ, prevA, prevB, prevG) = extractStates(prevState, totalK, getVariableMaxes(models))
            cumProbs = np.zeros(5)
        
            if cache[prevA][prevB][prevG].sum() > 0:
                pi[prevState] = cache[prevA][prevB][prevG]
        
            else:
                ## Sample f & j:
                for f in (0,1):
                    if prevA == 0 and prevB == 0 and f == 0:
                        continue
    
                    cumProbs[0] = (fork[prevB, prevG,f])
    
                    for j in (0,1):
                        ## At depth 1 -- no probability model for j
                        if prevA == 0 and prevB == 0:
                            ## index 1:
                            if j == 0:
                                cumProbs[1] = cumProbs[0]
                            else:
                                cumProbs[1] = 0
                                ## No point in continuing -- matrix is zero'd to start
                                continue
                
                        elif f == j:
                            cumProbs[1] = cumProbs[0]
                        else:
                            cumProbs[1] = 0
                            continue    
        
                        for a in range(1,a_max-1):
                            if f == 0 and j == 0:
                                ## active transition:
                                cumProbs[2] = cumProbs[1] * (act[prevA,0,a])
                            elif f == 1 and j == 0:
                                ## root -- technically depends on prevA and prevG
                                ## but in depth 1 this case only comes up at start
                                ## of sentence and prevA will always be 0
                                cumProbs[2] = cumProbs[1] * (root[0,prevG,a])
                            elif f == 1 and j == 1 and prevA == a:
                                cumProbs[2] = cumProbs[1]
                            else:
                                ## zero probability here
                                continue
        
                            if cumProbs[2] == 0:
                                continue

                            for b in range(1,b_max-1):
                                if j == 1:
                                    cumProbs[3] = cumProbs[2] * (cont[prevB, prevG, b])
                                else:
                                    cumProbs[3] = cumProbs[2] * (start[prevA, a, b])
            
                                # Multiply all the g's in one pass:
                                ## range gets the range of indices in the forward pass
                                ## that are contiguous in the state space
                                state_range = getStateRange(f,j,a,b, getVariableMaxes(models))
                                     
                                range_probs = cumProbs[3] * (pos[b,:-1])
                                pi[prevState, state_range] = range_probs

                cache[prevA][prevB][prevG] = pi[prevState,:]
                            
        time_spent = time.time() - t0
        logging.info("Done in %d s" % time_spent)
    #    return (np.matrix(pi, copy=False), np.matrix(phi, copy=False))
        return (np.matrix(pi,copy=False), np.matrix(phi,copy=False))
    
def unlog_models(models):
    fork = 10**models.fork[0].dist
    act = 10**models.act[0].dist
    root = 10**models.root[0].dist
    cont = 10**models.cont[0].dist
    start = 10**models.start[0].dist
    pos = 10**models.pos.dist
   
    return (fork,act,root,cont,start,pos)
