#!/usr/bin/env python3

import numpy as np
import sys

def log_vector_add(x, y):
    p = np.zeros(x.shape)
    for ind,val in enumerate(x):
        p[ind] = log_add(x[ind],y[ind])
    
    return p
    
## From here: https://facwiki.cs.byu.edu/nlp/index.php/Log_Domain_Computations
def log_add(x, y):
    ## Step 0: An addition -- if one of the values is 0 that means it has not been initialized yet
    if x == 0:
        return y
    
    if y == 0:
        return x
    
    if x == -np.inf:
        return y
        
    if y == -np.inf:
        return x
    
    ## Step 1: make x bigger
    if y > x:
        temp = y
        y = x
        x = temp
    
    ## Step 2:
    if x == sys.float_info[3]:   ## min value
        return x
        
    ## Step 3: How far down is y from x
    ## If it's small, ignore:
    neg_diff = y - x
    if neg_diff < -20:
        return x
    
    ## Step 4: Otherwise use algebra:
    return x + np.log10(1.0 + 10**neg_diff)

def normalize_from_log(log_dist):
    ## scale back up to a safe range
    log_dist -= log_dist.max()
    
    ## Back into prob space
    dist = 10**log_dist
    
    dist /= dist.sum()
    
    return dist

def log_boolean(v):
    if v.size == 1:
        return 0 if v else -np.inf
        
    valid_inds = np.where(v)
    effective_probs = np.zeros(v.shape) + -np.inf
    effective_probs[valid_inds] = 0
    return effective_probs
