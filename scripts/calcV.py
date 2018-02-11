#!/usr/bin/env python3.4

## To compute V-measure (as in vangael et al.)
## V = [(1 + Beta) * h * c ] / [(Beta*h) + c]
## where:
## h = 1 - H(C|K) / H(C)
## c = 1 - H(K/C) / H(K)
## H(X|Y) is conditional entropy:
## sum (x,y) p(x,y) * log [ p(x) / p(x,y) ]
## and H(X) is entropy:
## sum (x) p(x) * log p(x)
## so we need to read in taggings from two different systems 
## (usually automatically tagged vs. gold and calculate three
## probability distributions;
## p(x,y), p(x), and p(y)
## and everything else is derived from those.
## Beta is a parameter which balances the extent to which the 
## measure penalizes or rewards large clusters.
## Beta = 1 is V measure
## general term is V-beta.

## Expected input format is one sentence per line, with
## a sentence represented as space-separated POS tags.
## They should be mapped to integer indices, e.g.:
## 15 2 5 2 3
## is a five-word sentence.
## The system and gold input must be aligned or the output will
## be garbage.

import numpy as np
import uhhmm_io as io
import itertools
import sys
import pdb

def main(args):
    if sys.version_info[0] != 3:
        print("This script requires Python 3")
        exit()

    if len(args) < 2:
        sys.stderr.write("Two required arguments: <Gold pos file (same as system input)> <System pos file (sample format)>\n")
        exit()

    gold_pos = io.read_input_file(args[0])[1]
    sys_pos = io.read_sample_file(args[1])
    
    vm = get_v(gold_pos, sys_pos)
    
    print("Value of V measure is %f\n" % vm)
    
def get_v(gold_pos, sys_pos, beta=1):

    p_gold = get_distribution(gold_pos)
    p_sys = get_distribution(sys_pos)
    p_joint = get_joint_distribution(gold_pos, sys_pos)

    H_gold = entropy(p_gold)
    H_sys = entropy(p_sys)
    H_joint = entropy(p_joint)
    
    ## This is a bit tricky: We are dividing a 1d array by a 2d array (matrix)
    ## and x and y have to divide in different directions because the matrix is
    ## x/y in each dimension. Further, you can't take the transpose of a 1d array
    ## and have it change the outcome. So i had to tranpose the matrix, then do the
    ## divide, then transpose the outcome.
    
#    I_xy = get_mutual_information(p_x, p_y, p_xy)

    H_gold_given_sys = H_joint - H_sys
    H_sys_given_gold = H_joint - H_gold
    
    h = 1 - (H_gold_given_sys / H_gold)
    c = 1 - (H_sys_given_gold / H_sys)
    
    
    V = ((1 + beta) * h * c ) / ((beta*h) + c)
    
    return V

def get_vi(gold_pos, sys_pos):
    p_x = get_distribution(gold_pos)
    p_y = get_distribution(sys_pos)
    p_xy = get_joint_distribution(gold_pos, sys_pos)

    H_x = entropy(p_x)
    H_y = entropy(p_y)
    
    I_xy = get_mutual_information(p_x, p_y, p_xy)    
    
    H_x_given_y = H_y - I_xy
    H_y_given_x = H_x - I_xy
    
    vi = H_x_given_y + H_y_given_x
    return vi

## Takes in a list of lists of ints representing, e.g., a set of sentences with
## POS tags. Computes a distribution over tags by flattening the list, counting each
## tag and dividing by the total. Returns a list with len(out) = max(in)+1 so that
## the array can be indexed by the highest value in the tagset.
def get_distribution(seq_list):
    flat_seq = list(itertools.chain.from_iterable(seq_list))
    max_val = max(flat_seq)
    counts = np.bincount(flat_seq)
    
    return counts / counts.sum()

## Takes in 2 lists of lists, say for a gold and system tagging.
## Each list of list represents the sentences in a corpora and the tags for each word.
## Returns a distribution over tag pairs representing the joint distribution of each
## pair in the gold-system output. 
def get_joint_distribution(seqs1, seqs2):
    seq1_flat = list(itertools.chain.from_iterable(seqs1))
    seq2_flat = list(itertools.chain.from_iterable(seqs2))
    assert len(seq1_flat) == len(seq2_flat)

    ## Add 1 to size variable because we assume 0-indexing of input.
    counts = np.zeros((max(seq1_flat)+1, max(seq2_flat)+1))
    
    for i,val in enumerate(seq1_flat):
        counts[seq1_flat[i]][seq2_flat[i]] += 1

    return counts / counts.sum()

def get_mutual_information(dist_x, dist_y, dist_xy):
    sum = 0
    for ind,val in np.ndenumerate(dist_xy):
        if(dist_xy[ind] != 0.0):
            sum += (dist_xy[ind] * np.log2(dist_xy[ind] / (dist_x[ind[0]] * dist_y[ind[1]])))
    
    return sum
    
def entropy(dist):
    sum = 0
    for ind,val in np.ndenumerate(dist):
        if val != 0.0:
            sum += val * np.log2(val)
            
    return -sum
if __name__ == "__main__":
    main(sys.argv[1:])
