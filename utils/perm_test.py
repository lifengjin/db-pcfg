import random

import numpy as np
import argparse
import os
from nltk import tree
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument('-c1', '--candidate1', type=str, required=True, help='candidate 1 linetrees')
parser.add_argument('-c2', '--candidate2', type=str, required=True, help='candidate 2 linetrees')
parser.add_argument('-g', '--gold', type=str, required=True, help='gold linetrees')
parser.add_argument('-n', type=int, required=True, help='number of permutations')
parser.add_argument('-v', '--verbose', default=False, action='store_true', help='verbose output')
args = parser.parse_args()

tmp_dir = 'utils/'
tmp_cand1_fn = os.path.join(tmp_dir, 'cand1.txt')
tmp_cand2_fn = os.path.join(tmp_dir,'cand2.txt')
result_fn = os.path.join(tmp_dir, 'result.txt')

def get_f_diff(c1_fn, c2_fn, g_fn, res_fn):
    command = './utils/EVALB/evalb -p utils/EVALB/uhhmm.prm {} {} > {}'
    subprocess.call(command.format(g_fn, c1_fn, res_fn), shell=True)
    with open(res_fn) as r:
        for line in r:
            if 'FMeasure' in line:
                line = line.strip().split('=')
                c1_F = float(line[1].strip())
                break
        else:
            raise Exception('No FMeasure found!')
    subprocess.call(command.format(g_fn, c2_fn, res_fn), shell=True)
    with open(res_fn) as r:
        for line in r:
            if 'FMeasure' in line:
                line = line.strip().split('=')
                c2_F = float(line[1].strip())
                break
        else:
            raise Exception('No FMeasure found!')
    if args.verbose:
        print('the current pair gives Fs {} {}, and diff {}'.format(c1_F, c2_F, abs(c2_F - c1_F)))
    return abs(c2_F - c1_F)

c1_trees = []
c2_trees = []
with open(args.candidate1) as c1:
    for line in c1:
        t = tree.Tree.fromstring(line.strip())
        c1_trees.append(t)
with open(args.candidate2) as c2:
    for line in c2:
        t = tree.Tree.fromstring(line.strip())
        c2_trees.append(t)

assert len(c1_trees) == len(c2_trees)
bad_trees = 0
for index, (t1, t2) in enumerate(zip(c1_trees, c2_trees)):
    if len(t1.leaves()) != len(t2.leaves()):
        bad_trees += 1
    if len(t1.leaves()) > len(t2.leaves()):
        c2_trees[index] = t1
    elif len(t1.leaves()) < len(t2.leaves()):
        c1_trees[index] = t2
print('there are {} bad trees that don\'t match'.format(bad_trees))
print('fixed by using the longest trees')

diffs = []

with open(tmp_cand1_fn, 'w') as c1_fn, open(tmp_cand2_fn, 'w') as c2_fn:
    for t in c1_trees:
        string = t.pformat(margin=100000)
        print(string, file=c1_fn)
    for t in c2_trees:
        string = t.pformat(margin=100000)
        print(string, file=c2_fn)
real_diff = get_f_diff(tmp_cand1_fn, tmp_cand2_fn, args.gold, result_fn)

p_numerator = 1
for i in range(args.n):
    new_c1_trees = []
    new_c2_trees = []
    # if args.verbose:
    #     print('doing {}-th permutation now'.format(i+1))
    for tree1, tree2 in zip(c1_trees, c2_trees):
        coin = random.random()
        if coin > 0.5:
            new_c1_trees.append(tree1)
            new_c2_trees.append(tree2)
        else:
            new_c1_trees.append(tree2)
            new_c2_trees.append(tree1)
    with open(tmp_cand1_fn, 'w') as c1_fn, open(tmp_cand2_fn, 'w') as c2_fn:
        for t in new_c1_trees:
            string = t.pformat(margin=100000)
            print(string, file=c1_fn)
        c1_fn.flush()
        for t in new_c2_trees:
            string = t.pformat(margin=100000)
            print(string, file=c2_fn)
        c2_fn.flush()

    this_diff = get_f_diff(tmp_cand1_fn, tmp_cand2_fn, args.gold, result_fn)
    if this_diff > real_diff:
        p_numerator += 1
    if args.verbose:
        print('{}-th iter the running p value is {}'.format(i, p_numerator / (i + 1)))

# # diffs.sort()
# for i in range(args.n):
#     cur_comp = diffs[-i]
#     if cur_comp < real_diff:
#         print('the p value for this pair is {}'.format(i / args.n ))
#         break
