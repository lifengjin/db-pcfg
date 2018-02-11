#!/usr/bin/env python3
import re

import os

import sys
from nltk import tree
import string
import argparse
"""
slicing penn treebank to extract sentences of less than N words and
output the gold standard trees.
the treebank file must a single file with all the empty nodes and traces removed from the trees
this outputs trees, and one can use make item to convert into tagwords or words

this is needed because the punctuations are used in inference but not in counting word number
"""
parser = argparse.ArgumentParser()
parser.add_argument('-n', type=int, help='the number of words at most in a sent')
parser.add_argument('-c', default='', type=str, help='the path to the full treebank file')
parser.add_argument('-ref', default='', type=str, help='the path to the referred file for '
                                                       'filtering')
parser.add_argument('-positive-ref', action='store_true', help='use the ref for positive '
                                                               'reference or negative reference')
parser.add_argument('-word-only', default=False, action='store_true', help='use only words to '
                                                                           'determine if two '
                                                                           'sentences are the same')
parser.add_argument('-ordered', default=False, action='store_true', help='if the filtering is '
                                                                         'ordered')
args = parser.parse_args()
if args.n:
    sent_length = args.n  # the maximum number of words in a sentence
else:
    sent_length = 'X'

if not args.c:
    ptb_tree_file = '/home/jin.544/project_space/jin/corpora/ptb_trees/linetrees/wsj_full.linetrees'
else:
    ptb_tree_file = args.c

ptb_puncs = [',', '.', ':', 'LRB', 'RRB', '$', '#', '``', "''"]

def bracket_check(word):
    return re.match('[LR][A-Z]B', word)

if args.ref:
    with open(args.ref) as ref:
        referred_sents = ref.readlines()
        if args.word_only:
            referred_sents = [' '.join(tree.Tree.fromstring(x.strip()).leaves()) for x in
                              referred_sents]
    if not args.c:
        output_fn = 'generated/ptb_{}.ref.linetrees'.format(sent_length)
    else:
        _, fn = os.path.split(args.c)
        output_fn = 'generated/'+fn+'.{}.ref.linetrees'.format(sent_length)
else:
    referred_sents = []
    if not args.c:
        output_fn = 'generated/ptb_{}.linetrees'.format(sent_length)
    else:
        _, fn = os.path.split(args.c)
        output_fn = 'generated/'+fn+'.{}.linetrees'.format(sent_length)

with open(output_fn, 'w') as fi, open(ptb_tree_file) as ptb:
    for index, line in enumerate(ptb):
        this_line = line.strip()
        this_tree = tree.Tree.fromstring(this_line)
        terminals = this_tree.pos()
        length = len(terminals)
        for word, pos in terminals:
            if pos in ptb_puncs:
                length -= 1
        if  sent_length == 'X' or (length <= sent_length and length > 0):
            if args.word_only:
                line = ' '.join(this_tree.leaves())
            if not args.ordered:
                if not referred_sents or (line not in referred_sents and not args.positive_ref) or (
                    line in referred_sents and args.positive_ref):
                    print(this_line, file=fi)
            else:
                if not referred_sents or (line not in referred_sents and not args.positive_ref):
                    print(this_line, file=fi)
                elif line == referred_sents[0] and args.positive_ref:
                    print(this_line, file=fi)
                    referred_sents.pop(0)

if args.ordered and referred_sents:
    print('THERE ARE {} GOLD SENTS LEFT.'.format(len(referred_sents)))
    os.rmdir(output_fn)