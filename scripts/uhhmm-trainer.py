#!/usr/bin/env python3.4

import sys

if sys.version_info[0] != 3:
    print("This script requires Python 3")
    exit()

import calcV
import uhhmm_io as io
import configparser
import uhhmm
import numpy as np
import pdb
import os
import pickle
import logging
import signal
from random import randint


def main(argv):
    if len(argv) < 1:
        sys.stderr.write("One required argument: <Config file|Resume directory>\n")
        sys.exit(-1)

    path = argv[0]
    if not os.path.exists(path):
        sys.stderr.write("Input file/dir does not exist!")
        sys.exit(-1)

    config = configparser.ConfigParser()
    pickle_file = None
    input_seqs_file = None

    if os.path.isdir(path):
        ## Resume mode
        config.read(path + "/config.ini")
        pickle_file = path + "/sample.obj"
        out_dir = config.get('io', 'output_dir')
    else:
        config.read(argv[0])
        input_seqs_file = config.get('io', 'init_seqs', fallback=None)
        if not input_seqs_file is None:
            del config['io']['init_seqs']
        out_dir = config.get('io', 'output_dir')
        if not os.path.exists(out_dir):
            sys.stderr.write("Creating non-existent output directory.\n")
            os.makedirs(out_dir)

        with open(out_dir + "/config.ini", 'w') as configfile:
            config.write(configfile)

    ## Write git hash of current branch to out directory
    os.system('git rev-parse HEAD > %s/git-rev.txt' % (out_dir))

    input_file = config.get('io', 'input_file')
    working_dir = config.get('io', 'working_dir', fallback=out_dir)
    dict_file = config.get('io', 'dict_file')

    ## Read in input file to get sequence for X
    (pos_seq, word_seq) = io.read_input_file(input_file)
    num_types = max(map(max, word_seq)) + 1

    params = read_params(config)

    ## Store tag sequences of gold tagged sentences
    gold_seq = dict()
    if 'num_gold_sents' in params and params['num_gold_sents'] == 'all':
        for i in range(0, len(pos_seq)):
            gold_seq[i] = pos_seq[i]
    else:
        while len(gold_seq) < int(params.get('num_gold_sents', 0)) and len(gold_seq) < len(word_seq):
            rand = randint(0, len(word_seq) - 1)
            if rand not in gold_seq.keys():
                gold_seq[rand] = pos_seq[rand]

    word_vecs = None
    if 'word_vecs_file' in params:
        sys.stderr.write("This functionality is at alpha stage and disabled in master.\n")
        sys.exit(-1)
        word_vecs = io.read_word_vector_file(params.get('word_vecs_file'), io.read_dict_file(dict_file))

    (samples, stats) = uhhmm.sample_beam(word_seq, params, lambda x: io.write_output(x, None, config, pos_seq),
                                         lambda x: io.checkpoint(x,config), working_dir, pickle_file, gold_seq,
                                         input_seqs_file=input_seqs_file, word_vecs=word_vecs,  word_dict_file = dict_file)

    if len(samples) > 0:
        io.write_output(samples[-1], stats, config, pos_seq)


def read_params(config):
    params = {}
    for (key, val) in config.items('io'):
        params[key] = val
    for (key, val) in config.items('params'):
        params[key] = val

    return params


if __name__ == "__main__":
    main(sys.argv[1:])
