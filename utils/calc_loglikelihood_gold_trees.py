import sys
sys.path.append(sys.path[0] + "/../scripts/")
from pcfg_translator import load_gold_trees, calc_pcfg_loglikelihood
import argparse

# this scripts takes a gold linetrees file and calculate the loglikelihood of the TREES (not data)
# to calculate the loglikelihood of the data, one needs to at least do a forward pass
# which is not implemented here yet

parser = argparse.ArgumentParser()

parser.add_argument('--gold-trees-path', '-g', required=True, type=str, help='the path to the gold trees file')
parser.add_argument('--abp-domain-size', '-n', required=True, type=int, help='the size of ABP domain')
args = parser.parse_args()

grammar, counts = load_gold_trees(args.gold_trees_path, args.abp_domain_size)

print('loglikelihood of the TREES is', calc_pcfg_loglikelihood(grammar, counts))
