import argparse
import operator
import os

# this script is used for converting one model from the DBPCFG system to a format that synproc uses
# example command: python utils/extract_model_iter.py --iter 1000 --output-path outputs/abcabab_anneal6
# --iter provides the iteration index for the model
# --output-path provides the path where the model files live (pcfg_term and pcfg_nonterm)
# the output model file will be in the same folder as the output-path, and will be named
# pcfg_model_iterNNNN.txt

argparser = argparse.ArgumentParser()

argparser.add_argument('--iter', required=True, type=int, help='the iter number for the model')
argparser.add_argument('--output-path', required=True, type=str, help='the path of the pcfg model files')

args = argparser.parse_args()
output_file_path = os.path.join(args.output_path, 'pcfg_model_iter' + str(args.iter) + '.txt')
iter_number = str(args.iter)
terms_param_file = os.path.join(args.output_path, 'pcfg_terms.txt')
nonterms_param_file = os.path.join(args.output_path, 'pcfg_nonterms.txt')
# read in the params and dict
with open(terms_param_file) as terms_fh:
    terms_header = terms_fh.readline().strip().split('\t')[1:]
    for line in terms_fh:
        if line.startswith(str(iter_number)):
            terms_param = line.strip().split()[1:]
            break
    else:
        raise Exception('no iter number found in param files.')
with open(nonterms_param_file) as nonterms_fh:
    nonterms_header = nonterms_fh.readline().strip().split('\t')[1:]
    for line in nonterms_fh:
        if line.startswith(str(iter_number)):
            nonterms_param = line.strip().split()[1:]
            break
    else:
        raise Exception('no iter number found in param files.')

# construct a pcfg
pcfg = []
terms_probs = zip(terms_header, terms_param)
nonterms_probs = zip(nonterms_header, nonterms_param)
for rule_prob_pair in nonterms_probs:
    rule, prob = rule_prob_pair
    left, right = rule.split('->')
    right1, right2 = right.replace('(', '').replace(')', '').split(', ')
    full_rule = (left, right1, right2, float(prob))
    pcfg.append(full_rule)

for lex_prob_pair in terms_probs:
    lex, prob = lex_prob_pair
    pos, lex = lex.split('->')
    full_rule = (pos, lex, '-', float(prob))
    pcfg.append(full_rule)

pcfg.sort(key=operator.itemgetter(0, 3), reverse=True)

with open(output_file_path, 'w') as out:
    for rule in pcfg:
        print('G {} : {} {} = {}'.format(*rule), file=out)
