import nltk
import argparse
# this file is for fixing the different terminal symbols caused by tokenization.
# it takes a gold linetree and a predicted linetree file
# and replace the tokens in the predicted linetree file with gold if they are different.

parser = argparse.ArgumentParser()
parser.add_argument('--gold', '-g', required=True, help='the gold linetrees file')
parser.add_argument('--predicted', '-p', required=True, help='the predicted linetrees file')
args = parser.parse_args()

gold_trees = []
with open(args.gold) as g:
    for line in g:
        line = line.strip()
        gold_trees.append(nltk.tree.Tree.fromstring(line))

predicted_trees = []
with open(args.predicted) as p:
    for line in p:
        line = line.strip()
        predicted_trees.append(nltk.tree.Tree.fromstring(line))

for i in range(len(gold_trees)):
    this_gold_tree = gold_trees[i]
    this_predicted_tree = predicted_trees[i]
    gold_tokens = this_gold_tree.leaves()
    predicted_tokens = this_predicted_tree.leaves()
    assert len(gold_tokens) == len(predicted_tokens), "wrong length of tokens at {}".format(i)
    for j in range(len(gold_tokens)):
        if gold_tokens[j] != predicted_tokens[j]:
            this_predicted_tree[this_predicted_tree.leaf_treeposition(j)] = gold_tokens[j]

fn = args.predicted.split('.')

fn = fn[:-1]
fn.append('fixterms.linetrees')
fn = '.'.join(fn)
with open(fn, 'w') as ft:
    for tree in predicted_trees:
        string = tree.pformat(margin=100000)
        print(string, file=ft)