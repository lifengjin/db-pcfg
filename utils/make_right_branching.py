from nltk.tree import Tree
import argparse
def right_branching_strategy(line):
    new_tree = None
    for word in line[::-1]:
        new_lex = Tree('X', [word])
        if new_tree:
            new_tree = Tree('X', [new_lex, new_tree])
        else:
            new_tree = new_lex
    return new_tree

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--corpus', type=str, required=True, help='the linetoks to make right '
                                                                    'branching trees')
args = parser.parse_args()

with open(args.corpus) as c, open(args.corpus + '.RB', 'w') as RB:
    for line in c:
        line = line.strip().split(' ')
        tree = right_branching_strategy(line)
        string = tree.pformat(margin=100000)
        print(string, file=RB)
