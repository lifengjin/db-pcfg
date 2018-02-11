# this file is for add nonterminals 'X' to output trees from systems that do
# not have any labels, for example CCL and UPPARSE

import sys
import re
from nltk.tree import Tree
fn = sys.argv[1]

with open(fn) as i, open(fn+'.proper', 'w') as o:
    for line in i:

        line = line.replace('(', '(X ')
        line = line.replace(')', ' )')

        line = re.sub('(?<=\s)([^()\s]+)', '(X \g<1>)', line)
        print(line)

        # words = re.findall('[^()\s]+', line)
        # # words = set(words)
        # line = line.replace('(', '(X ')
        # replaced = []
        # for word in words:
        #     print(line, word)
        #     if '\\/' in word and word not in replaced:
        #         line = line.replace(word, '(X '+word+')')
        #         replaced.append(word)
        #         continue
        #
        #     line = re.sub('(?<!(X )|[\w\s]{2})'+str(word), '(X '+word+')', line, count=1)

        this_tree = Tree.fromstring(line)
        this_tree.collapse_unary(collapsePOS=True)
        string = this_tree.pformat(margin=100000)
        print(string, file=o)

