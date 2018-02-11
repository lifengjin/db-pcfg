from nltk import tree
import random
from left_corner2normal_tree_converter import full_chain_convert

# strategies for initializing the ints corpus
def left_branching_strategy(line, abp_domain_size,**kwargs):
    new_tree = None
    for word in line:
        new_lex = tree.Tree(str(random.randint(1, abp_domain_size)), [word])
        if new_tree:
            new_tree = tree.Tree(str(random.randint(1, abp_domain_size)), [new_tree, new_lex])
        else:
            new_tree = new_lex
    return new_tree

def right_branching_strategy(line, abp_domain_size,**kwargs):
    new_tree = None
    for word in line[::-1]:
        new_lex = tree.Tree(str(random.randint(1, abp_domain_size)), [word])
        if new_tree:
            new_tree = tree.Tree(str(random.randint(1, abp_domain_size)), [new_lex, new_tree])
        else:
            new_tree = new_lex
    return new_tree

def gold_pos_strategy(line, abp_domain_size, gold_pos_dict=None, **kwargs):
    if gold_pos_dict is None:
        raise Exception("Gold pos dictionary must be provided when using the gold pos strategy!")
    placholder = '-REPLACE-'
    new_tree = tree.Tree(str(random.randint(1, abp_domain_size)), [placholder,placholder])
    num_nodes = len(line) - 2
    for i in range(num_nodes):
        positions = new_tree.treepositions('leaves')
        random_pick = random.choice(positions)
        add_tree =  tree.Tree(str(random.randint(1, abp_domain_size)), [placholder,placholder])
        new_tree[random_pick] = add_tree
    for index, replace in enumerate(new_tree.treepositions('leaves')):
        new_tree[replace] = tree.Tree(str(gold_pos_dict[line[index]]), [line[index]])
    return new_tree

STRATEGY_STRINGS = {'left_branching':left_branching_strategy, 'right_branching':right_branching_strategy,
                    'gold_pos':gold_pos_strategy}
