import argparse
import os, sys, time
import re
import nltk
from collections import Counter, defaultdict, namedtuple
import scipy.stats as stats

# this file is NOT for UHHMM system output.
# it is for UPPARSE and CCL outputs.

parser = argparse.ArgumentParser()
parser.add_argument('-gold-trees', default='.', type=str) # UPPARSE or CCL bracketed trees with no nonterms
parser.add_argument('-test-trees', default='.', type=str)
parser.add_argument('-labeled', default=False, action='store_true')
args = parser.parse_args()
gold_trees_file = args.gold_trees
assert os.path.exists(gold_trees_file), 'no such file exists'

gold_trees = [nltk.Tree.fromstring(x.strip()) for x in open(gold_trees_file)]

phrases = ['NP', 'VP', 'PP']
gold_counters_dev = {'NP':[], 'VP':[], 'PP': []}
# np_count = 0
gold_counters_test = {'NP':[], 'VP':[], 'PP': []}
for index, t in enumerate(gold_trees):
    if index < 4000:
        gold_counters_dev['NP'].append(Counter())
        gold_counters_dev['VP'].append(Counter())
        gold_counters_dev['PP'].append(Counter())
        for sub_t in t.subtrees():
            if sub_t.label() in phrases and len(sub_t.leaves()) > 1:
                gold_counters_dev[sub_t.label()][index][' '.join(sub_t.leaves())] += 1
    else:
        local_index = index - 4000
        gold_counters_test['NP'].append(Counter())
        gold_counters_test['VP'].append(Counter())
        gold_counters_test['PP'].append(Counter())
        for sub_t in t.subtrees():
            if sub_t.label() in phrases and len(sub_t.leaves()) > 1:
                gold_counters_test[sub_t.label()][local_index][' '.join(sub_t.leaves())] += 1

gold_counters = {'NP':gold_counters_dev['NP'] + gold_counters_test['NP'],
                      'VP':gold_counters_dev['VP']+ gold_counters_test['VP'],
                      'PP': gold_counters_dev['PP']+ gold_counters_test['PP']}
scores = []
aggregate_scores = []
NP_length = sum([sum(x.values()) for x in gold_counters['NP']])
# assert NP_length == np_count
VP_length = sum([sum(x.values()) for x in gold_counters['VP']])
PP_length = sum([sum(x.values()) for x in gold_counters['PP']])
phrases_lengths = [NP_length, VP_length, PP_length]

NP_length = sum([sum(x.values()) for x in gold_counters_dev['NP']])
# assert NP_length == np_count
VP_length = sum([sum(x.values()) for x in gold_counters_dev['VP']])
PP_length = sum([sum(x.values()) for x in gold_counters_dev['PP']])
phrases_lengths_dev = [NP_length, VP_length, PP_length]

NP_length = sum([sum(x.values()) for x in gold_counters_test['NP']])
# assert NP_length == np_count
VP_length = sum([sum(x.values()) for x in gold_counters_test['VP']])
PP_length = sum([sum(x.values()) for x in gold_counters_test['PP']])
phrases_lengths_test = [NP_length, VP_length, PP_length]
print(phrases_lengths)
print(phrases_lengths_dev)
print(phrases_lengths_test)
def simple_brackets_converter(bracketed): # use for sents like ((no more) (milk shake))
    strings = []
    bracket_spans = []
    for index, char in enumerate(bracketed):
        if char == '(':
            bracket_spans.append([index, -1])
        elif char == ')':
            for span in bracket_spans[::-1]:
                if span[-1] == -1:
                    span[-1] = index
                    break
    # print(bracket_spans)
    for span in bracket_spans:
        strings.append(bracketed[span[0]:span[1]])
    # print(strings)
    strings = [re.sub('\([A-Z]*\d*', '', span) for span in strings]
    strings = [span.replace(')', '').strip() for span in strings]
    strings = [re.sub('\s+', ' ', span) for span in strings]
    if len(strings) > 1 and strings[0] == strings[1]:
        strings = strings[1:]
    return strings


def calc_phrase_stats(f_name):
    l_branch = 0
    r_branch = 0
    current_counter_nolabel = []
    num_total_trees = 0
    with open(f_name) as fh:
        for line in fh:
            if line.startswith('#'):
                continue
            num_total_trees += 1
            this_t = simple_brackets_converter(line.strip())
            current_counter_nolabel.append(Counter())
            for sub_t in this_t:
                current_counter_nolabel[num_total_trees - 1][sub_t] += 1

    total_acc_number = [Counter(), Counter(), Counter()]
    total_hyp_cat_number = [Counter(), Counter(), Counter()]
    nolabel_recalls = [0, 0, 0]
    for index_tree, tree_counter in enumerate(current_counter_nolabel):
        for index, phrase_cat in enumerate(phrases):
            acc_num = sum((tree_counter & gold_counters[phrase_cat][index_tree]).values())
            print(tree_counter, gold_counters[phrase_cat][index_tree])
            print((tree_counter & gold_counters[phrase_cat][index_tree]))
            nolabel_recalls[index] += acc_num
            # time.sleep(2)

    nolabel_recalls = [ x / phrases_lengths[index] for index, x in enumerate(nolabel_recalls)]

    return nolabel_recalls

def calc_branching_score(t):
    r_branch = 0
    l_branch = 0
    for position in t.treepositions():
        # print(t[position])
        if not (isinstance(t[position],str) or isinstance(t[position][0],str)):
            if len(t[position][0]) == 2:
                l_branch += 1
            if len(t[position][1]) == 2:
                r_branch += 1
    return l_branch, r_branch

def calc_phrase_stats_labeled(f_name, prec_thres=0.6):
    l_branch = 0
    r_branch = 0
    current_counters_dev = [] # dev
    current_counters_test = [] # test
    current_counter_nolabel = []
    num_d2_trees = 0
    num_total_trees = 0
    overall_label_counter = Counter() #used in label dist entropy calc
    non_term_only_label = defaultdict(bool) # used in non-preterm only label calc
    with open(f_name) as fh:
        for line in fh:
            num_total_trees += 1
            if num_total_trees >= 4001:
                counters = current_counters_test
                local_index = num_total_trees - 4001
            else:
                counters = current_counters_dev
                local_index = num_total_trees - 1
            d2_checked = 0
            this_t = nltk.Tree.fromstring(line.strip())
            this_l, this_r = calc_branching_score(this_t)
            l_branch += this_l
            r_branch += this_r
            counters.append({})
            current_counter_nolabel.append(Counter())
            for sub_t in this_t.subtrees():
                # if ' '.join(sub_t.leaves()) == 'want to play with them':
                #     print('1', counters[-1], sub_t)
                overall_label_counter[sub_t.label()] += 1
                if len(sub_t.leaves()) > 1:
                    non_term_only_label[sub_t.label()] &= True
                    if sub_t.label() not in counters[local_index]:
                        counters[local_index][sub_t.label()] = Counter()
                    try:
                        if not d2_checked and isinstance(sub_t[1][0][1], nltk.Tree):
                            d2_checked = 1
                            num_d2_trees += 1
                    except:
                        pass
                    counters[local_index][sub_t.label()][' '.join(sub_t.leaves())] += 1
                    current_counter_nolabel[num_total_trees - 1][' '.join(sub_t.leaves())] += 1
                    # if ' '.join(sub_t.leaves()) == 'want to play with them':
                    #     print('2', counters[-1])
                    #     exit()
                else:
                    non_term_only_label[sub_t.label()] &= False
    total_num_labels = sum(overall_label_counter.values())
    label_dist = [x/total_num_labels for x in overall_label_counter.values()]
    label_dist_entropy = stats.entropy(label_dist)
    num_non_term_only_label = sum(non_term_only_label.values())
    this_best_scores = [0,0,0]
    this_best_aggregate_scores = [0, 0, 0]
    best_cats = [0,0,0]
    total_acc_number = [Counter(), Counter(), Counter()]
    total_hyp_cat_number = [Counter(), Counter(), Counter()]
    total_acc_number_test = [Counter(), Counter(), Counter()]
    total_hyp_cat_number_test = [Counter(), Counter(), Counter()]
    # no label recall
    nolabel_recalls = [0, 0, 0]
    for index_tree, tree_counter in enumerate(current_counter_nolabel):
        for index, phrase_cat in enumerate(phrases):
            acc_num = sum((tree_counter & gold_counters[phrase_cat][index_tree]).values())
            nolabel_recalls[index] += acc_num

    nolabel_recalls = [ x / phrases_lengths[index] for index, x in enumerate(nolabel_recalls)]

    # aggregate recall
    # dev
    print(len(current_counters_test), len(current_counters_dev), len(gold_counters_test['NP']), len(gold_counters_dev['NP']))
    for index_tree, tree_counter in enumerate(current_counters_dev):
        for cat, cat_counter in tree_counter.items():
            for index, phrase_cat in enumerate(phrases):
                acc_num = sum(( cat_counter & gold_counters_dev[phrase_cat][index_tree]).values())
                total_acc_number[index][cat] += acc_num
                total_hyp_cat_number[index][cat] += sum(cat_counter.values())
                # if acc_num != sum(gold_counters_dev[phrase_cat][index_tree].values()):
                #     print(cat, cat_counter)
                #     print(phrase_cat, gold_counters_dev[phrase_cat][index_tree])
    # test
    for index_tree, tree_counter in enumerate(current_counters_test):
        for cat, cat_counter in tree_counter.items():
            for index, phrase_cat in enumerate(phrases):
                acc_num = sum(( cat_counter & gold_counters_test[phrase_cat][index_tree]).values())
                total_acc_number_test[index][cat] += acc_num
                total_hyp_cat_number_test[index][cat] += sum(cat_counter.values())
    #calc all metrics on dev
    Performance = namedtuple('Performance', 'id prec rec f1')
    performances = [[],[],[]] # compute the performances of each cat
    print(total_hyp_cat_number)
    for index, phrase_cat in enumerate(phrases):
        for cat, cat_acc in total_acc_number[index].items():
            prec = cat_acc / total_hyp_cat_number[index][cat]
            rec = cat_acc / phrases_lengths_dev[index]
            f1 = 0 if prec == 0 or rec == 0 else 1.0 / (0.5 / prec + (0.5 / rec))
            performances[index].append(Performance(cat, prec, rec, f1))

    for phrase in performances:
        phrase.sort(key=lambda x: x.f1, reverse=True)
        print(phrase)
    this_best_scores = [phrase[0].f1 for phrase in performances]
    this_best_aggregate_scores = [phrase[0].f1 for phrase in performances]
    this_best_aggregate_scores_index = [[phrase[0].id] for phrase in performances]
    for phrase in performances:
        phrase.sort(key=lambda x: x.prec, reverse=True)
    # calc f1 on dev
    for index, phrase_cat in enumerate(phrases):
        for k in range(len(performances[index]) - 1):
            best_f1 = 0
            best_id = None
            for best_cat_index, best_cat in enumerate(performances[index]):
                if best_cat.id in this_best_aggregate_scores_index[index]:
                    continue
                acc_num = sum([ total_acc_number[index][cat] for cat in this_best_aggregate_scores_index[index] + [best_cat.id]])
                total_num = sum([total_hyp_cat_number[index][cat] for cat in this_best_aggregate_scores_index[index] + [best_cat.id]])
                prec = acc_num / total_num
                rec = acc_num / phrases_lengths_dev[index]
                f1 = 0 if (prec == 0 or rec == 0 ) else 1.0 / (0.5 / prec + (0.5 / rec))
                # if best_cat_index == 0:
                #     assert f1 == this_best_aggregate_scores[index], "{}, {}, {}， {}".format(prec, rec, f1, this_best_aggregate_scores[index])
                if f1 > best_f1:
                    best_f1 = f1
                    best_id = best_cat.id
            if best_f1 > this_best_aggregate_scores[index]:
                this_best_aggregate_scores[index] = best_f1
                this_best_aggregate_scores_index[index].append(best_id)
            else: break

    aggregate_scores_test = [0,0,0]
    for index, phrase_cat in enumerate(phrases):

        acc_num = sum([ total_acc_number_test[index][cat] for cat in this_best_aggregate_scores_index[index]] )
        total_num = sum([total_hyp_cat_number_test[index][cat] for cat in this_best_aggregate_scores_index[index]])
        prec = acc_num / total_num
        rec = acc_num / phrases_lengths_test[index]
        f1 = 0 if (prec == 0 or rec == 0 ) else 1.0 / (0.5 / prec + (0.5 / rec))
        # if best_cat_index == 0:
        #     assert f1 == this_best_aggregate_scores[index], "{}, {}, {}， {}".format(prec, rec, f1, this_best_aggregate_scores[index])
        aggregate_scores_test[index] = f1


    return this_best_scores, aggregate_scores_test, r_branch / (l_branch+r_branch), num_d2_trees / num_total_trees, label_dist_entropy, num_non_term_only_label / len(non_term_only_label), nolabel_recalls


if not args.labeled:
    recalls = calc_phrase_stats(args.test_trees)
    print(recalls)
else:
    recalls = calc_phrase_stats_labeled(args.test_trees)
    print(recalls)
