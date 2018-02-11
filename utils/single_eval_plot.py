import os
import sys
import subprocess
from multiprocessing import Pool
import re
import numpy as np
import operator
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas
import time
import nltk
from collections import Counter, defaultdict, namedtuple
import scipy.stats as stats
import argparse
# EVALB folder needs to be placed into this folder

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('-folder', default='.', type=str, help='the path to the output folder')
arg_parser.add_argument('-gold-trees', default=None, type=str, help='the path to the gold trees')
arg_parser.add_argument('-first-n-iter', default=None, type=int, help='the first N of iterations considered')
arg_parser.add_argument('-burn-in', default=50, type=int, help='burn in period')
arg_parser.add_argument('-max-threads', default=20, type=int)
arg_parser.add_argument('-with-deps', action='store_true', default=False, help='no from deps trees calc')
args = arg_parser.parse_args()
last_sample_folder = args.folder
gold_trees_file = args.gold_trees
assert gold_trees_file is not None, "gold trees file is not defined!"
first_n_iter = args.first_n_iter
evalb_params_file = './utils/EVALB/uhhmm.prm'
total_process_limit = args.max_threads
process_pool = []
burn_in = args.burn_in
all_results = []

def bash_command(cmd):
    global process_pool
    # if len(process_pool) > 0:
    #     print(sum([(not bool(x.poll())) for x in process_pool]), process_pool[0].poll())
    while True:
        process_pool = [x for x in process_pool if x.poll() is None]
        if len(process_pool) >= total_process_limit:
            time.sleep(0.1)
            continue
        else:
            p = subprocess.Popen(['/bin/bash', '-c', cmd])
            process_pool.append(p)
            break
    return p

def check_finished():
    global process_pool
    while True:
        process_pool = [x for x in process_pool if x.poll() is None]
        if len(process_pool) == 0:
            return True

gold_trees = [nltk.Tree.fromstring(x.strip()) for x in open(gold_trees_file)]

modelblocks_path = open('user-modelblocks-location.txt').readline().strip()

build_rules_command = '''cat {}  |  perl {}/resource-linetrees/scripts/trees2rules.pl  >  {}'''
build_model_command = '''cat {0} | sort | uniq -c | sort -nr | awk '{{"wc -l {1} | cut -d \\" \\" -f1" | getline t; u = $1; $1 = u/t; print;}}' | awk '{{p = $1; for (i=1;i<NF;i++) $i=$(i+1);$NF="="; $(NF + 1)=p; tmp=$2;$2=$3;$3=tmp;$1="R";print;}}' > {2}'''
build_head_model_command =  '''cat {} | python3 {}/resource-linetrees/scripts/rules2headmodel.py > {}'''
convert_to_deps_command = '''cat {} | python3 {}/resource-linetrees/scripts/trees2deps.py {} | sed 's/<num>/-NUM-/g' | python {}/resource-linetrees/scripts/deps2trees.py -f stanford > {}'''

preprocess_command = '''cat {} | sed 's/(-DFL- \+E_S) *//g;s/  \+/ /g;s/\\t/ /g;s/\([^ ]\)(/\\1 (/g;s/_//g;s/-UNDERSCORE-//g;s/([^ ()]\+ \+\*[^ ()]*)//g;s/( *-NONE-[^ ()]\+ *[^ ()]* *)//g;s/([^ ()]\+ )//g;s/ )/)/g;s/( /(/g;s/  \+/ /g;' | awk '!/^\s*\(CODE/' | python {}/resource-linetrees/scripts/make-trees-lower.py | perl {}/resource-linetrees/scripts/killUnaries.pl | perl {}/resource-linetrees/scripts/make-trees-nounary.pl |  perl -pe "s/ \([^ ()]+ (,|\.|\`\`|\`|--|-RRB-|-LRB-|-LCB-|-RCB-|''|'|\.\.\.|\?|\!|\:|\;)\)//g" | perl {}/resource-linetrees/scripts/make-trees-nounary.pl > {}'''
# eval_command = '''python {}/resource-linetrees/scripts/constit_eval.py {} <(cat {} | python {}/resource-linetrees/scripts/filter_reannot_fails.py {}) > {}.nt-lower-nounary-nopunc.constiteval.txt'''
eval_command = './utils/EVALB/evalb -p {} {} {} > {}' # param, gold, test
constlist_command = '''python scripts/iters2constitevallist.py {} > {}'''
plot_command = '''python {}/resource-linetrees/scripts/constitevals2table.py {} > {} '''

output_files = os.listdir(last_sample_folder)
output_last_samples = [x for x in output_files if re.match('last_sample[0-9]+\.linetrees', x) and
                       (int(re.match('last_sample([0-9]+)\.linetrees', x).group(1)) >= burn_in) and (int(re.match('last_sample([0-9]+)\.linetrees', x).group(1)) < first_n_iter if first_n_iter is not None
                        else True)]

output_last_samples.sort(key=lambda x: int(re.match('last_sample([0-9]+)\.linetrees', x).group(1)))
# print(output_last_samples)


start_iter = burn_in
end_iter = burn_in + len(output_last_samples)
print("start iter: {}; end iter: {}".format(start_iter, end_iter))

rule_f_names = []
model_f_names = []
head_model_names = []
deps_f_names = []

if args.with_deps:

    for f in output_last_samples:
        f_name = os.path.join(last_sample_folder, f)
        rule_f_name = f_name.replace('linetrees', 'rules')
        p = bash_command(build_rules_command.format(f_name, modelblocks_path, rule_f_name))
        rule_f_names.append(rule_f_name)
    else:
        check_finished()

    for index, rule_f_name in enumerate(rule_f_names):
        model_f_name = rule_f_name.replace('rules', 'model')
        p = bash_command(build_model_command.format(rule_f_name, rule_f_name, model_f_name))
        model_f_names.append(model_f_name)
    else:
        check_finished()

    for index, model_f_name in enumerate(model_f_names):
        head_model_name = model_f_name.replace('model', 'head.model')
        p = bash_command(build_head_model_command.format(model_f_name, modelblocks_path, head_model_name))
        head_model_names.append(head_model_name)
    else:
        check_finished()


    for index, f_name in enumerate(output_last_samples):
        f_name = os.path.join(last_sample_folder, f_name)
        deps_f_name = f_name.replace('linetrees', 'fromdeps.linetrees')
        p = bash_command(convert_to_deps_command.format(f_name, modelblocks_path, head_model_names[index], modelblocks_path, deps_f_name))
        deps_f_names.append(deps_f_name)
    else:
        check_finished()

end_f_names = []

print('preprocessing the trees into nounary lower and nopunc')
for f in output_last_samples + deps_f_names:
    if 'fromdeps' not in f:
        f_name = os.path.join(last_sample_folder, f)
    else:
        f_name = f
    end_f_name = '{}.nt.lower.nounary.nopunc.linetrees'.format(f_name.replace('.linetrees', ''))
    end_f_names.append(end_f_name)
    # print(preprocess_command.format(f_name, modelblocks_path, modelblocks_path, modelblocks_path, modelblocks_path, end_f_name))
    p = bash_command(preprocess_command.format(f_name, modelblocks_path, modelblocks_path, modelblocks_path, modelblocks_path, end_f_name))
else:
    check_finished()

evalb_f_names = []
print('running EVALB on the line trees')
for end_f_name in end_f_names:
    evalb_name = end_f_name.replace('linetrees', 'evalb')
    # print(eval_command.format(evalb_params_file, gold_trees_file, end_f_name, evalb_name))
    p2 = bash_command(eval_command.format(evalb_params_file, gold_trees_file, end_f_name, evalb_name))
    evalb_f_names.append(evalb_name)
else:
    check_finished()

indices = []
prec = []
rec = []
f1 = []
deps_indices = []
deps_prec = []
deps_rec =[]
deps_f1 = []
print('plotting the p, r and f')
for evalb_name in evalb_f_names:
    index = re.search('(?<=last_sample)[0-9]+', evalb_name).group(0)
    if 'fromdeps' in evalb_name:
        deps_indices.append(int(index))
    else:
        indices.append(int(index))
    with open(evalb_name) as e:
        lines = e.readlines()
        for line in lines:
            if line.startswith('Bracketing Precision'):
                this_prec = re.search('[0-9\.]+', line).group(0)
                if 'fromdeps' in evalb_name:
                    deps_prec.append(float(this_prec))
                else:
                    prec.append(float(this_prec))
            if line.startswith('Bracketing Recall'):
                this_rec = re.search('[0-9\.]+', line).group(0)
                if 'fromdeps' in evalb_name:
                    deps_rec.append(float(this_rec))
                else:
                    rec.append(float(this_rec))
            if line.startswith('Bracketing FMeasure'):
                this_fm = re.search('[0-9\.]+', line).group(0)
                if 'fromdeps' in evalb_name:
                    deps_f1.append(float(this_fm))
                else:
                    f1.append(float(this_fm))
                break

titles = ['Precision', 'Recall', 'F1', 'Precision-deps', 'Recall-deps', 'F1-deps']
data = list(zip(indices, prec, rec, f1))
data.sort(key=operator.itemgetter(0))
data = np.array(data)
print(data.shape)
if args.with_deps:
    data_deps = list(zip(deps_indices, deps_prec, deps_rec, deps_f1))
    data_deps.sort(key=operator.itemgetter(0))
    data_deps = np.array(data_deps)
else:
    data_deps = np.zeros(data.shape)

data = np.hstack((data, data_deps[:, 1:]))
fig, ax = plt.subplots()
all_results.append(['iters']+titles)
all_results.append([data])
if args.with_deps:
    lines = ax.plot(data[:, 0], data[:,1], data[:,0], data[:,2], data[:,0], data[:,3], data[:,0], data[:,4], data[:,0], data[:,5], data[:,0], data[:,6])
else:
    lines = ax.plot(data[:, 0], data[:,1], data[:,0], data[:,2], data[:,0], data[:,3])
ax.set_ylabel('percentage')

if args.with_deps:
    ax.legend(lines, ('Precision', "Recall", "F1", 'Precision-deps', 'Recall-deps', 'F1-deps'))
else:
    ax.legend(lines, ('Precision', "Recall", "F1"))

pp = PdfPages(os.path.join(last_sample_folder, 'evalb' + '_'+ str(min(indices))+'_' + str(max(indices))) + '.pdf')
fig.savefig(pp, format='pdf')
pp.close()
plt.cla()
plt.clf()


# plot the log probs
hyperparams = pandas.read_table(os.path.join(last_sample_folder, 'pcfg_hypparams.txt'))

fig, ax = plt.subplots()
all_results[0].append('logprob')
# print(hyperparams.iter)
x_data = hyperparams.iter[ (hyperparams.iter >= start_iter) & (hyperparams.iter < end_iter)]

assert len(x_data) == len(output_last_samples), str(len(x_data)) + ' ' + str(len(output_last_samples))
all_results[1].append(hyperparams.logprob[(hyperparams.iter >= start_iter) & (hyperparams.iter < end_iter)].get_values())
line_1 = ax.plot(x_data, hyperparams.logprob[(hyperparams.iter >= start_iter) & (hyperparams.iter < end_iter)], 'b-')
ax.set_ylabel('logprob', color='b')
ax.tick_params('y', color='b')
if hasattr(hyperparams, 'val_logprob'):
    ax2 = ax.twinx()
    line_2 = ax2.plot(x_data, np.nan_to_num(hyperparams.val_logprob[(hyperparams.iter >= start_iter) & (hyperparams.iter < end_iter)]), 'r-')
    ax2.set_ylabel('dev logprob', color='r')
    ax2.tick_params('y',color='r')
ax.set_xlabel('iteration')
# ax.legend(lines, ('Training', 'Dev'))
fig.tight_layout()
pp = PdfPages(os.path.join(last_sample_folder, 'logprobs' + '_'+ str(min(x_data))+
                           '_' + str(max(x_data))) + '.pdf')
fig.savefig(pp, format='pdf')
pp.close()
plt.cla()
plt.clf()


# NP, VP and PP evaluations, branching scores
phrases = ['NP', 'VP', 'PP']
gold_counters = {'NP':[], 'VP':[], 'PP': []}
for index, t in enumerate(gold_trees):
    gold_counters['NP'].append(Counter())
    gold_counters['VP'].append(Counter())
    gold_counters['PP'].append(Counter())
    for sub_t in t.subtrees():
        if sub_t.label() in phrases and len(sub_t.leaves()) > 1:
            gold_counters[sub_t.label()][index][' '.join(sub_t.leaves())] += 1

scores = []
aggregate_scores = []
NP_length = sum([sum(x.values()) for x in gold_counters['NP']])
VP_length = sum([sum(x.values()) for x in gold_counters['VP']])
PP_length = sum([sum(x.values()) for x in gold_counters['PP']])
phrases_lengths = [NP_length, VP_length, PP_length]

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

def calc_phrase_stats(f_name, prec_thres=0.6):
    eps = 1e-6
    l_branch = 0
    r_branch = 0
    current_counters = []
    current_counter_nolabel = []
    num_d2_trees = 0
    num_total_trees = 0
    overall_label_counter = Counter() #used in label dist entropy calc
    non_term_only_label = defaultdict(bool) # used in non-preterm only label calc
    with open(f_name) as fh:
        for line in fh:
            num_total_trees += 1
            d2_checked = 0
            this_t = nltk.Tree.fromstring(line.strip())
            this_l, this_r = calc_branching_score(this_t)
            l_branch += this_l
            r_branch += this_r
            current_counters.append({})
            current_counter_nolabel.append(Counter())
            for sub_t in this_t.subtrees():
                overall_label_counter[sub_t.label()] += 1
                if len(sub_t.leaves()) > 1:
                    non_term_only_label[sub_t.label()] &= True
                    if sub_t.label() not in current_counters[num_total_trees - 1]:
                        current_counters[num_total_trees - 1][sub_t.label()] = Counter()
                    try:
                        if not d2_checked and isinstance(sub_t[1][0][1], nltk.Tree):
                            d2_checked = 1
                            num_d2_trees += 1
                    except:
                        pass
                    current_counters[num_total_trees - 1][sub_t.label()][' '.join(sub_t.leaves())] += 1
                    current_counter_nolabel[num_total_trees - 1][' '.join(sub_t.leaves())] += 1
                else:
                    non_term_only_label[sub_t.label()] &= False
    total_num_labels = sum(overall_label_counter.values())
    label_dist = [x/total_num_labels for x in overall_label_counter.values()]
    label_dist_entropy = stats.entropy(label_dist)
    num_non_term_only_label = sum(non_term_only_label.values())

    total_acc_number = [Counter(), Counter(), Counter()]
    total_hyp_cat_number = [Counter(), Counter(), Counter()]
    nolabel_recalls = [0, 0, 0]
    for index_tree, tree_counter in enumerate(current_counter_nolabel):
        for index, phrase_cat in enumerate(phrases):
            acc_num = sum((tree_counter & gold_counters[phrase_cat][index_tree]).values())
            nolabel_recalls[index] += acc_num

    nolabel_recalls = [ x / phrases_lengths[index] if phrases_lengths[index]
                        != 0  else 0 for index, x in enumerate(nolabel_recalls)]

    for index_tree, tree_counter in enumerate(current_counters):
        for cat, cat_counter in tree_counter.items():
            for index, phrase_cat in enumerate(phrases):
                acc_num = sum(( cat_counter & gold_counters[phrase_cat][index_tree]).values())
                total_acc_number[index][cat] += acc_num
                total_hyp_cat_number[index][cat] += sum(cat_counter.values())

    Performance = namedtuple('Performance', 'id prec rec f1')
    performances = [[],[],[]] # compute the performances of each cat
    for index, phrase_cat in enumerate(phrases):
        for cat, cat_acc in total_acc_number[index].items():
            prec = cat_acc / total_hyp_cat_number[index][cat]
            rec = cat_acc / phrases_lengths[index] if phrases_lengths[index] != 0 else 0
            f1 = 0 if prec == 0 or rec == 0 else 1.0 / (0.5 / prec + (0.5 / rec))
            performances[index].append(Performance(cat, prec, rec, f1))

    for phrase in performances:
        phrase.sort(key=lambda x: x.f1, reverse=True)
    this_best_scores = [phrase[0].f1 for phrase in performances]
    this_best_aggregate_scores = [phrase[0].f1 for phrase in performances]
    this_best_aggregate_scores_index = [[phrase[0].id] for phrase in performances]
    for phrase in performances:
        phrase.sort(key=lambda x: x.prec, reverse=True)


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
                rec = acc_num / phrases_lengths[index] if phrases_lengths[index] != 0 else 0
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

    return this_best_scores, this_best_aggregate_scores, r_branch / (l_branch+r_branch), num_d2_trees / num_total_trees, label_dist_entropy, num_non_term_only_label / len(non_term_only_label), nolabel_recalls

end_f_names_nodeps = [x for x in end_f_names if  'fromdeps' not in x]

with Pool(processes=total_process_limit) as pool:
    scores, aggregate_scores, r_branching_tendency, d2_proportion, label_dist_freq, num_non_term_label, nolabel_recalls= zip(*(pool.map(calc_phrase_stats, end_f_names_nodeps)))

# print(scores[0], aggregate_scores[0])
scores = np.array(scores)
len_iters = len(x_data)
aggregate_scores = np.array(aggregate_scores)
r_branching_tendency = np.array(r_branching_tendency)
d2_proportion = np.array(d2_proportion)
label_dist_freq = np.array(label_dist_freq)
num_non_term_label = np.array(num_non_term_label)
fig, ax = plt.subplots()
titles = [ "best NP F1", 'best VP F1', 'best PP F1', "best NP agg F1", "best VP agg F1", "best PP agg F1"]
all_results[0].extend(titles)
all_results[1].append(scores)
all_results[1].append(aggregate_scores)
lines = ax.plot(x_data, scores[:len_iters, 0], 'o', x_data, scores[:len_iters, 1], 'v', x_data, scores[:len_iters, 2]
                , 'x', x_data, aggregate_scores[:len_iters, 0], '+', x_data, aggregate_scores[:len_iters, 1], 's',
                x_data, aggregate_scores[:len_iters, 2], '*', alpha=0.3)
ax.set_ylabel('percentage')
ax.legend(lines, titles)
pp = PdfPages(
    os.path.join(last_sample_folder, 'phrases' + '_' + str(min(x_data)) + '_' + str(max(x_data))) + '.pdf')
fig.savefig(pp, format='pdf')
pp.close()
plt.cla()
plt.clf()

nolabel_recalls = np.array(nolabel_recalls)
fig, ax = plt.subplots()
# len_iters = len(output_last_samples)
titles = ["NP recall", "VP recall", "PP recall"]
all_results[0].extend(titles)
all_results[1].append(nolabel_recalls)
lines = ax.plot(x_data, nolabel_recalls[:len_iters, 0], x_data, nolabel_recalls[:len_iters, 1], x_data, nolabel_recalls[:len_iters, 2])
ax.set_ylabel('No label recalls')
ax.legend(lines, titles)
pp = PdfPages(
    os.path.join(last_sample_folder, 'nolabel_recall' + '_' + str(min(x_data)) + '_' + str(max(x_data))) + '.pdf')
fig.savefig(pp, format='pdf')
pp.close()
plt.cla()
plt.clf()


fig, ax = plt.subplots()
# len_iters = len(output_last_samples)
lines = ax.plot(x_data, r_branching_tendency[:len_iters])
all_results[0].append("R branching")
all_results[1].append(r_branching_tendency)
ax.set_ylabel('R branching tendency score')
ax.legend(lines, ( "R branching"))
pp = PdfPages(
    os.path.join(last_sample_folder, 'r_branching' + '_' + str(min(x_data)) + '_' + str(max(x_data))) + '.pdf')
fig.savefig(pp, format='pdf')
pp.close()
plt.cla()
plt.clf()

fig, ax = plt.subplots()
all_results[0].append("D2 trees")
all_results[1].append(d2_proportion)
lines = ax.plot(x_data, d2_proportion)
ax.set_ylabel('D2 trees proportion')
ax.legend(lines, ( "D2 trees"))
pp = PdfPages(
    os.path.join(last_sample_folder, 'd2_trees' + '_' + str(min(x_data)) + '_' + str(max(x_data))) + '.pdf')
fig.savefig(pp, format='pdf')
pp.close()
plt.cla()
plt.clf()

fig, ax = plt.subplots()
all_results[0].append("entropy")
all_results[1].append(label_dist_freq)
lines = ax.plot(x_data, label_dist_freq)
ax.set_ylabel('Label frequency distribution entropy')
ax.legend(lines, ( "Entropy"))
pp = PdfPages(
    os.path.join(last_sample_folder, 'label_dist_entropy' + '_' + str(min(x_data)) + '_' + str(max(x_data))) + '.pdf')
fig.savefig(pp, format='pdf')
pp.close()
plt.cla()
plt.clf()

fig, ax = plt.subplots()
all_results[0].append("nonterm_only")
all_results[1].append(num_non_term_label)
lines = ax.plot(x_data, num_non_term_label)
ax.set_ylabel('Percentage of non-termnial only labels')
ax.legend(lines, ( "Percentage"))
pp = PdfPages(
    os.path.join(last_sample_folder, 'non_term_only' + '_' + str(min(x_data)) + '_' + str(max(x_data))) + '.pdf')
fig.savefig(pp, format='pdf')
pp.close()
plt.cla()
plt.clf()
all_results[1] = [np.expand_dims(x, 1) if len(x.shape) < 2 else x for x in all_results[1]]
all_results[1] = np.hstack(all_results[1])
with open(os.path.join(last_sample_folder, 'results' + '_' + str(min(x_data)) + '_' + str(max(x_data))) + '.txt','w') as r:
    print('\t'.join(all_results[0]), file=r)
    for row in all_results[1]:
        print('\t'.join([str(x) for x in row]), file=r)

# #dependency phrase scores
# with Pool(processes=total_process_limit) as pool:
#     scores, aggregate_scores = zip(*(pool.map(calc_phrase_stats, deps_f_names)))
#
# print(scores[0], aggregate_scores[0])
# scores = np.array(scores)
# aggregate_scores = np.array(aggregate_scores)
# fig, ax = plt.subplots()
# lines = ax.plot(x_data, scores[:len_iters, 0], x_data, scores[:len_iters, 1], x_data, scores[:len_iters, 2]
#                 ,x_data, aggregate_scores[:len_iters, 0] , x_data, aggregate_scores[:len_iters, 1],
#                 x_data, aggregate_scores[:len_iters, 2])
# ax.set_ylabel('percentage')
# ax.legend(lines, ( "best NP F1 deps", 'best VP F1 deps', 'best PP F1 deps', "best NP agg F1 deps", "best VP agg F1 deps", "best PP agg F1 deps"))
# pp = PdfPages(
#     os.path.join(last_sample_folder, 'phrases_deps' + '_' + str(min(x_data)) + '_' + str(max(x_data))) + '.pdf')
# fig.savefig(pp, format='pdf')
# pp.close()
# plt.cla()
# plt.clf()

# delete the temp files
for f in rule_f_names + model_f_names + head_model_names + deps_f_names + end_f_names + evalb_f_names:
    bash_command('rm -f {}'.format(f))

# output_files = os.listdir(last_sample_folder)
# output_last_samples = [x for x in output_files if x.endswith('.constiteval.txt')]
# f_list = []
# for f in output_last_samples:
#     f_list.append(os.path.join(last_sample_folder, f))
# f_list_str = ' '.join(f_list)
# f_constlist = os.path.join(last_sample_folder, 'nt-lower-nounary-nopunc.uhhmm-iter.constitevallist')
# bash_command(constlist_command.format(f_list_str, f_constlist))
#
# f_const_table = os.path.join(last_sample_folder, f_constlist, 'nt-lower-nounary-nopunc.uhhmm-iter.constitevaltable.txt')
# bash_command(plot_command.format(modelblocks_path, f_const_table))