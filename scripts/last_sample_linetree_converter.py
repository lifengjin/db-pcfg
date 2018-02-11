import sys
import os
import shutil
output_folder = sys.argv[1]
corpus_name = sys.argv[2]

fs = os.listdir(output_folder)
last_sample = 'last_sample'

for src_f in fs:
    if last_sample in src_f and src_f.endswith('linetrees'):
        trg_f = src_f.replace(last_sample, corpus_name+'.')
        trg_f = trg_f.replace('.linetrees', '.uhhmm.linetrees')
        shutil.copyfile(os.path.join(output_folder, src_f), os.path.join(output_folder, trg_f))