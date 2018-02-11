import sys
import os
ccl_file = sys.argv[1]

assert os.path.exists(ccl_file)

with open(ccl_file) as ccl, open(ccl_file+'.linetrees', 'w') as ccl_line:
    single_sent = []
    l_brackets = 0
    r_brackets = 0
    for line in ccl:
        line = line.strip()
        if not line:
            continue
        l_brackets += line.count('(')
        r_brackets += line.count(')')
        if l_brackets != r_brackets:
            single_sent.append(line)
        else:
            single_sent.append(line)
            print(' '.join(single_sent), file=ccl_line)
            single_sent = []
            l_brackets = 0
            r_brackets = 0