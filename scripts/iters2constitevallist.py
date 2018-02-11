import sys, os, re, argparse

argparser = argparse.ArgumentParser('''
Outputs a table of UHHMM iteration number by pathname for a list of one or more *.constiteval.txt files.
''')
argparser.add_argument('constitevals', type=str, nargs='+', help='One or more *.constiteval.txt files from which to compute')
args, unknown = argparser.parse_known_args()

iternum = re.compile('([0-9]+)-uhhmm')
print('path iter')

for path in args.constitevals:
    with open(path, 'rb') as f:
        iter = iternum.search(path).group(1)
        print(os.path.abspath(path) + ' ' + str(iter))
