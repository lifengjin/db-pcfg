#!/usr/bin/python

import nltk
import sys
import codecs

f = codecs.open(sys.argv[1], encoding='utf-8')

for line in f:
    words = nltk.word_tokenize(line.rstrip())
    tags = nltk.pos_tag(words)
    print(" ".join(map(lambda x : x[1], tags)))
