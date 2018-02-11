#!/bin/bash

perl -pe 's/ /\n/g' | sort -n | uniq -c | sort -n
