#!/bin/bash

perl -pe 's/\(\S+//g;s/\)//g;s/^\s*//;s/  */ /g'
