#!/bin/bash

perl -pe 's/[^\s\/]+\/(\S+)/\1/g'
