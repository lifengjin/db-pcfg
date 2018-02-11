#!/bin/bash

## If it has as many slashes as tokens then we are in tagwords mode. Else we are in words mode.
perl -pe 'if(scalar(split(/\//)) >= scalar(split(/ /))){ s/(\S*)\/(\S*)/\1\/\L\2/g;}else{s/([^\s\/]+)/\L\1/g;}'

