#!/bin/bash

perl -pe "s/\S\S?\/\S\S?::(?:ACT\d+\/AWA\d+)?:(POS\d+);\S*/\1/g"
