#!/bin/bash

## REGEXP 1: (PHRASE (POS word)) -> (PHRASE [POS/word])  (find terminals and rewrite with brackets)
## REGEXP 2: (PHRASE ->                                  (remove phrase nodes)
## REGEXP 3: ) ->                                        (remove right parens and any traces that are with them)
## REGEXP 4: [pos/word]  [pos/word] -> [pos/word] [pos/word] (remove extra whitespace)
## REGEXP 5: [pos/word] -> pos/word                          (remove square brackets)

perl -pe 's/\(([^ -]+)\S* ([^\s\)\(]+)\)/[\1\/\2]/g;s/\(\S*//g;s/[^\[\]\s]*\)//g;s/^\s*//g;s/  +/ /g;s/[\[\]]//g;'

