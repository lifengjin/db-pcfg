import sys, model, tree, numpy as np
from RV_Discrete_Label import rv_discrete_label
_, n = sys.argv

n = int(n)

def sample_tree(t, m):
   if not t.c:
     t.c = m['T'].rvs()[0][0]
     sample_tree(t, m)
   else:
     if t.c in m:
       ch = m[t.c].rvs()
       if len(ch) == 2:
         lc = tree.Tree(ch[0][0])
         rc = tree.Tree(ch[1][0])
         sample_tree(lc, m)
         sample_tree(rc, m)
         t.ch = [lc, rc]
       elif len(ch) == 1:
         lc = tree.Tree(ch[0])
         t.ch = [lc]

R = model.CondModel('R')

for line in sys.stdin:
  if line.strip!= '':
    R.read(line)

m = {}
for c in R:
  vals = [None, None]
  vals[0] = list(R[c].keys())
  vals[1] = np.asarray(list(R[c].values()))
  m[c] = rv_discrete_label(values=vals)


for i in range(n):
  t = tree.Tree()
  sample_tree(t, m)
  print(str(t))  
