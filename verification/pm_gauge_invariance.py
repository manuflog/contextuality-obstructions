# V10 - PM six-context sign product = -1, invariant under ALL 512 sign gauges.
# Backs Paper A Thm E/F and Copenhagen-note S3: the -1 is gauge-invariant because
# every observable sits in exactly two contexts. Expected: PASS, product -1, 512/512.
import numpy as np, itertools
from weyl import build
X,Z,w,tau,W,N = build(2,2)
obs = dict(XI=(1,0,0,0),IX=(0,0,1,0),XX=(1,0,1,0),IZ=(0,0,0,1),ZI=(0,1,0,0),
           ZZ=(0,1,0,1),XZ=(1,0,0,1),ZX=(0,1,1,0),YY=(1,1,1,1))
names=list(obs)
fam=[['XI','IX','XX'],['IZ','ZI','ZZ'],['XZ','ZX','YY'],
     ['XI','IZ','XZ'],['IX','ZI','ZX'],['XX','ZZ','YY']]
def signs(g):
    out=[]
    for C in fam:
        M=np.eye(4,dtype=complex)
        for nm in C: M = M @ (g[nm]*W(np.array(obs[nm])))
        out.append(complex(np.round(np.trace(M)/4,6)))
    return out
base=signs({n:1 for n in names})
assert base==[1,1,1,1,1,-1], base
prods=set()
for bits in itertools.product([1,-1],repeat=9):
    g=dict(zip(names,bits)); prods.add(np.prod(signs(g)).real.round(6))
print(f"base context signs {base}")
print(f"six-sign product over all 512 sign gauges: {prods}")
print("PASS" if prods=={-1.0} else "FAIL")
