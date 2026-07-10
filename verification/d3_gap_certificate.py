# V13 - Certified instance of 'nontrivial class does NOT imply operational (AvN) contextuality':
#   (a) Heisenberg cocycle w(u,v)=zeta_3^{<u,v>} on Z3^2 is NOT a coboundary - exhaustive over
#       all 3^9=19683 cochains; same for w^2 => class has order exactly 3.
#   (b) qutrit (d=3) Weyl families: exhaustive m=1 (all 15 subfamilies of the 4 lines, every
#       cycle) and 2000 sampled m=2 isotropic-context families: self-pairing Q even ALWAYS
#       => AvN value 0 (Paper B Thm J, odd d).
# This machine-certifies the weakening adopted in the CMP repair plan. Expected: GAP CERTIFIED.
import numpy as np, itertools
from criterion import carry_data, left_kernel, sympinv
pts=[(a,b) for a in range(3) for b in range(3)]
sy=lambda u,v:(u[0]*v[1]-u[1]*v[0])%3
def cobound(lam):  # lam: dict pt->Z3, delta lam(u,v)=lam(u)+lam(v)-lam(u+v)
    return {(u,v):(lam[u]+lam[v]-lam[((u[0]+v[0])%3,(u[1]+v[1])%3)])%3 for u in pts for v in pts}
target1={(u,v):sy(u,v) for u in pts for v in pts}
target2={(u,v):(2*sy(u,v))%3 for u in pts for v in pts}
hits=[0,0]
for vals in itertools.product(range(3),repeat=9):
    lam=dict(zip(pts,vals)); d=cobound(lam)
    if d==target1: hits[0]+=1
    if d==target2: hits[1]+=1
print(f"(a) coboundary hits over 19683 cochains: omega:{hits[0]}, omega^2:{hits[1]} "
      f"=> class order exactly 3: {hits==[0,0]}")
lines=[[(1,0),(2,0)],[(0,1),(0,2)],[(1,1),(2,2)],[(1,2),(2,1)]]
odd=0; ncyc=0
for r in range(1,5):
    for sub in itertools.combinations(lines,r):
        A,b=carry_data([list(C) for C in sub],3); K=left_kernel(A)
        for k in K:
            ncyc+=1; odd+=sympinv([list(C) for C in sub],k,3)%2
rng=np.random.default_rng(2); odd2=0; ncyc2=0
for _ in range(2000):
    fam=[]
    for _ in range(rng.integers(4,9)):
        while True:
            v=tuple(rng.integers(0,3,4))
            if any(v): break
        fam.append([v,tuple((2*np.array(v))%3)])
    A,b=carry_data(fam,3); K=left_kernel(A)
    for k in K: ncyc2+=1; odd2+=sympinv(fam,k,3)%2
print(f"(b) m=1 exhaustive: {ncyc} cycles, odd Q: {odd}; m=2 sampled: {ncyc2} cycles, odd Q: {odd2}")
print("GAP CERTIFIED: nontrivial order-3 class AND AvN value {0}" if (hits==[0,0] and odd==0 and odd2==0) else "FAIL")
