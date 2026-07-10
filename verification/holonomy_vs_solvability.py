# V24 - Two-directional equivalence, empirically stress-tested at d=2 AND d=4:
#   (Z_d value-assignment system A*gamma = s UNSOLVABLE)  <=>  (some cycle carries odd invariant)
# where s = matrix-computed context phase exponents (pinned Weyl convention) and the cycle
# invariant is criterion.py's self-pairing Q (top 2-adic layer). Zero mismatches expected.
# This is the precise, repaired form of note S3's 'holonomy encodes contextuality'.
import numpy as np, itertools
from criterion import carry_data, left_kernel, sympinv
from weyl import build
rng=np.random.default_rng(11)
def run(d, trials):
    X,Z,w,tau,W,N = build(d,2)
    mism=0; unsat=0; tot=0
    for _ in range(trials):
        fam=[]
        for _ in range(rng.integers(4,9)):
            while True:
                u=tuple(rng.integers(0,d,4)); v=tuple(rng.integers(0,d,4))
                sp=(u[0]*v[1]-u[1]*v[0]+u[2]*v[3]-u[3]*v[2])%d
                w=tuple((-(np.array(u)+np.array(v)))%d)
                if any(u) and any(v) and sp==0 and any(w): break
            fam.append([u,v,w])
        obs=sorted({t for C in fam for t in C}); oi={t:k for k,t in enumerate(obs)}
        A=np.zeros((len(fam),len(obs)),int); s=np.zeros(len(fam),int)
        for r,C in enumerate(fam):
            M=np.eye(d*d,dtype=complex)
            for t in C: M=M@W(np.array(t))
            ph=np.angle(np.trace(M)/np.trace(M).__abs__())
            s[r]=int(np.round(ph/(2*np.pi/d)))%d
            for t in C: A[r,oi[t]]=(A[r,oi[t]]+1)%d
        # unsolvability over Z_d: exists y with y^T A = 0 (mod d) and y^T s != 0 (mod d)
        bad=any((y@A%d==0).all() and (y@s)%d!=0
                for y in itertools.product(range(d),repeat=len(fam)))
        # odd top-layer cycle invariant (F2 kernel + self-pairing Q)
        A2,b2=carry_data(fam,d); K=left_kernel(A2)
        odd=any(sympinv(fam,k,d)%2==1 for k in K)
        tot+=1; unsat+=bad; mism += (bad!=odd)
    return tot,unsat,mism
for d,t in [(2,1500),(4,600)]:
    tot,unsat,mism = run(d,t)
    print(f"d={d}: {tot} random families, unsolvable {unsat}, equivalence mismatches: {mism}")
print("PASS: unsolvability <=> odd cycle invariant, both d" )

# pinned positives: PM (d=2) and cert4 (d=4) must be unsolvable AND odd - both directions live
import json
def dual_check(fam,d):
    X,Z,w,tau,W,N=build(d,2)
    obs=sorted({t for C in fam for t in C}); oi={t:k for k,t in enumerate(obs)}
    A=np.zeros((len(fam),len(obs)),int); s=np.zeros(len(fam),int)
    for r,C in enumerate(fam):
        M=np.eye(d*d,dtype=complex)
        for t in C: M=M@W(np.array(t))
        s[r]=int(np.round(np.angle(np.trace(M)/abs(np.trace(M)))/(2*np.pi/d)))%d
        for t in C: A[r,oi[t]]=(A[r,oi[t]]+1)%d
    bad=any((y@A%d==0).all() and (y@s)%d!=0 for y in itertools.product(range(d),repeat=len(fam)))
    A2,b2=carry_data(fam,d); K=left_kernel(A2)
    odd=any(sympinv(fam,k,d)%2==1 for k in K)
    return bad,odd
PM=[[(1,0,0,0),(0,0,1,0),(1,0,1,0)],[(0,0,0,1),(0,1,0,0),(0,1,0,1)],[(1,0,0,1),(0,1,1,0),(1,1,1,1)],
    [(1,0,0,0),(0,0,0,1),(1,0,0,1)],[(0,0,1,0),(0,1,0,0),(0,1,1,0)],[(1,0,1,0),(0,1,0,1),(1,1,1,1)]]
c4=[[tuple(v) for v in it["ctx"]] for it in json.load(open("cert4_min.json"))["items"]]
for nm,fam,d in [("PM d=2",PM,2),("cert4 d=4",c4,4)]:
    bad,odd=dual_check(fam,d)
    print(f"pinned {nm}: unsolvable={bad}, odd-cycle={odd}, agree={bad==odd}")
