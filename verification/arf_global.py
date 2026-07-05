# Global Arf test: is W the Arf invariant of q on the CERTIFICATE KERNEL (matched-cycle space),
# not the per-context sum? Build one quadratic form over all fiber lift-bits, restrict to the
# kernel of the fiber incidence map, compute its Arf; compare to the attaining bit.
import json, numpy as np
from itertools import product
def symp(u,v):
    m=len(u)//2; return sum(u[2*i+1]*v[2*i]-u[2*i]*v[2*i+1] for i in range(m))
def load(path):
    c=json.load(open(path)); return [[tuple(v) for v in it["ctx"]] for it in c["items"]]

def fiber_all(C,d):
    """all lifted contexts (as tuples of lifted letters) over base context C."""
    from arf2 import fiber_solspace
    x0,ker=fiber_solspace(C,d)
    if x0 is None: return []
    m=len(C[0])//2;L=len(C)
    sols=[x0]
    for r in range(1,len(ker)+1):
        pass
    # enumerate full affine space
    space=[x0]
    for k in ker:
        space=[ (s^ (t*k))%2 for s in space for t in [0,1]]
    out=[]
    for s in set(map(tuple,space)):
        lifted=tuple(tuple((C[i][c]+d*s[i*2*m+c]) for c in range(2*m)) for i in range(L))
        out.append(lifted)
    return out

def build_fiber_pool(base,d):
    pool=[]; 
    for C in base:
        for lc in fiber_all(C,d):
            pool.append([tuple(v) for v in lc])
    # dedup
    seen=set(); uniq=[]
    for C in pool:
        k=tuple(sorted(C))
        if k not in seen: seen.add(k); uniq.append(C)
    return uniq

def carry(C,dd):  # obstruction bit at fiber level dd=2d
    return sum(symp(C[i],C[j])//dd for i in range(len(C)) for j in range(i+1,len(C)))%2

def attaining_bit(pool,dd):
    """does the fiber pool attain value dd/2? = exists GF2 left-kernel lambda of incidence with lambda.b=1"""
    obs=sorted({tuple(v) for C in pool for v in C}); oi={v:k for k,v in enumerate(obs)}
    A=np.zeros((len(pool),len(obs)),int); b=np.zeros(len(pool),int)
    for r,C in enumerate(pool):
        b[r]=carry(C,dd)
        for v in C: A[r,oi[tuple(v)]]^=1
    A%=2; rows,ncols=A.shape
    aug=np.concatenate([A,np.eye(rows,dtype=int)],axis=1)%2; r=0
    for c in range(ncols):
        pr=next((i for i in range(r,rows) if aug[i,c]),None)
        if pr is None: continue
        aug[[r,pr]]=aug[[pr,r]]
        for i in range(rows):
            if i!=r and aug[i,c]: aug[i]^=aug[r]
        r+=1
        if r==rows: break
    vb={int(aug[i,ncols:]@b%2) for i in range(rows) if not aug[i,:ncols].any()}
    return 1 if 1 in vb else 0

if __name__=='__main__':
    # attaining bases (should attain) vs trivial bases (should not) at base d=4 -> fiber dd=8
    d=4; dd=8
    fam4=load("cert4_min.json")
    pool=build_fiber_pool(fam4,d)
    print(f"cert4 fiber pool: {len(pool)} contexts -> attains value {dd//2}?  bit = {attaining_bit(pool,dd)}")
    
    rng=np.random.default_rng(5); m=2
    def rand_ctx():
        for _ in range(400):
            u=tuple(int(x) for x in rng.integers(0,d,2*m)); v=tuple(int(x) for x in rng.integers(0,d,2*m))
            if symp(u,v)%d==0:
                w=tuple((-(u[i]+v[i]))%d for i in range(2*m))
                if symp(u,w)%d==0 and symp(v,w)%d==0: return [u,v,w]
    for t in range(3):
        base=[c for c in (rand_ctx() for _ in range(6)) if c]
        pool=build_fiber_pool(base,d)
        print(f"trivial base {t}: fiber pool {len(pool)} ctx -> attaining bit = {attaining_bit(pool,dd)}")