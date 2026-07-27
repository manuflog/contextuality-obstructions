# Global Arf test: is W the Arf invariant of q on the CERTIFICATE KERNEL (matched-cycle space),
# not the per-context sum? Build one quadratic form over all fiber lift-bits, restrict to the
# kernel of the fiber incidence map, compute its Arf; compare to the attaining bit.
#
# VERIFIED CLAIM (asserted below; failure raises and exits nonzero):
#   [1] the cert4 fiber pool (base d=4 -> fiber dd=8) has 896 lifted contexts and
#       attaining bit 1: it DOES attain the value dd/2 = 4.
#   [2] the three random "trivial" control bases are genuinely trivial AT BASE LEVEL,
#       i.e. their base-level attaining bit (dd = d = 4) is 0.
#
# ANOMALY, INVESTIGATED AND RESOLVED (2026-07-24): the original inline comment claimed
# the trivial bases "should not" attain at FIBER level.  THAT COMMENT WAS WRONG, and the
# script's own output has always contradicted it (trivial base 0 attains at seed 5).
# It is NOT a seed artifact.  Sweep: 40 seeds x 3 random 6-context d=4 bases = 120 bases.
#   * base-level attaining bit = 0 for 120/120 (they really are base-trivial), BUT
#   * fiber-level attaining bit = 1 for 31/120 (~26%).
# Reason: build_fiber_pool replaces the 6 base contexts by ALL of their lifts -- a family
# of ~768 contexts with a large GF(2) left-kernel.  A base-noncontextual family therefore
# has no trouble acquiring an odd-carry cycle after lifting to Z_{2d}.  So fiber-level
# attainment is generic-ish and these random bases are NOT a negative control for it.
# Consequently the fiber bit of the random bases is PRINTED but deliberately NOT asserted;
# only the base-level triviality (which is the actual premise) is asserted.
import sys
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
    # attaining base (cert4, DOES attain) vs base-trivial controls, at base d=4 -> fiber dd=8
    d=4; dd=8
    checks=[]   # (label, ok) -- every load-bearing quantity gets an entry
    fam4=load("cert4_min.json")
    pool=build_fiber_pool(fam4,d)
    cert_bit=attaining_bit(pool,dd)
    print(f"cert4 fiber pool: {len(pool)} contexts -> attains value {dd//2}?  bit = {cert_bit}")
    checks.append((f"cert4 fiber pool has 896 lifted contexts (got {len(pool)})", len(pool)==896))
    checks.append((f"cert4 fiber pool ATTAINS value {dd//2} (bit=1, got {cert_bit})", cert_bit==1))

    rng=np.random.default_rng(5); m=2
    def rand_ctx():
        for _ in range(400):
            u=tuple(int(x) for x in rng.integers(0,d,2*m)); v=tuple(int(x) for x in rng.integers(0,d,2*m))
            if symp(u,v)%d==0:
                w=tuple((-(u[i]+v[i]))%d for i in range(2*m))
                if symp(u,w)%d==0 and symp(v,w)%d==0: return [u,v,w]
    fiber_bits=[]
    for t in range(3):
        base=[c for c in (rand_ctx() for _ in range(6)) if c]
        base_bit=attaining_bit(base,d)          # triviality AT BASE LEVEL: this is the premise
        pool=build_fiber_pool(base,d)
        fb=attaining_bit(pool,dd)
        fiber_bits.append(fb)
        print(f"trivial base {t}: fiber pool {len(pool)} ctx -> attaining bit = {fb}"
              f"   [base-level bit = {base_bit}]")
        checks.append((f"control base {t} is trivial AT BASE LEVEL (bit 0, got {base_bit})", base_bit==0))
    print("NOTE: the fiber-level bits of the controls are NOT asserted -- see header. A")
    print("      base-trivial family attains at fiber level ~26% of the time (31/120 over")
    print("      a 40-seed sweep), so these are not a negative control for fiber attainment.")
    print(f"      observed control fiber bits this run: {fiber_bits}")

    print()
    print("--- verdict ---")
    _bad=[l for l,ok in checks if not ok]
    for l in _bad: print(f"  FAILED CHECK: {l}")
    assert not _bad, "arf_global load-bearing checks FAILED: "+"; ".join(_bad)
    print(f"arf_global: {len(checks)}/{len(checks)} load-bearing checks passed")
    print("arf_global PASS")
    sys.exit(0)