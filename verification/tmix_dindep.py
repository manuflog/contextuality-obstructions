# Doubling-law support: the F2 system carrying T_mix is d-INDEPENDENT across d = 2 mod 4.
# See INDEX.md (Doubling law row; companion to close_T2_proof.py).
#
# EXTENDED 2026-07-24 -- this file tests only d = 2 mod 4, which is HALF the story.
# mod4_dichotomy.py proves that BOTH residue classes are internally constant, for a single
# reason: with u = h*p mod d and h = d/2, bilinearity gives qu = symp(u_a,u_b)/d = h*s where
# symp(p_a,p_b) = 2s, so every mod-2 quantity here depends on d only through the parity of h:
#     d = 0 mod 4 (h EVEN): every qu even => t == 0 IDENTICALLY (0 of 1536 lifted contexts)
#     d = 2 mod 4 (h ODD) : 5 of 18 qu odd => t != 0 on 128 of 768; vanishes only over a cycle
# So the d-independence verified below is not a coincidence of this sample, and the d = 0 mod 4
# class -- never tested here -- is constant too, with the OPPOSITE (and stronger) behaviour.
# VERIFIED CLAIMS (asserted below; failure raises and exits nonzero):
#   [1] the canonicalized lifted-context signature SET is byte-identical to the d=6 one
#       for every d in {6,10,14,18,22,26};
#   [2] the induced T vector is byte-identical to the d=6 one for every such d;
#   [3] the comparison is non-vacuous: T is not the zero vector (|T| = 128 nonzero
#       entries), so identity of T is a real constraint, not "0 == 0".
import sys
import numpy as np
from arf_global import symp, fiber_all
lbl={'XI':(1,0,0,0),'IX':(0,0,1,0),'XX':(1,0,1,0),'IY':(0,0,1,1),'YI':(1,1,0,0),
     'YY':(1,1,1,1),'XY':(1,0,1,1),'YX':(1,1,1,0),'ZZ':(0,1,0,1)}
R=[['XI','IX','XX'],['IY','YI','YY'],['XY','YX','ZZ']]
Co=[['XI','IY','XY'],['IX','YI','YX'],['XX','YY','ZZ']]
P={k:np.array(v) for k,v in lbl.items()}
def build(d):
    h=d//2; base=[[tuple((h*P[l])%d) for l in C] for C in R+Co]; pool=[]
    for C in base:
        for lc in fiber_all(C,d): pool.append([tuple(v) for v in lc])
    seen=set(); uniq=[]
    for C in pool:
        k=tuple(sorted(C))
        if k not in seen: seen.add(k); uniq.append(sorted(C))
    pool=uniq
    # canonicalize each lifted context by its (base-pauli-index, lift-bits) signature so ordering is d-independent
    def sig(C):
        s=[]
        for v in C:
            u=tuple(np.array(v)%d); x=tuple(((np.array(v)-(np.array(v)%d))//d)%2)
            up=tuple((np.array(u)//h)%2)  # recover pauli symbol (u = h*p mod d, h=d/2)
            s.append((up,x))
        return tuple(sorted(s))
    rows=[]
    for C in pool:
        sg=sig(C)
        # T value
        us=[np.array(v)%d for v in C]; xs=[((np.array(v)-(np.array(v)%d))//d)%2 for v in C]
        t=0; n=len(C)
        for a in range(n):
            for b in range(a+1,n):
                qu=symp(us[a],us[b])//d; Mab=symp(us[a],xs[b])+symp(xs[a],us[b])
                t^=((qu&1)*(Mab&1))&1
        rows.append((sg,t%2))
    rows.sort()
    T=np.array([t for _,t in rows])%2
    sigs=[sg for sg,_ in rows]
    return sigs,T
# compare across d = 2 mod 4
data={d:build(d) for d in [6,10,14,18,22,26]}
ref_sig,ref_T=data[6]
print("d≡2 mod4: same context signature-set as d=6?  and same T vector?")
checks=[]   # (label, ok) -- every load-bearing quantity gets an entry
for d,(sg,T) in data.items():
    same_ctx = (sg==ref_sig)
    same_T = bool(np.array_equal(T,ref_T)) if same_ctx else "n/a"
    print(f"  d={d:>3}: same_contexts={same_ctx}  same_T={same_T}  (|T|={T.sum()})")
    checks.append((f"d={d}: lifted-context signature set identical to d=6", bool(same_ctx)))
    checks.append((f"d={d}: T vector identical to d=6", bool(np.array_equal(T,ref_T))))
# non-vacuity: the compared object must not be trivially zero
checks.append((f"comparison non-vacuous: |T_ref| != 0 (got {int(ref_T.sum())} of {len(ref_T)})",
               int(ref_T.sum())>0))
checks.append((f"reference T has the pinned weight 128 (got {int(ref_T.sum())})",
               int(ref_T.sum())==128))
# also test whether T depends only on d mod 8 within 2 mod4
print("\nGroup by d mod 8:", {d:(d%8) for d in data})
print("(both residues 2 and 6 mod 8 are represented above and agree, so the invariance is")
print(" across d mod 4 = 2, not merely within a d mod 8 class.)")
checks.append(("both residues 2 and 6 mod 8 present in the sample",
               {d%8 for d in data}=={2,6}))

print("\n--- verdict ---")
_bad=[l for l,ok in checks if not ok]
for l in _bad: print(f"  FAILED CHECK: {l}")
assert not _bad, "tmix_dindep load-bearing checks FAILED: "+"; ".join(_bad)
print(f"tmix_dindep: {len(checks)}/{len(checks)} load-bearing checks passed")
print("tmix_dindep PASS")
# (no sys.exit here: a bare module-level exit kills any script that IMPORTS this one,
#  silently, with status 0. See KNOWN_LIMITATIONS.md. Falling off the end already exits 0.)
