#!/usr/bin/env python3
"""
B1 — TORSION LAYER: KS-colorability as the torsion invariant on rigid realizations
(session 2, 2026-07-16). Companion note: TORSION.md (precise trichotomy conjecture + status).

KS RULES (stated precisely). Given a finite ray set R subset C^d, its orthogonality graph
G(R) (edges = orthogonal pairs) and its basis hypergraph B(R) = all d-cliques of G(R)
(d mutually orthogonal nonzero vectors are linearly independent, hence automatically a
complete orthogonal basis of C^d), a KS coloring is  f : R -> {0,1}  such that
  (KS1)  sum_{r in b} f(r) = 1   for every complete basis b in B(R)   [exactly one 1 per basis]
  (KS2)  f(r) + f(s) <= 1        for every edge rs of G(R)            [at most one 1 per pair]
(KS2) restricted to pairs lying inside some basis is implied by (KS1); it is an independent
constraint exactly on the edges contained in NO basis (Yu-Oh has 12 such edges, the h-y ones).
R is KS-COLORABLE iff a KS coloring exists, KS-UNCOLORABLE otherwise.

TORSION INVARIANTS (defined for any realization; contentful on rigid ones — see TORSION.md):
  t0 in Z2 :  t0 = 1  iff the system (KS1)+(KS2) is infeasible (KS-uncolorable).
  tau      :  the class of the all-ones vector 1_B in GF(2)^B(R) / rowsp_GF2(M), where M is
              the ray-basis incidence matrix over GF(2).  tau != 0  <=>  there exists
              S subset B(R) with |S| ODD such that every ray lies in an EVEN number of
              members of S (V54's forced transversal parity / the AvN pattern: summing the
              (KS1) equations over S mod 2 yields 0 = 1).
  Lemma A (proved, elementary):  tau != 0  =>  t0 = 1, already at the (KS1)-only level.
  Lemma B (proved, elementary):  d odd  =>  tau = 0.
      [counting incidences of such an S: d*|S| = sum_r deg_S(r) = even, so |S| even.]
  Hence parity/AvN torsion is an even-dimension phenomenon: in d=3 the torsion content is
  t0 alone, and tau != 0 is STRICTLY finer than t0 = 1 (d=3 KS sets, e.g. the original
  117 rays, are uncolorable with tau = 0 by Lemma B). The two candidate definitions of the
  torsion invariant — "KS-uncolorable" and "forced transversal parity" — are NOT equivalent
  in general; they coincide on Peres 24 (verified below).

All arithmetic below is exact (integers / GF(2)); all searches are exhaustive (sound-pruned
backtracking, plus an independent 2^13 brute force for Yu-Oh and an independent parity PROOF
of uncolorability for Peres 24). Evidence label: exact-computational.
"""
from itertools import combinations, product
from fractions import Fraction

# ---------------- ray sets (integer representatives; identical to exact_rigidity.py) --------
def yuoh_rays():
    z = [[1,0,0],[0,1,0],[0,0,1]]
    y = [[0,1,1],[0,1,-1],[1,0,1],[1,0,-1],[1,1,0],[1,-1,0]]
    h = [[1,1,1],[-1,1,1],[1,-1,1],[1,1,-1]]
    return [tuple(v) for v in z+y+h]

def peres24_rays():
    rays=[]; seen=set()
    def add(v):
        v=list(v)
        for x in v:
            if x!=0:
                if x<0: v=[-t for t in v]
                break
        t=tuple(v)
        if t not in seen: seen.add(t); rays.append(t)
    for p in range(4):
        e=[0]*4; e[p]=1; add(e)
    for i,j in combinations(range(4),2):
        for s in (1,-1):
            e=[0]*4; e[i]=1; e[j]=s; add(e)
    for signs in product([1,-1],repeat=3):
        add([1,signs[0],signs[1],signs[2]])
    assert len(rays)==24
    return rays

def dot(u,v): return sum(a*b for a,b in zip(u,v))

def orthogonality_edges(rays):
    return [(i,j) for i,j in combinations(range(len(rays)),2)
            if dot(rays[i],rays[j])==0]

def complete_bases(rays):
    """All d-subsets of mutually orthogonal rays = all complete orthogonal bases in R."""
    d=len(rays[0])
    return [c for c in combinations(range(len(rays)),d)
            if all(dot(rays[i],rays[j])==0 for i,j in combinations(c,2))]

# ---------------- KS coloring: exhaustive backtracking counter ------------------------------
def count_ks_colorings(n, edges, bases, use_pairs=True):
    """Exhaustive count of f:V->{0,1} with (KS1) exactly one 1 per basis and, if use_pairs,
    (KS2) at most one 1 per edge. Pruning is sound (only provably-dead branches are cut),
    so count==0 is a certificate of uncolorability. Returns (count, one example|None)."""
    adj=[set() for _ in range(n)]
    for i,j in edges: adj[i].add(j); adj[j].add(i)
    binc=[[] for _ in range(n)]
    for bi,b in enumerate(bases):
        for r in b: binc[r].append(bi)
    order=sorted(range(n), key=lambda r:-len(binc[r]))   # most-constrained first
    color=[-1]*n
    ones=[0]*len(bases); unassigned=[len(b) for b in bases]
    count=0; example=None
    def rec(k):
        nonlocal count, example
        if k==n:
            count+=1
            if example is None: example=color.copy()
            return
        r=order[k]
        for c in (0,1):
            if c==1:
                if use_pairs and any(color[s]==1 for s in adj[r]): continue
                if any(ones[bi]>=1 for bi in binc[r]): continue      # KS1: never two 1s
            color[r]=c
            for bi in binc[r]:
                unassigned[bi]-=1; ones[bi]+=c
            dead=any(unassigned[bi]==0 and ones[bi]!=1 for bi in binc[r])  # KS1: exactly one
            if not dead: rec(k+1)
            for bi in binc[r]:
                unassigned[bi]+=1; ones[bi]-=c
            color[r]=-1
    rec(0)
    return count, example

def brute_count(n, edges, bases, use_pairs=True):
    """Independent full 2^n enumeration (used for Yu-Oh, n=13)."""
    cnt=0
    for m in range(1<<n):
        f=[(m>>i)&1 for i in range(n)]
        if use_pairs and any(f[i] and f[j] for i,j in edges): continue
        if any(sum(f[r] for r in b)!=1 for b in bases): continue
        cnt+=1
    return cnt

def verify_coloring(f, edges, bases):
    ok_e=all(f[i]+f[j]<=1 for i,j in edges)
    ok_b=all(sum(f[r] for r in b)==1 for b in bases)
    return ok_e and ok_b

# ---------------- parity torsion tau over GF(2) ----------------------------------------------
def gf2_row_reduce(rows):
    """Row-reduce bitmask rows over GF(2); return dict pivot_col -> reduced row."""
    piv={}
    for r in rows:
        cur=r
        while cur:
            lsb=(cur & -cur).bit_length()-1
            if lsb in piv: cur^=piv[lsb]
            else: piv[lsb]=cur; break
    return piv

def gf2_reduce_vector(vec, piv):
    cur=vec
    while cur:
        lsb=(cur & -cur).bit_length()-1
        if lsb in piv: cur^=piv[lsb]
        else: break
    return cur

def gf2_kernel(rows, ncols):
    """Kernel basis of the GF(2) map x -> (parity(row & x))_rows, x in GF(2)^ncols."""
    piv=gf2_row_reduce(rows)
    # full RREF so kernel vectors read off cleanly
    cols=sorted(piv)
    for c in cols:
        for c2 in cols:
            if c2!=c and (piv[c2]>>c)&1: piv[c2]^=piv[c]
    free=[c for c in range(ncols) if c not in piv]
    basis=[]
    for fc in free:
        x=1<<fc
        for c in cols:
            if (piv[c]>>fc)&1: x|=1<<c
        basis.append(x)
    for x in basis:                                   # exact verification
        assert all(bin(row & x).count('1')%2==0 for row in rows)
    return basis

def parity_torsion(nrays, bases):
    """tau != 0  <=>  1_B not in rowsp_GF2(M)  <=>  ker(M) contains an odd-weight vector.
    Returns (tau_nonzero, min-odd-weight certificate bitmask|None, ker_dim, odd-weight histogram)."""
    B=len(bases)
    rows=[sum(1<<bi for bi,b in enumerate(bases) if r in b) for r in range(nrays)]
    # test 1: reduce the all-ones vector against rowspace of M
    ones_vec=(1<<B)-1
    in_rowspace = gf2_reduce_vector(ones_vec, gf2_row_reduce(rows))==0
    # test 2 (independent): odd-weight vector in ker(M)
    ker=gf2_kernel(rows,B)
    dim=len(ker)
    tau_nonzero = any(bin(x).count('1')%2==1 for x in ker)
    assert tau_nonzero == (not in_rowspace), "GF(2) duality cross-check failed"
    cert=None; hist={}
    if tau_nonzero:
        assert dim<=22, f"kernel dim {dim} too large for exhaustive enumeration"
        x=0
        for g in range(1,1<<dim):                     # Gray-code walk over the whole kernel
            x^=ker[(g & -g).bit_length()-1]
            w=bin(x).count('1')
            if w%2==1:
                hist[w]=hist.get(w,0)+1
                if cert is None or w<bin(cert).count('1'): cert=x
    return tau_nonzero, cert, dim, hist

def verify_parity_certificate(cert, bases, nrays):
    """Check: |S| odd and every ray covered an even number of times => summing (KS1) over S
    mod 2 gives 0 = 1, an exact PROOF of KS-uncolorability (independent of any search)."""
    S=[bases[bi] for bi in range(len(bases)) if (cert>>bi)&1]
    deg=[0]*nrays
    for b in S:
        for r in b: deg[r]+=1
    return len(S)%2==1 and all(d%2==0 for d in deg), S, deg

# ---------------- tight-frame identity (mechanism behind SIC witnesses) ---------------------
def frame_constant(rays):
    """Exact check whether sum_i |v_i><v_i| = c*I; returns (True,c) or (False,None)."""
    d=len(rays[0])
    S=[[Fraction(0)]*d for _ in range(d)]
    for v in rays:
        n2=dot(v,v)
        for a in range(d):
            for b in range(d):
                S[a][b]+=Fraction(v[a]*v[b],n2)
    c=S[0][0]
    ok=all(S[a][b]==(c if a==b else Fraction(0)) for a in range(d) for b in range(d))
    return (ok, c if ok else None)

# ---------------- main -----------------------------------------------------------------------
def analyze(name, rays):
    d=len(rays[0]); n=len(rays)
    E=orthogonality_edges(rays)
    B=complete_bases(rays)
    covered=set()
    for b in B:
        for p in combinations(sorted(b),2): covered.add(p)
    free_edges=[e for e in E if e not in covered]
    cnt,example=count_ks_colorings(n,E,B,use_pairs=True)
    cnt1,_=count_ks_colorings(n,E,B,use_pairs=False)
    tau,cert,kdim,hist=parity_torsion(n,B)
    fr_ok,fr_c=frame_constant(rays)
    print(f"\n== {name}  (d={d}, V={n}, E={len(E)}, complete bases={len(B)}, "
          f"edges in no basis={len(free_edges)})")
    print(f"   KS colorings  (KS1+KS2): {cnt}     (KS1 only): {cnt1}   ->  "
          f"{'KS-COLORABLE, t0=0' if cnt>0 else 'KS-UNCOLORABLE, t0=1'}")
    if example is not None:
        assert verify_coloring(example,E,B)
        print(f"   example coloring (rays colored 1): {[i for i,c in enumerate(example) if c==1]}  [verified]")
    print(f"   parity torsion tau: {'NONZERO' if tau else 'zero'}   "
          f"(ker dim {kdim}; odd-weight kernel elements: "
          f"{sum(hist.values()) if hist else 0}{', min weight '+str(min(hist)) if hist else ''})")
    if cert is not None:
        ok,S,deg=verify_parity_certificate(cert,B,n)
        assert ok
        used=sorted({r for b in S for r in b})
        print(f"   parity certificate: {len(S)} bases (odd), every ray covered an even number "
              f"of times [verified] -> PROOF of uncolorability")
        print(f"   certificate uses {len(used)} distinct rays; bases: {S}")
    print(f"   tight-frame identity sum P_i = c*I: {'YES, c='+str(fr_c) if fr_ok else 'no'}")
    return dict(name=name,d=d,V=n,E=len(E),nB=len(B),free=len(free_edges),
                colorings=cnt,colorings_ks1=cnt1,t0=0 if cnt>0 else 1,tau=tau,
                cert=cert,kdim=kdim,hist=hist,frame=fr_c)

if __name__=="__main__":
    print("B1 TORSION LAYER — KS-colorability / forced-parity computations "
          "(exact, exhaustive)")
    print("="*96)

    results={}
    # controls
    onb=[(1,0,0),(0,1,0),(0,0,1)]
    results['ONB']=analyze("ONB d=3 (control)", onb)
    c5_edges=[(i,(i+1)%5) for i in range(5)]
    c5_cnt,_=count_ks_colorings(5,c5_edges,[],use_pairs=True)
    print(f"\n== KCBS C5 (d=3, V=5, E=5, complete bases=0)  [abstract orthogonality graph of the "
          f"umbrella realization: no triangles => no triads]")
    print(f"   KS colorings: {c5_cnt} (= number of independent sets of C5); trivially "
          f"KS-COLORABLE, t0=0; tau vacuously 0 (no bases)")

    # the two ends of the torsion axis
    results['YO']=analyze("Yu-Oh 13", yuoh_rays())
    results['P24']=analyze("Peres 24", peres24_rays())

    # independent brute force for Yu-Oh (2^13 full enumeration)
    yo=yuoh_rays(); Eyo=orthogonality_edges(yo); Byo=complete_bases(yo)
    bf =brute_count(13,Eyo,Byo,use_pairs=True)
    bf1=brute_count(13,Eyo,Byo,use_pairs=False)
    print(f"\n   [independent check] Yu-Oh full 2^13 enumeration: {bf} colorings (KS1+KS2), "
          f"{bf1} (KS1 only)")

    # optional cross-check of the rigidity layer (numerical; exact certificates live in
    # exact_rigidity.py) — import guarded so torsion results never depend on it
    flexes={}
    try:
        import flex_dimension as fd
        import io, contextlib
        buf=io.StringIO()
        with contextlib.redirect_stdout(buf):
            flexes['ONB']=fd.flex_dimension(fd.onb(3),name="ONB")
            flexes['C5'] =fd.flex_dimension(fd.odd_cycle(5),name="C5")
            flexes['YO'] =fd.flex_dimension(fd.yu_oh(),name="Yu-Oh")
            flexes['P24']=fd.flex_dimension(fd.peres24(),name="Peres24")
        print(f"\n   [cross-check] flex recomputed: ONB={flexes['ONB']} C5={flexes['C5']} "
              f"Yu-Oh={flexes['YO']} Peres24={flexes['P24']}  "
              f"(exact-arithmetic versions in exact_rigidity.py)")
    except Exception as e:
        print(f"\n   [cross-check] flex recomputation skipped ({e}); "
              f"exact values: Yu-Oh 0, Peres24 0 (exact_rigidity.py)")

    # ---------------- trichotomy table -------------------------------------------------------
    print("\n"+"="*96)
    print("TRICHOTOMY TABLE (dictionary cells; flex from flex_dimension/exact_rigidity)")
    print(f"{'scenario':<12}{'d':<3}{'flex':<6}{'t0 (KS-colorable?)':<22}{'#colorings':<12}"
          f"{'tau (parity)':<14}{'cell':<26}{'contextuality [lit.]'}")
    rows=[
      ("ONB",      3, 0, "0 (colorable)",   3,        "0",       "— (non-contextual)",   "none — control"),
      ("KCBS C5",  3, 2, "0 (colorable)",   c5_cnt,   "0 (vac.)","A: FLEXIBLE",          "state-DEPENDENT"),
      ("C7, C9",   3,"6,10","0 (colorable)","—",      "0 (vac.)","A: FLEXIBLE",          "state-DEPENDENT"),
      ("Yu-Oh 13", 3, 0, f"{results['YO']['t0']} ({'colorable' if results['YO']['t0']==0 else 'UNCOLORABLE'})",
                         results['YO']['colorings'],
                         "0 (Lem.B)","B: RIGID, torsion-free","state-INDEP., inequality"),
      ("Peres 24", 4, 0, f"{results['P24']['t0']} ({'colorable' if results['P24']['t0']==0 else 'UNCOLORABLE'})",
                         results['P24']['colorings'],
                         "NONZERO", "C: RIGID + TORSION",    "state-indep., AvN/parity"),
    ]
    for r in rows:
        print(f"{r[0]:<12}{r[1]:<3}{str(r[2]):<6}{r[3]:<22}{str(r[4]):<12}{r[5]:<14}{r[6]:<26}{r[7]}")

    # ---------------- PASS/FAIL --------------------------------------------------------------
    print("\n"+"="*96)
    checks=[
     ("Yu-Oh 13 has exactly 4 complete triads (z-triad + three z/y triads)",
        results['YO']['nB']==4),
     ("Yu-Oh 13 has 12 orthogonal pairs in no triad (the h-y edges; KS2 is contentful)",
        results['YO']['free']==12),
     ("Yu-Oh 13 is KS-COLORABLE (t0=0) — trichotomy's key empirical claim",
        results['YO']['t0']==0),
     ("Yu-Oh 13: backtracking count == independent 2^13 brute force (both rule sets)",
        results['YO']['colorings']==bf and results['YO']['colorings_ks1']==bf1),
     ("Yu-Oh 13: 24 colorings under (KS1+KS2), 192 under KS1-only (hand count: 12 z/y "
      "colorings x exactly-one-admissible-h x 2; 12 x 16 without KS2)",
        results['YO']['colorings']==24 and results['YO']['colorings_ks1']==192),
     ("Yu-Oh 13: tau = 0 (forced by Lemma B, d=3 odd) — machine-confirmed",
        results['YO']['tau']==False),
     ("Peres 24 is KS-UNCOLORABLE (t0=1), exhaustive backtracking, both rule sets",
        results['P24']['t0']==1 and results['P24']['colorings_ks1']==0),
     ("Peres 24: tau NONZERO — parity certificate found and verified (odd #bases, all "
      "ray-degrees even): an arithmetic PROOF of uncolorability, independent of the search",
        results['P24']['tau']==True and results['P24']['cert'] is not None),
     ("Peres 24: minimal parity certificate has 9 bases (reproduces the Cabello-Estebaranz-"
      "Garcia-Alcaine 18-ray / 9-basis parity proof inside Peres 24 [lit.])",
        results['P24']['hist'] and min(results['P24']['hist'])==9),
     ("Lemma A instance: tau != 0 and t0 = 1 agree on Peres 24; tau = 0 consistent with "
      "t0 = 0 on Yu-Oh",
        (results['P24']['tau'] and results['P24']['t0']==1) and
        ((not results['YO']['tau']) and results['YO']['t0']==0)),
     ("Both SIC sets are exact tight frames: sum P = (13/3) I (Yu-Oh), 6 I (Peres 24) — "
      "the operator identity behind state-independent witnesses",
        results['YO']['frame']==Fraction(13,3) and results['P24']['frame']==Fraction(6)),
     ("ONB control: single basis, 3 colorings; C5: colorable (11 = independent sets of C5)",
        results['ONB']['colorings']==3 and c5_cnt==11),
    ]
    if flexes:
        checks.append(("Rigidity cross-check: flex(Yu-Oh)=0, flex(Peres24)=0, flex(C5)=2",
                       flexes.get('YO')==0 and flexes.get('P24')==0 and flexes.get('C5')==2))
    allpass=True
    for msg,ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {msg}")
        allpass&=ok
    print("\n"+("torsion_layer PASS — both ends of the torsion axis pinned: "
          "Yu-Oh 13 = rigid & KS-colorable (torsion-free SIC, inequality-type); "
          "Peres 24 = rigid & KS-uncolorable with forced parity (torsion / AvN-type)."
          if allpass else "torsion_layer FAIL — inspect above"))
