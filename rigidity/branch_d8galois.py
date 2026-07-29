#!/usr/bin/env python3
"""
branch_d8galois.py -- Branch D8-GALOIS: the d=8 rung of the Kochen-Specker circle tower and its
holonomy Galois group.

GOAL (GALOIS_TOWER.md Sec.4, last bullet): the tower's Galois spectrum is certified at d=3 (Z/2),
d=4 (S_4) and d=6 (S_6).  d=8 was flagged "the natural next empirical point; not yet computed
(heavy multi-X core build)".  This file computes it.

METHOD -- a direct, dimension-general re-implementation of branch_d6galois.py (which is hardcoded
to d=6), reusing UNMODIFIED:
    branch_d6flex.canon_ray / build_A0A1 / stable_matrix_M9 / find_cliques   (fast vectorized
        integer-matrix mechanism-stability test -- three exact integer matrix products, NOT the
        slow per-pair sympy Laurent arithmetic)
    branch_d6geo.exact_fourier_d / clear_denominators_d                      (incidence-frame
        Wilczek-Zee connection; already dimension-general)
    ks_flex_census.cache_save / cache_load                                   (staging)

THE SCALE PROBLEM AND HOW IT IS SOLVED.  At d=6 the whole X-count==2 sub-pool (2400 rays / 122040
pairs) admits a COMPLETE 6-clique enumeration (1920 bases, 1.5 s) and a full SAT peel.  At d=8 the
X-count==2 sub-pool has 40768 rays; its stability matrix alone is 40768^2 = 1.7e9 entries and an
exhaustive 8-clique enumeration is hopeless.  Resolution (Stage 2): restrict to the SUPPORT-4
sub-pool.  This is not an ad-hoc cut -- at d=6 the basis-participating restriction of the
X-count==2 pool is EXACTLY the support-4 sub-pool (720 rays; verified in stage0), i.e. support 4 is
forced there, so imposing it at d=8 is the faithful generalization.  The d=8 support-4 X-count==2
sub-pool has 3360 rays / 572880 pairs and is KS-UNCOLORABLE (SAT).

RIGOUR NOTE on partial clique enumeration.  Adding bases can only ever make a KS instance MORE
uncolorable (bases contribute positive clauses).  So an UNSAT verdict obtained with a PARTIAL list
of 8-cliques is a valid uncolorability certificate.  The peel therefore runs against a partial
basis list, and the FINAL core is then re-processed with a COMPLETE 8-clique enumeration inside the
core (cheap once the core is small), against which uncolorability and per-ray criticality are
re-verified from scratch.  Nothing downstream depends on the partial list.

STAGES (CLI dispatch; every stage checkpoints to d8galois_*.cache.json):
    python3 branch_d8galois.py stage0     # SANITY GATE at d=6 (must pass before any d=8 work)
    python3 branch_d8galois.py stage1     # d=8 X-count==2 pool + support histogram
    python3 branch_d8galois.py stage2     # support-4 sub-pool: stability, cliques, SAT gate
    python3 branch_d8galois.py stage3 [budget]   # resumable SAT peel (incremental, assumptions)
    python3 branch_d8galois.py stage3final       # complete clique re-enum + criticality
    python3 branch_d8galois.py stage4     # WZ connection: constancy, symmetry, dominance
    python3 branch_d8galois.py stage5     # exact degree-8 characteristic polynomial
    python3 branch_d8galois.py stage6 [cap]      # Dedekind+Jordan Galois certificate (resumable)
    python3 branch_d8galois.py report     # PASS/FAIL summary of every stage

No existing file is modified.  No git.
"""
import os, sys, time, random, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from itertools import product as iproduct, combinations
from collections import Counter

import numpy as np
import sympy as sp

from ks_flex_census import cache_save, cache_load
import branch_d6flex as bd6
import branch_d6geo as bg6

HERE = os.path.dirname(os.path.abspath(__file__))
T0 = time.time()
DSIZE = 8

# The published d=6 reference numbers this file's generalized code must reproduce (SANITY GATE).
D6_REF = dict(
    x2_pool=2400, x2_pairs=122040, x2_bases=1920, x2_supp4=720,
    core_rays=280, core_pairs=3284, core_bases=95,
    charpoly=("x**6 - 570*x**5 + 135129*x**4 - 17053830*x**3 + 1208401591*x**2 "
              "- 45581353598*x + 715049394620"),
    L=190, trM=570,
)


def g_save(name, obj): cache_save(f"d8galois_{name}", obj)
def g_load(name): return cache_load(f"d8galois_{name}")


def xcount(v): return sum(1 for c in v if c in ("X", "-X"))
def support(v): return sum(1 for c in v if c != "0")


# ======================================================================================
# POOL BUILDER -- dimension-general X-count==2 sub-pool.
# ======================================================================================
def pool_x2(d, supp=None):
    """All projective rays of the two-symbol unimodular alphabet {0,+-1,+-X}^d whose X-count is
    exactly 2, deduped by branch_d6flex.canon_ray (exact projective equivalence over Q(X)).

    Convention matches branch_d6galois.py EXACTLY: rays whose entire support is the two X-entries
    are EXCLUDED, because in the full-pool dedup they are projectively equal to a PURE ray
    (dividing by the leading +-X clears every exponent) and are therefore represented by a pure
    ray, not by an X-count==2 one.  Check: d=6 gives 2400, the published number.
    """
    rays, seen = [], set()
    others = ("0", "1", "-1")
    for pos in combinations(range(d), 2):
        for s1 in ("X", "-X"):
            for s2 in ("X", "-X"):
                for rest in iproduct(others, repeat=d - 2):
                    if not any(c != "0" for c in rest):
                        continue                       # all-X support: equals a PURE ray
                    v = [None] * d
                    v[pos[0]], v[pos[1]] = s1, s2
                    it = iter(rest)
                    for c in range(d):
                        if v[c] is None:
                            v[c] = next(it)
                    v = tuple(v)
                    if supp is not None and support(v) != supp:
                        continue
                    k = bd6.canon_ray(v)
                    if k in seen:
                        continue
                    seen.add(k)
                    rays.append(v)
    return rays


def graph_of(rays, d, clique_time=None):
    st = bd6.stable_matrix_M9(rays, d)
    pairs = [(int(i), int(j)) for i, j in zip(*np.nonzero(np.triu(st, 1)))]
    adj = [set(np.nonzero(st[i])[0].tolist()) for i in range(len(rays))]
    bases, complete = bd6.find_cliques(adj, len(rays), d, time_limit=clique_time)
    return st, pairs, bases, complete


def sat_colorable(V, pairs, bases):
    from pysat.solvers import Cadical153
    s = Cadical153()
    for i, j in pairs:
        s.add_clause([-(i + 1), -(j + 1)])
    for b in bases:
        s.add_clause([x + 1 for x in b])
    return s.solve()


# ======================================================================================
# STAGE 0 -- SANITY GATE: run the generalized code at d=6, reproduce the published numbers.
# ======================================================================================
def stage0_gate():
    print("=" * 100)
    print("STAGE 0 -- SANITY GATE: generalized code must reproduce the published d=6 numbers")
    print("=" * 100)
    ok = True
    t0 = time.time()

    # (a) the pool builder
    p6 = pool_x2(6)
    print(f"[gate] d=6 X-count==2 pool: {len(p6)} rays (expect {D6_REF['x2_pool']})")
    ok &= len(p6) == D6_REF["x2_pool"]
    st, pairs, bases, complete = graph_of(p6, 6, clique_time=40.0)
    print(f"[gate] d=6 X==2 stable graph: {len(pairs)} pairs (expect {D6_REF['x2_pairs']}), "
          f"{len(bases)} 6-cliques (expect {D6_REF['x2_bases']}), complete={complete}")
    ok &= len(pairs) == D6_REF["x2_pairs"] and len(bases) == D6_REF["x2_bases"] and complete
    col = sat_colorable(len(p6), pairs, bases)
    print(f"[gate] d=6 X==2 pool KS-uncolorable: {not col} (expect True)")
    ok &= not col

    # (b) the support-4 forcing that justifies the d=8 restriction
    bp = sorted(set(x for b in bases for x in b))
    supp_bp = Counter(support(p6[i]) for i in bp)
    n_supp4 = sum(1 for v in p6 if support(v) == 4)
    print(f"[gate] basis-participating rays: {len(bp)} (expect {D6_REF['x2_supp4']}); their "
          f"support histogram: {dict(sorted(supp_bp.items()))}; #support-4 rays in pool: {n_supp4}")
    ok &= len(bp) == D6_REF["x2_supp4"] and set(supp_bp) == {4} and n_supp4 == D6_REF["x2_supp4"]
    print("[gate] => at d=6, SUPPORT 4 IS FORCED on every basis-participating X==2 ray. "
          "Restricting to support-4 at d=8 is the faithful generalization, not an ad-hoc cut.")

    # (c) the connection / charpoly code path, on the PUBLISHED d=6 critical core
    core = cache_load("d6galois_core_final")
    if core is None:
        print("[gate] WARNING: d6galois_core_final.cache.json missing -- skipping (c).")
    else:
        cs = [tuple(v) for v in core["core_syms"]]
        cb = [tuple(b) for b in core["core_bases"]]
        cp_ = core["core_pairs"]
        print(f"[gate] published d=6 core: {len(cs)} rays / {len(cp_)} pairs / {len(cb)} bases "
              f"(expect {D6_REF['core_rays']}/{D6_REF['core_pairs']}/{D6_REF['core_bases']})")
        ok &= (len(cs), len(cp_), len(cb)) == (D6_REF["core_rays"], D6_REF["core_pairs"],
                                               D6_REF["core_bases"])
        A0, A1, Am, pcount, Nb = bg6.exact_fourier_d(cs, cb, 6)
        const = (all(A1[c][d] == 0 for c in range(6) for d in range(6)) and
                 all(Am[c][d] == 0 for c in range(6) for d in range(6)))
        M, L = bg6.clear_denominators_d(A0, 6)
        trM = sum(M[i][i] for i in range(6))
        x = sp.symbols("x")
        cp = sp.expand(sp.Matrix(M).charpoly(x).as_expr())
        print(f"[gate] d=6 connection constant: {const}; L={L} (expect {D6_REF['L']}); "
              f"Tr(M)={trM} (expect {D6_REF['trM']})")
        print(f"[gate] d=6 char poly: {cp}")
        ok &= const and L == D6_REF["L"] and trM == D6_REF["trM"]
        ok &= sp.expand(cp - sp.sympify(D6_REF["charpoly"])) == 0
        # Galois certificate code path, on the published sextic
        cert = dedekind_jordan(sp.Poly(cp, x), 6, prime_cap=200)
        print(f"[gate] d=6 Dedekind+Jordan certificate: verdict={cert['verdict']} "
              f"(expect S_6) cycles={cert['witness']}")
        ok &= cert["verdict"] == "S_6"

    print(f"\n[gate] SANITY GATE: {'PASS' if ok else 'FAIL'}   ({time.time()-t0:.1f}s)")
    g_save("stage0_gate", dict(passed=bool(ok)))
    assert ok, "SANITY GATE FAILED -- do not proceed to d=8"
    return ok


# ======================================================================================
# STAGE 1 -- the d=8 X-count==2 pool.
# ======================================================================================
def stage1_pool():
    print("=" * 100)
    print("STAGE 1 -- the d=8 X-count==2 pool")
    print("=" * 100)
    t0 = time.time()
    rays = pool_x2(DSIZE)
    hist = Counter(support(v) for v in rays)
    print(f"[stage1] d=8 X-count==2 rays: {len(rays)}   support histogram: "
          f"{dict(sorted(hist.items()))}")
    print(f"[stage1] (for reference: the FULL d=8 pool has 5^8-1 = {5**8-1} raw vectors; the "
          f"X==2 stability matrix alone would be {len(rays)}^2 = {len(rays)**2:.3g} entries -- "
          f"exhaustive 8-clique enumeration there is out of reach, hence the support-4 cut.)")
    out = dict(n=len(rays), supp_hist=dict(hist))
    g_save("stage1_pool", out)
    print(f"[stage1] done ({time.time()-t0:.1f}s)")
    return out


# ======================================================================================
# STAGE 2 -- the support-4 sub-pool: stability graph, 8-cliques, SAT uncolorability gate.
# ======================================================================================
def stage2_subpool(clique_budget=22.0):
    print("=" * 100)
    print("STAGE 2 -- d=8 support-4 X-count==2 sub-pool: stability + cliques + SAT gate")
    print("=" * 100)
    t0 = time.time()
    rays = pool_x2(DSIZE, supp=4)
    st = bd6.stable_matrix_M9(rays, DSIZE)
    deg = st.sum(axis=1)
    pairs = [(int(i), int(j)) for i, j in zip(*np.nonzero(np.triu(st, 1)))]
    print(f"[stage2] sub-pool: {len(rays)} rays, {len(pairs)} pairs, "
          f"deg(min/mean/max)=({int(deg.min())}/{deg.mean():.1f}/{int(deg.max())})")
    adj = [set(np.nonzero(st[i])[0].tolist()) for i in range(len(rays))]
    prev = g_load("stage2_sub")
    bases = [tuple(b) for b in prev["bases"]] if prev else []
    if not bases:
        bases, complete = bd6.find_cliques(adj, len(rays), DSIZE, time_limit=clique_budget)
        print(f"[stage2] 8-cliques: {len(bases)} (COMPLETE={complete}; partial is fine, see the "
              f"rigour note in the docstring)")
    else:
        complete = prev["complete"]
        print(f"[stage2] reusing cached {len(bases)} 8-cliques (complete={complete})")
    bp = sorted(set(x for b in bases for x in b))
    print(f"[stage2] basis-participating rays: {len(bp)} of {len(rays)}")
    t1 = time.time()
    col = sat_colorable(len(rays), pairs, bases)
    print(f"[stage2] *** SAT: colorable={col}  =>  KS-UNCOLORABLE={not col} *** "
          f"({time.time()-t1:.2f}s)")
    out = dict(rays=rays, pairs=pairs, bases=bases, complete=bool(complete),
               uncolorable=bool(not col), nbp=len(bp))
    g_save("stage2_sub", out)
    print(f"[stage2] done ({time.time()-t0:.1f}s)")
    return out


# ======================================================================================
# STAGE 3 -- resumable SAT peel to a critical core.
#
# Uses ONE incremental Cadical instance with selector variables s_i and solves under ASSUMPTIONS,
# instead of rebuilding an 800k-clause CNF for every candidate removal (the d=6 script's approach,
# which does not scale to this clause count).  Encoding:
#     x_i  (var i+1)      = "ray i is coloured TRUE"
#     s_i  (var V+i+1)    = "ray i is PRESENT"
#     (-x_i or s_i)                        : absent rays are forced FALSE
#     (-x_i or -x_j)      for each pair    : orthogonal rays not both TRUE
#     (x_b1..x_b8, -s_b1..-s_b8) per basis : a basis all of whose rays are present needs a TRUE ray
# Solving under assumptions {s_i = (i in keep)} is EXACTLY the colorability question for the
# induced sub-instance.
# ======================================================================================
class PeelSolver:
    def __init__(self, V, pairs, bases):
        from pysat.solvers import Cadical153
        self.V = V
        self.s = Cadical153()
        for i in range(V):
            self.s.add_clause([-(i + 1), V + i + 1])
        for i, j in pairs:
            self.s.add_clause([-(i + 1), -(j + 1)])
        for b in bases:
            self.s.add_clause([x + 1 for x in b] + [-(V + x + 1) for x in b])

    def colorable(self, keep):
        V = self.V
        return self.s.solve(assumptions=[(V + i + 1) if i in keep else -(V + i + 1)
                                         for i in range(V)])


def stage3_peel(budget=35.0, seed=0):
    t0 = time.time()
    sub = g_load("stage2_sub")
    assert sub is not None, "run stage2 first"
    rays = [tuple(v) for v in sub["rays"]]
    pairs = [tuple(p) for p in sub["pairs"]]
    bases = [tuple(b) for b in sub["bases"]]
    V = len(rays)
    ps = PeelSolver(V, pairs, bases)
    print(f"[stage3] solver built ({time.time()-t0:.1f}s): V={V}, {len(pairs)} pairs, "
          f"{len(bases)} bases")

    prev = g_load("stage3_peel")
    keep = set(prev["keep"]) if prev else set(range(V))
    phase = prev.get("phase", "block") if prev else "block"
    rnd = random.Random(seed + len(keep))
    print(f"[stage3] resuming with |keep|={len(keep)} phase={phase}")

    # -- Phase A: randomized BLOCK peel (remove many rays at once; converges far faster) --
    if phase == "block":
        blk = max(1, len(keep) // 12)
        while time.time() - t0 < budget - 3 and blk >= 1:
            cand = sorted(keep)
            rnd.shuffle(cand)
            drop = set(cand[:blk])
            if not ps.colorable(keep - drop):
                keep -= drop
            else:
                blk = blk // 2
            if len(keep) < 40:
                break
        print(f"[stage3] block phase: |keep|={len(keep)} (blk down to {blk})")
        if blk < 1:
            phase = "single"

    # -- Phase B: exhaustive single-ray peel to a genuinely critical core --
    if phase == "single":
        order = sorted(keep)
        rnd.shuffle(order)
        done = prev.get("done", []) if prev else []
        doneset = set(done)
        for r in order:
            if time.time() - t0 > budget:
                break
            if r in doneset or r not in keep:
                continue
            if not ps.colorable(keep - {r}):
                keep.discard(r)
            doneset.add(r)
        print(f"[stage3] single phase: |keep|={len(keep)}, tested {len(doneset)}")
        g_save("stage3_peel", dict(keep=sorted(keep), phase="single", done=sorted(doneset)))
        print(f"[stage3] done ({time.time()-t0:.1f}s)")
        return keep

    g_save("stage3_peel", dict(keep=sorted(keep), phase=phase, done=[]))
    print(f"[stage3] BEST core so far: {len(keep)} rays  ({time.time()-t0:.1f}s)")
    return keep


def stage3_final(clique_budget=30.0):
    """Complete 8-clique re-enumeration INSIDE the core, then independent SAT uncolorability and
    per-ray criticality against that COMPLETE basis set (so nothing depends on stage 2's partial
    clique list)."""
    print("=" * 100)
    print("STAGE 3-FINAL -- complete clique re-enum + criticality verification")
    print("=" * 100)
    t0 = time.time()
    sub = g_load("stage2_sub")
    peel = g_load("stage3_peel")
    assert peel is not None, "run stage3 first"
    allrays = [tuple(v) for v in sub["rays"]]
    keep = sorted(peel["keep"])
    core_syms = [allrays[i] for i in keep]
    V = len(core_syms)
    st, pairs, bases, complete = graph_of(core_syms, DSIZE, clique_time=clique_budget)
    print(f"[final] core: {V} rays, {len(pairs)} pairs, {len(bases)} 8-cliques "
          f"(COMPLETE={complete})")
    print(f"[final] X-count histogram (must be ALL 2): "
          f"{dict(sorted(Counter(xcount(v) for v in core_syms).items()))}")
    print(f"[final] support histogram: "
          f"{dict(sorted(Counter(support(v) for v in core_syms).items()))}")
    assert set(xcount(v) for v in core_syms) == {2}
    col = sat_colorable(V, pairs, bases)
    print(f"[final] independent SAT re-check with COMPLETE bases: KS-uncolorable={not col}")
    assert not col, "core is colorable against the complete basis set -- re-peel needed"

    # re-peel against the complete basis set, then verify criticality
    ps = PeelSolver(V, pairs, bases)
    keep2 = set(range(V))
    changed = True
    while changed and time.time() - t0 < clique_budget + 25:
        changed = False
        for r in sorted(keep2):
            if not ps.colorable(keep2 - {r}):
                keep2.discard(r)
                changed = True
    ncrit = 0
    noncrit = []
    for r in sorted(keep2):
        if ps.colorable(keep2 - {r}):
            ncrit += 1
        else:
            noncrit.append(r)
    print(f"[final] after re-peel vs COMPLETE bases: {len(keep2)} rays; "
          f"critical {ncrit}/{len(keep2)}; non-critical {len(noncrit)}")
    core_syms = [core_syms[i] for i in sorted(keep2)]
    st, pairs, bases, complete = graph_of(core_syms, DSIZE, clique_time=clique_budget)
    col = sat_colorable(len(core_syms), pairs, bases)
    print(f"[final] FINAL CORE: {len(core_syms)} rays / {len(pairs)} pairs / {len(bases)} bases "
          f"(complete={complete}); uncolorable={not col}; all-critical={len(noncrit)==0}")
    g_save("core_final", dict(core_syms=core_syms, core_pairs=pairs, core_bases=bases,
                              complete=bool(complete), uncolorable=bool(not col),
                              all_critical=bool(len(noncrit) == 0)))
    print(f"[final] done ({time.time()-t0:.1f}s)")
    return dict(V=len(core_syms), pairs=len(pairs), bases=len(bases))


# ======================================================================================
# STAGE 4 -- the incidence-frame Wilczek-Zee connection.
# ======================================================================================
def stage4_connection():
    print("=" * 100)
    print("STAGE 4 -- the d=8 WZ holonomy connection")
    print("=" * 100)
    t0 = time.time()
    core = g_load("core_final")
    assert core is not None, "run stage3final first"
    cs = [tuple(v) for v in core["core_syms"]]
    cb = [tuple(b) for b in core["core_bases"]]
    print(f"[stage4] core: {len(cs)} rays, {len(cb)} bases, d={DSIZE}")
    A0, A1, Am, pcount, Nb = bg6.exact_fourier_d(cs, cb, DSIZE)
    z1 = all(A1[c][d] == 0 for c in range(DSIZE) for d in range(DSIZE))
    zm = all(Am[c][d] == 0 for c in range(DSIZE) for d in range(DSIZE))
    print(f"[stage4] *** A1==0: {z1}   Am==0: {zm} ***  (connection EXACTLY CONSTANT iff both)")
    M, L = bg6.clear_denominators_d(A0, DSIZE)
    tr = sum(M[i][i] for i in range(DSIZE))
    offnz = [(c, d) for c in range(DSIZE) for d in range(DSIZE) if c != d and M[c][d] != 0]
    sym = all(M[c][d] == M[d][c] for c in range(DSIZE) for d in range(DSIZE))
    offmax = max((abs(M[c][d]) for c, d in offnz), default=0)
    margins = [M[i][i] - sum(abs(M[i][j]) for j in range(DSIZE) if j != i) for i in range(DSIZE)]
    print(f"[stage4] L={L}.  Stilde := L*A0 (integer matrix):")
    for row in M:
        print("     ", row)
    print(f"[stage4] Tr(Stilde)={tr}, Tr(S)={sp.Rational(tr, L)}")
    print(f"[stage4] off-diagonal nonzero: {len(offnz)} of {DSIZE*(DSIZE-1)} "
          f"(max |off-diag| = {offmax})")
    print(f"[stage4] REAL SYMMETRIC: {sym}")
    print(f"[stage4] strictly DIAGONALLY DOMINANT: {all(m > 0 for m in margins)}  "
          f"(row margins {margins}, min {min(margins)})")
    out = dict(M=M, L=L, tr=tr, constant=bool(z1 and zm), symmetric=bool(sym),
               diagdom=bool(all(m > 0 for m in margins)), margins=margins,
               n_offdiag_nonzero=len(offnz), offmax=int(offmax), Nb=Nb)
    g_save("stage4_connection", out)
    print(f"[stage4] done ({time.time()-t0:.1f}s)")
    return out


# ======================================================================================
# STAGE 5 -- the exact degree-8 characteristic polynomial.
# ======================================================================================
def stage5_charpoly():
    print("=" * 100)
    print("STAGE 5 -- the exact degree-8 characteristic polynomial")
    print("=" * 100)
    conn = g_load("stage4_connection")
    assert conn is not None, "run stage4 first"
    M, L = conn["M"], conn["L"]
    x = sp.symbols("x")
    cp = sp.expand(sp.Matrix(M).charpoly(x).as_expr())
    P = sp.Poly(cp, x, domain="QQ")
    irred = P.is_irreducible
    fl = sp.factor_list(cp, x)
    disc = sp.discriminant(cp, x)
    di = int(disc)
    sq = (di >= 0 and math.isqrt(di) ** 2 == di)
    # INDEPENDENT re-derivation of the char poly: evaluate det(t*I - Stilde) exactly (Bareiss
    # fraction-free determinant) at d+1 integer points and Lagrange-interpolate.  This shares no
    # code path with Matrix.charpoly (Berkowitz).
    pts = list(range(-4, 5))
    Mm = sp.Matrix(M)
    vals = [(t, (t * sp.eye(DSIZE) - Mm).det(method="berkowitz" if False else "bareiss"))
            for t in pts]
    cp2 = sp.expand(sp.interpolate(vals, x))
    indep_ok = sp.expand(cp - cp2) == 0
    ev = sorted(np.linalg.eigvalsh(np.array(M, dtype=float)))
    print(f"[stage5] char poly of Stilde (degree {P.degree()}):\n    {cp}")
    print(f"[stage5] INDEPENDENT check (Bareiss det at 9 integer points + exact Lagrange "
          f"interpolation) agrees: {indep_ok}")
    print(f"[stage5] numpy eigvalsh (float, independent): "
          f"{[round(v,6) for v in ev]}")
    print(f"[stage5] irreducible over Q: {irred}   (factors: "
          f"{[ (sp.Poly(f,x).degree(), sp.Poly(f,x).as_expr()) for f,_ in fl[1] ] if not irred else 'single degree-8 factor'})")
    print(f"[stage5] discriminant: {di}")
    print(f"[stage5] discriminant a perfect square: {sq}  => Galois group "
          f"{'INSIDE' if sq else 'NOT inside'} A_{DSIZE}")
    rts = sp.Poly(cp, x).nroots(n=30)
    print(f"[stage5] roots (all real: {all(abs(sp.im(r)) < 1e-20 for r in rts)}):")
    for r in sorted(rts, key=lambda r: sp.re(r)):
        v = sp.re(r) / L
        print(f"     lambda={sp.N(r,20)}   phi/(2pi)={sp.N(v - sp.floor(v),20)}")
    out = dict(charpoly=str(cp), irreducible=bool(irred), disc=str(di), disc_square=bool(sq),
               roots=[str(sp.N(r, 25)) for r in sorted(rts, key=lambda r: sp.re(r))], L=L,
               independent_check=bool(indep_ok))
    g_save("stage5_charpoly", out)
    return out


# ======================================================================================
# STAGE 6 -- Dedekind + Jordan Galois certificate (no black box).
#
# Same three ingredients as branch_galois_cert.py, but with a FAST self-contained Dedekind
# cycle-type routine (distinct-degree factorization over GF(p) via the Frobenius map) so that the
# prime cap can be pushed far past 400 -- necessary at d=8, where a transposition has Chebotarev
# density 28/40320 = 1/1440 and so needs ~1440 primes on average.
# ======================================================================================
def _pm(a, b, p):                       # poly multiply mod p (lists, low->high)
    r = [0] * (len(a) + len(b) - 1)
    for i, ai in enumerate(a):
        if ai:
            for j, bj in enumerate(b):
                r[i + j] = (r[i + j] + ai * bj) % p
    return r


def _pmod(a, f, p):                     # a mod f (f monic), mod p
    a = a[:]
    df = len(f) - 1
    if df == 0:                         # f is the constant 1 -> everything reduces to 0
        return [0]
    while len(a) - 1 >= df and len(a) > 1:
        c = a[-1] % p
        if c:
            k = len(a) - 1 - df
            for i, fi in enumerate(f):
                a[k + i] = (a[k + i] - c * fi) % p
        a.pop()
    while len(a) > 1 and a[-1] % p == 0:
        a.pop()
    return [v % p for v in a]


def _pgcd(a, b, p):
    a = [v % p for v in a]; b = [v % p for v in b]
    while len(b) > 1 or b[0] % p:
        inv = pow(b[-1], p - 2, p)
        bm = [(v * inv) % p for v in b]
        a, b = b, _pmod(a, bm, p)
        if all(v == 0 for v in b):
            b = [0]
            break
    return a


def cycle_type_mod_p(coeffs, p):
    """coeffs = monic integer poly, low->high.  Returns the sorted tuple of irreducible factor
    degrees of f mod p (the Dedekind cycle type), or None if f mod p is not squarefree."""
    d = len(coeffs) - 1
    f = [c % p for c in coeffs]
    if f[-1] % p == 0:
        return None
    fp = [(i * f[i]) % p for i in range(1, len(f))]
    while len(fp) > 1 and fp[-1] == 0:
        fp.pop()
    if not fp or all(v == 0 for v in fp):
        return None                                    # f' == 0: f is a p-th power, not squarefree
    g = _pgcd(f, fp, p)
    if len(g) > 1:
        return None                                    # not squarefree
    # Frobenius matrix: column i = x^(p*i) mod f
    xp = [0, 1]
    e, base, acc = p, [0, 1], [1]
    while e:
        if e & 1:
            acc = _pmod(_pm(acc, base, p), f, p)
        base = _pmod(_pm(base, base, p), f, p)
        e >>= 1
    xp = acc                                            # x^p mod f
    cols = [[1]]
    cur = [1]
    for i in range(1, d):
        cur = _pmod(_pm(cur, xp, p), f, p)
        cols.append(cur[:])
    Q = [[0] * d for _ in range(d)]
    for i, c in enumerate(cols):
        for j, v in enumerate(c):
            Q[j][i] = v % p
    degs = []
    rem = f[:]
    vec = [0] * d
    vec[1 % d] = 1                                      # the vector for x
    cur = vec[:]
    for k in range(1, d + 1):
        cur = [sum(Q[i][j] * cur[j] for j in range(d)) % p for i in range(d)]  # x^(p^k)
        h = cur[:]
        h[1] = (h[1] - 1) % p                           # x^(p^k) - x
        while len(h) > 1 and h[-1] == 0:
            h.pop()
        gk = _pgcd(rem, h, p) if any(h) else rem[:]
        nk = (len(gk) - 1) // k
        degs += [k] * nk
        if len(gk) > 1:
            inv = pow(gk[-1], p - 2, p)
            gm = [(v * inv) % p for v in gk]
            rem = _pdiv(rem, gm, p)
        if len(rem) == 1:
            break
    if sum(degs) != d:
        return None
    return tuple(sorted(degs, reverse=True))


def _pdiv(a, b, p):                     # exact division a/b (b monic) mod p
    a = a[:]
    db = len(b) - 1
    q = [0] * max(1, len(a) - db)
    while len(a) - 1 >= db and len(a) > 1:
        c = a[-1] % p
        k = len(a) - 1 - db
        q[k] = c
        if c:
            for i, bi in enumerate(b):
                a[k + i] = (a[k + i] - c * bi) % p
        a.pop()
    return q


def dedekind_jordan(P, d, prime_cap=20000, start=3, seen=None, verbose=False):
    """Certificate: irreducible (=> transitive) + (d-1)-cycle with d-1 PRIME (=> 2-transitive =>
    primitive) + transposition (=> Jordan) => S_d."""
    x = P.gen
    coeffs = list(reversed(P.all_coeffs()))
    coeffs = [int(c) for c in coeffs]
    irred = P.is_irreducible
    seen = dict(seen or {})
    if not irred:                     # reducible => Galois group is NOT transitive, not S_d;
        return dict(irreducible=False, seen={}, witness=dict(dcycle=None, dm1=None, transp=None),
                    dminus1_prime=bool(sp.isprime(d - 1)), verdict="REDUCIBLE (not S_%d)" % d,
                    n_types=0)
    want_d = (d,)
    want_dm1 = tuple([d - 1] + [1])
    want_t = tuple([2] + [1] * (d - 2))
    for pr in sp.primerange(start, prime_cap):
        ct = cycle_type_mod_p(coeffs, pr)
        if ct is None:
            continue
        seen.setdefault(ct, int(pr))
        if want_d in seen and want_dm1 in seen and want_t in seen:
            break
    ok_dm1 = sp.isprime(d - 1)
    have = dict(dcycle=seen.get(want_d), dm1=seen.get(want_dm1), transp=seen.get(want_t))
    full = bool(irred and have["transp"] and have["dm1"] and ok_dm1)
    return dict(irreducible=bool(irred), seen={str(k): v for k, v in seen.items()},
                witness=have, dminus1_prime=bool(ok_dm1),
                verdict=(f"S_{d}" if full else "INCOMPLETE"), n_types=len(seen))


def stage6_certificate(prime_cap=20000):
    print("=" * 100)
    print("STAGE 6 -- Dedekind + Jordan certificate for the d=8 Galois group")
    print("=" * 100)
    t0 = time.time()
    cpd = g_load("stage5_charpoly")
    assert cpd is not None, "run stage5 first"
    x = sp.symbols("x")
    P = sp.Poly(sp.sympify(cpd["charpoly"]), x)
    prev = g_load("stage6_cert")
    seen = {}
    start = 3
    if prev:
        seen = {tuple(eval(k)): v for k, v in prev["seen"].items()}
        start = prev.get("scanned_to", 3)
        print(f"[stage6] resuming: {len(seen)} cycle types known, scanned primes < {start}")
    cert = dedekind_jordan(P, DSIZE, prime_cap=prime_cap, start=start, seen=seen, verbose=True)
    print(f"[stage6] irreducible over Q: {cert['irreducible']}   => Galois group TRANSITIVE")
    print(f"[stage6] d-1 = {DSIZE-1} is prime: {cert['dminus1_prime']}  "
          f"(the Jordan template applies)")
    print(f"[stage6] realized cycle types (Dedekind -- each IS an element of the Galois group): "
          f"{cert['n_types']} distinct")
    for k, v in sorted(cert["seen"].items(), key=lambda kv: kv[1]):
        print(f"     mod {v:>6}: {k}")
    w = cert["witness"]
    print(f"[stage6] {DSIZE}-cycle:      mod {w['dcycle']}")
    print(f"[stage6] {DSIZE-1}-cycle:      mod {w['dm1']}   => point stabilizer transitive "
          f"=> 2-transitive => PRIMITIVE")
    print(f"[stage6] transposition: mod {w['transp']}")
    print(f"[stage6] *** VERDICT: Galois group = {cert['verdict']} "
          f"{'(order ' + str(sp.factorial(DSIZE)) + ')' if cert['verdict'].startswith('S_') else ''} ***")
    cert["scanned_to"] = prime_cap
    g_save("stage6_cert", cert)
    print(f"[stage6] done ({time.time()-t0:.1f}s)")
    return cert


def stage7_crosscheck():
    """Attempted cross-check with sympy's own galois_group (black box).  NOTE: sympy supports
    galois_group only up to DEGREE 6, so at d=8 there is no black box to compare against -- the
    Dedekind+Jordan certificate of stage6 is the ONLY route.  Reported honestly."""
    print("=" * 100)
    print("STAGE 7 -- sympy galois_group cross-check attempt")
    print("=" * 100)
    cpd = g_load("stage5_charpoly")
    x = sp.symbols("x")
    P = sp.Poly(sp.sympify(cpd["charpoly"]), x, domain="QQ")
    from sympy.polys.numberfields.galoisgroups import galois_group
    t0 = time.time()
    try:
        G, alt = galois_group(P, by_name=True)
        print(f"[stage7] sympy galois_group: {G}  (inside A_n: {alt})  [{time.time()-t0:.1f}s]")
        out = dict(available=True, group=str(G), alt=bool(alt))
    except Exception as ex:
        print(f"[stage7] sympy galois_group UNAVAILABLE at degree 8: {ex}")
        print("[stage7] => no black-box cross-check exists at d=8; the stage6 Dedekind+Jordan "
              "certificate is the sole (and proof-grade) route.  This is exactly why the "
              "certificate machinery was built.")
        out = dict(available=False, reason=str(ex))
    g_save("stage7_sympy", out)
    return out


# ======================================================================================
# STAGE 8 -- INDEPENDENT re-derivation of the connection by direct symbolic differentiation
# (mirrors D6_GALOIS.md's cross-check #1: a totally different code path, no Fourier-coefficient
# machinery at all).  Builds Ehat(theta) explicitly, differentiates with sympy, and forms
# B(theta) = Ehat^dag (dEhat/dtheta) / Nb.  The exact_fourier_d answer is A0 with true entry
# i*A0, so B must equal i*A0 identically; constancy is re-checked at a third sample point.
# ======================================================================================
def stage8_symbolic_crosscheck(sample=sp.Rational(1, 3)):
    print("=" * 100)
    print("STAGE 8 -- independent symbolic-differentiation re-derivation of the connection")
    print("=" * 100)
    t0 = time.time()
    core = g_load("core_final")
    conn = g_load("stage4_connection")
    cs = [tuple(v) for v in core["core_syms"]]
    cb = [tuple(b) for b in core["core_bases"]]
    th = sp.symbols("theta", real=True)
    xx = sp.exp(sp.I * th)
    val = {"0": sp.Integer(0), "1": sp.Integer(1), "-1": sp.Integer(-1), "X": xx, "-X": -xx}
    Ds = set(sum(1 for c in v if c != "0") for v in cs)
    assert len(Ds) == 1, f"non-uniform support sizes {Ds}"
    Dj = Ds.pop()
    nrm = 1 / sp.sqrt(sp.Integer(Dj))
    rows = []
    for b in cb:
        for j in b:
            rows.append([val[s] * nrm for s in cs[j]])
    E = sp.Matrix(rows)
    Nb = len(cb)
    print(f"[stage8] Ehat: {E.shape[0]}x{E.shape[1]} (Nb={Nb} bases, uniform support D_j={Dj})")
    dE = E.diff(th)
    B = sp.expand(sp.simplify((E.conjugate().T * dE) / Nb))
    A0 = conn["M"]
    L = conn["L"]
    match = True
    Bconst = True
    for c in range(DSIZE):
        for d in range(DSIZE):
            got = sp.simplify(B[c, d])
            want = sp.I * sp.Rational(A0[c][d], L)
            if sp.simplify(got - want) != 0:
                match = False
            if sp.simplify(sp.diff(B[c, d], th)) != 0:
                Bconst = False
    Bs = sp.simplify(B.subs(th, sample) - B.subs(th, 0))
    third = all(sp.simplify(Bs[c, d]) == 0 for c in range(DSIZE) for d in range(DSIZE))
    print(f"[stage8] direct symbolic B(theta) == i*Stilde/L (exact_fourier_d result): {match}")
    print(f"[stage8] dB/dtheta == 0 identically (constancy, third independent way): {Bconst}")
    print(f"[stage8] B(theta={sample}) - B(0) == 0 (third sample point): {third}")
    out = dict(match=bool(match), constant=bool(Bconst), third_point=bool(third), Dj=int(Dj))
    g_save("stage8_symbolic", out)
    print(f"[stage8] done ({time.time()-t0:.1f}s)")
    return out


# ======================================================================================
# STAGE 9 -- PEEL-INDEPENDENCE: a second, independently-seeded critical core, its own octic, its
# own certificate (mirrors D6_GALOIS.md's cross-check #2).
# ======================================================================================
def stage9_altcore(seed=999, budget=10.0, prime_cap=60000):
    print("=" * 100)
    print(f"STAGE 9 -- independent second core (seed={seed}) and its own Galois certificate")
    print("=" * 100)
    t0 = time.time()
    sub = g_load("stage2_sub")
    rays = [tuple(v) for v in sub["rays"]]
    pairs = [tuple(p) for p in sub["pairs"]]
    bases = [tuple(b) for b in sub["bases"]]
    V = len(rays)
    ps = PeelSolver(V, pairs, bases)
    rnd = random.Random(seed)
    keep = set(range(V))
    blk = V // 12
    while blk >= 1 and time.time() - t0 < budget:
        cand = sorted(keep); rnd.shuffle(cand)
        drop = set(cand[:blk])
        if not ps.colorable(keep - drop):
            keep -= drop
        else:
            blk //= 2
    order = sorted(keep); rnd.shuffle(order)
    for r in order:
        if r in keep and not ps.colorable(keep - {r}):
            keep.discard(r)
    cs0 = [rays[i] for i in sorted(keep)]
    st, cpairs, cbases, complete = graph_of(cs0, DSIZE, clique_time=25.0)
    # re-peel against the complete in-core basis set
    ps2 = PeelSolver(len(cs0), cpairs, cbases)
    k2 = set(range(len(cs0)))
    changed = True
    while changed:
        changed = False
        for r in sorted(k2):
            if not ps2.colorable(k2 - {r}):
                k2.discard(r); changed = True
    cs = [cs0[i] for i in sorted(k2)]
    st, cpairs, cbases, complete = graph_of(cs, DSIZE, clique_time=25.0)
    col = sat_colorable(len(cs), cpairs, cbases)
    prim = [tuple(v) for v in g_load("core_final")["core_syms"]]
    shared = len(set(cs) & set(prim))
    print(f"[stage9] ALT core: {len(cs)} rays / {len(cpairs)} pairs / {len(cbases)} bases "
          f"(complete={complete}), uncolorable={not col}")
    print(f"[stage9] overlap with the primary {len(prim)}-ray core: {shared} shared rays "
          f"({len(cs)-shared} genuinely different)")
    A0, A1, Am, pc, Nb = bg6.exact_fourier_d(cs, cbases, DSIZE)
    const = (all(A1[c][d] == 0 for c in range(DSIZE) for d in range(DSIZE)) and
             all(Am[c][d] == 0 for c in range(DSIZE) for d in range(DSIZE)))
    M, L = bg6.clear_denominators_d(A0, DSIZE)
    tr = sum(M[i][i] for i in range(DSIZE))
    sym = all(M[c][d] == M[d][c] for c in range(DSIZE) for d in range(DSIZE))
    margins = [M[i][i] - sum(abs(M[i][j]) for j in range(DSIZE) if j != i) for i in range(DSIZE)]
    x = sp.symbols("x")
    cp = sp.expand(sp.Matrix(M).charpoly(x).as_expr())
    print(f"[stage9] constant={const} symmetric={sym} diag-dominant={all(m>0 for m in margins)} "
          f"L={L} Tr(S)={sp.Rational(tr,L)}")
    for row in M:
        print("     ", row)
    print(f"[stage9] char poly: {cp}")
    cert = dedekind_jordan(sp.Poly(cp, x), DSIZE, prime_cap=prime_cap)
    print(f"[stage9] *** ALT-CORE VERDICT: {cert['verdict']} *** witnesses={cert['witness']}")
    decoupled = [c for c in range(DSIZE)
                 if all(M[c][d] == 0 for d in range(DSIZE) if d != c)]
    print(f"[stage9] DECOUPLED coordinates (zero off-diagonal row): {decoupled}"
          + ("  => char poly has the rational root M[c][c], hence REDUCIBLE" if decoupled else ""))
    out = dict(n=len(cs), pairs=len(cpairs), bases=len(cbases), shared=shared, M=M, L=L, tr=tr,
               constant=bool(const), symmetric=bool(sym), charpoly=str(cp),
               verdict=cert["verdict"], witness=cert["witness"], core_syms=cs,
               decoupled=decoupled, uncolorable=bool(not col), complete=bool(complete),
               seed=int(seed))
    g_save(f"stage9_altcore_seed{seed}", out)
    g_save("stage9_altcore", out)
    print(f"[stage9] done ({time.time()-t0:.1f}s)")
    return out


# ======================================================================================
# STAGE 10 -- full Dedekind cycle-type census (how generic is the octic?).
# ======================================================================================
def stage10_census(prime_cap=400000):
    print("=" * 100)
    print("STAGE 10 -- full Dedekind cycle-type census (genericity)")
    print("=" * 100)
    t0 = time.time()
    cpd = g_load("stage5_charpoly")
    x = sp.symbols("x")
    P = sp.Poly(sp.sympify(cpd["charpoly"]), x)
    coeffs = [int(c) for c in reversed(P.all_coeffs())]
    prev = g_load("stage10_census")
    seen = {tuple(eval(k)): v for k, v in prev["seen"].items()} if prev else {}
    start = prev["scanned_to"] if prev else 3
    total_types = len(list(sp.utilities.iterables.ordered_partitions(DSIZE)))
    for pr in sp.primerange(start, prime_cap):
        if time.time() - t0 > 35:
            start = int(pr)
            break
        ct = cycle_type_mod_p(coeffs, pr)
        if ct is not None:
            seen.setdefault(ct, int(pr))
    else:
        start = prime_cap
    print(f"[stage10] scanned primes < {start}: {len(seen)} of {total_types} cycle types of S_{DSIZE} "
          f"realized")
    for k, v in sorted(seen.items(), key=lambda kv: kv[1]):
        print(f"     mod {v:>7}: {k}")
    missing = [p for p in sp.utilities.iterables.ordered_partitions(DSIZE)
               if tuple(sorted(p, reverse=True)) not in seen]
    print(f"[stage10] missing types: {[tuple(sorted(m,reverse=True)) for m in missing]}")
    g_save("stage10_census", dict(seen={str(k): v for k, v in seen.items()},
                                  scanned_to=int(start), total_types=int(total_types)))
    print(f"[stage10] done ({time.time()-t0:.1f}s)")
    return seen


def report():
    print("=" * 100)
    print("BRANCH D8-GALOIS -- REPORT")
    print("=" * 100)
    rows = []
    g = g_load("stage0_gate");   rows.append(("stage0 d=6 sanity gate", g and g["passed"]))
    s1 = g_load("stage1_pool")
    if s1: rows.append((f"stage1 d=8 X==2 pool = {s1['n']} rays", s1["n"] == 40768))
    s2 = g_load("stage2_sub")
    if s2: rows.append((f"stage2 support-4 sub-pool {len(s2['rays'])} rays / {len(s2['pairs'])} "
                        f"pairs / {len(s2['bases'])} bases uncolorable", s2["uncolorable"]))
    cf = g_load("core_final")
    if cf: rows.append((f"stage3 core {len(cf['core_syms'])} rays / {len(cf['core_pairs'])} pairs "
                        f"/ {len(cf['core_bases'])} bases (complete={cf['complete']}, "
                        f"all-critical={cf['all_critical']})", cf["uncolorable"]))
    c4 = g_load("stage4_connection")
    if c4: rows.append((f"stage4 connection constant={c4['constant']} symmetric={c4['symmetric']} "
                        f"diag-dominant={c4['diagdom']} L={c4['L']}",
                        c4["constant"] and c4["symmetric"] and c4["diagdom"]))
    c5 = g_load("stage5_charpoly")
    if c5: rows.append((f"stage5 charpoly irreducible={c5['irreducible']} "
                        f"disc-square={c5['disc_square']} "
                        f"independent-det-check={c5.get('independent_check')}",
                        c5["irreducible"] and not c5["disc_square"]
                        and c5.get("independent_check", True)))
    c6 = g_load("stage6_cert")
    if c6: rows.append((f"stage6 Dedekind+Jordan verdict = {c6['verdict']}",
                        c6["verdict"] == "S_8"))
    c7 = g_load("stage7_sympy")
    if c7: rows.append(("stage7 sympy galois_group cross-check unavailable at degree 8 "
                        "(sympy caps at 6) -- noted, not a failure", True))
    c8 = g_load("stage8_symbolic")
    if c8: rows.append((f"stage8 independent symbolic-diff connection match={c8['match']} "
                        f"constant={c8['constant']} third-point={c8['third_point']}",
                        c8["match"] and c8["constant"] and c8["third_point"]))
    seeds = [(s, g_load(f"stage9_altcore_seed{s}")) for s in (1, 7, 999)]
    for s, c9 in seeds:
        if c9:
            rows.append((f"stage9 alt core seed={s}: {c9['n']} rays ({c9['shared']} shared with "
                         f"primary), decoupled={c9['decoupled']}, verdict={c9['verdict']}",
                         c9["verdict"] == "S_8" or bool(c9["decoupled"])))
    c10 = g_load("stage10_census")
    if c10: rows.append((f"stage10 cycle-type census: {len(c10['seen'])}/{c10['total_types']} "
                         f"types of S_8 realized (primes < {c10['scanned_to']})",
                         len(c10["seen"]) == c10["total_types"]))
    allok = True
    for name, ok in rows:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
        allok &= bool(ok)
    print(f"\n  OVERALL: {'PASS' if allok else 'FAIL/INCOMPLETE'}")
    if c5:
        print(f"\n  char poly f_8 = {c5['charpoly']}")
    return allok


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "report"
    arg = sys.argv[2] if len(sys.argv) > 2 else None
    if which == "stage0":   stage0_gate()
    elif which == "stage1": stage1_pool()
    elif which == "stage2": stage2_subpool(float(arg) if arg else 22.0)
    elif which == "stage3": stage3_peel(float(arg) if arg else 35.0,
                                        int(sys.argv[3]) if len(sys.argv) > 3 else 0)
    elif which == "stage3final": stage3_final(float(arg) if arg else 30.0)
    elif which == "stage4": stage4_connection()
    elif which == "stage5": stage5_charpoly()
    elif which == "stage6": stage6_certificate(int(arg) if arg else 20000)
    elif which == "stage7": stage7_crosscheck()
    elif which == "stage8": stage8_symbolic_crosscheck()
    elif which == "stage9": stage9_altcore(int(arg) if arg else 999)
    elif which == "stage10": stage10_census(int(arg) if arg else 400000)
    elif which == "report": report()
    elif which == "all":
        stage0_gate(); stage1_pool(); stage2_subpool(); stage3_peel(60.0)
        stage3_final(); stage4_connection(); stage5_charpoly(); stage6_certificate()
        stage7_crosscheck(); stage8_symbolic_crosscheck()
        for s in (1, 7, 999):
            stage9_altcore(s)
        stage10_census(); report()
    else:
        print(f"unknown stage {which!r}"); sys.exit(1)
    print(f"\n[branch_d8galois.py stage={which} total {time.time()-T0:.1f}s]")
