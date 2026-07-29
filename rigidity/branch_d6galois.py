#!/usr/bin/env python3
"""
branch_d6galois.py -- Branch D6-GALOIS: does a genuinely non-diagonal (multi-X) d=6 holonomy
Galois core exist?

QUESTION (D6_GALOIS.md). D6_GEOMETRY.md PROVED (Sec.2.1) a clean lemma: any mechanism-stable core
built ONLY from X-count<=1 rays has a DIAGONAL incidence-frame WZ connection A0 (no ray carries
exponent-1 on two coordinates simultaneously => no off-diagonal Fourier-degree-0 term can ever be
nonzero) => the holonomy Galois group is trivial BY CONSTRUCTION, not by luck. Every d=6
uncolorable stable core found so far (D6_CIRCLE.md, D6_GEOMETRY.md) used ONLY X-count<=1 rays,
purely because that was the only pool judged SAT/clique-tractable at the time. THE QUESTION: does a
d=6 mechanism-stable KS-uncolorable core exist that USES rays with >=2 X-entries (X-count>=2),
with those rays genuinely LOAD-BEARING (not decorative), giving a real shot at a non-diagonal
connection and a nontrivial Galois group -- the first genuinely-coupled d=6 holonomy?

READ FIRST (per the task brief): D6_CIRCLE.md + branch_d6flex.py (the d=6 unimodular pool + the
mechanism-stability/SAT-colorability machinery -- REUSED UNMODIFIED here: stable_matrix_M9,
find_cliques, POINTS, exact_flex_hermitian_at_point). D6_GEOMETRY.md + branch_d6geo.py (the KEY
LEMMA -- X-count<=1 => diagonal connection => trivial Galois -- and the incidence-frame connection
builder exact_fourier_d, REUSED UNMODIFIED here: it is already fully general, per-COORDINATE
exponents e_jc/e_jd in {0,1} regardless of a ray's total X-count, so it applies to X-count>=2 rays
without modification). M9_GEOMETRY.md (d=4's S4 precedent: multi-X coupling gave a constant but
NON-diagonal connection -- the structure to try to reproduce here).

STRATEGY: instead of trying to fold multi-X rays into the ALREADY-SUFFICIENT X-count<=1 pool
(where they end up decorative/non-load-bearing -- any peel just drops them, since X-count<=1 alone
already suffices for uncolorability, exactly like the ALREADY-sufficient pure sub-pool made single-X
rays "optional" until D6_CIRCLE.md Stage 2c deliberately excluded pure rays to prove genuine
theta-dependence), this branch tests the CLEAN, analogous restriction: is the X-COUNT==2 ONLY
sub-pool (pure rays AND single-X rays BOTH excluded entirely) KS-uncolorable on its own? If yes,
every ray in the resulting obstruction is *forced* to carry exactly two X-entries -- genuine,
structural, non-optional coupling of two coordinates via the free phase X, not just permitted.

No existing file is modified. No git. Machinery reused UNMODIFIED (imported): branch_d6flex.py
(stable_matrix_M9, find_cliques, POINTS, exact_flex_hermitian_at_point), branch_d6geo.py
(exact_fourier_d, clear_denominators_d), ks_flex_census.py (cache_save/cache_load).

STAGES (CLI dispatch, each checkpoints to d6galois_*.cache.json):
    python3 branch_d6galois.py stage1                  # enumerate multi-X rays (X-count>=2)
    python3 branch_d6galois.py stage2                  # X-count==2-ONLY pool: build + SAT-check
                                                         #   uncolorability (the existence gate)
    python3 branch_d6galois.py stage3 [--trials N --seed0 S]  # SAT-based greedy critical-core peel
    python3 branch_d6galois.py stage3final              # rigorous closing verification (criticality)
    python3 branch_d6galois.py stage4                   # WZ holonomy connection on the critical core
    python3 branch_d6galois.py stage5                   # char poly + GALOIS GROUP (the prize)
    python3 branch_d6galois.py stage6                   # det W / abelian layer
    python3 branch_d6galois.py all
"""
import os, sys, time, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from itertools import combinations
from collections import Counter

import numpy as np
import sympy as sp

from ks_flex_census import cache_save, cache_load
import branch_d6flex as bd6
import branch_d6geo as bg6

HERE = os.path.dirname(os.path.abspath(__file__))
T0 = time.time()
DSIZE = 6


def g_save(name, obj): cache_save(f"d6galois_{name}", obj)
def g_load(name): return cache_load(f"d6galois_{name}")


def xcount(v): return sum(1 for c in v if c in ("X", "-X"))
def is_pure(v): return all(c in ("0", "1", "-1") for c in v)


# ======================================================================================
# STAGE 1 -- ENUMERATE the multi-X rays (X-count>=2) of the d=6 unimodular stable pool.
# ======================================================================================
def stage1_enumerate():
    print("=" * 100)
    print("STAGE 1 -- ENUMERATE multi-X (X-count>=2) rays of the d=6 unimodular pool")
    print("=" * 100)
    t0 = time.time()
    rays = cache_load("d6flex_pool_rays")
    assert rays is not None, "run branch_d6flex.py stage1 first"
    rays = [tuple(v) for v in rays]
    V = len(rays)
    xc = Counter(xcount(v) for v in rays)
    print(f"[stage1] full pool: {V} rays. X-count histogram: {dict(sorted(xc.items()))}")
    multi = sum(n for k, n in xc.items() if k >= 2)
    print(f"[stage1] X-count>=2 (multi-X) total: {multi}  "
          f"(X==2:{xc.get(2,0)} X==3:{xc.get(3,0)} X==4:{xc.get(4,0)} X==5:{xc.get(5,0)} "
          f"X==6:{xc.get(6,0)})")
    x2_idx = [i for i, v in enumerate(rays) if xcount(v) == 2]
    # which coordinate-pairs carry the two X's, across the 2400 X==2 rays
    pair_hist = Counter()
    for i in x2_idx:
        cs = tuple(c for c, s in enumerate(rays[i]) if s in ("X", "-X"))
        pair_hist[cs] += 1
    print(f"[stage1] X==2 rays: {len(x2_idx)}, distributed over {len(pair_hist)} coordinate-pairs "
          f"(of C(6,2)=15 possible): counts per pair = "
          f"{sorted(set(pair_hist.values()))} (min/max {min(pair_hist.values())}/"
          f"{max(pair_hist.values())})")
    out = dict(V=V, xc_hist=dict(xc), multi_total=multi, x2_total=len(x2_idx),
               pair_hist={str(k): v for k, v in pair_hist.items()})
    g_save("stage1_enum", out)
    print(f"[stage1] done ({time.time()-t0:.1f}s)")
    return out


# ======================================================================================
# STAGE 2 -- THE EXISTENCE GATE: is the X-COUNT==2 ONLY sub-pool (pure AND single-X rays BOTH
# excluded) KS-uncolorable on its own? (direct analogue of D6_CIRCLE.md Stage 2c's X-count==1-only
# test, one X-count level up)
# ======================================================================================
def _sat_colorable(V, pairs, bases):
    from pysat.solvers import Cadical153
    solver = Cadical153()
    for i, j in pairs:
        solver.add_clause([-(i + 1), -(j + 1)])
    for b in bases:
        solver.add_clause([x + 1 for x in b])
    return solver.solve()


def stage2_existence():
    print("=" * 100)
    print("STAGE 2 -- EXISTENCE GATE: X-count==2 ONLY pool (no pure, no single-X)")
    print("=" * 100)
    t0 = time.time()
    rays = cache_load("d6flex_pool_rays")
    rays = [tuple(v) for v in rays]
    x2_idx = [i for i, v in enumerate(rays) if xcount(v) == 2]
    stable_full = bd6.stable_matrix_M9(rays, DSIZE)
    sub = stable_full[np.ix_(x2_idx, x2_idx)]
    deg = sub.sum(axis=1)
    npairs = int(sub.sum()) // 2
    print(f"[stage2] X==2-only stable subgraph: {len(x2_idx)} rays, {npairs} pairs, "
          f"degree(min/mean/max)=({int(deg.min())}/{deg.mean():.1f}/{int(deg.max())})")
    adj = [set(np.nonzero(sub[i])[0].tolist()) for i in range(len(x2_idx))]
    bases, complete = bd6.find_cliques(adj, len(x2_idx), DSIZE, time_limit=25.0)
    pairs = [(int(i), int(j)) for i in range(len(x2_idx)) for j in range(i + 1, len(x2_idx)) if sub[i, j]]
    print(f"[stage2] bases (6-cliques): {len(bases)} (complete enum={complete})  "
          f"({time.time()-t0:.1f}s)")
    t1 = time.time()
    col = _sat_colorable(len(x2_idx), pairs, bases)
    print(f"[stage2] SAT colorability check: colorable={col}  =>  "
          f"KS-UNCOLORABLE={not col}  (SAT call: {time.time()-t1:.2f}s)")
    if not col:
        print("[stage2] *** HEADLINE: the X-count==2 ONLY pool (every ray couples exactly two "
              "coordinates via the free phase X, ZERO help from pure or single-X rays) is "
              "KS-UNCOLORABLE ON ITS OWN. A multi-X-only KS obstruction EXISTS at d=6. ***")
    out = dict(idx=x2_idx, pairs=pairs, bases=bases, complete=complete, uncolorable=not col)
    g_save("stage2_x2only", out)
    print(f"[stage2] done ({time.time()-t0:.1f}s total)")
    return out


# ======================================================================================
# STAGE 3 -- SAT-based greedy critical-core peel from the X-count==2-ONLY pool (mirrors
# branch_d6flex.py's own stage3 exactly, applied to this new pool).
# ======================================================================================
def _colorable_sat_restricted(cand_set, pairs, bases):
    from pysat.solvers import Cadical153
    solver = Cadical153()
    for i, j in pairs:
        if i in cand_set and j in cand_set:
            solver.add_clause([-(i + 1), -(j + 1)])
    for b in bases:
        if all(x in cand_set for x in b):
            solver.add_clause([x + 1 for x in b])
    return solver.solve()


def _greedy_peel_sat(V, pairs, bases, seed, wall_budget):
    order = list(range(V))
    random.Random(seed).shuffle(order)
    keep = set(range(V))
    t0 = time.time()
    for r in order:
        if time.time() - t0 > wall_budget:
            return keep, False
        cand = keep - {r}
        if not _colorable_sat_restricted(cand, pairs, bases):
            keep = cand
    return keep, True


def _repair_sat(keep, pairs, bases, wall_budget):
    t0 = time.time()
    changed = True
    while changed:
        changed = False
        for r in sorted(keep):
            if time.time() - t0 > wall_budget:
                return keep
            cand = keep - {r}
            if not _colorable_sat_restricted(cand, pairs, bases):
                keep = cand
                changed = True
    return keep


def stage3_core(trials=10, seed_start=0, wall_budget_total=32.0):
    t0 = time.time()
    x2 = g_load("stage2_x2only")
    if x2 is None:
        x2 = stage2_existence()
    idx, pairs, bases = x2["idx"], [tuple(p) for p in x2["pairs"]], [tuple(b) for b in x2["bases"]]
    bp = sorted(set(x for b in bases for x in b))
    sub_idx = [idx[i] for i in bp]
    remap = {old: new for new, old in enumerate(bp)}
    sub_pairs = [(remap[i], remap[j]) for i, j in pairs if i in remap and j in remap]
    sub_bases = [tuple(remap[x] for x in b) for b in bases if all(x in remap for x in b)]
    V = len(bp)
    print(f"[stage3] basis-participating restriction of X==2-only pool: {V} rays, "
          f"{len(sub_pairs)} pairs, {len(sub_bases)} bases")

    prev = g_load("stage3_best")
    best = set(prev["keep"]) if prev else set(range(V))
    if prev:
        print(f"[stage3] resuming: cached best core so far = {len(best)} rays")

    for t in range(trials):
        if time.time() - t0 > wall_budget_total:
            print(f"[stage3] wall budget reached after {t} trials this call")
            break
        seed = seed_start + t
        keep, done = _greedy_peel_sat(V, sub_pairs, sub_bases, seed=seed,
                                       wall_budget=min(6.0, wall_budget_total - (time.time() - t0)))
        keep = _repair_sat(keep, sub_pairs, sub_bases,
                            wall_budget=min(3.0, wall_budget_total - (time.time() - t0)))
        print(f"[stage3]   seed {seed}: core size {len(keep)} done={done} "
              f"(t={time.time()-t0:.1f}s)")
        if len(keep) < len(best):
            best = keep

    core_global = [sub_idx[i] for i in sorted(best)]
    g_save("stage3_best", dict(sub_idx=sub_idx, pairs=sub_pairs, sub_bases=sub_bases,
                                keep=sorted(best), core_global_idx=core_global))
    print(f"[stage3] BEST core so far: {len(best)} rays (global pool indices cached)  "
          f"({time.time()-t0:.1f}s total)")
    return best


def stage3_finalize():
    """Rigorous final verification: independent SAT re-check of uncolorability, per-ray
       criticality, composition report (X-count histogram -- should be ALL 2)."""
    rays = cache_load("d6flex_pool_rays")
    best = g_load("stage3_best")
    keep = sorted(best["keep"])
    sub_idx, pairs, sub_bases = best["sub_idx"], [tuple(p) for p in best["pairs"]], [tuple(b) for b in best["sub_bases"]]
    remap = {old: new for new, old in enumerate(keep)}
    core_pairs = [(remap[i], remap[j]) for i, j in pairs if i in remap and j in remap]
    core_bases = [tuple(remap[x] for x in b) for b in sub_bases if all(x in remap for x in b)]
    global_idx = [sub_idx[i] for i in keep]
    core_syms = [tuple(rays[i]) for i in global_idx]
    V = len(keep)

    supp = Counter(sum(1 for c in v if c != "0") for v in core_syms)
    xc = Counter(xcount(v) for v in core_syms)
    print(f"[stage3-final] critical core: {V} rays, {len(core_pairs)} pairs, {len(core_bases)} bases")
    print(f"[stage3-final] support-size histogram: {dict(sorted(supp.items()))}")
    print(f"[stage3-final] X-count histogram (should be ALL 2): {dict(sorted(xc.items()))}")
    assert set(xc.keys()) == {2}, "core contains non-X==2 rays -- unexpected"

    col = _sat_colorable(V, core_pairs, core_bases)
    print(f"[stage3-final] independent SAT re-check, KS-uncolorable: {not col}")
    assert not col

    all_critical = True
    for r in range(V):
        cand_pairs = [(i, j) for i, j in core_pairs if i != r and j != r]
        cand_bases = [b for b in core_bases if r not in b]
        rm2 = {old: new for new, old in enumerate(x for x in range(V) if x != r)}
        cp2 = [(rm2[i], rm2[j]) for i, j in cand_pairs]
        cb2 = [tuple(rm2[x] for x in b) for b in cand_bases]
        if not _sat_colorable(V - 1, cp2, cb2):
            all_critical = False
            print(f"[stage3-final]   NOT CRITICAL: ray {r} removable-check failed")
    print(f"[stage3-final] ALL {V} rays independently verified CRITICAL "
          f"(removing any single one restores colorability): {all_critical}")
    assert all_critical

    # Report coordinate-pair / sign structure of the critical core (relevant to whether the
    # off-diagonal sign-symmetry-cancellation of D6_GEOMETRY.md Sec.2.2 is broken here).
    pair_sign = Counter()
    for v in core_syms:
        cs = tuple((c, s) for c, s in enumerate(v) if s in ("X", "-X"))
        pair_sign[cs] += 1
    print(f"[stage3-final] (coord-pair, sign-pattern) multiplicities in the critical core:")
    for k, n in sorted(pair_sign.items()):
        print(f"     {k}: {n}")

    g_save("core_final", dict(core_syms=core_syms, core_pairs=core_pairs, core_bases=core_bases,
                               global_idx=global_idx, pair_sign={str(k): v for k, v in pair_sign.items()}))
    return dict(V=V, pairs=len(core_pairs), bases=len(core_bases), core_syms=core_syms)


# ======================================================================================
# STAGE 4 -- THE WZ HOLONOMY CONNECTION on the X-count==2 critical core. Reuses
# branch_d6geo.exact_fourier_d UNMODIFIED -- it is already fully general (per-coordinate exponents
# e_jc,e_jd in {0,1}; the Fourier-degree decomposition delta=e_jd-e_jc in {-1,0,1} holds for ANY
# ray regardless of its total X-count, since each raw alphabet symbol is still degree<=1 in X per
# coordinate). This is the FIRST time this exact code path is fed rays with X-count==2 -- the
# off-diagonal term (c,d) with BOTH e_jc=e_jd=1 can now be genuinely nonzero (D6_GEOMETRY.md
# Sec.2.1's diagonal-forcing lemma explicitly does NOT apply here, by construction).
# ======================================================================================
def stage4_connection():
    print("=" * 100)
    print("STAGE 4 -- WZ HOLONOMY CONNECTION on the X-count==2 critical core")
    print("=" * 100)
    t0 = time.time()
    core = g_load("core_final")
    assert core is not None, "run stage3final first"
    core_syms = [tuple(v) for v in core["core_syms"]]
    core_bases = [tuple(b) for b in core["core_bases"]]
    print(f"[stage4] core: {len(core_syms)} rays, {len(core_bases)} bases, dsize={DSIZE}")

    A0, A1, Am, pcount, Nb = bg6.exact_fourier_d(core_syms, core_bases, DSIZE)
    zeroA1 = all(A1[c][d] == 0 for c in range(DSIZE) for d in range(DSIZE))
    zeroAm = all(Am[c][d] == 0 for c in range(DSIZE) for d in range(DSIZE))
    print(f"[stage4] Nb={Nb} bases")
    print(f"[stage4] *** A1==0: {zeroA1}   Am==0: {zeroAm} *** (constant connection iff both True)")

    print("\n[stage4] A0 (coefficient of i; true entry = i*A0):")
    for row in A0:
        print("   ", [str(x) for x in row])

    off_diag_nonzero = [(c, d) for c in range(DSIZE) for d in range(DSIZE)
                         if c != d and A0[c][d] != 0]
    print(f"\n[stage4] *** OFF-DIAGONAL NONZERO ENTRIES: {off_diag_nonzero} ***")
    is_diagonal = len(off_diag_nonzero) == 0
    print(f"[stage4] connection is DIAGONAL: {is_diagonal}")

    M, L = bg6.clear_denominators_d(A0, DSIZE)
    tr = sum(M[i][i] for i in range(DSIZE))
    print(f"\n[stage4] Common denominator L={L}. Integer matrix M := L*A0:")
    for row in M:
        print("    ", row)
    print(f"[stage4] Tr(M)={tr}   Tr(S)=Tr(M)/L={sp.Rational(tr, L)}")

    out = dict(A0=[[str(x) for x in row] for row in A0], zeroA1=zeroA1, zeroAm=zeroAm,
               is_diagonal=is_diagonal, off_diag_nonzero=off_diag_nonzero,
               M=M, L=L, tr=tr, Nb=Nb, dsize=DSIZE)
    g_save("stage4_connection", out)
    print(f"\n[stage4] done in {time.time()-t0:.1f}s")
    return out


# ======================================================================================
# STAGE 5 -- THE GALOIS GROUP (the prize, if the connection is constant).
# If A(theta) is NOT constant, this stage instead reports the theta=0 generator's spectrum as a
# fallback and flags the honest caveat (a genuinely rotating connection needs either a rotating-
# frame reduction (d=3 style) or a path-ordered/holonomy-at-2pi computation, not attempted unless
# needed).
# ======================================================================================
def stage5_galois():
    print("=" * 100)
    print("STAGE 5 -- THE d=6 MULTI-X HOLONOMY GALOIS GROUP")
    print("=" * 100)
    t0 = time.time()
    conn = g_load("stage4_connection")
    assert conn is not None, "run stage4 first"
    M, L, dsize = conn["M"], conn["L"], conn["dsize"]
    if not (conn["zeroA1"] and conn["zeroAm"]):
        print("[stage5] *** WARNING: connection is NOT constant (A1 or Am nonzero). The char-poly/"
              "Galois computation below is of the theta=0 generator M ONLY -- honestly flagged as "
              "NOT the full-loop holonomy generator. ***")

    x = sp.symbols("x")
    Msp = sp.Matrix(M)
    cp = sp.expand(Msp.charpoly(x).as_expr())
    print(f"[stage5] exact characteristic polynomial of M (dsize={dsize}):")
    print(f"    {cp}")
    P = sp.Poly(cp, x, domain="QQ")
    print(f"[stage5] degree: {P.degree()}")
    factors = sp.factor_list(cp, x)
    print(f"[stage5] factorization over Q: {factors}")

    irred_factors = [f for f, mult in factors[1]]
    print(f"[stage5] irreducible factors: {len(irred_factors)}")
    for f in irred_factors:
        Pf = sp.Poly(f, x, domain="QQ")
        print(f"    factor (deg {Pf.degree()}): {f}")

    from sympy.polys.numberfields.galoisgroups import galois_group
    galois_groups = []
    for f in irred_factors:
        Pf = sp.Poly(f, x, domain="QQ")
        if Pf.degree() <= 1:
            print(f"    degree<=1 factor {f}: Galois group TRIVIAL (rational root)")
            galois_groups.append((str(f), "trivial (degree<=1)"))
            continue
        try:
            G, alt = galois_group(Pf, by_name=True)
            print(f"    factor {f}: Galois group = {G}  (subgroup of A_n: {alt})")
            galois_groups.append((str(f), str(G)))
        except Exception as ex:
            print(f"    factor {f}: galois_group computation FAILED: {ex}")
            galois_groups.append((str(f), f"FAILED: {ex}"))

    roots = sp.Poly(cp, x).nroots(n=40)
    roots = sorted(roots, key=lambda r: sp.re(r))
    print(f"\n[stage5] numeric roots (40-digit precision):")
    eigenphases = []
    for r in roots:
        val = sp.re(r) / L
        frac = val - sp.floor(val)
        eigenphases.append(str(frac))
        print(f"    lambda={sp.N(r,20)}   phi/(2pi)=lambda/{L} mod 1 = {sp.N(frac,20)}")

    out = dict(charpoly=str(cp), factors=[str(f) for f in irred_factors],
               galois_groups=galois_groups, roots=[str(sp.N(r, 20)) for r in roots],
               eigenphases=eigenphases)
    g_save("stage5_galois", out)
    print(f"\n[stage5] done in {time.time()-t0:.1f}s")
    return out


# ======================================================================================
# STAGE 6 -- det W / abelian layer
# ======================================================================================
def stage6_abelian():
    print("=" * 100)
    print("STAGE 6 -- det W / abelian layer")
    print("=" * 100)
    conn = g_load("stage4_connection")
    assert conn is not None, "run stage4 first"
    tr, L = conn["tr"], conn["L"]
    trS = sp.Rational(tr, L)
    detW_phase = trS - sp.floor(trS)
    print(f"[stage6] Tr(S) = {trS}")
    print(f"[stage6] det W(2pi) = exp(2pi i * Tr(S)) = exp(2pi i * {detW_phase})")
    out = dict(trS=str(trS), detW_phase=str(detW_phase), denom=int(sp.fraction(detW_phase)[1]))
    g_save("stage6_abelian", out)
    return out


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which == "stage1":
        stage1_enumerate()
    elif which == "stage2":
        stage2_existence()
    elif which == "stage3":
        trials = int(sys.argv[2]) if len(sys.argv) > 2 else 10
        seed0 = int(sys.argv[3]) if len(sys.argv) > 3 else 0
        stage3_core(trials=trials, seed_start=seed0)
    elif which == "stage3final":
        stage3_finalize()
    elif which == "stage4":
        stage4_connection()
    elif which == "stage5":
        stage5_galois()
    elif which == "stage6":
        stage6_abelian()
    elif which == "all":
        stage1_enumerate()
        stage2_existence()
        stage3_core(trials=20, seed_start=0)
        stage3_finalize()
        stage4_connection()
        stage5_galois()
        stage6_abelian()
    else:
        print(f"unknown stage {which!r}")
        sys.exit(1)
    print(f"\n[branch_d6galois.py stage={which} total {time.time()-T0:.1f}s]")
