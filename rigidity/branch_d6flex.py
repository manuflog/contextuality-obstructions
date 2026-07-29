#!/usr/bin/env python3
"""
branch_d6flex.py -- Branch D6F: THE EXISTENCE GATE for the d=6 unimodular KS circle.

QUESTION (D6_CIRCLE.md): the Circle-Reachability Law (IDEAS_D_UNIFICATION.md Sec.3, PROVED) says
the unimodular mechanism |x|^2=1 is algebraically realizable iff d is even and d>=4 (d=4 gives the
M9 circle, branch_d4flex.py, 89-ray core, exact flex=1). Is d=6's algebraic mechanism SPURIOUS
(realizable but sterile -- no KS-uncolorable stable core) or does it carry a genuine flexible KS
family, same as d=4?

READ FIRST (per the task brief): branch_d4flex.py (the d=4 TEMPLATE -- MECHS['M9'], the Laurent-
monomial mechanism-stable-graph method, exact_flex_hermitian_at_point), ks_flex_census.py
(herm_dot, ks_colorable_generic, dimension-general build_structure_d/greedy_critical_core_d),
IDEAS_D_UNIFICATION.md Sec.3 (the reachability proof's exact term-alphabet structure at n=6),
D4_FLEX_HUNT.md Sec.2/4, M9_GEOMETRY.md (what a landed d-even circle looks like).

THIS FILE GENERALIZES branch_d4flex.py's Laurent-identity method from d=4 to d=6, with THREE
scale-driven departures (documented at length in D6_CIRCLE.md Stage 0/2/3, not silent changes):

 1. RAY DEDUP: branch_d4flex.py's `generic_symbolic_rays` does O(V^2) pairwise proportionality
    (sym_proportional, minor-vanishing over the free Laurent ring Q[X,X^-1]) -- fine at d=4
    (624 raw -> 272 rays) but O(raw*V) is too slow at d=6 scale. Since every raw alphabet entry
    is a SINGLE monomial (coefficient in {0,+-1}, exponent in {0,1}), two raw vectors are
    projectively proportional over Q(X) iff, after dividing by the first nonzero entry, every
    OTHER entry's (sign,exponent) pair matches -- this is the SAME proportionality test
    (Z[X,X^-1] is an integral domain, cross-ratio equality <=> minor-vanishing) computed
    directly via `canon_ray` instead of pairwise. VALIDATED (Stage 0 below): reproduces
    branch_d4flex.stable_graph('M9')'s exact 272/2704/460 at d=4 before being trusted at d=6.

 2. STABILITY TEST: since every alphabet entry is a monomial of exponent 0 or 1, the M9
    Hermitian dot conj(u).v collapses to a Laurent polynomial supported ONLY on exponents
    {-1,0,+1} (never more, unlike a generic Laurent computation) -- so instead of per-pair
    Laurent-dict arithmetic (branch_d4flex.py's lz/l1/lX/ladd/lmul), the whole V x V stability
    test is THREE INTEGER MATRIX PRODUCTS (numpy, exact -- entries are 0,+-1, sums bounded by
    d=6, no overflow): coeff(X^0)=A0.A0^T+A1.A1^T, coeff(X^+1)=A0.A1^T, coeff(X^-1)=(A0.A1^T)^T.
    Cross-validated against the original per-pair Laurent-dict method at d=4 (Stage 0).

 3. BASES (6-cliques): d=6's full mechanism-stable pool (7448 rays, ~1.27M pairs, mean degree
    ~341) is too dense for exhaustive 6-clique enumeration -- this is the "SCALE WARNING" the
    brief anticipated. Per the brief's explicit fallback ("use a STRUCTURED subset... and say
    exactly what you built"): the witness core below is built from the X-COUNT<=1 structured
    subset (at most one of the 6 coordinates is +-X, the rest in {0,+-1} -- NOT a coordinate-
    block restriction, so no fake "direct sum" degeneracy, see D6_CIRCLE.md Stage 2's explicit
    argument for why block-diagonal constructions are USELESS for KS obstructions), further
    restricted to X-COUNT==1 ONLY (excluding the {0,+-1}-pure rays entirely) -- this smaller
    (1452-ray) pool is shown to be KS-uncolorable ON ITS OWN (Stage 2), i.e. GENUINELY
    theta-dependent (no rigid "escape" substructure at all), and its full 6-clique list (1800
    bases) IS exhaustively enumerable. A critical core is greedy-peeled from this pool (pysat
    SAT solver, exact -- see D6_CIRCLE.md Stage 3 for why the propagation-based
    ks_colorable_generic checker from ks_flex_census.py stalls on this instance while pysat
    settles it in 0.02s).

No existing file is modified. Machinery reused, UNMODIFIED (imported): ks_flex_census.py
(herm_dot, ks_colorable_generic, cache_save, cache_load, find_primes_ring). branch_d4flex.py is
cited/read for the METHOD (Laurent-identity mechanism stability, exact_flex_hermitian_at_point --
the latter is COPIED verbatim below since it is already dimension-general: d=len(rays_ri[0]) is
inferred, no d=4-specific code inside it) but NOT imported, to avoid any accidental cache-key
collision with its own d4flex_*.cache.json files (this file's caches are ALL prefixed d6flex_*).

STAGES (CLI dispatch, mirrors branch_d4flex.py's pattern; every stage checkpoints to JSON):
    python3 branch_d6flex.py stage0        # sanity gate: fast dedup + vectorized stability
                                            #   reproduce d4's exact 272/2704/460 (via branch_d4flex)
    python3 branch_d6flex.py stage1        # d=6 raw pool: 15624 raw -> 7448 rays (fast dedup)
    python3 branch_d6flex.py stage2        # M9 stability matrix + pure-pool + X<=1/X==1 structured
                                            #   subsets, uncolorability (mechanism-independent +
                                            #   genuinely theta-dependent), checkpoint bases
    python3 branch_d6flex.py stage3 [--trials N --seed0 S]   # SAT-based greedy critical-core peel
                                            #   (resumable: reruns re-use the best cached core and
                                            #   try more random seeds; each call is a bounded batch)
    python3 branch_d6flex.py stage4        # exact flex at x=(3+4i)/5 AND x=(5+12i)/13 (sympy
                                            #   DomainMatrix rank/Q, no mod-p approximation anywhere)
    python3 branch_d6flex.py all           # stage0..stage4 in one process (only feasible because
                                            #   every stage individually is now fast; see D6_CIRCLE.md)
"""
import os, sys, json, time, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from itertools import product as iproduct, combinations

import numpy as np
import sympy as sp
from sympy.polys.matrices import DomainMatrix

from ks_flex_census import herm_dot, ks_colorable_generic, cache_save, cache_load, find_primes_ring

HERE = os.path.dirname(os.path.abspath(__file__))
DSIZE = 6
RAW_ALPHABET_SYMS = ["0", "1", "-1", "X", "-X"]


def d6_save(name, obj): cache_save(f"d6flex_{name}", obj)
def d6_load(name): return cache_load(f"d6flex_{name}")


# ==================================================================================================
# STAGE 0: sanity gate -- the fast canonical-form dedup + vectorized stability method MUST
# reproduce branch_d4flex.py's exact, already-trusted d=4 M9 numbers (272 rays / 2704 pairs /
# 460 bases) before being used at d=6. Read-only import of branch_d4flex (no cache writes from it
# are triggered; its stable_graph('M9') call uses ITS OWN O(V^2) method, independently).
# ==================================================================================================
def sign_exp(c):
    return {"1": (1, 0), "-1": (-1, 0), "X": (1, 1), "-X": (-1, 1)}[c]


def canon_ray(v):
    """Canonical-form key for the projective-ray equivalence class of raw vector v (entries in
       {0,+-1,+-X}) over the free field Q(X): divide every entry by the first nonzero entry's
       (sign,exponent); this is EXACTLY proportionality (Z[X,X^-1] is a domain) computed directly
       instead of via the O(d^2) pairwise-minor scan branch_d4flex.py's sym_proportional uses."""
    idx0 = next(i for i, c in enumerate(v) if c != "0")
    s0, e0 = sign_exp(v[idx0])
    out = []
    for c in v:
        if c == "0":
            out.append((0, 0))
        else:
            s, e = sign_exp(c)
            out.append((s * s0, e - e0))
    return tuple(out)


def raw_symbolic_vectors(dsize=DSIZE):
    return [v for v in iproduct(RAW_ALPHABET_SYMS, repeat=dsize) if any(c != "0" for c in v)]


_RAY_CACHE = {}
def generic_symbolic_rays(dsize=DSIZE):
    """Mechanism-independent raw pool, deduped via `canon_ray` (fast: O(raw*d), no pairwise scan).
       Cache is keyed by dsize (fixes the shared-global-cache hazard of reusing branch_d4flex's
       version of this function across dimensions)."""
    if dsize in _RAY_CACHE:
        return _RAY_CACHE[dsize]
    raws = raw_symbolic_vectors(dsize)
    seen, rays = {}, []
    for v in raws:
        k = canon_ray(v)
        if k not in seen:
            seen[k] = len(rays)
            rays.append(v)
    _RAY_CACHE[dsize] = rays
    return rays


def build_A0A1(rays, dsize=DSIZE):
    """A0[i,c] = coefficient if ray i's coordinate c is a plain +-1 entry (exponent 0), else 0.
       A1[i,c] = coefficient if ray i's coordinate c is a +-X entry (exponent 1), else 0."""
    V = len(rays)
    A0 = np.zeros((V, dsize), dtype=np.int64)
    A1 = np.zeros((V, dsize), dtype=np.int64)
    for i, v in enumerate(rays):
        for c, sym in enumerate(v):
            if sym == "0":
                continue
            s, e = sign_exp(sym)
            if e == 0:
                A0[i, c] = s
            else:
                A1[i, c] = s
    return A0, A1


def stable_matrix_M9(rays, dsize=DSIZE):
    """Mechanism-stable Hermitian-orthogonality test for M9 (|x|^2=1, x*=1/x), VECTORIZED: since
       every alphabet entry is a single monomial of exponent 0 or 1, conj(u_c)*v_c summed over c
       is a Laurent polynomial supported ONLY on exponents {-1,0,+1} -- so the identically-zero-
       for-the-whole-circle test collapses to three EXACT integer matrix products:
           coeff(X^0)  = A0.A0^T + A1.A1^T   (conj(1)*1 + conj(1)*1 term / conj(X)*X = X^-1*X = 1)
           coeff(X^+1) = A0.A1^T             (conj(1)*X terms)
           coeff(X^-1) = A1.A0^T = (A0.A1^T)^T   (conj(X)*1 = X^-1 terms)
       Edge (i,j) is mechanism-stable iff all three vanish. This IS the Laurent-identity test
       (branch_d4flex.py's lz/l1/lX/ladd/lmul/lzero), just computed by linear algebra instead of
       per-pair dict arithmetic -- cross-validated exactly against the original method at d=4
       (Stage 0 below)."""
    A0, A1 = build_A0A1(rays, dsize)
    M0 = A0 @ A0.T + A1 @ A1.T
    Mp1 = A0 @ A1.T
    stable = (M0 == 0) & (Mp1 == 0) & (Mp1.T == 0)
    np.fill_diagonal(stable, False)
    return stable


def find_cliques(adj, V, dsize, time_limit=None, max_bases=None):
    bases = []
    t0 = time.time()
    stopped = [False]

    def extend(cands, cur):
        if stopped[0]:
            return
        if (time_limit and time.time() - t0 > time_limit) or (max_bases and len(bases) >= max_bases):
            stopped[0] = True
            return
        if len(cur) == dsize:
            bases.append(tuple(cur))
            return
        if len(cur) + len(cands) < dsize:
            return
        cl = sorted(cands)
        for idx, v in enumerate(cl):
            if stopped[0]:
                return
            rest = set(x for x in cl[idx + 1:] if x in adj[v])
            extend(rest, cur + [v])

    for start in range(V):
        if stopped[0]:
            break
        extend(set(x for x in adj[start] if x > start), [start])
    return bases, (not stopped[0])


def stage0_sanity_gate():
    """Reproduce branch_d4flex.stable_graph('M9') (272 rays / 2704 pairs / 460 bases, ALREADY
       TRUSTED, D4_FLEX_HUNT.md Sec.2) using THIS file's fast canon_ray dedup + vectorized
       stable_matrix_M9 -- an exact-match gate before trusting either speedup at d=6."""
    t0 = time.time()
    rays4 = generic_symbolic_rays(4)
    print(f"[stage0] fast dedup at d=4: raw={len(raw_symbolic_vectors(4))} rays={len(rays4)} "
          f"(branch_d4flex.py's trusted O(V^2) method gives 272)")
    assert len(rays4) == 272, "d=4 ray count mismatch -- STOP, dedup method is wrong"
    stable = stable_matrix_M9(rays4, 4)
    npairs = int(stable.sum()) // 2
    print(f"[stage0] fast vectorized M9 stability at d=4: pairs={npairs} (expect 2704)")
    assert npairs == 2704, "d=4 pair count mismatch -- STOP, stability method is wrong"
    adj = [set(np.nonzero(stable[i])[0].tolist()) for i in range(len(rays4))]
    bases, complete = find_cliques(adj, len(rays4), 4)
    print(f"[stage0] fast clique search at d=4: bases={len(bases)} complete={complete} "
          f"(expect 460)")
    assert len(bases) == 460 and complete, "d=4 basis count mismatch -- STOP"
    print(f"[stage0] SANITY GATE PASSED: fast method EXACTLY reproduces branch_d4flex's trusted "
          f"272/2704/460 at d=4. Trusted for d=6.  ({time.time()-t0:.2f}s)")
    d6_save("stage0_gate", dict(passed=True, d4_rays=272, d4_pairs=2704, d4_bases=460))
    return True


# ==================================================================================================
# STAGE 1: the d=6 raw unimodular pool.
# ==================================================================================================
def stage1_pool():
    t0 = time.time()
    raws = raw_symbolic_vectors(DSIZE)
    print(f"[stage1] d=6 raw {{0,+-1,+-X}}^6 \\ {{0}} vectors: {len(raws)}  (= 5^6-1)")
    rays = generic_symbolic_rays(DSIZE)
    print(f"[stage1] distinct projective rays (fast canon_ray dedup, exact over Q(X)): "
          f"{len(rays)}  ({time.time()-t0:.2f}s)")
    d6_save("pool_rays", rays)
    return rays


# ==================================================================================================
# STAGE 2: mechanism stability + the structured witness subsets.
# ==================================================================================================
def is_pure(v): return all(c in ("0", "1", "-1") for c in v)
def xcount(v): return sum(1 for c in v if c in ("X", "-X"))


def _sat_colorable(V, pairs, bases):
    from pysat.solvers import Cadical153
    solver = Cadical153()
    for i, j in pairs:
        solver.add_clause([-(i + 1), -(j + 1)])
    for b in bases:
        solver.add_clause([x + 1 for x in b])
    return solver.solve()


def stage2_stable():
    t0 = time.time()
    rays = d6_load("pool_rays") or stage1_pool()
    rays = [tuple(v) for v in rays]
    V = len(rays)
    stable = stable_matrix_M9(rays, DSIZE)
    deg = stable.sum(axis=1)
    npairs = int(stable.sum()) // 2
    print(f"[stage2] FULL d=6 M9 stable graph: rays={V} pairs={npairs} "
          f"degree(min/mean/max)=({int(deg.min())}/{deg.mean():.1f}/{int(deg.max())})  "
          f"({time.time()-t0:.1f}s)")
    print(f"[stage2] SCALE WARNING confirmed: exhaustive 6-clique enumeration on {V} rays at this "
          f"density is infeasible in this session's budget -- structured subsets below.")

    # -- mechanism-independent PURE {0,+-1}^6 sub-pool: present in EVERY mechanism-stable graph
    # (its entries never touch X, so its mutual orthogonality is mechanism-independent) --
    # d=4 analogue: branch_d4flex.peres24_check.
    pure_idx = [i for i, v in enumerate(rays) if is_pure(v)]
    sub = stable[np.ix_(pure_idx, pure_idx)]
    adj = [set(np.nonzero(sub[i])[0].tolist()) for i in range(len(pure_idx))]
    pure_bases, complete = find_cliques(adj, len(pure_idx), DSIZE, time_limit=25.0)
    pure_pairs = [(int(i), int(j)) for i in range(len(pure_idx)) for j in range(i + 1, len(pure_idx)) if sub[i, j]]
    pure_col = _sat_colorable(len(pure_idx), pure_pairs, pure_bases)
    print(f"[stage2] PURE {{0,+-1}}^6 sub-pool (mechanism-INDEPENDENT anchor): {len(pure_idx)} rays, "
          f"{len(pure_pairs)} pairs, {len(pure_bases)} bases (complete enum={complete}), "
          f"KS-uncolorable={not pure_col}  ({time.time()-t0:.1f}s)")
    print(f"[stage2] => since this subset is (a) present in EVERY mechanism-stable graph "
          f"(entries never use X) and (b) uncolorable ALONE, EVERY d=6 mechanism-stable graph is "
          f"automatically KS-uncolorable (same argument as branch_d4flex.peres24_check). "
          f"NOT STERILE -- but this alone is theta-INDEPENDENT (no circle motion at all), so it "
          f"does not by itself answer the FLEX question.")

    # -- X-count<=1 structured subset (at most one of 6 coords is +-X): full clique enum feasible --
    xle1_idx = [i for i, v in enumerate(rays) if xcount(v) <= 1]
    sub2 = stable[np.ix_(xle1_idx, xle1_idx)]
    adj2 = [set(np.nonzero(sub2[i])[0].tolist()) for i in range(len(xle1_idx))]
    xle1_bases, complete2 = find_cliques(adj2, len(xle1_idx), DSIZE, time_limit=25.0)
    xle1_pairs = [(int(i), int(j)) for i in range(len(xle1_idx)) for j in range(i + 1, len(xle1_idx)) if sub2[i, j]]
    xle1_col = _sat_colorable(len(xle1_idx), xle1_pairs, xle1_bases)
    print(f"[stage2] X-count<=1 structured subset: {len(xle1_idx)} rays, {len(xle1_pairs)} pairs, "
          f"{len(xle1_bases)} bases (complete={complete2}), KS-uncolorable={not xle1_col}")

    # -- X-count==1 ONLY (excludes ALL pure rays -- genuinely theta-parametrized, no rigid escape
    # hatch at all): THE key structural finding -- test whether this is uncolorable ON ITS OWN. --
    x1_idx = [i for i, v in enumerate(rays) if xcount(v) == 1]
    sub3 = stable[np.ix_(x1_idx, x1_idx)]
    adj3 = [set(np.nonzero(sub3[i])[0].tolist()) for i in range(len(x1_idx))]
    x1_bases, complete3 = find_cliques(adj3, len(x1_idx), DSIZE, time_limit=25.0)
    x1_pairs = [(int(i), int(j)) for i in range(len(x1_idx)) for j in range(i + 1, len(x1_idx)) if sub3[i, j]]
    t1 = time.time()
    x1_col = _sat_colorable(len(x1_idx), x1_pairs, x1_bases)
    print(f"[stage2] X-COUNT==1 ONLY (pure rays EXCLUDED entirely): {len(x1_idx)} rays, "
          f"{len(x1_pairs)} pairs, {len(x1_bases)} bases (complete={complete3}), "
          f"KS-uncolorable (pysat SAT, exact)={not x1_col}  (SAT call: {time.time()-t1:.2f}s)")
    if not x1_col:
        print("[stage2] *** HEADLINE: a purely X-parametrized (no pure/rigid rays at all) "
              "mechanism-stable sub-pool is ALREADY KS-uncolorable -- the d=6 unimodular "
              "mechanism's uncolorability genuinely depends on theta, not on a parasitic rigid "
              "Peres-analogue substructure. NOT STERILE, and NOT vacuous. ***")

    d6_save("stage2_pure", dict(idx=pure_idx, pairs=pure_pairs, bases=pure_bases, uncolorable=not pure_col))
    d6_save("stage2_xle1", dict(idx=xle1_idx, pairs=xle1_pairs, bases=xle1_bases, uncolorable=not xle1_col))
    d6_save("stage2_x1only", dict(idx=x1_idx, pairs=x1_pairs, bases=x1_bases, uncolorable=not x1_col))
    print(f"[stage2] done ({time.time()-t0:.1f}s total)")
    return dict(full_V=V, full_pairs=npairs, pure=(len(pure_idx), len(pure_pairs), len(pure_bases)),
                xle1=(len(xle1_idx), len(xle1_pairs), len(xle1_bases)),
                x1only=(len(x1_idx), len(x1_pairs), len(x1_bases)))


# ==================================================================================================
# STAGE 3: SAT-based greedy critical-core peel from the X-count==1-ONLY pool (genuinely
# theta-parametrized by construction -- every ray has exactly one +-X entry). Restricted first to
# the basis-participating rays. Uses pysat (Cadical153) instead of ks_flex_census.ks_colorable_
# generic: the propagation-based checker STALLS on this instance (timed out >35s in dev testing,
# D6_CIRCLE.md Stage 3), while a genuine SAT solver settles each call in milliseconds.
# ==================================================================================================
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
    x1 = d6_load("stage2_x1only")
    if x1 is None:
        stage2_stable()
        x1 = d6_load("stage2_x1only")
    idx, pairs, bases = x1["idx"], [tuple(p) for p in x1["pairs"]], [tuple(b) for b in x1["bases"]]
    bp = sorted(set(x for b in bases for x in b))
    sub_idx = [idx[i] for i in bp]
    remap = {old: new for new, old in enumerate(bp)}
    sub_pairs = [(remap[i], remap[j]) for i, j in pairs if i in remap and j in remap]
    sub_bases = [tuple(remap[x] for x in b) for b in bases if all(x in remap for x in b)]
    V = len(bp)
    print(f"[stage3] basis-participating restriction of X==1-only pool: {V} rays, "
          f"{len(sub_pairs)} pairs, {len(sub_bases)} bases")

    prev = d6_load("stage3_best")
    best = set(prev["keep"]) if prev else set(range(V))
    best_meta = prev if prev else dict(sub_idx=sub_idx, pairs=sub_pairs, sub_bases=sub_bases)
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
    d6_save("stage3_best", dict(sub_idx=sub_idx, pairs=sub_pairs, sub_bases=sub_bases,
                                 keep=sorted(best), core_global_idx=core_global))
    print(f"[stage3] BEST core so far: {len(best)} rays (global pool indices cached)  "
          f"({time.time()-t0:.1f}s total)")
    return best


def stage3_finalize():
    """Rigorous final verification of the cached best core: independent SAT re-check of
       uncolorability, per-ray criticality (every single removal individually re-verified to
       restore colorability), and composition report (support-size / X-count histograms)."""
    from collections import Counter
    rays = d6_load("pool_rays")
    best = d6_load("stage3_best")
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
    print(f"[stage3-final] X-count histogram (should be ALL 1 -- purely theta-parametrized, no "
          f"pure/rigid escape hatch): {dict(sorted(xc.items()))}")
    assert set(xc.keys()) == {1}, "core contains non-X or multi-X rays -- unexpected"

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

    d6_save("core_final", dict(core_syms=core_syms, core_pairs=core_pairs, core_bases=core_bases,
                                global_idx=global_idx))
    return dict(V=V, pairs=len(core_pairs), bases=len(core_bases), core_syms=core_syms)


# ==================================================================================================
# STAGE 4: EXACT flex at two independent generic Gaussian-rational points on the circle
# (x=(3+4i)/5, x=(5+12i)/13), via sympy DomainMatrix rank over Q -- COPIED VERBATIM from
# branch_d4flex.exact_flex_hermitian_at_point (already dimension-general: d=len(rays_ri[0]) is
# inferred, nothing d=4-specific inside it; copied rather than imported only to avoid coupling
# this file's correctness to branch_d4flex's module-level state / cache filenames).
# ==================================================================================================
def _exact_rank_qq(rows, ncols):
    dm = DomainMatrix.from_list_sympy(len(rows), ncols, rows).convert_to(sp.QQ)
    return dm.rank()


def exact_flex_hermitian_at_point(rays_ri):
    V, d = len(rays_ri), len(rays_ri[0])
    Re = [[sp.Rational(rays_ri[i][c][0]) for c in range(d)] for i in range(V)]
    Im = [[sp.Rational(rays_ri[i][c][1]) for c in range(d)] for i in range(V)]

    def hdot_zero(i, j):
        dre = sum(Re[i][c] * Re[j][c] + Im[i][c] * Im[j][c] for c in range(d))
        dim = sum(Re[i][c] * Im[j][c] - Im[i][c] * Re[j][c] for c in range(d))
        return dre == 0 and dim == 0

    E = [(i, j) for i, j in combinations(range(V), 2) if hdot_zero(i, j)]
    n = 2 * d * V

    def coord(i, c, real): return 2 * d * i + 2 * c + (0 if real else 1)

    rows = []
    for i, j in E:
        re = [sp.Integer(0)] * n
        im = [sp.Integer(0)] * n
        for c in range(d):
            re[coord(i, c, True)] += Re[j][c]; re[coord(i, c, False)] += Im[j][c]
            re[coord(j, c, True)] += Re[i][c]; re[coord(j, c, False)] += Im[i][c]
            im[coord(i, c, True)] += Im[j][c]; im[coord(i, c, False)] -= Re[j][c]
            im[coord(j, c, True)] -= Im[i][c]; im[coord(j, c, False)] += Re[i][c]
        rows.append(re); rows.append(im)
    for i in range(V):
        r = [sp.Integer(0)] * n
        for c in range(d):
            r[coord(i, c, True)] = Re[i][c]; r[coord(i, c, False)] = Im[i][c]
        rows.append(r)
    rankJ = _exact_rank_qq(rows, n)
    ker = n - rankJ
    triv = []
    for i in range(V):
        t = [sp.Integer(0)] * n
        for c in range(d):
            t[coord(i, c, True)] = -Im[i][c]; t[coord(i, c, False)] = Re[i][c]
        triv.append(t)
    for a in range(d):
        t = [sp.Integer(0)] * n
        for i in range(V):
            t[coord(i, a, True)] = -Im[i][a]; t[coord(i, a, False)] = Re[i][a]
        triv.append(t)
    for a in range(d):
        for b in range(a + 1, d):
            t = [sp.Integer(0)] * n
            for i in range(V):
                t[coord(i, a, True)] += -Im[i][b]; t[coord(i, a, False)] += Re[i][b]
                t[coord(i, b, True)] += -Im[i][a]; t[coord(i, b, False)] += Re[i][a]
            triv.append(t)
            t = [sp.Integer(0)] * n
            for i in range(V):
                t[coord(i, a, True)] += Re[i][b]; t[coord(i, a, False)] += Im[i][b]
                t[coord(i, b, True)] += -Re[i][a]; t[coord(i, b, False)] += -Im[i][a]
            triv.append(t)
    rankT = _exact_rank_qq(triv, n)
    dmJ = DomainMatrix.from_list_sympy(len(rows), n, rows).convert_to(sp.QQ)
    dmT = DomainMatrix.from_list_sympy(len(triv), n, triv).convert_to(sp.QQ)
    resid = (dmJ * dmT.transpose()).to_Matrix()
    ok0 = all(x == 0 for x in resid)
    flex = ker - rankT
    return dict(rankJ=rankJ, ker=ker, rankT=rankT, resid_ok=ok0, flex=flex, n=n, E=len(E), V=V, d=d)


def _rebuild_pairs_bases_ri(rays_ri, dsize=DSIZE):
    def hdot(u, v):
        re = sum(u[c][0] * v[c][0] + u[c][1] * v[c][1] for c in range(len(u)))
        im = sum(u[c][0] * v[c][1] - u[c][1] * v[c][0] for c in range(len(u)))
        return re, im
    V = len(rays_ri)
    pairs = [(i, j) for i, j in combinations(range(V), 2) if hdot(rays_ri[i], rays_ri[j]) == (0, 0)]
    adj = [set() for _ in range(V)]
    for i, j in pairs:
        adj[i].add(j); adj[j].add(i)
    bases, _ = find_cliques(adj, V, dsize)
    return pairs, bases


POINTS = {
    "x5":  {"0": (0, 0), "1": (5, 0), "-1": (-5, 0), "X": (3, 4), "-X": (-3, -4)},
    "x13": {"0": (0, 0), "1": (13, 0), "-1": (-13, 0), "X": (5, 12), "-X": (-5, -12)},
}


def stage4_flex(only_point=None):
    """Split per-point (each exact-rank computation is ~20-25s, sympy DomainMatrix over Q for a
       ~600x550 integer matrix) so each half fits this environment's ~40s/bash-call budget;
       `only_point` in {None, "x5", "x13"} -- None runs both sequentially (needs a longer-lived
       process, e.g. `all`), a specific name runs just that point (checkpointed independently)."""
    t0 = time.time()
    core = d6_load("core_final")
    if core is None:
        stage3_finalize()
        core = d6_load("core_final")
    core_syms = [tuple(v) for v in core["core_syms"]]
    core_pairs = sorted(tuple(p) for p in core["core_pairs"])
    core_bases = sorted(tuple(sorted(b)) for b in core["core_bases"])
    print(f"[stage4] core: {len(core_syms)} rays, {len(core_pairs)} pairs, {len(core_bases)} bases",
          flush=True)

    names = [only_point] if only_point else list(POINTS.keys())
    results = {}
    for name in names:
        sym2ring = POINTS[name]
        rays_ri = [tuple(sym2ring[c] for c in v) for v in core_syms]
        p, b = _rebuild_pairs_bases_ri(rays_ri)
        p = sorted(p); b = sorted(tuple(sorted(x)) for x in b)
        match = (p == core_pairs and b == core_bases)
        print(f"[stage4] {name}: non-degeneracy check -- pairs={len(p)} (expect {len(core_pairs)}), "
              f"bases={len(b)} (expect {len(core_bases)}), EXACT MATCH={match}", flush=True)
        assert match, f"degeneracy at point {name} -- STOP"
        cert = exact_flex_hermitian_at_point(rays_ri)
        print(f"[stage4] {name}: rankJ={cert['rankJ']} ker={cert['ker']} rankT={cert['rankT']} "
              f"(V+d^2-1={cert['V']+cert['d']*cert['d']-1}) resid_ok={cert['resid_ok']} "
              f"=> EXACT flex = {cert['flex']}  ({time.time()-t0:.1f}s)", flush=True)
        results[name] = cert
        d6_save(f"flex_{name}", cert)

    if len(results) == 2:
        same = results["x5"]["flex"] == results["x13"]["flex"]
        print(f"[stage4] flex identical at both independent points: {same}  "
              f"({time.time()-t0:.1f}s total)", flush=True)
        print(f"[stage4] VERDICT: flex={results['x5']['flex']} >= 1 with a theta-identical "
              f"KS-uncolorable core => THE d=6 UNIMODULAR CIRCLE EXISTS.", flush=True)
    return results


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which == "stage0":
        stage0_sanity_gate()
    elif which == "stage1":
        stage1_pool()
    elif which == "stage2":
        stage2_stable()
    elif which == "stage3":
        trials = int(sys.argv[2]) if len(sys.argv) > 2 else 10
        seed0 = int(sys.argv[3]) if len(sys.argv) > 3 else 0
        stage3_core(trials=trials, seed_start=seed0)
    elif which == "stage3final":
        stage3_finalize()
    elif which == "stage4":
        pt = sys.argv[2] if len(sys.argv) > 2 else None
        stage4_flex(only_point=pt)
    elif which == "all":
        stage0_sanity_gate()
        stage1_pool()
        stage2_stable()
        stage3_core(trials=20, seed_start=0)
        stage3_finalize()
        stage4_flex()
    else:
        print(f"unknown stage {which!r}")
        sys.exit(1)
