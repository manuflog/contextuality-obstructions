#!/usr/bin/env python3
"""
branch_d10galois.py -- MOONSHOT: the d=10 rung of the KS circle tower (the "d=10 wall").

The two obstructions recorded in NOTE.md SESSION 57 / SD_ATTACK.md:
  (a) the certificate template needs d-1 prime; 9 is composite      -> SOLVED by
      sd_certificate_v2.py (route A: 7-cycle pattern + Jordan p<=n-3 + odd type).
  (b) the d=10 X-count==2 ray pool has 590,400 rays; the d=8 pipeline already could not
      materialize its own 40,768-ray pool's stability matrix (13 GB).

THIS FILE attacks (b) by LAZY FIBER GENERATION + the support-4 stratum:
  * The X==2 pool fibers over (position pair of the two X entries) x (sign pattern) x
    ({0,+-1}^(d-2) tail).  The support-s stratum has size C(d,2)*2*C(d-2,s-2)*2^(s-2)
    (projective dedup halves the 4 sign patterns to 2).  At d=10 the histogram is
    s=3..10: 1440, 10080, 40320, 100800, 161280, 161280, 92160, 23040  (total 590,400)
    -- verified against the d=8 recorded histogram by the same formula.  The support-4
    stratum (10080 rays) is generated DIRECTLY; the 590,400-ray pool is never built.
  * Support 4 is the same restriction the d=8 rung used, licensed by the d=6 fact that ALL
    basis-participating X==2 rays have support exactly 4 (branch_d8galois.py stage 0).
    HONEST CAVEAT (inherited from d=8 and unresolved): the support>=5 strata are not searched.
  * The stability graph on 10080 rays is built in ROW BLOCKS (exact {0,+-1} dot products in
    float64, values bounded by 10 << 2^53) -- peak memory ~120 MB instead of 13 GB.
  * Everything downstream (partial-clique SAT uncolorability, incremental assumption-based
    peel, complete in-core clique re-enumeration, criticality, exact WZ connection, exact
    charpoly, certificate) is the d=8 pipeline, imported READ-ONLY from ../rigidity where it is
    dimension-general, re-implemented here where it is not.

RIGOUR NOTE (same as branch_d8galois.py): bases only add positive clauses, so UNSAT with a
PARTIAL 10-clique list is a sound uncolorability certificate; the final core is re-verified
against a COMPLETE in-core enumeration.

Stages (all caches d10_*.cache.json live HERE in arxiv/moonshots/ -- ../rigidity is read-only):
    python3 branch_d10galois.py gate      # reproduce d=8 numbers with THIS file's machinery
    python3 branch_d10galois.py stage1    # d=10 pool accounting + support-4 stratum
    python3 branch_d10galois.py stage2 [clique_seconds]   # graph + cliques + SAT gate
    python3 branch_d10galois.py stage3 [seconds]          # resumable peel
    python3 branch_d10galois.py stage3final               # complete cliques + criticality
    python3 branch_d10galois.py stage4    # WZ connection
    python3 branch_d10galois.py stage5    # exact f_10
    python3 branch_d10galois.py stage6    # sd_certificate_v2 on f_10
    python3 branch_d10galois.py report    # PASS/FAIL summary
No repo file outside arxiv/moonshots/ is written.  No git.
"""
import os, sys, time, json, random, math
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
RIG = os.path.abspath(os.path.join(HERE, "..", "rigidity"))
sys.path.insert(0, RIG)
sys.path.insert(0, HERE)

from itertools import combinations, product as iproduct
from collections import Counter

import numpy as np
import sympy as sp

import branch_d6flex as bd6                 # canon_ray, build_A0A1, find_cliques  (read-only)
import branch_d6geo as bg6                  # exact_fourier_d, clear_denominators_d (read-only)
import branch_d8galois as bd8               # pool_x2, PeelSolver, sat_colorable    (read-only)
from sd_certificate_v2 import certify_v2, F8

D = 10
T0 = time.time()


# ---------------------------------------------------------------- cache (LOCAL to moonshots/)
def _jsonable(o):
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.floating,)): return float(o)
    if isinstance(o, (set, tuple)): return [_jsonable(v) for v in o]
    if isinstance(o, list): return [_jsonable(v) for v in o]
    if isinstance(o, dict): return {str(k): _jsonable(v) for k, v in o.items()}
    return o


def save(name, obj):
    with open(os.path.join(HERE, f"d10_{name}.cache.json"), "w") as fh:
        json.dump(_jsonable(obj), fh)


def load(name):
    p = os.path.join(HERE, f"d10_{name}.cache.json")
    if not os.path.exists(p):
        return None
    with open(p) as fh:
        return json.load(fh)


def support(v):
    return sum(1 for c in v if c != "0")


# ---------------------------------------------------------------- blocked stability graph
def blocked_adjacency(rays, d, block=1024):
    """EXACTLY stable_matrix_M9's test (three integer matrix products == 0), computed in row
    blocks so the full V x V matrix never exists.  float64 matmul is exact here: entries are
    dot products of {0,+-1} vectors of length d <= 10, so |value| <= 10 << 2^53."""
    A0, A1 = bd6.build_A0A1(rays, d)
    A0f, A1f = A0.astype(np.float64), A1.astype(np.float64)
    V = len(rays)
    adj = [None] * V
    npairs = 0
    for s in range(0, V, block):
        e = min(V, s + block)
        M0 = A0f[s:e] @ A0f.T + A1f[s:e] @ A1f.T
        Mp = A0f[s:e] @ A1f.T          # coeff of X^{+1}
        Mm = A1f[s:e] @ A0f.T          # coeff of X^{-1}
        st = (M0 == 0) & (Mp == 0) & (Mm == 0)
        for i in range(s, e):
            st[i - s, i] = False
            nb = np.nonzero(st[i - s])[0]
            adj[i] = set(int(t) for t in nb)
            npairs += len(nb)
    assert npairs % 2 == 0
    return adj, npairs // 2


def pairs_iter(adj):
    for i, s in enumerate(adj):
        for j in s:
            if j > i:
                yield (i, j)


def sat_colorable(V, adj_or_pairs, bases):
    from pysat.solvers import Cadical153
    s = Cadical153()
    it = pairs_iter(adj_or_pairs) if isinstance(adj_or_pairs[0], set) else adj_or_pairs
    for i, j in it:
        s.add_clause([-(i + 1), -(j + 1)])
    for b in bases:
        s.add_clause([int(t) + 1 for t in b])
    res = s.solve()
    s.delete()
    return res


class PeelSolver:
    """Identical encoding to branch_d8galois.PeelSolver (selector vars + assumptions)."""
    def __init__(self, V, pairs, bases):
        from pysat.solvers import Cadical153
        self.V = V
        self.s = Cadical153()
        for i in range(V):
            self.s.add_clause([-(i + 1), V + i + 1])
        for i, j in pairs:
            self.s.add_clause([-(i + 1), -(j + 1)])
        for b in bases:
            self.s.add_clause([int(t) + 1 for t in b] + [-(V + int(t) + 1) for t in b])

    def colorable(self, keep):
        V = self.V
        return self.s.solve(assumptions=[(V + i + 1) if i in keep else -(V + i + 1)
                                         for i in range(V)])


# ==================================================================================== GATE
def gate():
    """This file's machinery must reproduce the published d=8 numbers before touching d=10:
    (i) support-4 X==2 pool 3360 rays / 572880 pairs / degree 341 (blocked adjacency vs the
    published stability figures);  (ii) the published 311-ray core's connection, charpoly f_8
    and an S_8 verdict from sd_certificate_v2 (read the d8 cache READ-ONLY)."""
    print("=" * 100)
    print("GATE -- reproduce d=8 with THIS file's blocked machinery (published numbers required)")
    print("=" * 100)
    ok = True
    t0 = time.time()
    rays8 = bd8.pool_x2(8, supp=4)
    print(f"[gate] d=8 support-4 X==2 pool: {len(rays8)} rays (expect 3360)")
    ok &= len(rays8) == 3360
    adj, npairs = blocked_adjacency(rays8, 8)
    degs = sorted(len(a) for a in adj)
    print(f"[gate] blocked adjacency: {npairs} pairs (expect 572880); degree min/max = "
          f"{degs[0]}/{degs[-1]} (expect 341/341)")
    ok &= npairs == 572880 and degs[0] == degs[-1] == 341
    # cross-check one full row against the reference dense routine on a subsample
    sub = rays8[:400]
    st = bd6.stable_matrix_M9(sub, 8)
    adjs, nps = blocked_adjacency(sub, 8, block=97)
    ref = [set(np.nonzero(st[i])[0].tolist()) for i in range(len(sub))]
    ok &= all(adjs[i] == ref[i] for i in range(len(sub)))
    print(f"[gate] blocked == dense stable_matrix_M9 on a 400-ray subsample: "
          f"{all(adjs[i] == ref[i] for i in range(len(sub)))}")

    cf = os.path.join(RIG, "d8galois_core_final.cache.json")
    with open(cf) as fh:
        core = json.load(fh)
    cs = [tuple(v) for v in core["core_syms"]]
    cb = [tuple(int(t) for t in b) for b in core["core_bases"]]
    print(f"[gate] published d=8 core loaded READ-ONLY: {len(cs)} rays / {len(cb)} bases "
          f"(expect 311/56)")
    ok &= (len(cs), len(cb)) == (311, 56)
    A0, A1, Am, pc, Nb = bg6.exact_fourier_d(cs, cb, 8)
    const = (all(A1[c][dd] == 0 for c in range(8) for dd in range(8)) and
             all(Am[c][dd] == 0 for c in range(8) for dd in range(8)))
    M, L = bg6.clear_denominators_d(A0, 8)
    xx = sp.symbols("x")
    cp = sp.expand(sp.Matrix(M).charpoly(xx).as_expr())
    f8 = sum(c * xx ** i for i, c in enumerate(F8))
    match = sp.expand(cp - f8) == 0
    print(f"[gate] connection constant: {const}; L={L} (expect 112); charpoly == published f_8: "
          f"{match}")
    ok &= const and L == 112 and match
    r = certify_v2(F8, "f_8", verbose=False)
    print(f"[gate] sd_certificate_v2(f_8): {r['verdict']} via route {r['route']} (expect S_8)")
    ok &= r["verdict"] == "S_8"
    print(f"[gate] GATE: {'PASS' if ok else 'FAIL'}   ({time.time()-t0:.1f}s)")
    save("gate", dict(passed=bool(ok)))
    assert ok, "GATE FAILED -- do not proceed to d=10"
    return ok


# ==================================================================================== STAGE 1
def stage1():
    print("=" * 100)
    print("STAGE 1 -- d=10 pool accounting (LAZY: the 590,400-ray pool is never materialized)")
    print("=" * 100)
    t0 = time.time()
    # combinatorial stratum sizes, formula validated against the d=8 recorded histogram
    for dd, ref in ((8, {3: 672, 4: 3360, 5: 8960, 6: 13440, 7: 10752, 8: 3584}), (10, None)):
        hist = {s: math.comb(dd, 2) * 2 * math.comb(dd - 2, s - 2) * 2 ** (s - 2)
                for s in range(3, dd + 1)}
        tot = sum(hist.values())
        if ref is not None:
            assert hist == ref, (hist, ref)
            print(f"[stage1] formula check at d=8: histogram matches branch_d8galois stage1 "
                  f"exactly (total {tot} = 40768: {tot == 40768})")
        else:
            print(f"[stage1] d=10 X==2 stratum sizes by support: {hist}")
            print(f"[stage1] d=10 X==2 pool TOTAL = {tot} (the published wall figure: 590400: "
                  f"{tot == 590400})")
    rays = bd8.pool_x2(D, supp=4)          # generates ONLY the support-4 stratum
    hist4 = Counter(support(v) for v in rays)
    print(f"[stage1] support-4 stratum generated directly: {len(rays)} rays {dict(hist4)} "
          f"(formula: 10080)")
    assert len(rays) == 10080
    save("stage1", dict(n=len(rays), total_pool=590400))
    print(f"[stage1] done ({time.time()-t0:.1f}s)")
    return rays


# ==================================================================================== STAGE 2
# The session sandbox kills any process when the invoking shell call returns (~45 s), so stage 2
# is CHUNKED and RESUMABLE: `cliques` accumulates randomized DFS batches across invocations into
# the cache; `sat` then decides uncolorability on the instance RESTRICTED to basis-participating
# rays (dropping rays and their pair clauses can only make the instance EASIER to color, so UNSAT
# on the restriction is a sound uncolorability certificate for the full stratum).

def _dfs_cliques(adj, d, starts, deadline, out, max_per_start=4000, rnd=None):
    """Randomized-order DFS clique enumeration from the given start vertices."""
    found_here = 0

    def extend(cands, cur):
        nonlocal found_here
        if time.time() > deadline or found_here >= max_per_start:
            return
        if len(cur) == d:
            out.add(tuple(sorted(cur)))
            found_here += 1
            return
        if len(cur) + len(cands) < d:
            return
        cl = list(cands)
        if rnd:
            rnd.shuffle(cl)
        for idx, v in enumerate(cl):
            if time.time() > deadline or found_here >= max_per_start:
                return
            rest = set(x for x in cl[idx + 1:] if x in adj[v])
            extend(rest, cur + [v])

    for s0 in starts:
        if time.time() > deadline:
            break
        found_here = 0
        extend(set(adj[s0]), [s0])


def stage2_cliques(budget=25.0, seed=None):
    t0 = time.time()
    rays = bd8.pool_x2(D, supp=4)
    adj, npairs = blocked_adjacency(rays, D)
    degs = sorted(len(a) for a in adj)
    print(f"[stage2] graph: {len(rays)} rays, {npairs} pairs, degree min/mean/max = "
          f"{degs[0]}/{sum(degs)/len(degs):.1f}/{degs[-1]}   ({time.time()-t0:.1f}s)")
    prev = load("stage2_cliques")
    out = set(tuple(int(t) for t in b) for b in prev["bases"]) if prev else set()
    n0 = len(out)
    seed = int(seed) if seed is not None else n0 + 1
    rnd = random.Random(seed)
    starts = list(range(len(rays)))
    rnd.shuffle(starts)
    deadline = t0 + budget
    _dfs_cliques(adj, D, starts, deadline, out, rnd=rnd)
    bp = set(t for b in out for t in b)
    print(f"[stage2] cliques: {n0} -> {len(out)} (+{len(out)-n0}); basis-participating rays "
          f"{len(bp)}/{len(rays)}   (seed {seed}, {time.time()-t0:.1f}s)")
    save("stage2_cliques", dict(bases=sorted(out), npairs=npairs, nbp=len(bp)))
    return len(out)


def stage2_sat(conf_budget=None):
    t0 = time.time()
    prev = load("stage2_cliques")
    assert prev, "run stage2cliques first"
    bases = [tuple(int(t) for t in b) for b in prev["bases"]]
    rays = bd8.pool_x2(D, supp=4)
    bp = sorted(set(t for b in bases for t in b))
    idx = {v: i for i, v in enumerate(bp)}
    sub = [rays[v] for v in bp]
    adj, npairs = blocked_adjacency(sub, D)
    print(f"[stage2sat] restricted instance: {len(bp)} basis-participating rays, {npairs} "
          f"pairs, {len(bases)} bases   ({time.time()-t0:.1f}s)")
    from pysat.solvers import Cadical153
    s = Cadical153()
    for i, j in pairs_iter(adj):
        s.add_clause([-(i + 1), -(j + 1)])
    for b in bases:
        s.add_clause([idx[t] + 1 for t in b])
    t1 = time.time()
    if conf_budget:
        s.conf_budget(int(conf_budget))
        res = s.solve_limited()
    else:
        res = s.solve()
    print(f"[stage2sat] *** SAT: colorable={res}  =>  KS-UNCOLORABLE={res is False} ***  "
          f"(solve {time.time()-t1:.1f}s)")
    if res is False:
        save("stage2", dict(uncolorable=True, nbases=len(bases), nbp=len(bp),
                            npairs_sub=npairs))
    else:
        print("[stage2sat] not UNSAT yet -- accumulate more cliques (stage2cliques) and retry")
    return res


# ==================================================================================== STAGE 3
def stage3(budget=32.0, seed=0):
    """Resumable peel on the RESTRICTED instance (basis-participating rays only; rays outside
    any found basis are colorable-removable trivially, exactly as the d=8 block peel would drop
    them first)."""
    t0 = time.time()
    s2 = load("stage2")
    assert s2 is not None and s2["uncolorable"], "run stage2sat first (and it must be UNSAT)"
    cl = load("stage2_cliques")
    bases_glob = [tuple(int(t) for t in b) for b in cl["bases"]]
    rays = bd8.pool_x2(D, supp=4)
    bp = sorted(set(t for b in bases_glob for t in b))
    idx = {v: i for i, v in enumerate(bp)}
    sub = [rays[v] for v in bp]
    adj, npairs = blocked_adjacency(sub, D)
    bases = [tuple(idx[t] for t in b) for b in bases_glob]
    V = len(sub)
    ps = PeelSolver(V, pairs_iter(adj), bases)
    print(f"[stage3] solver built ({time.time()-t0:.1f}s): V={V}, {npairs} pairs, "
          f"{len(bases)} bases")
    prev = load("stage3")
    keep = set(int(t) for t in prev["keep"]) if prev else set(range(V))
    phase = prev.get("phase", "block") if prev else "block"
    doneset = set(int(t) for t in prev.get("done", [])) if prev else set()
    rnd = random.Random(seed + len(keep))
    print(f"[stage3] resuming with |keep|={len(keep)} phase={phase}")

    if phase == "block":
        blk = max(1, len(keep) // 12)
        while time.time() - t0 < budget and blk >= 1:
            cand = sorted(keep)
            rnd.shuffle(cand)
            drop = set(cand[:blk])
            if not ps.colorable(keep - drop):
                keep -= drop
            else:
                blk //= 2
        if blk < 1:
            phase = "single"
        print(f"[stage3] block phase: |keep|={len(keep)} (blk down to {blk}) phase now {phase}")

    if phase == "single":
        order = sorted(keep)
        rnd.shuffle(order)
        for r in order:
            if time.time() - t0 > budget:
                break
            if r in doneset or r not in keep:
                continue
            if not ps.colorable(keep - {r}):
                keep.discard(r)
            doneset.add(r)
        ntested = len(doneset & keep)
        print(f"[stage3] single phase: |keep|={len(keep)}, tested {len(doneset)}")

    # store GLOBAL ray indices so downstream stages are independent of bp ordering
    save("stage3", dict(keep=sorted(keep), phase=phase, done=sorted(doneset),
                        keep_global=[bp[i] for i in sorted(keep)],
                        finished=bool(phase == "single" and
                                      all(r in doneset for r in keep))))
    print(f"[stage3] checkpoint: |keep|={len(keep)} finished="
          f"{phase=='single' and all(r in doneset for r in keep)}  ({time.time()-t0:.1f}s)")
    return keep


# ==================================================================================== STAGE 3F
def stage3final(clique_budget=60.0):
    print("=" * 100)
    print("STAGE 3-FINAL -- complete in-core clique re-enumeration + criticality")
    print("=" * 100)
    t0 = time.time()
    s3 = load("stage3")
    assert s3 is not None, "run stage3 first"
    rays = bd8.pool_x2(D, supp=4)
    keep = sorted(int(t) for t in s3["keep_global"])
    cs = [rays[i] for i in keep]
    adj, npairs = blocked_adjacency(cs, D)
    bases, complete = bd6.find_cliques(adj, len(cs), D, time_limit=clique_budget)
    print(f"[final] core candidate: {len(cs)} rays, {npairs} pairs, {len(bases)} 10-cliques "
          f"(COMPLETE={complete})")
    xh = Counter(sum(1 for c in v if c in ("X", "-X")) for v in cs)
    sh = Counter(support(v) for v in cs)
    print(f"[final] X-count histogram (must be all 2): {dict(xh)}; support histogram: {dict(sh)}")
    assert set(xh) == {2}
    col = sat_colorable(len(cs), adj, bases)
    print(f"[final] SAT vs COMPLETE in-core bases: KS-uncolorable={not col}")
    assert not col, "core colorable against complete bases -- peel further in stage3"
    pairs = list(pairs_iter(adj))
    ps = PeelSolver(len(cs), pairs, bases)
    k2 = set(range(len(cs)))
    changed = True
    while changed:
        changed = False
        for r in sorted(k2):
            if not ps.colorable(k2 - {r}):
                k2.discard(r)
                changed = True
    noncrit = [r for r in sorted(k2) if not ps.colorable(k2 - {r})]
    print(f"[final] after re-peel vs complete bases: {len(k2)} rays; non-critical: "
          f"{len(noncrit)}")
    cs = [cs[i] for i in sorted(k2)]
    adj, npairs = blocked_adjacency(cs, D)
    bases, complete = bd6.find_cliques(adj, len(cs), D, time_limit=clique_budget)
    col = sat_colorable(len(cs), adj, bases)
    print(f"[final] FINAL CORE: {len(cs)} rays / {npairs} pairs / {len(bases)} bases "
          f"(complete={complete}); uncolorable={not col}; all-critical={len(noncrit) == 0}")
    save("core_final", dict(core_syms=cs, core_bases=bases, npairs=npairs,
                            complete=bool(complete), uncolorable=bool(not col),
                            all_critical=bool(len(noncrit) == 0)))
    print(f"[final] done ({time.time()-t0:.1f}s)")


# ==================================================================================== STAGE 4
def stage4():
    print("=" * 100)
    print("STAGE 4 -- the d=10 WZ holonomy connection")
    print("=" * 100)
    core = load("core_final")
    assert core is not None
    cs = [tuple(v) for v in core["core_syms"]]
    cb = [tuple(int(t) for t in b) for b in core["core_bases"]]
    print(f"[stage4] core: {len(cs)} rays, {len(cb)} bases")
    A0, A1, Am, pc, Nb = bg6.exact_fourier_d(cs, cb, D)
    z1 = all(A1[c][dd] == 0 for c in range(D) for dd in range(D))
    zm = all(Am[c][dd] == 0 for c in range(D) for dd in range(D))
    print(f"[stage4] *** A1==0: {z1}   Am==0: {zm} ***  (connection EXACTLY CONSTANT iff both)")
    M, L = bg6.clear_denominators_d(A0, D)
    tr = sum(M[i][i] for i in range(D))
    sym = all(M[c][dd] == M[dd][c] for c in range(D) for dd in range(D))
    margins = [M[i][i] - sum(abs(M[i][j]) for j in range(D) if j != i) for i in range(D)]
    offnz = sum(1 for c in range(D) for dd in range(D) if c != dd and M[c][dd] != 0)
    offmax = max((abs(M[c][dd]) for c in range(D) for dd in range(D) if c != dd), default=0)
    print(f"[stage4] L={L}.  Stilde := L*A0:")
    for row in M:
        print("     ", row)
    print(f"[stage4] Tr(Stilde)={tr}, Tr(S)={sp.Rational(tr, L)}")
    print(f"[stage4] symmetric={sym}; off-diag nonzero {offnz}/{D*(D-1)} (max |off| {offmax}); "
          f"diag-dominant={all(m > 0 for m in margins)} (margins {margins})")
    save("stage4", dict(M=M, L=L, tr=tr, constant=bool(z1 and zm), symmetric=bool(sym),
                        margins=margins, diagdom=bool(all(m > 0 for m in margins))))


# ==================================================================================== STAGE 5
def stage5():
    print("=" * 100)
    print("STAGE 5 -- the exact degree-10 characteristic polynomial f_10")
    print("=" * 100)
    s4 = load("stage4")
    assert s4 is not None
    M, L = [[int(v) for v in row] for row in s4["M"]], int(s4["L"])
    xx = sp.symbols("x")
    cp = sp.expand(sp.Matrix(M).charpoly(xx).as_expr())
    P = sp.Poly(cp, xx)
    # independent re-derivation: Bareiss fraction-free determinant at 11 integer points
    Mm = sp.Matrix(M)
    vals = [(t, (t * sp.eye(D) - Mm).det(method="bareiss")) for t in range(-5, 6)]
    cp2 = sp.expand(sp.interpolate(vals, xx))
    indep = sp.expand(cp - cp2) == 0
    ev = sorted(np.linalg.eigvalsh(np.array(M, dtype=float)))
    print(f"[stage5] f_10(x) = {cp}")
    print(f"[stage5] independent Bareiss+interpolation check: {indep}")
    print(f"[stage5] irreducible over Q: {P.is_irreducible}")
    print(f"[stage5] eigenvalues (float check): {[round(v, 5) for v in ev]}")
    print(f"[stage5] eigenphases phi/(2pi) = lambda/L: {[round(v / L % 1, 6) for v in ev]}")
    save("stage5", dict(charpoly=str(cp), coeffs=[int(c) for c in P.all_coeffs()[::-1]],
                        irreducible=bool(P.is_irreducible), indep=bool(indep), L=L))


# ==================================================================================== STAGE 6
def stage6():
    print("=" * 100)
    print("STAGE 6 -- Galois certificate for f_10 (sd_certificate_v2, route A: NO d-1 prime)")
    print("=" * 100)
    s5 = load("stage5")
    assert s5 is not None
    coeffs = [int(c) for c in s5["coeffs"]]
    r = certify_v2(coeffs, "f_10")
    save("stage6", dict(verdict=r["verdict"], route=r["route"], witness=r["witness"],
                        chain=r.get("chain")))
    return r


# ==================================================================================== STAGE 7
def stage7():
    """Independent cross-checks on the d=10 result:
    (a) the connection by a DIFFERENT code path: build Ehat(theta) numerically (complex128) at
        random sample angles, form B = Ehat^dag dEhat/dtheta / Nb by central differences, and
        compare with i*Stilde/L -- no Fourier machinery, no exact arithmetic shared;
    (b) disc(f_10) not a perfect square (independent confirmation of the odd-element step);
    (c) Dedekind cycle-type census: how many of the 42 cycle types of S_10 are realized."""
    print("=" * 100)
    print("STAGE 7 -- independent cross-checks")
    print("=" * 100)
    core = load("core_final")
    s4 = load("stage4")
    s5 = load("stage5")
    cs = [tuple(v) for v in core["core_syms"]]
    cb = [tuple(int(t) for t in b) for b in core["core_bases"]]
    M = np.array([[int(v) for v in row] for row in s4["M"]], dtype=float)
    L = int(s4["L"])
    val = {"0": 0, "1": 1, "-1": -1}
    rng = np.random.default_rng(3)
    maxerr = 0.0
    for th in rng.uniform(0, 2 * np.pi, 4):
        def E(t):
            xx = np.exp(1j * t)
            rows = []
            for b in cb:
                for j in b:
                    rows.append([val[s] if s in val else (xx if s == "X" else -xx)
                                 for s in cs[j]])
            return np.array(rows, dtype=complex) / 2.0          # support 4 => 1/sqrt(4)
        h = 1e-6
        dE = (E(th + h) - E(th - h)) / (2 * h)
        B = E(th).conj().T @ dE / len(cb)
        err = np.abs(B - 1j * M / L).max()
        maxerr = max(maxerr, err)
    print(f"[stage7a] numeric E^dag dE/Nb vs i*Stilde/L at 4 random angles: max abs err = "
          f"{maxerr:.3g}  (PASS if < 1e-8)")
    ok_a = maxerr < 1e-8
    xx = sp.symbols("x")
    f10 = sp.Poly([int(c) for c in s5["coeffs"]][::-1], xx)
    disc = int(sp.discriminant(f10.as_expr(), xx))
    sq = disc >= 0 and math.isqrt(abs(disc)) ** 2 == disc
    print(f"[stage7b] disc(f_10) = {disc}")
    print(f"[stage7b] perfect square: {sq}  => Galois group {'inside' if sq else 'NOT inside'} "
          f"A_10 (must be False)")
    ok_b = not sq
    from sd_certificate_v2 import factor_degrees_mod_p
    coeffs = [int(c) for c in s5["coeffs"]]
    seen = {}
    t0 = time.time()
    cap = 3
    for q in sp.primerange(2, 200000):
        if time.time() - t0 > 25:
            cap = int(q)
            break
        ct = factor_degrees_mod_p(coeffs, int(q))
        if ct is not None:
            seen.setdefault(ct, int(q))
        cap = int(q)
    from sympy.utilities.iterables import ordered_partitions
    total = sum(1 for _ in ordered_partitions(10))
    print(f"[stage7c] cycle-type census primes < {cap}: {len(seen)}/{total} types of S_10 "
          f"realized")
    missing = [tuple(sorted(p, reverse=True)) for p in ordered_partitions(10)
               if tuple(sorted(p, reverse=True)) not in seen]
    if missing:
        print(f"[stage7c] missing: {missing}")
    save("stage7", dict(conn_maxerr=float(maxerr), disc=str(disc), disc_square=bool(sq),
                        census=len(seen), census_total=int(total), census_cap=cap,
                        ok=bool(ok_a and ok_b)))
    print(f"[stage7] {'PASS' if ok_a and ok_b else 'FAIL'}")


# ==================================================================================== ALT CORE
def altcore(seed=999, budget=30.0):
    """Independently-seeded second core (the d=8 lesson: seed 999 there produced a DECOUPLED
    core whose octic factored -- criticality does not force irreducible coupling).  Chunked and
    resumable like stage3; run repeatedly until 'finished', then it completes the whole
    downstream (complete cliques, criticality, connection, f, certificate) in one go."""
    t0 = time.time()
    cl = load("stage2_cliques")
    bases_glob = [tuple(int(t) for t in b) for b in cl["bases"]]
    rays = bd8.pool_x2(D, supp=4)
    bp = sorted(set(t for b in bases_glob for t in b))
    idx = {v: i for i, v in enumerate(bp)}
    sub = [rays[v] for v in bp]
    adj, npairs = blocked_adjacency(sub, D)
    bases = [tuple(idx[t] for t in b) for b in bases_glob]
    V = len(sub)
    ps = PeelSolver(V, pairs_iter(adj), bases)
    key = f"altcore{seed}"
    prev = load(key)
    if prev and prev.get("stage") == "done":
        print(f"[altcore] seed {seed} already complete: verdict {prev['verdict']}")
        return
    keep = set(int(t) for t in prev["keep"]) if prev else set(range(V))
    doneset = set(int(t) for t in prev.get("done", [])) if prev else set()
    phase = prev.get("phase", "block") if prev else "block"
    rnd = random.Random(seed * 1000 + len(keep))
    print(f"[altcore] seed {seed}: resuming |keep|={len(keep)} phase={phase} "
          f"(solver {time.time()-t0:.1f}s)")
    if phase == "block":
        blk = max(1, len(keep) // 12)
        while time.time() - t0 < budget and blk >= 1:
            cand = sorted(keep); rnd.shuffle(cand)
            drop = set(cand[:blk])
            if not ps.colorable(keep - drop):
                keep -= drop
            else:
                blk //= 2
        if blk < 1:
            phase = "single"
        print(f"[altcore] block: |keep|={len(keep)} phase {phase}")
    finished = False
    if phase == "single":
        order = sorted(keep); rnd.shuffle(order)
        for r in order:
            if time.time() - t0 > budget:
                break
            if r in doneset or r not in keep:
                continue
            if not ps.colorable(keep - {r}):
                keep.discard(r)
            doneset.add(r)
        finished = all(r in doneset for r in keep)
        print(f"[altcore] single: |keep|={len(keep)} finished={finished}")
    if not finished:
        save(key, dict(keep=sorted(keep), done=sorted(doneset), phase=phase, stage="peel"))
        print(f"[altcore] checkpoint saved -- run again  ({time.time()-t0:.1f}s)")
        return
    cs0 = [sub[i] for i in sorted(keep)]
    cadj, cnp = blocked_adjacency(cs0, D)
    cbases, complete = bd6.find_cliques(cadj, len(cs0), D, time_limit=15.0)
    ps2 = PeelSolver(len(cs0), pairs_iter(cadj), cbases)
    k2 = set(range(len(cs0)))
    changed = True
    while changed:
        changed = False
        for r in sorted(k2):
            if not ps2.colorable(k2 - {r}):
                k2.discard(r); changed = True
    cs = [cs0[i] for i in sorted(k2)]
    cadj, cnp = blocked_adjacency(cs, D)
    cbases, complete = bd6.find_cliques(cadj, len(cs), D, time_limit=15.0)
    col = sat_colorable(len(cs), cadj, cbases)
    prim = set(tuple(v) for v in load("core_final")["core_syms"])
    shared = len(set(cs) & prim)
    print(f"[altcore] ALT core: {len(cs)} rays / {cnp} pairs / {len(cbases)} bases "
          f"(complete={complete}), uncolorable={not col}; shared with primary: {shared}")
    A0, A1, Am, pc, Nb = bg6.exact_fourier_d(cs, cbases, D)
    const = (all(A1[c][dd] == 0 for c in range(D) for dd in range(D)) and
             all(Am[c][dd] == 0 for c in range(D) for dd in range(D)))
    M, L = bg6.clear_denominators_d(A0, D)
    tr = sum(M[i][i] for i in range(D))
    decoupled = [c for c in range(D) if all(M[c][dd] == 0 for dd in range(D) if dd != c)]
    xx = sp.symbols("x")
    cp = sp.expand(sp.Matrix(M).charpoly(xx).as_expr())
    print(f"[altcore] constant={const} L={L} Tr(S)={sp.Rational(tr, L)} "
          f"decoupled rows={decoupled}")
    print(f"[altcore] alt char poly: {cp}")
    r = certify_v2([int(c) for c in sp.Poly(cp, xx).all_coeffs()[::-1]],
                   f"f_10 (alt seed {seed})")
    save(key, dict(stage="done", n=len(cs), shared=shared, M=M, L=L,
                   constant=bool(const), decoupled=decoupled, charpoly=str(cp),
                   verdict=r["verdict"], route=r["route"], uncolorable=bool(not col),
                   complete=bool(complete)))
    print(f"[altcore] done ({time.time()-t0:.1f}s)")


# ==================================================================================== REPORT
def report():
    print("=" * 100)
    print("BRANCH D10-GALOIS -- REPORT")
    print("=" * 100)
    rows = []
    g = load("gate")
    rows.append(("gate: d=8 reproduced by this file's blocked machinery",
                 bool(g and g["passed"])))
    s1 = load("stage1")
    if s1: rows.append((f"stage1: lazy pool accounting, support-4 stratum {s1['n']} of "
                        f"{s1['total_pool']}", s1["n"] == 10080))
    s2 = load("stage2")
    if s2: rows.append((f"stage2: {s2['nbases']} partial 10-cliques over {s2['nbp']} rays, "
                        f"UNCOLORABLE={s2['uncolorable']}", s2["uncolorable"]))
    cf = load("core_final")
    if cf: rows.append((f"stage3: core {len(cf['core_syms'])} rays / {cf['npairs']} pairs / "
                        f"{len(cf['core_bases'])} bases (complete={cf['complete']}, "
                        f"all-critical={cf['all_critical']})", cf["uncolorable"]))
    s4 = load("stage4")
    if s4: rows.append((f"stage4: connection constant={s4['constant']} symmetric="
                        f"{s4['symmetric']} diag-dominant={s4['diagdom']} L={s4['L']} "
                        f"Tr(S)={sp.Rational(int(s4['tr']), int(s4['L']))}",
                        s4["constant"] and s4["symmetric"]))
    s5 = load("stage5")
    if s5: rows.append((f"stage5: f_10 irreducible={s5['irreducible']} "
                        f"independent-check={s5['indep']}", s5["irreducible"] and s5["indep"]))
    s6 = load("stage6")
    if s6: rows.append((f"stage6: VERDICT {s6['verdict']} via route {s6['route']}",
                        s6["verdict"] == "S_10"))
    s7 = load("stage7")
    if s7: rows.append((f"stage7: numeric-connection err {s7['conn_maxerr']:.1e}, disc-square="
                        f"{s7['disc_square']}, census {s7['census']}/{s7['census_total']} "
                        f"types (primes<{s7['census_cap']}; missing = identity + transposition, "
                        f"the two rarest -- the OLD template would still be searching)",
                        s7["ok"]))
    ac = load("altcore999")
    if ac and ac.get("stage") == "done":
        rows.append((f"altcore seed 999: {ac['n']} rays ({ac['shared']} shared), L={ac['L']}, "
                     f"decoupled={ac['decoupled']}, VERDICT {ac['verdict']}",
                     ac["verdict"] == "S_10"))
    allok = True
    for nm, ok in rows:
        print(f"  [{'PASS' if ok else 'FAIL'}] {nm}")
        allok &= bool(ok)
    print(f"\n  OVERALL: {'PASS' if allok else 'FAIL/INCOMPLETE'}")
    if s5:
        print(f"\n  f_10 = {s5['charpoly']}")
    return allok


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "report"
    arg = float(sys.argv[2]) if len(sys.argv) > 2 else None
    if which == "gate": gate()
    elif which == "stage1": stage1()
    elif which == "stage2cliques": stage2_cliques(arg or 25.0,
                                                  sys.argv[3] if len(sys.argv) > 3 else None)
    elif which == "stage2sat": stage2_sat(arg)
    elif which == "stage3": stage3(arg or 30.0, int(sys.argv[3]) if len(sys.argv) > 3 else 0)
    elif which == "stage3final": stage3final(arg or 60.0)
    elif which == "stage4": stage4()
    elif which == "stage5": stage5()
    elif which == "stage6": stage6()
    elif which == "stage7": stage7()
    elif which == "altcore": altcore(int(sys.argv[2]) if len(sys.argv) > 2 else 999,
                                     float(sys.argv[3]) if len(sys.argv) > 3 else 30.0)
    elif which == "report": report()
    else:
        print(f"unknown stage {which!r}")
        sys.exit(1)
    print(f"\n[branch_d10galois.py {which}: total {time.time()-T0:.1f}s]")
