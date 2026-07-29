#!/usr/bin/env python3
"""
branch_d12galois.py -- MOONSHOT: the d=12 rung of the KS circle tower.

Direct successor of branch_d10galois.py (read that file + D10_WALL.md first).  Same route:
  * the X-count==2 pool at d=12 fibers by support; the support-4 stratum has
        C(12,2) * 2 * C(10,2) * 2^2 = 66*2*45*4 = 23,760 rays
    (first-principles recount below, formula re-validated against the d=8 recorded histogram
    AND the d=10 published numbers).  The full pool, 66*2*(3^10-1) = 7,794,336 rays, is NEVER
    materialized.
  * blocked exact adjacency; partial-clique SAT uncolorability (sound: bases only add positive
    clauses, so UNSAT with a PARTIAL 12-clique list certifies stratum uncolorability, and
    a fortiori pool uncolorability); assumption-based incremental peel; complete in-core clique
    re-enumeration + criticality; exact WZ connection (Fraction); exact charpoly (Berkowitz +
    independent Bareiss-at-13-points + Lagrange); sd_certificate_v2 route A (p=7).
  * NEW vs d=10 (pure engineering, same mathematics):
      - the stratum is generated DIRECTLY (gen_s4: 47,520 raw sign patterns -> canon_ray dedup
        -> 23,760), gated against bd8.pool_x2 at d=8 by exact canon-key set equality;
      - adjacency is stored BIT-PACKED (23,760 x 2,970 uint8 = 70 MB) in d12_adj.cache.npz,
        because list-of-set adjacency at this scale would be GBs;
      - the peel REBUILDS THE SOLVER ON THE CURRENT KEEP-SET each chunk (sound: a pair clause
        with a deactivated endpoint and a basis clause with a dropped ray are both inactive in
        the selector encoding, so the restricted instance is EQUIVALENT for subsets of keep);
        this keeps every 45 s chunk inside budget as the instance shrinks.
  * NEW science stage `headtohead`: at d=12, d-1=11 is PRIME, so the OLD published template
    (irreducible + 11-cycle + literal transposition) is applicable IN PRINCIPLE for the first
    time since d=8 -- both templates are run on the same f_12 and their witness primes compared.

Stages (all caches d12_*.cache.json / d12_adj.cache.npz live HERE in arxiv/moonshots/):
    python3 branch_d12galois.py gate            # reproduce d=8 AND d=10 published numbers
    python3 branch_d12galois.py stage1          # pool accounting + the 23,760-ray stratum
    python3 branch_d12galois.py stage2graph     # blocked adjacency -> packed cache + stats
    python3 branch_d12galois.py stage2cliques [budget] [seed]   # accumulate 12-cliques
    python3 branch_d12galois.py stage2sat [conf_budget] [min_cnt]  # UNSAT gate
    python3 branch_d12galois.py stage3 [budget] [seed]          # resumable peel
    python3 branch_d12galois.py stage3final [clique_budget]     # complete cliques + criticality
    python3 branch_d12galois.py stage4          # WZ connection (+ trace law Tr S = d/2, L = 2Nb)
    python3 branch_d12galois.py stage5          # exact f_12
    python3 branch_d12galois.py stage6          # sd_certificate_v2 route A
    python3 branch_d12galois.py headtohead [budget]             # old vs new template on f_12
    python3 branch_d12galois.py stage7          # independent cross-checks
    python3 branch_d12galois.py altcore [seed] [budget]         # alt-seed peel (chunked)
    python3 branch_d12galois.py altcorefinal [seed]             # alt core downstream
    python3 branch_d12galois.py report          # PASS/FAIL summary
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
from fractions import Fraction as F

import numpy as np
import sympy as sp

import branch_d6flex as bd6                 # canon_ray, build_A0A1, find_cliques  (read-only)
import branch_d6geo as bg6                  # exact_fourier_d, clear_denominators_d (read-only)
import branch_d8galois as bd8               # pool_x2 (read-only; gate only -- too slow at d=12)
from sd_certificate_v2 import certify_v2, factor_degrees_mod_p, F8

D = 12
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
    with open(os.path.join(HERE, f"d12_{name}.cache.json"), "w") as fh:
        json.dump(_jsonable(obj), fh)


def load(name):
    p = os.path.join(HERE, f"d12_{name}.cache.json")
    if not os.path.exists(p):
        return None
    with open(p) as fh:
        return json.load(fh)


def support(v):
    return sum(1 for c in v if c != "0")


# ---------------------------------------------------------------- the stratum, generated directly
def gen_s4(d):
    """The support-4, X-count==2 stratum, generated DIRECTLY: two +-X entries, two +-1 entries.
    Raw count C(d,2)*4*C(d-2,2)*4; projective dedup (only +-1 global scalings keep the alphabet,
    since X*(+-1)=+-X but X*(+-X)=+-X^2 leaves it) via bd6.canon_ray halves it.  Gate: exact
    canon-key set equality with bd8.pool_x2(8, supp=4).  Deterministic order (sorted)."""
    seen = {}
    for (i, j) in combinations(range(d), 2):
        rest_pos = [p for p in range(d) if p != i and p != j]
        for s1 in ("X", "-X"):
            for s2 in ("X", "-X"):
                for (k, l) in combinations(rest_pos, 2):
                    for t1 in ("1", "-1"):
                        for t2 in ("1", "-1"):
                            v = ["0"] * d
                            v[i], v[j], v[k], v[l] = s1, s2, t1, t2
                            v = tuple(v)
                            key = bd6.canon_ray(v)
                            if key not in seen:
                                seen[key] = v
    return sorted(seen.values())


# ---------------------------------------------------------------- packed blocked adjacency
def packed_adjacency(rays, d, block=512):
    """EXACTLY stable_matrix_M9's test (three integer matrix products == 0) in row blocks;
    float64 matmul is exact ({0,+-1} dot products, |value| <= 12 << 2^53).  Returns the
    BIT-PACKED adjacency matrix (V x ceil(V/8) uint8), degree vector, and pair count."""
    A0, A1 = bd6.build_A0A1(rays, d)
    A0f, A1f = A0.astype(np.float64), A1.astype(np.float64)
    V = len(rays)
    P = np.zeros((V, (V + 7) // 8), dtype=np.uint8)
    degs = np.zeros(V, dtype=np.int64)
    for s in range(0, V, block):
        e = min(V, s + block)
        M0 = A0f[s:e] @ A0f.T + A1f[s:e] @ A1f.T
        Mp = A0f[s:e] @ A1f.T          # coeff of X^{+1}
        Mm = A1f[s:e] @ A0f.T          # coeff of X^{-1}
        st = (M0 == 0) & (Mp == 0) & (Mm == 0)
        for r in range(s, e):
            st[r - s, r] = False
        degs[s:e] = st.sum(axis=1)
        P[s:e] = np.packbits(st, axis=1)
    npairs = int(degs.sum())
    assert npairs % 2 == 0
    return P, degs, npairs // 2


def row_bool(P, v, V):
    return np.unpackbits(P[v])[:V].astype(bool)


def iter_pair_blocks(P, V, block=1024):
    """Yield (ii, jj) python lists of upper-triangle adjacent pairs, blockwise."""
    for s in range(0, V, block):
        e = min(V, s + block)
        st = np.unpackbits(P[s:e], axis=1)[:, :V].astype(bool)
        for r in range(e - s):
            st[r, :s + r + 1] = False          # strict upper triangle only
        ii, jj = np.nonzero(st)
        yield (ii + s).tolist(), jj.tolist()


def adj_sets_from_packed(P, V):
    return [set(np.nonzero(row_bool(P, i, V))[0].tolist()) for i in range(V)]


def pairs_iter_sets(adj):
    for i, s in enumerate(adj):
        for j in s:
            if j > i:
                yield (i, j)


def sat_colorable_packed(P, V, bases, conf_budget=None):
    from pysat.solvers import Cadical153
    s = Cadical153()
    add = s.add_clause
    for ii, jj in iter_pair_blocks(P, V):
        for i, j in zip(ii, jj):
            add([-(i + 1), -(j + 1)])
    for b in bases:
        add([int(t) + 1 for t in b])
    if conf_budget:
        s.conf_budget(int(conf_budget))
        res = s.solve_limited()          # None = inconclusive (only False is used as evidence)
    else:
        res = s.solve()
    s.delete()
    return res


class PeelSolver:
    """Identical encoding to branch_d10galois.PeelSolver (selector vars + assumptions)."""
    def __init__(self, V, pair_blocks, bases):
        from pysat.solvers import Cadical153
        self.V = V
        self.s = Cadical153()
        add = self.s.add_clause
        for i in range(V):
            add([-(i + 1), V + i + 1])
        for ii, jj in pair_blocks:
            for i, j in zip(ii, jj):
                add([-(i + 1), -(j + 1)])
        for b in bases:
            add([int(t) + 1 for t in b] + [-(V + int(t) + 1) for t in b])

    def colorable(self, keep, conf=None):
        """True/False = decided; None = conflict budget exhausted (callers must treat None as
        'colorable', which is the sound direction: only PROVEN UNSAT justifies a drop)."""
        V = self.V
        A = [(V + i + 1) if i in keep else -(V + i + 1) for i in range(V)]
        if conf:
            self.s.conf_budget(int(conf))
            return self.s.solve_limited(assumptions=A)
        return self.s.solve(assumptions=A)


def peel_solver_sets(adj, bases):
    V = len(adj)
    ii, jj = [], []
    for i, j in pairs_iter_sets(adj):
        ii.append(i); jj.append(j)
    return PeelSolver(V, [(ii, jj)], bases)


# ==================================================================================== GATE
def gate():
    """This file's machinery must reproduce BOTH published rungs before touching d=12:
    (i)  d=8:  gen_s4 == pool_x2 canon-key sets (3360); packed adjacency 572,880 pairs, degree
         341 regular; packed vs dense stable_matrix_M9 on a subsample; the published 311-ray
         core's connection (L=112), f_8, S_8 verdict.
    (ii) d=10: gen_s4 gives 10,080 rays with 7,282,800 pairs, degree 1445 regular (the published
         d10 stage2 figures); the cached d10 stage4 matrix reproduces the published f_10."""
    print("=" * 100)
    print("GATE -- reproduce d=8 AND d=10 published numbers with THIS file's machinery")
    print("=" * 100)
    ok = True
    t0 = time.time()
    rays8 = gen_s4(8)
    pool8 = bd8.pool_x2(8, supp=4)
    same = set(map(bd6.canon_ray, rays8)) == set(map(bd6.canon_ray, pool8))
    print(f"[gate] d=8 gen_s4: {len(rays8)} rays; canon-key set == bd8.pool_x2(8,4): {same} "
          f"(expect 3360, True)")
    ok &= len(rays8) == 3360 and same
    P8, degs8, np8 = packed_adjacency(rays8, 8)
    print(f"[gate] d=8 packed adjacency: {np8} pairs (expect 572880); degree min/max = "
          f"{degs8.min()}/{degs8.max()} (expect 341/341)")
    ok &= np8 == 572880 and degs8.min() == degs8.max() == 341
    sub = rays8[:400]
    st = bd6.stable_matrix_M9(sub, 8)
    Ps, _, _ = packed_adjacency(sub, 8, block=97)
    agree = all(set(np.nonzero(row_bool(Ps, i, 400))[0]) ==
                set(np.nonzero(st[i])[0]) - {i} for i in range(400))
    print(f"[gate] packed == dense stable_matrix_M9 on a 400-ray subsample: {agree}")
    ok &= agree

    with open(os.path.join(RIG, "d8galois_core_final.cache.json")) as fh:
        core = json.load(fh)
    cs = [tuple(v) for v in core["core_syms"]]
    cb = [tuple(int(t) for t in b) for b in core["core_bases"]]
    ok &= (len(cs), len(cb)) == (311, 56)
    A0, A1, Am, pc, Nb = bg6.exact_fourier_d(cs, cb, 8)
    const = (all(A1[c][dd] == 0 for c in range(8) for dd in range(8)) and
             all(Am[c][dd] == 0 for c in range(8) for dd in range(8)))
    M, L = bg6.clear_denominators_d(A0, 8)
    xx = sp.symbols("x")
    cp = sp.expand(sp.Matrix(M).charpoly(xx).as_expr())
    f8 = sum(c * xx ** i for i, c in enumerate(F8))
    match = sp.expand(cp - f8) == 0
    r = certify_v2(F8, "f_8", verbose=False)
    print(f"[gate] d=8 published core: constant={const}, L={L} (expect 112), charpoly==f_8: "
          f"{match}, certificate: {r['verdict']} route {r['route']}")
    ok &= const and L == 112 and match and r["verdict"] == "S_8"

    rays10 = gen_s4(10)
    P10, degs10, np10 = packed_adjacency(rays10, 10)
    print(f"[gate] d=10 gen_s4: {len(rays10)} rays (expect 10080); packed adjacency {np10} "
          f"pairs (expect 7282800), degree {degs10.min()}/{degs10.max()} (expect 1445/1445)")
    ok &= len(rays10) == 10080 and np10 == 7282800 and degs10.min() == degs10.max() == 1445
    s4_10 = json.load(open(os.path.join(HERE, "d10_stage4.cache.json")))
    s5_10 = json.load(open(os.path.join(HERE, "d10_stage5.cache.json")))
    M10 = [[int(v) for v in row] for row in s4_10["M"]]
    cp10 = sp.expand(sp.Matrix(M10).charpoly(xx).as_expr())
    match10 = sp.expand(cp10 - sp.sympify(s5_10["charpoly"])) == 0
    print(f"[gate] d=10 cached stage4 matrix reproduces the published f_10: {match10}")
    ok &= match10
    print(f"[gate] GATE: {'PASS' if ok else 'FAIL'}   ({time.time()-t0:.1f}s)")
    save("gate", dict(passed=bool(ok)))
    assert ok, "GATE FAILED -- do not proceed to d=12"
    return ok


# ==================================================================================== STAGE 1
def stage1():
    print("=" * 100)
    print("STAGE 1 -- d=12 pool accounting (LAZY: the 7,794,336-ray pool is never materialized)")
    print("=" * 100)
    t0 = time.time()
    # first-principles recount: two X positions C(d,2); 4 X-sign patterns halved by the global
    # sign (the ONLY alphabet-preserving projective scaling) -> 2; tail = choose s-2 of d-2
    # positions, each +-1 -> C(d-2,s-2)*2^(s-2).
    refs = {8: {3: 672, 4: 3360, 5: 8960, 6: 13440, 7: 10752, 8: 3584},      # recorded d=8
            10: {3: 1440, 4: 10080, 5: 40320, 6: 100800, 7: 161280, 8: 161280,
                 9: 92160, 10: 23040}}                                        # published d=10
    for dd in (8, 10, 12):
        hist = {s: math.comb(dd, 2) * 2 * math.comb(dd - 2, s - 2) * 2 ** (s - 2)
                for s in range(3, dd + 1)}
        tot = sum(hist.values())
        closed = math.comb(dd, 2) * 2 * (3 ** (dd - 2) - 1)
        assert tot == closed
        if dd in refs:
            assert hist == refs[dd], (dd, hist)
            print(f"[stage1] formula check at d={dd}: histogram matches the recorded one "
                  f"exactly (total {tot})")
        else:
            print(f"[stage1] d=12 X==2 stratum sizes by support: {hist}")
            print(f"[stage1] d=12 X==2 pool TOTAL = {tot} = C(12,2)*2*(3^10-1) "
                  f"(D10_WALL predicted support-4 ~ 23760: {hist[4] == 23760})")
    rays = gen_s4(D)
    hist4 = Counter(support(v) for v in rays)
    xh = Counter(sum(1 for c in v if c in ("X", "-X")) for v in rays)
    print(f"[stage1] support-4 stratum generated directly: {len(rays)} rays; support hist "
          f"{dict(hist4)}; X-count hist {dict(xh)}")
    assert len(rays) == 23760 and set(hist4) == {4} and set(xh) == {2}
    # dedup sanity: canon keys pairwise distinct and closed under global sign
    keys = set(map(bd6.canon_ray, rays))
    neg = {bd6.canon_ray(tuple({"X": "-X", "-X": "X", "1": "-1", "-1": "1", "0": "0"}[c]
                              for c in v)) for v in rays}
    print(f"[stage1] canon keys distinct: {len(keys) == len(rays)}; closed under global sign: "
          f"{neg == keys}")
    assert len(keys) == len(rays) and neg == keys
    save("stage1", dict(n=len(rays), total_pool=7794336, hist={str(k): v
                                                               for k, v in hist4.items()}))
    print(f"[stage1] done ({time.time()-t0:.1f}s)")
    return rays


# ==================================================================================== STAGE 2
def stage2_graph():
    t0 = time.time()
    rays = gen_s4(D)
    P, degs, npairs = packed_adjacency(rays, D)
    print(f"[stage2graph] graph: {len(rays)} rays, {npairs} pairs, degree min/mean/max = "
          f"{degs.min()}/{degs.mean():.1f}/{degs.max()}   ({time.time()-t0:.1f}s)")
    np.savez(os.path.join(HERE, "d12_adj.cache.npz"), P=P, degs=degs, npairs=npairs,
             V=len(rays))
    save("stage2graph", dict(V=len(rays), npairs=npairs, degmin=int(degs.min()),
                             degmax=int(degs.max())))
    print(f"[stage2graph] packed adjacency cached ({P.nbytes/1e6:.0f} MB)   "
          f"({time.time()-t0:.1f}s)")


def load_packed():
    z = np.load(os.path.join(HERE, "d12_adj.cache.npz"))
    return z["P"], int(z["V"]), int(z["npairs"])


def _dfs_cliques_packed(P, V, d, starts, deadline, out, max_per_start=3000, rnd=None):
    """Randomized-order DFS 12-clique enumeration on the packed adjacency."""
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
        cl = np.array(cl, dtype=np.int64)
        for idx in range(len(cl)):
            if time.time() > deadline or found_here >= max_per_start:
                return
            v = int(cl[idx])
            rb = row_bool(P, v, V)
            rest = cl[idx + 1:]
            rest = rest[rb[rest]]
            extend(rest.tolist(), cur + [v])

    for s0 in starts:
        if time.time() > deadline:
            break
        found_here = 0
        extend(np.nonzero(row_bool(P, s0, V))[0].tolist(), [s0])


def stage2_cliques(budget=25.0, seed=None):
    t0 = time.time()
    P, V, npairs = load_packed()
    prev = load("stage2_cliques")
    out = set(tuple(int(t) for t in b) for b in prev["bases"]) if prev else set()
    n0 = len(out)
    seed = int(seed) if seed is not None else n0 + 1
    rnd = random.Random(seed)
    starts = list(range(V))
    rnd.shuffle(starts)
    deadline = T0 + budget
    _dfs_cliques_packed(P, V, D, starts, deadline, out, rnd=rnd)
    bp = set(t for b in out for t in b)
    print(f"[stage2] cliques: {n0} -> {len(out)} (+{len(out)-n0}); basis-participating rays "
          f"{len(bp)}/{V}   (seed {seed}, {time.time()-t0:.1f}s)")
    save("stage2_cliques", dict(bases=sorted(out), npairs=npairs, nbp=len(bp)))
    return len(out)


def weak_pair_keys(bases_g, V):
    """The co-basis orthogonal pairs (i<j sharing at least one found 12-clique), as unique
    int64 keys i*V+j.  Cached in d12_weakpairs.cache.npz.  SOUNDNESS: any SUBSET of the pair
    clauses only makes the instance easier to color, so UNSAT with co-basis pairs alone is a
    valid uncolorability certificate; 'colorable' answers become optimistic, which only makes
    the peel more conservative (stage3final re-verifies with the FULL in-core adjacency)."""
    p = os.path.join(HERE, "d12_weakpairs.cache.npz")
    if os.path.exists(p):
        z = np.load(p)
        if int(z["nbases"]) == len(bases_g):
            return z["keys"]
    B = np.array(bases_g, dtype=np.int64)
    ij = np.array(list(combinations(range(D), 2)), dtype=np.int64)
    keys = (B[:, ij[:, 0]] * V + B[:, ij[:, 1]]).ravel()      # bases are sorted tuples: i<j
    keys = np.unique(keys)
    np.savez(p, keys=keys, nbases=len(bases_g))
    return keys


def stage2_sat(conf_budget=None, mode="weak"):
    """UNSAT gate on the instance RESTRICTED to basis-participating rays.  mode='weak' uses only
    co-basis pair clauses (see weak_pair_keys: sound for UNSAT, and it is what fits in 3.9 GB at
    this scale -- the full 46M-pair-clause instance OOMs); mode='full' uses all pairs.
    Restriction to bp rays is sound as at d=10: dropping rays/clauses only helps the colorer, so
    UNSAT certifies uncolorability of the full stratum, a fortiori of the 7,794,336-ray pool."""
    t0 = time.time()
    prev = load("stage2_cliques")
    assert prev, "run stage2cliques first"
    bases_g = [tuple(int(t) for t in b) for b in prev["bases"]]
    Vfull = 23760
    bp = sorted(set(t for b in bases_g for t in b))
    idx = np.full(Vfull, -1, dtype=np.int64)
    idx[bp] = np.arange(len(bp))
    bases = [tuple(int(idx[t]) for t in b) for b in bases_g]
    from pysat.solvers import Cadical153
    s = Cadical153()
    add = s.add_clause
    if mode == "weak":
        keys = weak_pair_keys(bases_g, Vfull)
        ii = idx[keys // Vfull].tolist()
        jj = idx[keys % Vfull].tolist()
        npairs = len(ii)
        for i, j in zip(ii, jj):
            add([-(i + 1), -(j + 1)])
    else:
        rays = gen_s4(D)
        sub = [rays[v] for v in bp]
        P, degs, npairs = packed_adjacency(sub, D)
        for ii, jj in iter_pair_blocks(P, len(bp)):
            for i, j in zip(ii, jj):
                add([-(i + 1), -(j + 1)])
    for b in bases:
        add([int(t) + 1 for t in b])
    print(f"[stage2sat] restricted instance (mode={mode}): {len(bp)} rays, {npairs} pair "
          f"clauses, {len(bases)} bases   ({time.time()-t0:.1f}s)")
    t1 = time.time()
    if conf_budget:
        s.conf_budget(int(conf_budget))
        res = s.solve_limited()
    else:
        res = s.solve()
    s.delete()
    print(f"[stage2sat] *** SAT: colorable={res}  =>  KS-UNCOLORABLE={res is False} ***  "
          f"(solve {time.time()-t1:.1f}s)")
    if res is False:
        save("stage2", dict(uncolorable=True, nbases=len(bases), nbp=len(bp),
                            npairs_sub=npairs, mode=mode, bp=bp))
    else:
        print("[stage2sat] not UNSAT yet -- accumulate more cliques (stage2cliques) and retry")
    return res


# ==================================================================================== STAGE 3
FULL_THRESHOLD = 8500          # above this many rays the full-pair PeelSolver would OOM/overrun


def topup_cliques(kg, rays, budget, rnd):
    """Enumerate extra 12-cliques INSIDE the current keep set and merge them (as GLOBAL ids)
    into the stage2_cliques cache.  Bases only ADD positive clauses -- always sound -- and this
    keeps the peel instance basis-rich along the peel trajectory (without it, the restriction
    of the global random clique list thins out and the peel stalls)."""
    sub = [rays[v] for v in kg]
    P, degs, npairs = packed_adjacency(sub, D)
    out = set()
    starts = list(range(len(kg)))
    rnd.shuffle(starts)
    _dfs_cliques_packed(P, len(kg), D, starts, time.time() + budget, out, rnd=rnd)
    cl = load("stage2_cliques")
    bases = set(tuple(int(t) for t in b) for b in cl["bases"])
    n0 = len(bases)
    for b in out:
        bases.add(tuple(sorted(kg[i] for i in b)))
    save("stage2_cliques", dict(bases=sorted(bases), npairs=cl["npairs"],
                                nbp=len(set(t for b in bases for t in b))))
    return len(bases) - n0


def _build_restricted(keep_global, bases_g, rays):
    """Instance restricted to keep_global (sorted list of GLOBAL ray ids).  Equivalent to the
    full selector encoding for all subsets of keep_global (inactive clauses removed).  Uses the
    FULL pair set when |keep| <= FULL_THRESHOLD, else the sound weak co-basis pair set."""
    kg = sorted(keep_global)
    kset = set(kg)
    idx = np.full(23760, -1, dtype=np.int64)
    idx[kg] = np.arange(len(kg))
    bases = [tuple(int(idx[t]) for t in b) for b in bases_g if all(t in kset for t in b)]
    if len(kg) <= FULL_THRESHOLD:
        sub = [rays[v] for v in kg]
        P, degs, npairs = packed_adjacency(sub, D)
        return kg, iter_pair_blocks(P, len(kg)), npairs, bases, "full"
    keys = weak_pair_keys(bases_g, 23760)
    gi, gj = keys // 23760, keys % 23760
    m = (idx[gi] >= 0) & (idx[gj] >= 0)
    ii, jj = idx[gi[m]].tolist(), idx[gj[m]].tolist()
    return kg, [(ii, jj)], len(ii), bases, "weak"


def _peel_chunk(cache_name, seed, budget, t0, tag):
    """One resumable peel chunk, shared by stage3 (cache 'stage3') and altcore.
    PHASE 'core': solve(keep) must be UNSAT; Cadical's failed-assumption core gives a subset
      S of the selector assumptions with instance(S) UNSAT.  Setting keep' = {positive selectors
      in S} and deactivating EVERYTHING else is at least as constraining as the core's own
      assumption set, so keep' stays uncolorable.  [SOUND, and far faster than block peeling.]
    PHASE 'single': classical one-ray criticality peel, entered only on the FULL pair set."""
    s2 = load("stage2")
    assert s2 is not None and s2["uncolorable"], "run stage2sat first (and it must be UNSAT)"
    rays = gen_s4(D)
    prev = load(cache_name)
    if prev:
        keep_global = [int(t) for t in prev["keep_global"]]
        phase = prev.get("phase", "core")
        if phase == "block":
            phase = "core"          # migrate pre-core-phase checkpoints
        done_global = set(int(t) for t in prev.get("done_global", []))
    else:
        keep_global = [int(t) for t in s2["bp"]]
        phase, done_global = "core", set()
    rnd = random.Random(seed * 1009 + len(keep_global))
    nnew = topup_cliques(sorted(keep_global), rays, min(8.0, budget / 4), rnd)
    bases_g = [tuple(int(t) for t in b) for b in load("stage2_cliques")["bases"]]
    kg, pair_blocks, npairs, bases, pmode = _build_restricted(keep_global, bases_g, rays)
    V = len(kg)
    ps = PeelSolver(V, pair_blocks, bases)
    print(f"[{tag}] solver rebuilt ({time.time()-t0:.1f}s): V={V}, {npairs} pair clauses "
          f"({pmode}), {len(bases)} bases (+{nnew} topped up), phase={phase}")
    keep = set(range(V))

    if phase == "core":
        stall = 0
        while time.time() - t0 < budget and stall < 2:
            ts = time.time()
            res = ps.colorable(keep, conf=1000000)
            if res is False:
                core = ps.s.get_core() or []
                pos = set(l - V - 1 for l in core if l > V)
                if len(pos) < len(keep):
                    # randomize which rays get retried next round
                    keep = pos
                    stall = 0
                else:
                    stall += 1
            elif res is None:
                stall += 1
            else:
                raise RuntimeError("keep set became colorable -- soundness bug")
            print(f"[{tag}]   core step: res={res} |keep|={len(keep)} "
                  f"({time.time()-ts:.1f}s)", flush=True)
        if stall >= 2 and pmode == "full":
            phase = "single"
        print(f"[{tag}] core phase: |keep|={len(keep)} phase now {phase}")

    if phase == "single":
        order = sorted(keep)
        rnd.shuffle(order)
        for r in order:
            if time.time() - t0 > budget:
                break
            if kg[r] in done_global or r not in keep:
                continue
            if ps.colorable(keep - {r}, conf=200000) is False:
                keep.discard(r)
            done_global.add(kg[r])
        print(f"[{tag}] single phase: |keep|={len(keep)}, done={len(done_global)}")

    keep_global = [kg[i] for i in sorted(keep)]
    finished = bool(phase == "single" and all(g in done_global for g in keep_global))
    prevd = load(cache_name) or {}
    prevd.update(dict(keep_global=keep_global, phase=phase,
                      done_global=sorted(done_global), finished=finished))
    save(cache_name, prevd)
    print(f"[{tag}] checkpoint: |keep|={len(keep_global)} finished={finished}  "
          f"({time.time()-t0:.1f}s)")
    return keep_global, finished


def stage3(budget=30.0, seed=0):
    return _peel_chunk("stage3", seed, budget, time.time(), "stage3")


# ==================================================================================== STAGE 3F
def stage3final(clique_budget=25.0):
    print("=" * 100)
    print("STAGE 3-FINAL -- complete in-core clique re-enumeration + criticality")
    print("=" * 100)
    t0 = time.time()
    s3 = load("stage3")
    assert s3 is not None and s3.get("finished"), "run stage3 until finished=True first"
    rays = gen_s4(D)
    keep = sorted(int(t) for t in s3["keep_global"])
    cs = [rays[i] for i in keep]
    P, degs, npairs = packed_adjacency(cs, D)
    adj = adj_sets_from_packed(P, len(cs))
    bases, complete = bd6.find_cliques(adj, len(cs), D, time_limit=clique_budget)
    print(f"[final] core candidate: {len(cs)} rays, {npairs} pairs, {len(bases)} 12-cliques "
          f"(COMPLETE={complete})")
    xh = Counter(sum(1 for c in v if c in ("X", "-X")) for v in cs)
    sh = Counter(support(v) for v in cs)
    print(f"[final] X-count histogram (must be all 2): {dict(xh)}; support histogram: {dict(sh)}")
    assert set(xh) == {2}
    assert complete, "in-core clique enumeration did not finish -- raise clique_budget"
    ps = peel_solver_sets(adj, bases)
    col = ps.colorable(set(range(len(cs))))
    print(f"[final] SAT vs COMPLETE in-core bases: KS-uncolorable={not col}")
    assert not col, "core colorable against complete bases -- peel further in stage3"
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
    P, degs, npairs = packed_adjacency(cs, D)
    adj = adj_sets_from_packed(P, len(cs))
    bases, complete = bd6.find_cliques(adj, len(cs), D, time_limit=clique_budget)
    ps = peel_solver_sets(adj, bases)
    col = ps.colorable(set(range(len(cs))))
    ncrit2 = [r for r in range(len(cs)) if not ps.colorable(set(range(len(cs))) - {r})]
    print(f"[final] FINAL CORE: {len(cs)} rays / {npairs} pairs / {len(bases)} bases "
          f"(complete={complete}); uncolorable={not col}; all-critical={len(ncrit2) == 0}")
    save("core_final", dict(core_syms=cs, core_bases=bases, npairs=npairs,
                            complete=bool(complete), uncolorable=bool(not col),
                            all_critical=bool(len(ncrit2) == 0)))
    print(f"[final] done ({time.time()-t0:.1f}s)")


# ==================================================================================== STAGE 4
def stage4():
    print("=" * 100)
    print("STAGE 4 -- the d=12 WZ holonomy connection (+ the trace law Tr S = d/2, L = 2*Nb)")
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
    print(f"[stage4] L={L}, Nb={Nb}.  Stilde := L*A0:")
    for row in M:
        print("     ", row)
    trS = sp.Rational(tr, L)
    print(f"[stage4] Tr(Stilde)={tr}, Tr(S)={trS}")
    print(f"[stage4] *** TRACE LAW Tr(S) == d/2 == 6: {trS == 6} ***")
    print(f"[stage4] *** L == 2*Nb ({L} == {2*Nb}): {L == 2 * Nb} ***")
    print(f"[stage4] symmetric={sym}; off-diag nonzero {offnz}/{D*(D-1)} (max |off| {offmax}); "
          f"diag-dominant={all(m > 0 for m in margins)} (margins {margins})")
    save("stage4", dict(M=M, L=L, Nb=Nb, tr=tr, constant=bool(z1 and zm), symmetric=bool(sym),
                        margins=margins, diagdom=bool(all(m > 0 for m in margins)),
                        trace_law=bool(trS == 6), L_law=bool(L == 2 * Nb)))


# ==================================================================================== STAGE 5
def stage5():
    print("=" * 100)
    print("STAGE 5 -- the exact degree-12 characteristic polynomial f_12")
    print("=" * 100)
    s4 = load("stage4")
    assert s4 is not None
    M, L = [[int(v) for v in row] for row in s4["M"]], int(s4["L"])
    xx = sp.symbols("x")
    cp = sp.expand(sp.Matrix(M).charpoly(xx).as_expr())
    P = sp.Poly(cp, xx)
    # independent re-derivation: Bareiss fraction-free determinant at 13 integer points
    Mm = sp.Matrix(M)
    vals = [(t, (t * sp.eye(D) - Mm).det(method="bareiss")) for t in range(-6, 7)]
    cp2 = sp.expand(sp.interpolate(vals, xx))
    indep = sp.expand(cp - cp2) == 0
    ev = sorted(np.linalg.eigvalsh(np.array(M, dtype=float)))
    print(f"[stage5] f_12(x) = {cp}")
    print(f"[stage5] independent Bareiss+interpolation check: {indep}")
    print(f"[stage5] irreducible over Q: {P.is_irreducible}")
    print(f"[stage5] eigenvalues (float check): {[round(v, 5) for v in ev]}")
    print(f"[stage5] eigenphases phi/(2pi) = lambda/L: {[round(v / L % 1, 6) for v in ev]}")
    save("stage5", dict(charpoly=str(cp), coeffs=[int(c) for c in P.all_coeffs()[::-1]],
                        irreducible=bool(P.is_irreducible), indep=bool(indep), L=L))


# ==================================================================================== STAGE 6
def stage6():
    print("=" * 100)
    print("STAGE 6 -- Galois certificate for f_12 (sd_certificate_v2, route A: p=7)")
    print("=" * 100)
    s5 = load("stage5")
    assert s5 is not None
    coeffs = [int(c) for c in s5["coeffs"]]
    r = certify_v2(coeffs, "f_12")
    save("stage6", dict(verdict=r["verdict"], route=r["route"], witness=r["witness"],
                        chain=r.get("chain")))
    return r


# ==================================================================================== HEAD2HEAD
def headtohead(budget=25.0):
    """d=12 is the first rung since d=8 where d-1=11 is PRIME, so the OLD published template
    (irreducible + (d-1)-cycle + LITERAL transposition (2,1^10)) applies in principle.  Run both
    templates on the same f_12 and compare witness primes.  Also track the power-trick
    transposition (cycle type with exactly one part 2, all other parts odd -> the lcm(odd)-th
    power is a pure transposition), which repairs the old template's density problem."""
    print("=" * 100)
    print("HEAD-TO-HEAD -- old template (11-cycle + transposition) vs route A (7-cycle) on f_12")
    print("=" * 100)
    s5 = load("stage5")
    assert s5 is not None
    coeffs = [int(c) for c in s5["coeffs"]]
    st = load("headtohead") or dict(scanned_to=2, found={}, nprimes=0)
    found = {k: (int(v[0]), tuple(int(t) for t in v[1])) for k, v in st["found"].items()}
    start, nprimes = int(st["scanned_to"]), int(st["nprimes"])
    t0 = time.time()
    q = start
    for q in sp.primerange(start, 10 ** 9):
        if time.time() - t0 > budget:
            break
        ct = factor_degrees_mod_p(coeffs, int(q))
        nprimes += 1
        if ct is None:
            continue
        if ct == (12,):
            found.setdefault("dcycle", (int(q), ct))
        if 11 in ct:
            found.setdefault("d1cycle", (int(q), ct))       # (11,1) => 11-cycle (power trick)
        if 7 in ct:
            found.setdefault("p7", (int(q), ct))            # route A ingredient
        if (12 - len(ct)) % 2 == 1:
            found.setdefault("odd", (int(q), ct))
        if ct == tuple([2] + [1] * 10):
            found.setdefault("transp_literal", (int(q), ct))
        if list(ct).count(2) == 1 and all(p % 2 == 1 for p in ct if p != 2):
            found.setdefault("transp_power", (int(q), ct))
        have_all = all(k in found for k in
                       ("dcycle", "d1cycle", "p7", "odd", "transp_literal", "transp_power"))
        if have_all:
            break
    save("headtohead", dict(scanned_to=int(q), nprimes=nprimes,
                            found={k: [v[0], list(v[1])] for k, v in found.items()}))
    # Chebotarev densities (exact): literal transposition C(12,2)/12!; power-trick transposition
    # sum over partitions of 10 into odd parts of 1/(2 * prod k^m_k * m_k!).
    dens_lit = F(math.comb(12, 2), math.factorial(12))
    dens_pt = F(0)
    from sympy.utilities.iterables import ordered_partitions
    for part in ordered_partitions(10):
        if all(p % 2 == 1 for p in part):
            cnt = Counter(part)
            z = 2                                            # the single 2-cycle
            for k, mk in cnt.items():
                z *= k ** mk * math.factorial(mk)
            dens_pt += F(1, z)
    print(f"[h2h] primes scanned: {nprimes} (up to {q})")
    for k in ("dcycle", "d1cycle", "p7", "odd", "transp_literal", "transp_power"):
        v = found.get(k)
        print(f"[h2h]   {k:15s}: " + (f"mod {v[0]:>7} type {v[1]}" if v else "NOT FOUND yet"))
    print(f"[h2h] Chebotarev density literal transposition = {dens_lit} ~ "
          f"{float(dens_lit):.3g}  (expected ~{int(1/float(dens_lit))} usable primes needed)")
    print(f"[h2h] Chebotarev density power-trick transposition = {dens_pt} ~ "
          f"{float(dens_pt):.3g}")
    old_ok = all(k in found for k in ("dcycle", "d1cycle", "transp_literal"))
    oldp_ok = all(k in found for k in ("dcycle", "d1cycle", "transp_power"))
    newA_ok = all(k in found for k in ("dcycle", "p7", "odd"))
    wa = max(found[k][0] for k in ("dcycle", "p7", "odd")) if newA_ok else None
    wb = max(found[k][0] for k in ("dcycle", "d1cycle", "transp_literal")) if old_ok else None
    wbp = max(found[k][0] for k in ("dcycle", "d1cycle", "transp_power")) if oldp_ok else None
    print(f"[h2h] ROUTE A  (7-cycle + odd):                 "
          f"{'COMPLETE, max witness prime ' + str(wa) if newA_ok else 'incomplete'}")
    print(f"[h2h] OLD TEMPLATE (11-cycle + literal transp): "
          f"{'COMPLETE, max witness prime ' + str(wb) if old_ok else 'INCOMPLETE at cap'}")
    print(f"[h2h] OLD + power trick (11-cycle + PT transp): "
          f"{'COMPLETE, max witness prime ' + str(wbp) if oldp_ok else 'incomplete'}")
    save("headtohead", dict(scanned_to=int(q), nprimes=nprimes,
                            found={k: [v[0], list(v[1])] for k, v in found.items()},
                            dens_lit=str(dens_lit), dens_pt=str(dens_pt),
                            routeA=bool(newA_ok), old_literal=bool(old_ok),
                            old_power=bool(oldp_ok), wA=wa, wB=wb, wBp=wbp))


# ==================================================================================== STAGE 7
def stage7():
    """Independent cross-checks (same battery as d=10):
    (a) numeric E^dag dE/dtheta / Nb vs i*Stilde/L (no Fourier machinery shared);
    (b) disc(f_12) not a perfect square;
    (c) Dedekind cycle-type census over the 77 types of S_12."""
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
    f12 = sp.Poly([int(c) for c in s5["coeffs"]][::-1], xx)
    disc = int(sp.discriminant(f12.as_expr(), xx))
    sq = disc >= 0 and math.isqrt(abs(disc)) ** 2 == disc
    print(f"[stage7b] disc(f_12) = {disc}")
    print(f"[stage7b] perfect square: {sq}  => Galois group {'inside' if sq else 'NOT inside'} "
          f"A_12 (must be False)")
    ok_b = not sq
    coeffs = [int(c) for c in s5["coeffs"]]
    prev = load("stage7census") or dict(seen={}, cap=2)
    seen = {tuple(int(t) for t in json.loads(k)): int(v) for k, v in prev["seen"].items()}
    t0 = time.time()
    cap = int(prev["cap"])
    for q in sp.primerange(cap, 10 ** 9):
        if time.time() - t0 > 20:
            break
        ct = factor_degrees_mod_p(coeffs, int(q))
        if ct is not None:
            seen.setdefault(ct, int(q))
        cap = int(q)
    save("stage7census", dict(seen={json.dumps(list(k)): v for k, v in seen.items()}, cap=cap))
    from sympy.utilities.iterables import ordered_partitions
    total = sum(1 for _ in ordered_partitions(12))
    missing = [tuple(sorted(p, reverse=True)) for p in ordered_partitions(12)
               if tuple(sorted(p, reverse=True)) not in seen]
    print(f"[stage7c] cycle-type census primes < {cap}: {len(seen)}/{total} types of S_12 "
          f"realized; missing: {len(missing)}")
    if missing and len(missing) <= 12:
        print(f"[stage7c] missing: {missing}")
    save("stage7", dict(conn_maxerr=float(maxerr), disc=str(disc), disc_square=bool(sq),
                        census=len(seen), census_total=int(total), census_cap=cap,
                        n_missing=len(missing), ok=bool(ok_a and ok_b)))
    print(f"[stage7] {'PASS' if ok_a and ok_b else 'FAIL'}")


# ==================================================================================== ALT CORE
def altcore(seed=999, budget=28.0):
    """Independently-seeded second peel (the d=8 seed-999 lesson), via the same chunked
    core-shrink machinery as stage3; `altcorefinal` runs the downstream."""
    key = f"altcore{seed}"
    prev = load(key)
    if prev and prev.get("stage") in ("peeled", "done"):
        print(f"[altcore] seed {seed} peel already finished (stage={prev['stage']}) -- run "
              f"altcorefinal")
        return
    _, finished = _peel_chunk(key, seed, budget, time.time(), f"altcore{seed}")
    if finished:
        d = load(key)
        d["stage"] = "peeled"
        save(key, d)


def altcorefinal(seed=999, clique_budget=20.0):
    t0 = time.time()
    key = f"altcore{seed}"
    prev = load(key)
    assert prev and prev.get("stage") in ("peeled", "done"), "run altcore until finished first"
    if prev.get("stage") == "done":
        print(f"[altcore] seed {seed} already complete: verdict {prev['verdict']}")
        return
    rays = gen_s4(D)
    cs0 = [rays[int(i)] for i in sorted(int(t) for t in prev["keep_global"])]
    P, degs, cnp = packed_adjacency(cs0, D)
    cadj = adj_sets_from_packed(P, len(cs0))
    cbases, complete = bd6.find_cliques(cadj, len(cs0), D, time_limit=clique_budget)
    ps2 = peel_solver_sets(cadj, cbases)
    k2 = set(range(len(cs0)))
    changed = True
    while changed:
        changed = False
        for r in sorted(k2):
            if not ps2.colorable(k2 - {r}):
                k2.discard(r)
                changed = True
    cs = [cs0[i] for i in sorted(k2)]
    P, degs, cnp = packed_adjacency(cs, D)
    cadj = adj_sets_from_packed(P, len(cs))
    cbases, complete = bd6.find_cliques(cadj, len(cs), D, time_limit=clique_budget)
    ps3 = peel_solver_sets(cadj, cbases)
    col = ps3.colorable(set(range(len(cs))))
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
    trS = sp.Rational(tr, L)
    print(f"[altcore] constant={const} L={L} Nb={Nb} Tr(S)={trS} "
          f"(trace law =6: {trS == 6}; L==2Nb: {L == 2 * Nb}) decoupled rows={decoupled}")
    print(f"[altcore] alt char poly: {cp}")
    r = certify_v2([int(c) for c in sp.Poly(cp, xx).all_coeffs()[::-1]],
                   f"f_12 (alt seed {seed})")
    save(key, dict(stage="done", n=len(cs), shared=shared, M=M, L=L, Nb=Nb,
                   constant=bool(const), decoupled=decoupled, charpoly=str(cp),
                   trace_law=bool(trS == 6), L_law=bool(L == 2 * Nb),
                   verdict=r["verdict"], route=r["route"], uncolorable=bool(not col),
                   complete=bool(complete)))
    print(f"[altcore] done ({time.time()-t0:.1f}s)")


# ==================================================================================== REPORT
def report():
    print("=" * 100)
    print("BRANCH D12-GALOIS -- REPORT")
    print("=" * 100)
    rows = []
    g = load("gate")
    rows.append(("gate: d=8 AND d=10 published numbers reproduced", bool(g and g["passed"])))
    s1 = load("stage1")
    if s1: rows.append((f"stage1: lazy pool accounting, support-4 stratum {s1['n']} of "
                        f"{s1['total_pool']}", s1["n"] == 23760))
    sg = load("stage2graph")
    if sg: rows.append((f"stage2graph: {sg['V']} rays / {sg['npairs']} pairs, degree "
                        f"{sg['degmin']}..{sg['degmax']}", sg["V"] == 23760))
    s2 = load("stage2")
    if s2: rows.append((f"stage2: {s2['nbases']} partial 12-cliques over {s2['nbp']} rays, "
                        f"UNCOLORABLE={s2['uncolorable']}", s2["uncolorable"]))
    cf = load("core_final")
    if cf: rows.append((f"stage3: core {len(cf['core_syms'])} rays / {cf['npairs']} pairs / "
                        f"{len(cf['core_bases'])} bases (complete={cf['complete']}, "
                        f"all-critical={cf['all_critical']})",
                        cf["uncolorable"] and cf["complete"] and cf["all_critical"]))
    s4 = load("stage4")
    if s4: rows.append((f"stage4: constant={s4['constant']} symmetric={s4['symmetric']} "
                        f"diag-dominant={s4['diagdom']} L={s4['L']} Nb={s4['Nb']} "
                        f"Tr(S)={sp.Rational(int(s4['tr']), int(s4['L']))} "
                        f"[trace law =6: {s4['trace_law']}; L==2Nb: {s4['L_law']}]",
                        s4["constant"] and s4["symmetric"]))
    s5 = load("stage5")
    if s5: rows.append((f"stage5: f_12 irreducible={s5['irreducible']} "
                        f"independent-check={s5['indep']}", s5["irreducible"] and s5["indep"]))
    s6 = load("stage6")
    if s6: rows.append((f"stage6: VERDICT {s6['verdict']} via route {s6['route']}",
                        s6["verdict"] == "S_12"))
    h = load("headtohead")
    if h and "routeA" in h:
        rows.append((f"headtohead: route A complete={h['routeA']} (max witness {h.get('wA')}); "
                     f"old literal-transposition template complete={h['old_literal']} "
                     f"(scanned {h['nprimes']} primes to {h['scanned_to']}); "
                     f"old+power-trick complete={h['old_power']} (max witness {h.get('wBp')})",
                     h["routeA"]))
    s7 = load("stage7")
    if s7: rows.append((f"stage7: numeric-connection err {s7['conn_maxerr']:.1e}, disc-square="
                        f"{s7['disc_square']}, census {s7['census']}/{s7['census_total']} "
                        f"types (primes<{s7['census_cap']}, {s7['n_missing']} missing)",
                        s7["ok"]))
    ac = load("altcore999")
    if ac and ac.get("stage") == "done":
        rows.append((f"altcore seed 999: {ac['n']} rays ({ac['shared']} shared), L={ac['L']}, "
                     f"Nb={ac['Nb']}, trace law: {ac['trace_law']}, L==2Nb: {ac['L_law']}, "
                     f"decoupled={ac['decoupled']}, VERDICT {ac['verdict']}",
                     ac["verdict"] == "S_12"))
    allok = True
    for nm, ok in rows:
        print(f"  [{'PASS' if ok else 'FAIL'}] {nm}")
        allok &= bool(ok)
    print(f"\n  OVERALL: {'PASS' if allok else 'FAIL/INCOMPLETE'}")
    if s5:
        print(f"\n  f_12 = {s5['charpoly']}")
    return allok


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "report"
    arg = float(sys.argv[2]) if len(sys.argv) > 2 else None
    if which == "gate": gate()
    elif which == "stage1": stage1()
    elif which == "stage2graph": stage2_graph()
    elif which == "stage2cliques": stage2_cliques(arg or 25.0,
                                                  sys.argv[3] if len(sys.argv) > 3 else None)
    elif which == "stage2sat": stage2_sat(arg, sys.argv[3] if len(sys.argv) > 3 else "weak")
    elif which == "stage3": stage3(arg or 30.0, int(sys.argv[3]) if len(sys.argv) > 3 else 0)
    elif which == "stage3final": stage3final(arg or 25.0)
    elif which == "stage4": stage4()
    elif which == "stage5": stage5()
    elif which == "stage6": stage6()
    elif which == "headtohead": headtohead(arg or 25.0)
    elif which == "stage7": stage7()
    elif which == "altcore": altcore(int(sys.argv[2]) if len(sys.argv) > 2 else 999,
                                     float(sys.argv[3]) if len(sys.argv) > 3 else 28.0)
    elif which == "altcorefinal": altcorefinal(int(sys.argv[2]) if len(sys.argv) > 2 else 999)
    elif which == "report": report()
    else:
        print(f"unknown stage {which!r}")
        sys.exit(1)
    print(f"\n[branch_d12galois.py {which}: total {time.time()-T0:.1f}s]")
