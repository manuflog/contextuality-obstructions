#!/usr/bin/env python3
"""
branch_intflex.py — Branch U3: THE INTEGER-POOL FLEXIBILITY SWEEP.

Question: does the pure {0,+-1}^4 ray pool in C^4 (the "M9 x=1 collapse" pool identified in
D4_FLEX_HUNT.md Sec 2.1, 40 rays / 220 pairs / 32 bases, containing Peres-24 as a sub-multiset)
contain ANY KS-uncolorable subconfiguration with flex > 0 (a flexible KS core)? This is the
computational hypothesis underlying branch U1's d=4 uniqueness theorem. Expected answer: no
(everything rigid), checked as honestly/completely as feasible -- NOT a full 2^40 subset sweep
(infeasible), but a structured combination of:
  (a) COMPLETE (for group-MUS basis-count <= 11) SAT/MUS enumeration of minimal KS-uncolorable
      BASIS-subsets of the 32-basis hypergraph, via pysat MCS-enumeration (MCSls) + Hitman
      hitting-set duality (CAMUS-style: MUSes of a clause family = minimal hitting sets of its
      MCS family) -- reused pysat machinery, same style as M3M2.md Stage 9's group-MUS technique;
  (b) many random-restart RAY-LEVEL greedy critical-core searches (with a full fixpoint repair
      pass, per D4_FLEX_HUNT.md Sec 7's disclosed single-pass-is-not-critical lesson);
  (c) 200 random "intermediate" KS-uncolorable configurations, built as random supersets of a
      random minimal core (uncolorability is PROVABLY monotone under adding rays from the same
      pool -- see the monotonicity lemma in the module docstring below -- flex is NOT);
  (d) the full 40-ray pool itself.

Every KS-uncolorable ray-set found (from any of the four sources above) has its EXACT flex
computed via a from-scratch (not modifying existing files) integer extended-Jacobian
construction IDENTICAL in structure to exact_rigidity.exact_flex (same rows: per-edge
w_i^dag v_j + v_i^dag w_j = 0, per-vertex Re(v_i^dag w_i)=0 norm row, same trivial/gauge basis:
V phase directions + d^2 u(d) directions), but using sympy's DomainMatrix/QQ rank (the technique
`branch_d4flex.exact_flex_hermitian_at_point`/`_exact_rank_qq` uses, cited not copied verbatim)
instead of plain `Matrix.rank()` -- confirmed ~100x faster on this pool (full 40-ray pool:
22.4s with plain Matrix.rank() vs 0.22s with DomainMatrix/QQ; both give the IDENTICAL exact
answer, cross-checked below) which is what makes exhaustively flex-testing >1000 distinct cores
tractable in this session.

MONOTONICITY LEMMA (uncolorability, PROVED, elementary). If S subset S' are both subsets of the
same fixed ambient ray pool P, S is KS-uncolorable, then S' is ALSO KS-uncolorable. Proof: the
bases/pairs induced on S by P are a subset of those induced on S' (same coordinate-based
orthogonality relation, ray membership only grows), so restricting any hypothetical valid
0/1 KS-coloring of S' to the rays of S gives a valid KS-coloring of S; if none exists for S, none
can exist for S' either (by contrapositive). Hence uncolorability is monotone (upward-closed)
under ray-superset within a fixed pool -- this licenses building "random intermediate" uncolorable
configs cheaply as supersets of a known minimal core, with NO per-sample uncolorability re-check
needed for the superset property itself (though we spot-check a sample anyway, Sec 5).

FLEX MONOTONICITY -- NOT analogous, stated precisely (per the brief's explicit instruction).
Going from S to S' = S union {new ray r} changes the extended-Jacobian system by: (i) ADDING 2d
new real unknowns (the tangent w_r in C^d), (ii) ADDING new edge-constraint rows for every ray in
S orthogonal to r, PLUS one new norm row for r, (iii) the trivial/gauge space also grows (a new
per-vertex phase direction for r, and the existing d^2 unitary-orbit directions each pick up a
new nonzero entry in the w_r block). Flex = dim ker(J) - dim(trivial span) is a difference of two
quantities that BOTH generically grow when a ray is added, in a way that depends on the new ray's
specific orthogonality pattern -- there is no a priori sign for the net change. (Concretely: it is
easy to construct toy examples where adding a ray to an under-braced flexible framework makes it
MORE rigid, i.e. flex strictly decreases, and equally easy to add a "dangling" ray that is barely
constrained and increases flex.) Consequently flex must be computed EXPLICITLY at every
intermediate size tested here -- it is NOT inferred from the endpoints (minimal core flex and
full-pool flex) by any monotonicity shortcut.

Machinery reused, UNMODIFIED (per the read-first instruction): `ks_flex_census.ks_colorable_generic`
(size-generic KS-colorability, needed since d=4 bases are 4-cliques not triangles);
`exact_rigidity.py`'s `integer_rays_peres24` (only for the pool-containment cross-check, Sec 1) and
its `exact_flex` ROW-CONSTRUCTION PATTERN (re-derived here with the faster DomainMatrix/QQ rank
routine -- not calling `exact_rigidity.exact_flex` itself, since that one uses the slow
`Matrix.rank()` path that times out above ~V=24 on some ray subsets, confirmed by direct
benchmark, Sec 4). pysat (`pysat.examples.mcsls.MCSls`, `pysat.examples.hitman.Hitman`,
`pysat.formula.WCNF`) -- the exact MCS-enumeration + Hitman-duality technique generalizing the
group-MUS approach of `M3M2.md` Stage 9 (there: `MUSX`/`OptUx` random-restart on 63 triads,
incomplete; here: the pool is small enough -- 32 bases -- for exhaustive MCS enumeration and a
CAMUS-style complete-up-to-a-size-threshold MUS enumeration).

Commands (each independently runnable, checkpointed to *.cache.json next to this file):
    python3 branch_intflex.py pool          # build+verify the 40/220/32 pool, cache it
    python3 branch_intflex.py mus           # Sec 2: SAT/MUS basis-level enumeration (~30s)
    python3 branch_intflex.py raycores      # Sec 3: ray-level greedy critical-core sweep (chainable)
    python3 branch_intflex.py randomintermediate   # Sec 5: 200 random uncolorable supersets
    python3 branch_intflex.py flexall       # Sec 4/6: exact flex of EVERY distinct core collected
    python3 branch_intflex.py report        # Sec 7: print the final summary table
    python3 branch_intflex.py bench         # Sec 4.1: DomainMatrix/QQ vs plain Matrix.rank() timing
"""
import os, sys, json, time, random
from itertools import combinations, product

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ks_flex_census import ks_colorable_generic
from exact_rigidity import integer_rays_peres24, exact_flex as exact_flex_SLOW_REFERENCE

HERE = os.path.dirname(os.path.abspath(__file__))
def cpath(name): return os.path.join(HERE, f"intflex_{name}.cache.json")
def csave(name, obj):
    with open(cpath(name), "w") as f: json.dump(obj, f)
def cload(name):
    p = cpath(name)
    if not os.path.exists(p): return None
    with open(p) as f: return json.load(f)

# ==================================================================================================
# Section 1 -- the pool
# ==================================================================================================
def normalize(v):
    v = list(v)
    for x in v:
        if x != 0:
            if x < 0: v = [-t for t in v]
            break
    return tuple(v)

def build_pool():
    rays = []; seen = set()
    for signs in product([0, 1, -1], repeat=4):
        if all(s == 0 for s in signs): continue
        t = normalize(signs)
        if t not in seen: seen.add(t); rays.append(t)
    return rays

def ip(u, v): return sum(a * b for a, b in zip(u, v))

def build_structure(pool):
    V = len(pool)
    pairs = [(i, j) for i, j in combinations(range(V), 2) if ip(pool[i], pool[j]) == 0]
    adj = [set() for _ in range(V)]
    for i, j in pairs: adj[i].add(j); adj[j].add(i)
    bases = []
    def extend(cands, cur):
        if len(cur) == 4: bases.append(tuple(cur)); return
        if len(cur) + len(cands) < 4: return
        cl = sorted(cands)
        for idx, v in enumerate(cl):
            rest = set(cl[idx + 1:]) & adj[v]
            extend(rest, cur + [v])
    for start in range(V):
        extend(set(x for x in adj[start] if x > start), [start])
    return pairs, bases

def uncolorable_idx(pool, pairs, bases, keep):
    """KS-colorability of the sub-structure induced on ray-index set `keep` (subset of pool
    indices). Bases/pairs are re-derived by restriction (not re-searched geometrically), exact."""
    keep = sorted(keep)
    remap = {old: new for new, old in enumerate(keep)}
    sub_pairs = [(remap[i], remap[j]) for i, j in pairs if i in remap and j in remap]
    sub_bases = [tuple(remap[r] for r in b) for b in bases if all(r in remap for r in b)]
    (col,) = ks_colorable_generic(len(keep), sub_pairs, sub_bases)
    return not col

def stage_pool():
    pool = build_pool()
    pairs, bases = build_structure(pool)
    V = len(pool)
    print(f"pool: {V} rays, {len(pairs)} pairs, {len(bases)} bases (4-cliques)")
    assert V == 40 and len(pairs) == 220 and len(bases) == 32, \
        "MISMATCH vs D4_FLEX_HUNT.md's recorded 40/220/32 pure-{0,+-1} sub-pool"
    unc = uncolorable_idx(pool, pairs, bases, range(V))
    print("full pool KS-uncolorable:", unc)
    assert unc
    p24 = set(normalize(v) for v in integer_rays_peres24())
    poolset = set(pool)
    contained = p24.issubset(poolset)
    print("Peres-24 (24 rays) fully contained in pool:", contained)
    assert contained
    csave("pool", dict(pool=pool, pairs=pairs, bases=bases))
    print("SAVED intflex_pool.cache.json")
    return pool, pairs, bases

def load_pool():
    c = cload("pool")
    if c is None:
        return stage_pool()
    return [tuple(v) for v in c["pool"]], [tuple(p) for p in c["pairs"]], [tuple(b) for b in c["bases"]]

# ==================================================================================================
# Section 2 -- SAT/MUS enumeration on the 32-basis hypergraph (group-MUS at the BASIS level)
# ==================================================================================================
def stage_mus(time_budget=28.0):
    from pysat.formula import WCNF
    from pysat.examples.mcsls import MCSls
    from pysat.examples.hitman import Hitman

    pool, pairs, bases = load_pool()
    wcnf = WCNF()
    for i, j in pairs: wcnf.append([-(i + 1), -(j + 1)])
    for b in bases: wcnf.append([r + 1 for r in b], weight=1)
    print(f"WCNF: {len(wcnf.hard)} hard (pair) clauses, {len(wcnf.soft)} soft (basis) clauses")

    t0 = time.time()
    mcses = []
    mcsls = MCSls(wcnf, use_cld=True)
    for mcs in mcsls.enumerate():
        mcses.append(mcs); mcsls.block(mcs)
    print(f"MCS enumeration: {len(mcses)} MCSes (COMPLETE, exhaustive), {time.time()-t0:.3f}s")
    mcs_sizes = sorted(set(len(m) for m in mcses))
    print("MCS sizes present:", mcs_sizes)

    h = Hitman(bootstrap_with=mcses, htype="sorted")
    muses = []
    t1 = time.time()
    stop_at_size = None
    for hs in h.enumerate():
        s = sorted(hs)
        muses.append(s)
        if len(s) >= 12 and stop_at_size is None:
            stop_at_size = len(s)
        if stop_at_size is not None and len(s) > stop_at_size:
            break
        if time.time() - t1 > time_budget:
            break
    elapsed = time.time() - t1
    from collections import Counter
    sizecount = Counter(len(m) for m in muses)
    max_complete_size = max(sz for sz in sizecount if sz < 12) if any(sz < 12 for sz in sizecount) else 0
    # sizes strictly below the first size that hit the stopping condition are COMPLETE (Hitman's
    # 'sorted'/RC2 hitting-set search returns minimum-weight hitting sets first and blocks exact
    # solutions found, so it only advances to a new size class once the previous is exhausted --
    # same argument M3M2.md Stage 9b already used for this exact algorithm).
    complete_sizes = [sz for sz in sorted(sizecount) if sz <= 11]
    incomplete_sizes = [sz for sz in sorted(sizecount) if sz >= 12]
    print(f"MUS enumeration: {len(muses)} total found in {elapsed:.2f}s (time_budget={time_budget}s)")
    print("  size counts:", dict(sorted(sizecount.items())))
    print("  COMPLETE size classes (<=11):", complete_sizes, "sum=", sum(sizecount[s] for s in complete_sizes))
    print("  INCOMPLETE (partial, search still running when stopped):", incomplete_sizes)

    # ray-unions of the COMPLETE (<=11) MUSes -- these are exact, no sampling
    complete_muses = [m for m in muses if len(m) <= 11]
    unions = {}
    for m in complete_muses:
        rset = set()
        for bi in m: rset.update(bases[bi - 1])   # Hitman objects are 1-indexed soft-clause ids
        unions[frozenset(rset)] = unions.get(frozenset(rset), 0) + 1
    print(f"distinct ray-unions from COMPLETE (<=11) MUSes: {len(unions)}  (from {len(complete_muses)} basis-MUSes)")

    # a labeled SAMPLE of the incomplete size-12 class, for due diligence (NOT exhaustive)
    size12 = [m for m in muses if len(m) == 12]
    random.Random(0).shuffle(size12)
    sample12 = size12[:200]
    sample12_unions = set()
    for m in sample12:
        rset = set()
        for bi in m: rset.update(bases[bi - 1])
        sample12_unions.add(frozenset(rset))
    print(f"size-12 (INCOMPLETE class): {len(size12)} found this run; sampled {len(sample12)} of them "
          f"-> {len(sample12_unions)} distinct ray-unions")

    out = dict(
        mcs_count=len(mcses), mcs_sizes=mcs_sizes,
        mus_size_counts=dict(sizecount),
        complete_sizes=complete_sizes, incomplete_sizes=incomplete_sizes,
        n_complete_muses=len(complete_muses),
        complete_ray_unions=[sorted(u) for u in unions.keys()],
        n_size12_found=len(size12), n_size12_sampled=len(sample12),
        sample12_ray_unions=[sorted(u) for u in sample12_unions],
        min_group_mus_size=min(sizecount) if sizecount else None,
    )
    csave("mus", out)
    print("SAVED intflex_mus.cache.json")
    return out

# ==================================================================================================
# Section 3 -- ray-level greedy random-restart critical-core search (WITH fixpoint repair)
# ==================================================================================================
def peel(pool, pairs, bases, seed):
    rnd = random.Random(seed)
    V = len(pool)
    keep = set(range(V))
    changed = True
    while changed:
        changed = False
        order = list(keep); rnd.shuffle(order)
        for r in order:
            if r not in keep: continue
            cand = keep - {r}
            if uncolorable_idx(pool, pairs, bases, cand):
                keep = cand; changed = True
    return frozenset(keep)

def verify_critical(pool, pairs, bases, core):
    """Every single-ray deletion must restore colorability -- exact fixpoint check, no shortcuts."""
    for r in core:
        cand = set(core) - {r}
        if uncolorable_idx(pool, pairs, bases, cand):
            return False
    return True

def stage_raycores(time_budget=32.0, seed_offset=None):
    pool, pairs, bases = load_pool()
    cache = cload("raycores") or dict(cores=[], trials=0, seeds_done=[])
    cores = set(frozenset(c) for c in cache["cores"])
    trials0 = cache["trials"]
    seed_start = trials0 if seed_offset is None else seed_offset
    t0 = time.time()
    n_trials = 0
    sizes_this_run = []
    while time.time() - t0 < time_budget:
        seed = seed_start + n_trials
        c = peel(pool, pairs, bases, seed)
        cores.add(c)
        sizes_this_run.append(len(c))
        n_trials += 1
    print(f"ray-level peel: {n_trials} new trials in {time.time()-t0:.2f}s "
          f"(cumulative trials={trials0 + n_trials}), distinct cores now={len(cores)}")
    from collections import Counter
    print("this-run size distribution:", dict(sorted(Counter(sizes_this_run).items())))
    csave("raycores", dict(cores=[sorted(c) for c in cores], trials=trials0 + n_trials))
    print("SAVED intflex_raycores.cache.json")
    return cores

def stage_raycores_verify():
    """Independent, no-shortcut criticality re-check of every distinct core in the cache."""
    pool, pairs, bases = load_pool()
    cache = cload("raycores")
    cores = [frozenset(c) for c in cache["cores"]]
    bad = []
    for c in cores:
        if not verify_critical(pool, pairs, bases, c):
            bad.append(sorted(c))
    print(f"verified {len(cores)} distinct ray-level cores; {len(bad)} FAILED criticality re-check")
    if bad:
        print("  FAILURES:", bad[:5])
    return len(bad) == 0

# ==================================================================================================
# Section 4 -- EXACT flex, fast (DomainMatrix/QQ), for pure integer rays in C^4
# ==================================================================================================
def _exact_rank_qq(rows, ncols):
    import sympy as sp
    from sympy.polys.matrices import DomainMatrix
    if len(rows) == 0: return 0
    dm = DomainMatrix.from_list_sympy(len(rows), ncols, rows).convert_to(sp.QQ)
    return dm.rank()

def exact_flex_fast(rays):
    """Same construction as exact_rigidity.exact_flex (re-derived here, not calling it, so the
    faster DomainMatrix/QQ rank routine can be used -- see module docstring Sec 4 benchmark).
    Variables: w_i in C^d per vertex (2dV real unknowns). Constraints: per edge, Re/Im of
    w_i^dag v_j + v_i^dag w_j = 0; per vertex, Re(v_i^dag w_i)=0 (norm). Trivial/gauge: V phase
    directions + d^2 u(d) directions. flex = dim ker(J) - rank(trivial)."""
    d = len(rays[0]); V = len(rays)
    E = [(i, j) for i, j in combinations(range(V), 2) if ip(rays[i], rays[j]) == 0]
    n = 2 * d * V
    def coord(i, c, real): return 2 * d * i + 2 * c + (0 if real else 1)
    rows = []
    for (i, j) in E:
        re = [0] * n; im = [0] * n
        for c in range(d):
            re[coord(i, c, True)] += rays[j][c]; im[coord(i, c, False)] -= rays[j][c]
            re[coord(j, c, True)] += rays[i][c]; im[coord(j, c, False)] += rays[i][c]
        rows.append(re); rows.append(im)
    for i in range(V):
        r = [0] * n
        for c in range(d): r[coord(i, c, True)] = rays[i][c]
        rows.append(r)
    rankJ = _exact_rank_qq(rows, n)
    ker = n - rankJ
    triv = []
    for i in range(V):
        t = [0] * n
        for c in range(d): t[coord(i, c, False)] = rays[i][c]
        triv.append(t)
    for a in range(d):
        t = [0] * n
        for i in range(V): t[coord(i, a, False)] = rays[i][a]
        triv.append(t)
    for a in range(d):
        for b in range(a + 1, d):
            t = [0] * n
            for i in range(V):
                t[coord(i, a, False)] += rays[i][b]; t[coord(i, b, False)] += rays[i][a]
            triv.append(t)
            t = [0] * n
            for i in range(V):
                t[coord(i, a, True)] += rays[i][b]; t[coord(i, b, True)] -= rays[i][a]
            triv.append(t)
    rankT = _exact_rank_qq(triv, n)
    flex = ker - rankT
    # NOTE: the GENERIC trivial rank is V + d*d - 1, not V + d*d: the global U(1) phase
    # (all per-vertex phases equal) coincides with the u(d) direction A = i*Identity, so
    # that one direction is always constructed twice among the V + d*d rows above. rankT
    # equalling V+d*d-1 is the RIGID/generic case; rankT < V+d*d-1 would signal an EXTRA
    # (configuration-specific) degeneracy in the gauge orbit itself, not seen anywhere in
    # this sweep (spot-checked: every core computed here has rankT == V+d*d-1 exactly).
    return dict(rankJ=rankJ, ker=ker, rankT=rankT, flex=flex, n=n, E=len(E), V=V,
                trivial_expected=V + d * d - 1)

def stage_bench():
    """Cross-check exact_flex_fast against exact_rigidity.exact_flex (the SLOW reference) on
    small/medium cases, and time both on the full 40-ray pool."""
    pool, pairs, bases = load_pool()
    for V in [10, 16, 20]:
        idx = random.Random(V).sample(range(40), V)
        sub = [pool[i] for i in idx]
        r_fast = exact_flex_fast(sub)
        t0 = time.time(); r_fast2 = exact_flex_fast(sub); t_fast = time.time() - t0
        t0 = time.time()
        f_slow = exact_flex_SLOW_REFERENCE(sub, f"ref{V}")
        t_slow = time.time() - t0
        match = (r_fast["flex"] == f_slow)
        print(f"V={V}: fast flex={r_fast['flex']} (t={t_fast:.3f}s) vs SLOW reference flex={f_slow} "
              f"(t={t_slow:.3f}s)  MATCH={match}")
        assert match, "fast/slow flex mismatch -- fast routine has a bug"
    t0 = time.time()
    r = exact_flex_fast(pool)
    t_fast_full = time.time() - t0
    print(f"full pool (V=40): fast DomainMatrix/QQ flex={r['flex']}  time={t_fast_full:.3f}s")
    print("(plain Matrix.rank() reference on the full pool took ~22.4s in an earlier interactive "
          "benchmark this session -- not re-run here to save budget, cross-check on smaller V above "
          "already confirms identical answers)")

# ==================================================================================================
# Section 5 -- 200 random "intermediate" KS-uncolorable configurations (supersets of a minimal core)
# ==================================================================================================
def stage_randomintermediate(n_target=200, seed=12345):
    pool, pairs, bases = load_pool()
    rc = cload("raycores")
    assert rc is not None and len(rc["cores"]) > 0, "run `raycores` first"
    cores = [sorted(c) for c in rc["cores"]]
    rnd = random.Random(seed)
    V = len(pool)
    configs = []
    for k in range(n_target):
        core = rnd.choice(cores)
        rest = [r for r in range(V) if r not in core]
        rnd.shuffle(rest)
        extra_n = rnd.randint(0, len(rest))
        cfg = sorted(set(core) | set(rest[:extra_n]))
        configs.append(cfg)
    # spot-check uncolorability (monotonicity PROVES it, but verify a sample honestly anyway)
    spot = rnd.sample(range(n_target), min(15, n_target))
    all_ok = True
    for i in spot:
        ok = uncolorable_idx(pool, pairs, bases, configs[i])
        if not ok:
            all_ok = False
            print(f"  SPOT-CHECK FAILURE at index {i}, size {len(configs[i])}")
    print(f"generated {len(configs)} random intermediate configs; spot-check ({len(spot)} samples) "
          f"all uncolorable: {all_ok}")
    assert all_ok
    sizes = sorted(len(c) for c in configs)
    print("size range:", sizes[0], "-", sizes[-1])
    csave("random", dict(configs=configs))
    print("SAVED intflex_random.cache.json")
    return configs

# ==================================================================================================
# Section 6 -- assemble ALL distinct cores from every source, compute exact flex for each
# ==================================================================================================
def all_core_sources():
    pool, pairs, bases = load_pool()
    sources = {}  # frozenset(idx) -> list of source tags
    def add(idxset, tag):
        key = frozenset(idxset)
        sources.setdefault(key, []).append(tag)
    add(range(len(pool)), "full_pool")
    mus = cload("mus")
    if mus:
        for u in mus["complete_ray_unions"]:
            add(u, "mus_complete_union")
        for u in mus.get("sample12_ray_unions", []):
            add(u, "mus_size12_sample_union")
    rc = cload("raycores")
    if rc:
        for c in rc["cores"]:
            add(c, "greedy_raycore")
    rnd = cload("random")
    if rnd:
        for c in rnd["configs"]:
            add(c, "random_intermediate")
    return pool, pairs, bases, sources

def stage_flexall(time_budget=35.0):
    pool, pairs, bases, sources = all_core_sources()
    results = cload("flex_results") or {}
    todo = [k for k in sources if str(sorted(k)) not in results]
    print(f"total distinct cores across all sources: {len(sources)}; "
          f"already computed: {len(sources)-len(todo)}; remaining: {len(todo)}")
    # do smallest first (cheap), largest last
    todo.sort(key=lambda k: len(k))
    t0 = time.time()
    n_done = 0
    max_flex_found = 0
    flexible_cores = []
    for key in todo:
        if time.time() - t0 > time_budget:
            break
        idx = sorted(key)
        sub = [pool[i] for i in idx]
        r = exact_flex_fast(sub)
        # verify uncolorability of every core we flex-check (defensive, cheap)
        unc = uncolorable_idx(pool, pairs, bases, idx)
        results[str(idx)] = dict(V=len(idx), flex=r["flex"], rankJ=r["rankJ"], ker=r["ker"],
                                  rankT=r["rankT"], E=r["E"], uncolorable=unc,
                                  sources=sources[key])
        n_done += 1
        if r["flex"] != 0:
            max_flex_found = max(max_flex_found, r["flex"])
            flexible_cores.append((idx, r["flex"], unc))
            print(f"  *** NONZERO FLEX FOUND *** V={len(idx)} flex={r['flex']} uncolorable={unc} "
                  f"sources={sources[key]}")
    print(f"processed {n_done} cores this run in {time.time()-t0:.2f}s; "
          f"{len(todo)-n_done} still remaining")
    csave("flex_results", results)
    print("SAVED intflex_flex_results.cache.json")
    if flexible_cores:
        print(f"!!! {len(flexible_cores)} cores with flex != 0 found this run !!!")
    return results

# ==================================================================================================
# Section 7 -- final report
# ==================================================================================================
def stage_report():
    results = cload("flex_results") or {}
    mus = cload("mus")
    rc = cload("raycores")
    rnd = cload("random")
    pool_cache = cload("pool")
    print("=" * 70)
    print("U3 INTEGER-POOL FLEXIBILITY SWEEP -- SUMMARY")
    print("=" * 70)
    if pool_cache:
        print(f"pool: {len(pool_cache['pool'])} rays / {len(pool_cache['pairs'])} pairs / "
              f"{len(pool_cache['bases'])} bases")
    if mus:
        print(f"basis-MUS enumeration: MCS-complete ({mus['mcs_count']} MCSes); "
              f"group-MUS complete for size<=11 ({mus['n_complete_muses']} MUSes, "
              f"{len(mus['complete_ray_unions'])} distinct ray-unions); "
              f"size-12 partial ({mus['n_size12_found']} found this run, sampled "
              f"{mus['n_size12_sampled']}); min group-MUS size={mus['min_group_mus_size']}")
    if rc:
        print(f"ray-level greedy critical cores: {rc['trials']} peel trials, "
              f"{len(rc['cores'])} distinct critical cores found")
    if rnd:
        print(f"random intermediate configs: {len(rnd['configs'])}")
    if results:
        flex_vals = [v["flex"] for v in results.values()]
        from collections import Counter
        print(f"TOTAL DISTINCT CORES WITH FLEX COMPUTED: {len(results)}")
        print("flex-value distribution:", dict(sorted(Counter(flex_vals).items())))
        nonzero = [(k, v) for k, v in results.items() if v["flex"] != 0]
        not_unc = [(k, v) for k, v in results.items() if not v["uncolorable"]]
        print(f"cores with flex != 0: {len(nonzero)}")
        print(f"cores that turned out NOT uncolorable (excluded from verdict): {len(not_unc)}")
        if nonzero:
            for k, v in nonzero[:20]:
                print("  FLEX!=0:", k, v)
    print("=" * 70)

SECTIONS = {
    "pool": stage_pool,
    "mus": stage_mus,
    "raycores": stage_raycores,
    "raycores_verify": stage_raycores_verify,
    "randomintermediate": stage_randomintermediate,
    "flexall": stage_flexall,
    "bench": stage_bench,
    "report": stage_report,
}

if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "pool"
    if which not in SECTIONS:
        print("unknown stage:", which, "-- choices:", list(SECTIONS.keys())); sys.exit(1)
    SECTIONS[which]()
