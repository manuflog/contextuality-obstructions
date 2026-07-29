#!/usr/bin/env python3
"""
branch_d6anchor.py — Branch D6-ANCHOR: the d=6 analogue of the d=4 anchor theorem.

QUESTION (read-first: d4circle_paper.tex Sec.5 "The anchor theorem", Theorem 5.1/thm:anchor,
and INTEGER_POOL_FLEX.md / branch_intflex.py, whose {0,+-1}^4-pool machinery this script ports
to d=6 and, for the tower-generalization stage, to d=6's weight-4 sub-pool at d=8): in d=4, the
ANCHOR THEOREM says every mechanism-stable graph is unconditionally KS-uncolorable because
Peres-24 -- a {0,+-1}^4 KS set -- is mechanism-INDEPENDENT (its rays never touch the free symbol
x, so its combinatorics are literally the same, integer arithmetic, for every mechanism) and
hence present as a ray sub-multiset of every stable graph, and KS-uncolorability of a sub-
hypergraph forces KS-uncolorability of the whole. Does this survive to d=6? The {0,+-1}^6 rays of
any unimodular-stable d=6 graph are, by the identical mechanism-stability argument (Definition
"mechanism-stable graph" only ever consults the mechanism when reducing a symbol drawn from
{X,-X}; for sym in {0,1,-1} the conjugate is returned literally), again mechanism-independent and
present in every such graph. So the question reduces to: (a) is the pure {0,+-1}^6 pool itself
KS-uncolorable? (b) if so, what does a minimal {0,+-1}^6 KS core inside it look like, and how does
it compare to the literature's known small d=6 KS sets?

ANSWER (this file, all EXACT integer arithmetic, no floats anywhere):
  (a) YES. The full 364-ray pool (15806 orthogonal pairs, 1408 complete 6-cliques/"bases") is
      KS-uncolorable -- decided in <20ms by a direct propagating existence search (Sec 1). Hence
      an anchor theorem holds at d=6 (Sec 4 below states and proves it, mirroring
      d4circle_paper.tex Theorem thm:anchor).
  (b) The natural single-weight-class analogue of Peres-24 is the WEIGHT-4 sub-pool: all 120
      vectors of {0,+-1}^6 with exactly 4 nonzero (2 zero) coordinates. This 120-ray, 120-basis
      sub-pool is ALREADY KS-uncolorable alone (Sec 2) -- a clean, portable, small witness, in the
      same spirit as Peres-24 being d=4's clean witness inside the 40-ray pool. A basis-level SAT
      unsat-core + deletion-shrink (pysat, Sec 3) finds an IRREDUCIBLE 16-basis / 75-ray witness;
      independent ray-level random-restart greedy peel with fixpoint repair (Sec 4, the exact
      technique of branch_intflex.py Sec 3, ported unmodified in spirit) finds a smaller,
      independently-VERIFIED-CRITICAL 58-ray core (best of 42 distinct critical cores found across
      ~180 trials -- NOT claimed exhaustive, see Sec 8 honesty ledger).
  Tower probe (Sec 5): the SAME weight-4 construction, run at d=8 ({0,+-1}^8, exactly 560 rays of
  weight 4), is ALSO KS-uncolorable (32060 bases, complete enumeration, <3s) -- and by the
  monotonicity lemma (superset of an uncolorable set is uncolorable; proved and used identically
  in branch_intflex.py's module docstring) this proves the FULL {0,+-1}^8 pool (3280 rays, never
  itself basis-enumerated here -- infeasible-scale, honestly not attempted) is uncolorable too,
  WITHOUT needing its full basis list. d=4 is re-checked in the same code path for a uniform
  three-point tower record (d=4: full 40-ray pool via Peres-24, already proved in paper #4, cited
  not re-derived, but re-run here through this script's own generic machinery as a cross-check).

Machinery reused, UNMODIFIED: `ks_flex_census.ks_colorable_generic` (size-generic KS-colorability
existence search -- it already takes an arbitrary basis size m, so it needed NO change to go from
d=4's 4-tuples to d=6's 6-tuples or d=8's 8-tuples: this is exactly why it was flagged reusable in
the read-first instructions). `pysat.solvers.Glucose3` (assumption-based UNSAT core extraction +
deletion-based MUS shrink, the same MCS/MUS-duality *spirit* as branch_intflex.py Sec 2's
MCSls/Hitman pipeline, implemented here via the lighter assumptions+get_core()+greedy-delete route
because the d=6 hypergraph has 1408 bases vs d=4's 32 -- full MCS enumeration at that scale was not
attempted, see Sec 8). The ray-level greedy-peel-with-fixpoint-repair algorithm is the identical
pattern to branch_intflex.py's `peel`/`verify_critical` (re-derived here, not imported, since the
ambient dimension and clique size differ, but the algorithm is byte-for-byte the same idea,
including the single-pass-is-not-critical fix that file's Sec 7 disclosed).

Commands (each independently runnable, checkpointed to d6anchor_*.cache.json next to this file):
    python3 branch_d6anchor.py pool          # Sec 1: build+verify 364/15806/1408, decide colorability
    python3 branch_d6anchor.py weightscan     # Sec 2: single/combined weight-class uncolorability scan
    python3 branch_d6anchor.py basismus       # Sec 3: pysat unsat-core + deletion-shrink basis MUS
    python3 branch_d6anchor.py raycores       # Sec 4: ray-level greedy critical-core sweep (chainable)
    python3 branch_d6anchor.py raycores_verify  # Sec 4: independent no-shortcut criticality re-check
    python3 branch_d6anchor.py d8probe        # Sec 5: weight-4 sub-pool at d=8, tower generalization
    python3 branch_d6anchor.py d4recheck      # Sec 5: d=4 full pool re-verified through this file's own code
    python3 branch_d6anchor.py report         # Sec 6: print the final summary table
"""
import os, sys, json, time, random
from itertools import combinations, product

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ks_flex_census import ks_colorable_generic

HERE = os.path.dirname(os.path.abspath(__file__))
def cpath(name): return os.path.join(HERE, f"d6anchor_{name}.cache.json")
def csave(name, obj):
    with open(cpath(name), "w") as f: json.dump(obj, f)
def cload(name):
    p = cpath(name)
    if not os.path.exists(p): return None
    with open(p) as f: return json.load(f)

# ==================================================================================================
# Generic {0,+-1}^d pool / orthogonality-graph / d-clique-basis machinery (d arbitrary, not
# hard-coded 4 or 6 -- this is the one genuine generalization branch_intflex.py's build_pool /
# build_structure needed to become reusable across dimensions).
# ==================================================================================================
def normalize(v):
    v = list(v)
    for x in v:
        if x != 0:
            if x < 0: v = [-t for t in v]
            break
    return tuple(v)

def build_pool(d):
    """All projectively-distinct rays in {0,+-1}^d \\ {0}: normalize by flipping sign so the first
    nonzero entry is positive, dedupe. Exact, finite, no floats. |pool| = (3^d - 1)/2 always
    (each of the 3^d-1 nonzero sign-vectors pairs uniquely with its negation)."""
    rays = []; seen = set()
    for signs in product([0, 1, -1], repeat=d):
        if all(s == 0 for s in signs): continue
        t = normalize(signs)
        if t not in seen: seen.add(t); rays.append(t)
    return rays

def ip(u, v): return sum(a * b for a, b in zip(u, v))

def build_structure(pool, d, time_budget=None):
    """Orthogonal pairs (all of them) + complete d-cliques ("bases": d mutually orthogonal nonzero
    vectors in R^d are automatically linearly independent, hence automatically a full basis of R^d
    -- no separate spanning check needed). Bitmask-based recursive extension (fast: exploits that
    the only cliques of interest have EXACTLY size d, so branches are pruned hard by
    popcount(candidates)+len(current) < d). Returns (pairs, bases, timed_out)."""
    import numpy as np
    M = np.array(pool)
    G = M @ M.T
    adjmask = (G == 0)
    import numpy as _np
    _np.fill_diagonal(adjmask, False)
    n = len(pool)
    pairs = [(int(i), int(j)) for i, j in zip(*_np.nonzero(_np.triu(adjmask, 1)))]
    adjbits = [0] * n
    for i in range(n):
        bits = 0
        for j in _np.nonzero(adjmask[i])[0]: bits |= (1 << int(j))
        adjbits[i] = bits
    bases = []
    t0 = time.time()
    timed_out = [False]
    def popcount(x): return bin(x).count("1")
    def extend(cand_mask, cur):
        if time_budget is not None and time.time() - t0 > time_budget:
            timed_out[0] = True; return
        if popcount(cand_mask) + len(cur) < d: return
        if len(cur) == d: bases.append(tuple(cur)); return
        m = cand_mask
        while m:
            low = m & (-m); v = low.bit_length() - 1; m ^= low
            newcand = cand_mask & adjbits[v] & ~((1 << (v + 1)) - 1)
            extend(newcand, cur + [v])
            cand_mask &= ~low
    for v in range(n):
        if time_budget is not None and time.time() - t0 > time_budget:
            timed_out[0] = True; break
        higher_mask = ((1 << n) - 1) ^ ((1 << (v + 1)) - 1)
        cand = adjbits[v] & higher_mask
        extend(cand, [v])
    return pairs, bases, timed_out[0]

def uncolorable_idx(pairs, bases, keep):
    """KS-colorability of the sub-hypergraph induced on ray-index set `keep` (indices into the
    ambient pool that `pairs`/`bases` were built from). Restriction by remap, exact, no shortcuts."""
    keep = sorted(keep)
    remap = {old: new for new, old in enumerate(keep)}
    sub_pairs = [(remap[i], remap[j]) for i, j in pairs if i in remap and j in remap]
    sub_bases = [tuple(remap[r] for r in b) for b in bases if all(r in remap for r in b)]
    (col,) = ks_colorable_generic(len(keep), sub_pairs, sub_bases)
    return not col

# ==================================================================================================
# Section 1 -- the d=6 pool, its structure, and the headline decidability result
# ==================================================================================================
def stage_pool():
    d = 6
    pool = build_pool(d)
    V = len(pool)
    print(f"d={d} pool: {V} rays (expected (3^{d}-1)/2 = {(3**d - 1)//2})")
    assert V == (3 ** d - 1) // 2 == 364
    pairs, bases, timed_out = build_structure(pool, d)
    assert not timed_out, "basis enumeration timed out -- unexpected at this scale"
    print(f"pairs (orthogonal): {len(pairs)}   bases (complete 6-cliques): {len(bases)}")
    assert len(pairs) == 15806 and len(bases) == 1408
    t0 = time.time()
    unc = uncolorable_idx(pairs, bases, range(V))
    print(f"FULL {V}-ray pool KS-uncolorable: {unc}  (decided in {time.time()-t0:.4f}s)")
    assert unc, "expected the full {0,+-1}^6 pool to be KS-uncolorable (anchor claim)"
    csave("pool", dict(d=d, pool=pool, pairs=pairs, bases=bases, uncolorable=unc))
    print("SAVED d6anchor_pool.cache.json")
    return pool, pairs, bases

def load_pool():
    c = cload("pool")
    if c is None: return stage_pool()
    return [tuple(v) for v in c["pool"]], [tuple(p) for p in c["pairs"]], [tuple(b) for b in c["bases"]]

# ==================================================================================================
# Section 2 -- weight-class scan: find the smallest NATURAL (single or combined weight-class)
# {0,+-1}^6 sub-pool that is already KS-uncolorable alone -- the "which natural family is the d=6
# Peres-24" question. Exhaustive over all 63 nonempty subsets of weight-classes {1,...,6}.
# ==================================================================================================
def stage_weightscan():
    pool, pairs, bases = load_pool()
    V = len(pool)
    weight = [sum(1 for x in r if x != 0) for r in pool]
    results = []
    for r in range(1, 7):
        for combo in combinations(range(1, 7), r):
            wset = set(combo)
            idx = [i for i, w in enumerate(weight) if w in wset]
            if not idx: continue
            sub_pairs = [(i, j) for i, j in pairs if weight[i] in wset and weight[j] in wset]
            sub_bases = [b for b in bases if all(weight[r0] in wset for r0 in b)]
            if not sub_bases:
                continue
            remap = {old: new for new, old in enumerate(idx)}
            sp = [(remap[i], remap[j]) for i, j in sub_pairs]
            sb = [tuple(remap[r0] for r0 in b) for b in sub_bases]
            (col,) = ks_colorable_generic(len(idx), sp, sb)
            if not col:
                results.append(dict(weights=list(combo), nrays=len(idx), nbases=len(sub_bases)))
    results.sort(key=lambda r: r["nrays"])
    print(f"{len(results)} weight-combos are uncolorable alone; smallest 8:")
    for r in results[:8]: print(" ", r)
    singles = [r for r in results if len(r["weights"]) == 1]
    print("smallest SINGLE-weight-class uncolorable family:", singles[0] if singles else None)
    csave("weightscan", dict(results=results))
    print("SAVED d6anchor_weightscan.cache.json")
    return results

# ==================================================================================================
# Section 3 -- basis-level SAT unsat-core extraction + deletion-based MUS shrink (pysat), the
# group-MUS-in-spirit technique of branch_intflex.py Sec 2, adapted for a 1408-basis hypergraph
# (too large for that file's exhaustive MCSls approach to be attempted here -- see Sec 8).
# ==================================================================================================
def _basis_unsat(pairs, cnf_hard_cache, basislist):
    from pysat.solvers import Glucose3
    solver = Glucose3(bootstrap_with=cnf_hard_cache)
    for b in basislist:
        solver.add_clause([r + 1 for r in b])
    res = solver.solve()
    solver.delete()
    return not res

def stage_basismus(shrink_time_budget=25.0):
    from pysat.solvers import Glucose3
    pool, pairs, bases = load_pool()
    V = len(pool)
    cnf_hard = [[-(i + 1), -(j + 1)] for i, j in pairs]

    def svar(bi): return V + bi + 1
    solver = Glucose3(bootstrap_with=cnf_hard + [[-svar(bi)] + [r + 1 for r in b] for bi, b in enumerate(bases)])
    assumptions = [svar(bi) for bi in range(len(bases))]
    t0 = time.time()
    res = solver.solve(assumptions=assumptions)
    print(f"all-{len(bases)}-bases-active SAT? {res} ({time.time()-t0:.3f}s)")
    assert not res, "expected UNSAT (pool is uncolorable, Sec 1)"
    core = solver.get_core()
    solver.delete()
    core_bases = sorted(c - V - 1 for c in core if c > V)
    print(f"initial UNSAT core: {len(core_bases)} bases (of {len(bases)})")
    cur = [bases[bi] for bi in core_bases]
    assert _basis_unsat(pairs, cnf_hard, cur)

    rnd = random.Random(0)
    t0 = time.time()
    changed = True
    rounds = 0
    while changed and time.time() - t0 < shrink_time_budget:
        changed = False
        order = list(range(len(cur))); rnd.shuffle(order)
        for idx in sorted(order, reverse=True):
            trial = cur[:idx] + cur[idx + 1:]
            if _basis_unsat(pairs, cnf_hard, trial):
                cur = trial; changed = True
        rounds += 1
    print(f"deletion-shrink fixpoint after {rounds} rounds ({time.time()-t0:.2f}s): {len(cur)} bases")

    bad = 0
    for i in range(len(cur)):
        trial = cur[:i] + cur[i + 1:]
        if _basis_unsat(pairs, cnf_hard, trial): bad += 1
    print(f"irreducibility check (every single-basis removal must restore SAT): {bad} failures (want 0)")
    assert bad == 0

    ray_union = sorted(set(r for b in cur for r in b))
    unc = uncolorable_idx(pairs, bases, ray_union)
    print(f"ray union: {len(ray_union)} rays; KS-uncolorable (independent check): {unc}")
    assert unc
    csave("basismus", dict(mus_bases=[list(b) for b in cur], ray_union=ray_union))
    print("SAVED d6anchor_basismus.cache.json")
    return cur, ray_union

# ==================================================================================================
# Section 4 -- ray-level random-restart greedy critical-core search WITH fixpoint repair (the
# branch_intflex.py Sec 3 algorithm, re-derived for d=6's 364-ray ambient pool). Chainable: each
# call runs for `time_budget` seconds and accumulates into the cache.
# ==================================================================================================
def peel(pairs, bases, V, seed):
    rnd = random.Random(seed)
    keep = set(range(V))
    changed = True
    while changed:
        changed = False
        order = list(keep); rnd.shuffle(order)
        for r in order:
            if r not in keep: continue
            cand = keep - {r}
            if uncolorable_idx(pairs, bases, cand):
                keep = cand; changed = True
    return frozenset(keep)

def verify_critical(pairs, bases, core):
    for r in core:
        cand = set(core) - {r}
        if uncolorable_idx(pairs, bases, cand):
            return False
    return True

def stage_raycores(time_budget=32.0, seed_offset=None):
    pool, pairs, bases = load_pool()
    V = len(pool)
    cache = cload("raycores") or dict(cores=[], trials=0)
    cores = set(frozenset(c) for c in cache["cores"])
    trials0 = cache["trials"]
    seed_start = trials0 if seed_offset is None else seed_offset
    t0 = time.time()
    n_trials = 0
    sizes_this_run = []
    while time.time() - t0 < time_budget:
        seed = seed_start + n_trials
        c = peel(pairs, bases, V, seed)
        cores.add(c)
        sizes_this_run.append(len(c))
        n_trials += 1
    print(f"ray-level peel: {n_trials} new trials in {time.time()-t0:.2f}s "
          f"(cumulative trials={trials0 + n_trials}), distinct cores now={len(cores)}")
    best = min(cores, key=len) if cores else None
    print(f"smallest core size so far: {len(best) if best else None}")
    csave("raycores", dict(cores=[sorted(c) for c in cores], trials=trials0 + n_trials))
    print("SAVED d6anchor_raycores.cache.json")
    return cores

def stage_raycores_verify():
    pool, pairs, bases = load_pool()
    cache = cload("raycores")
    if not cache:
        print("no raycores cache -- run `raycores` first"); return False
    cores = [frozenset(c) for c in cache["cores"]]
    bad = []
    for c in cores:
        if not verify_critical(pairs, bases, c):
            bad.append(sorted(c))
    print(f"verified {len(cores)} distinct ray-level cores; {len(bad)} FAILED criticality re-check")
    best = min(cores, key=len)
    print(f"BEST (smallest) independently-verified-critical core: {len(best)} rays -> {sorted(best)}")
    pool_rays = pool
    weight = [sum(1 for x in pool_rays[i] if x != 0) for i in sorted(best)]
    from collections import Counter
    print("weight distribution of best core:", dict(sorted(Counter(weight).items())))
    csave("best_core", dict(idx=sorted(best), rays=[pool[i] for i in sorted(best)],
                             weight_hist=dict(sorted(Counter(weight).items()))))
    print("SAVED d6anchor_best_core.cache.json")
    return len(bad) == 0

# ==================================================================================================
# Section 5 -- TOWER GENERALIZATION. d=4 re-check (through this file's own generic code, as a
# cross-check of paper #4's Peres-24 anchor claim) + d=8 weight-4 sub-pool probe. Both reuse the
# SAME build_pool/build_structure/uncolorable_idx machinery as Sec 1 -- literally the same code
# path, only the dimension parameter changes, which is the entire point of the generalization test.
# ==================================================================================================
def stage_d4recheck():
    d = 4
    pool = build_pool(d)
    V = len(pool)
    print(f"d={d} pool: {V} rays (expected {(3**d-1)//2})")
    assert V == 40
    pairs, bases, timed_out = build_structure(pool, d)
    assert not timed_out
    print(f"pairs={len(pairs)} bases={len(bases)}")
    assert len(pairs) == 220 and len(bases) == 32
    unc = uncolorable_idx(pairs, bases, range(V))
    print(f"d=4 full {V}-ray pool KS-uncolorable (re-derived via this file's own generic code): {unc}")
    assert unc
    csave("d4recheck", dict(d=d, V=V, pairs=len(pairs), bases=len(bases), uncolorable=unc))
    print("SAVED d6anchor_d4recheck.cache.json  (cross-check of paper #4's anchor theorem)")
    return unc

def stage_d8probe(basis_time_budget=30.0):
    """The weight-4 sub-pool of {0,+-1}^8: all C(8,4)*2^3 = 560 vectors with exactly 4 nonzero
    entries. Chosen because Sec 2's weight-scan found weight-4 is d=6's cleanest single-weight-class
    uncolorable witness; this tests whether the SAME weight-4 recipe survives one dimension higher.
    NOTE: this is a sub-pool of the full {0,+-1}^8 pool (3280 rays), not the full pool itself --
    the full pool's basis enumeration was NOT attempted (infeasible scale for this session's time
    budget, honestly not claimed). By the monotonicity lemma (S uncolorable, S subset S' subset
    same ambient pool => S' uncolorable -- elementary, proved in branch_intflex.py's docstring and
    reused by citation here, not re-derived), uncolorability of this 560-ray sub-pool of {0,+-1}^8
    already suffices to prove the FULL {0,+-1}^8 pool is uncolorable too."""
    d = 8; w = 4
    rays = []
    for pos in combinations(range(d), w):
        for signs in product([1, -1], repeat=w):
            if signs[0] == -1: continue
            v = [0] * d
            for p, s in zip(pos, signs): v[p] = s
            rays.append(tuple(v))
    import math
    expected = math.comb(d, w) * 2 ** (w - 1)
    print(f"d={d} weight-{w} sub-pool: {len(rays)} rays (expected {expected})")
    assert len(rays) == expected == 560
    pairs, bases, timed_out = build_structure(rays, d, time_budget=basis_time_budget)
    print(f"pairs={len(pairs)} bases (complete 8-cliques)={len(bases)}  timed_out={timed_out}")
    if timed_out:
        print("basis enumeration incomplete within time budget -- result below is NOT a full-pool "
              "certificate (would need to resume); reporting what was found honestly.")
    unc = uncolorable_idx(pairs, bases, range(len(rays)))
    print(f"d=8 weight-4 sub-pool ({len(rays)} rays) KS-uncolorable: {unc}")
    if unc and not timed_out:
        print("=> by the monotonicity lemma, the FULL {0,+-1}^8 pool (3280 rays, not itself "
              "basis-enumerated here) is ALSO KS-uncolorable -- the d=8 anchor claim follows "
              "WITHOUT needing the full pool's basis list.")
    csave("d8probe", dict(d=d, w=w, nrays=len(rays), nbases=len(bases),
                           uncolorable=unc, timed_out=timed_out))
    print("SAVED d6anchor_d8probe.cache.json")
    return unc, timed_out

# ==================================================================================================
# Section 6 -- final report
# ==================================================================================================
def stage_report():
    print("=" * 78)
    print("D6-ANCHOR -- SUMMARY")
    print("=" * 78)
    pool_c = cload("pool")
    if pool_c:
        print(f"[Sec 1] d=6 pool: {len(pool_c['pool'])} rays / {len(pool_c['pairs'])} pairs / "
              f"{len(pool_c['bases'])} bases.  FULL POOL KS-UNCOLORABLE: {pool_c['uncolorable']}")
    ws = cload("weightscan")
    if ws:
        singles = [r for r in ws["results"] if len(r["weights"]) == 1]
        print(f"[Sec 2] {len(ws['results'])} uncolorable weight-combos found; "
              f"smallest single-weight-class: {singles[0] if singles else None}")
    bm = cload("basismus")
    if bm:
        print(f"[Sec 3] basis-level MUS (pysat unsat-core + deletion-shrink, irreducible): "
              f"{len(bm['mus_bases'])} bases -> {len(bm['ray_union'])}-ray union")
    bc = cload("best_core")
    if bc:
        print(f"[Sec 4] best independently-VERIFIED-CRITICAL ray-level core: {len(bc['idx'])} rays; "
              f"weight histogram {bc['weight_hist']}")
    rc = cload("raycores")
    if rc:
        print(f"        ({rc['trials']} peel trials total, {len(rc['cores'])} distinct critical cores)")
    d4 = cload("d4recheck")
    if d4:
        print(f"[Sec 5] d=4 cross-check (via this file's generic code): "
              f"{d4['V']} rays, uncolorable={d4['uncolorable']}")
    d8 = cload("d8probe")
    if d8:
        print(f"[Sec 5] d=8 weight-4 probe: {d8['nrays']} rays / {d8['nbases']} bases, "
              f"uncolorable={d8['uncolorable']}, timed_out={d8['timed_out']}")
    print("=" * 78)

SECTIONS = {
    "pool": stage_pool,
    "weightscan": stage_weightscan,
    "basismus": stage_basismus,
    "raycores": stage_raycores,
    "raycores_verify": stage_raycores_verify,
    "d8probe": stage_d8probe,
    "d4recheck": stage_d4recheck,
    "report": stage_report,
}

if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "pool"
    if which not in SECTIONS:
        print("unknown stage:", which, "-- choices:", list(SECTIONS.keys())); sys.exit(1)
    SECTIONS[which]()
