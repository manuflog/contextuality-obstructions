#!/usr/bin/env python3
"""
branch_d4flex.py -- Branch D4F: THE d=4 FLEXIBLE-FAMILY HUNT.

Does any two-symbol d=4 mechanism (M6-M13, ALPHABET_D4.md Sec.2) admit a positive-dimensional
flexible KS family -- a d=4 analogue of the Gould-Aravind circle (PERES_PENROSE.md)? See
D4_FLEX_HUNT.md for the full write-up; this file is the reproducible spine.

READ FIRST: ALPHABET_D4.md (the 16-mechanism d=4 classification + the Peres-24-containment
theorem), MECHANISM_MODULI.md (the "mechanism-stable graph" / Laurent-identity method, Sec.4's
M3-line hunt), ks_flex_census.py (exact flex machinery), peres_penrose.py (the GA circle's exact
Laurent form -- the positive target shape).

STRUCTURE OF THIS FILE:
  Stage 1 (triage):        mechanism_triage()      -- dim of each of the 16 mechanisms' variety
  Stage 2 (stable graph):  stable_graph(mech)       -- Laurent-identity-stable graph per mechanism,
                            peres24_check()          the KEY structural question: is Peres-24 always
                                                      inside the stable graph? (settled immediately,
                                                      by construction, then computationally confirmed)
  Stage 3 (concrete flex): concrete_flex(name,...)  -- exact flex at a representative point of each
                                                      tested mechanism, reusing ks_flex_census.py
  Stage 4 (GA ansatz):     ga_ansatz_check(mech)    -- does the stable graph's basis-participating
                                                      part realize a genuine flex family (no
                                                      degeneration along the whole variety)?

Laurent-dict convention (generalizing mechanism_moduli.py's (a,b)-polynomial line-stable method to
BOTH lines and circles, exponents rather than just degree<=2): a "Laurent polynomial in the free
symbol X" is a dict {exponent: Fraction(coeff)}, exponents any integers (lines only ever produce
non-negative exponents 0,1,2; circles centered at the origin produce -2..2 via X* = R^2/X). Two
polys are added by summing coefficients per exponent (dropping zeros); multiplied by convolution;
"identically zero on the WHOLE mechanism variety" iff the dict, after collecting, is empty -- this
is EXACT (Fraction arithmetic, no floats): a Laurent polynomial with infinitely many roots on an
infinite variety (line or circle, both uncountable) must be the zero Laurent polynomial (clear
denominators to reduce to an ordinary polynomial of bounded degree with infinitely many roots).

No existing file is modified. Machinery reused, UNMODIFIED: ks_flex_census.py (qmul, qneg, qconj,
qz, ZERO, herm_dot, bil_dot, raw_vectors, collect_rays, build_structure_d, uncolorable_d,
basis_participating, restrict, greedy_critical_core_d, exact_flex_hermitian_quadratic,
exact_flex_real_quadratic, find_primes_ring, t0_tau, proportional, ks_colorable_generic,
cache_save, cache_load), exact_rigidity.py (integer_rays_peres24), alphabet_d4.py (derive4, as a
function call, for the Stage-1 mechanism list -- not re-derived), mechanism_moduli.py (cited, not
imported, for the line-stable METHOD which this file generalizes to circles).
"""
import os, sys, time, io, contextlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fractions import Fraction as Fr
from itertools import product as iproduct, combinations

from ks_flex_census import (
    qmul, qneg, qconj, qz, ZERO, herm_dot, bil_dot, raw_vectors, collect_rays,
    build_structure_d, uncolorable_d, basis_participating, restrict, greedy_critical_core_d,
    exact_flex_hermitian_quadratic, exact_flex_real_quadratic, find_primes_ring, t0_tau,
    proportional, ks_colorable_generic, cache_save, cache_load,
)
from exact_rigidity import integer_rays_peres24
import alphabet_d4 as ad4

# ==================================================================================================
# STAGE 1: mechanism_triage -- dim of the solution variety {x valid} for each of the 16 d=4
# mechanisms (M6-M13), read off `alphabet_d4.derive4()`'s exact classification UNMODIFIED (no
# re-derivation -- this stage just tabulates dim(point)=0 vs dim(line/circle)=1).
# ==================================================================================================
def mechanism_triage():
    t0 = time.time()
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        found = ad4.derive4()
    rows = []
    for key in sorted(found):
        if not key.startswith("NEW"):
            continue
        a1, aRx, aIx, aN = found[key][0][2]
        if key.startswith("NEW(rational)") or key.startswith("NEW(real quadratic"):
            dim = 0
            shape = "point(s)"
        else:
            dim = 1
            shape = "circle" if "modulus" in key or "circle" in key else "line"
        rows.append((key, dim, shape, (a1, aRx, aN)))
    print("[triage] 16 genuinely-new d=4 mechanisms (alphabet_d4.derive4, EXACT, exhaustive):")
    print(f"{'mechanism':<62} {'dim':>3}  {'shape':<8}  (a1,aRx,aN)")
    for key, dim, shape, coef in rows:
        print(f"  {key:<60} {dim:>3}  {shape:<8}  {coef}")
    n_pos = sum(1 for _, dim, _, _ in rows if dim == 1)
    n_pt = sum(1 for _, dim, _, _ in rows if dim == 0)
    print(f"[triage] {n_pos} positive-dimensional (line/circle) mechanisms, {n_pt} point-like "
          f"(dim 0, isolated rational points or real-quadratic-irrational pairs). "
          f"({time.time()-t0:.2f}s)")
    return rows


# ==================================================================================================
# STAGE 2: the Laurent-identity mechanism-stable-graph method, generalizing mechanism_moduli.py's
# M3 LINE-stable method (a,b)-polynomials in X) to BOTH lines and circles via exponent dicts.
# ==================================================================================================
def lz(): return {}                              # Laurent zero
def l1(c): return {0: Fr(c)} if c != 0 else {}    # constant c
def lX(c=1): return {1: Fr(c)} if c != 0 else {}  # c*X

def ladd(a, b):
    out = dict(a)
    for e, c in b.items():
        out[e] = out.get(e, 0) + c
        if out[e] == 0: del out[e]
    return out

def lneg(a): return {e: -c for e, c in a.items()}
def lsub(a, b): return ladd(a, lneg(b))

def lmul(a, b):
    out = {}
    for e1, c1 in a.items():
        for e2, c2 in b.items():
            e = e1 + e2
            out[e] = out.get(e, 0) + c1 * c2
    return {e: c for e, c in out.items() if c != 0}

def lzero(a): return len(a) == 0

# ---- mechanism descriptors: each gives conj(X) as a Laurent dict, and a short tag ----
def line_mech(center, name):
    """Re(x) = center: x* = 2*center - x identically."""
    xstar = ladd(l1(2 * center), lneg(lX()))
    return dict(name=name, kind="line", conj_X=xstar, param=center)

def circle_mech(R2, name):
    """|x|^2 = R2 (centered at the origin): x* = R2 / x identically -- a genuine LAURENT
       (negative-exponent) relation, the direct d=4 analogue of the M2 mechanism that produced
       the Gould-Aravind circle (PERES_PENROSE.md): x = sqrt(R2)*e^{i theta} parametrizes the
       WHOLE variety directly."""
    xstar = {-1: Fr(R2)}
    return dict(name=name, kind="circle", conj_X=xstar, param=R2)

MECHS = {
    "M9":   circle_mech(1, "M9: |x|^2=1 (unimodular)"),
    "M8":   circle_mech(3, "M8: |x|^2=3"),
    "M8p":  circle_mech(Fr(1, 3), "M8': |x|^2=1/3"),
    "M10":  line_mech(0, "M10: Re(x)=0"),
    "M11a": line_mech(1, "M11a: Re(x)=1"),
    "M11b": line_mech(-1, "M11b: Re(x)=-1"),
    # M12a/M12b/M13a/M13b (off-center circles) are Mobius, NOT pure Laurent monomials -- see
    # D4_FLEX_HUNT.md's honesty section for why these are NOT attempted via this method.
}

RAW_ALPHABET_SYMS = ["0", "1", "-1", "X", "-X"]
SYM_DICT = {"0": lz(), "1": l1(1), "-1": l1(-1), "X": lX(1), "-X": lX(-1)}

def conj_symbol(sym, mech):
    if sym == "0": return lz()
    if sym == "1": return l1(1)
    if sym == "-1": return l1(-1)
    if sym == "X": return mech["conj_X"]
    if sym == "-X": return lneg(mech["conj_X"])

def raw_symbolic_vectors(dsize=4):
    return [v for v in iproduct(RAW_ALPHABET_SYMS, repeat=dsize) if any(c != "0" for c in v)]

def sym_minor(u, v, i, j):
    """Plain (non-conjugated) 2x2 minor u_i*v_j - u_j*v_i, as a Laurent dict -- used ONLY for
       generic (mechanism-INDEPENDENT) ray dedup: X is treated as a free indeterminate, exactly
       matching ks_flex_census.proportional's qminor but over the abstract polynomial ring Q[X]
       (never invoking any mechanism relation)."""
    return lsub(lmul(SYM_DICT[u[i]], SYM_DICT[v[j]]), lmul(SYM_DICT[u[j]], SYM_DICT[v[i]]))

def sym_proportional(u, v):
    return all(lzero(sym_minor(u, v, i, j)) for i, j in combinations(range(len(u)), 2))

_RAY_CACHE = None
def generic_symbolic_rays(dsize=4):
    """The mechanism-INDEPENDENT raw pool: distinct rays of {0,+-1,+-X}^4 up to proportionality
       treating X as a free indeterminate (i.e. BEFORE any mechanism relation is imposed). This
       ray SET is the same for every mechanism (dedup never uses conj_X)."""
    global _RAY_CACHE
    if _RAY_CACHE is not None: return _RAY_CACHE
    raws = raw_symbolic_vectors(dsize)
    rays = []
    for v in raws:
        if not any(sym_proportional(v, r) for r in rays):
            rays.append(v)
    _RAY_CACHE = rays
    return rays

def stable_dot(u, v, mech):
    acc = lz()
    for uc, vc in zip(u, v):
        term = lmul(conj_symbol(uc, mech), SYM_DICT[vc])
        acc = ladd(acc, term)
    return acc

def stable_graph(mech_key, dsize=4, verbose=True):
    """Build the maximal MECHANISM-STABLE graph: same ray set as `generic_symbolic_rays` (mechanism
       -independent), edges = pairs whose Hermitian dot is IDENTICALLY zero for the WHOLE mechanism
       variety (i.e. the Laurent dict collapses to {}), bases = complete dsize-cliques."""
    t0 = time.time()
    mech = MECHS[mech_key]
    rays = generic_symbolic_rays(dsize)
    V = len(rays)
    pairs = [(i, j) for i, j in combinations(range(V), 2)
             if lzero(stable_dot(rays[i], rays[j], mech))]
    adj = [set() for _ in range(V)]
    for i, j in pairs: adj[i].add(j); adj[j].add(i)
    bases = []
    def extend(cands, cur):
        if len(cur) == dsize: bases.append(tuple(cur)); return
        if len(cur) + len(cands) < dsize: return
        cl = sorted(cands)
        for idx, v in enumerate(cl):
            rest = set(cl[idx + 1:]) & adj[v]
            extend(rest, cur + [v])
    for start in range(V):
        extend(set(x for x in adj[start] if x > start), [start])
    bp = basis_participating(V, bases)
    n_pure = sum(1 for r in rays if all(c in ("0", "1", "-1") for c in r))
    n_x_in_bp = sum(1 for i in bp if any(rays[i][c] in ("X", "-X") for c in range(dsize)))
    if verbose:
        print(f"[stable_graph:{mech_key}] {mech['name']}: rays={V} pairs={len(pairs)} "
              f"bases={len(bases)} basis-participating={len(bp)} "
              f"(pure-{{0,+-1}} rays in pool: {n_pure}, X-using rays IN a basis: {n_x_in_bp}) "
              f"({time.time()-t0:.2f}s)")
    return dict(mech=mech_key, rays=rays, pairs=pairs, bases=bases, bp=bp, adj=adj)


# ==================================================================================================
# The KEY structural question (settle immediately, per the brief): is Peres-24 -- built purely
# from {0,+-1} entries, NO x-dependence at all -- automatically inside every mechanism-stable
# graph? ARGUMENT (immediate, no computation needed): Peres-24's own rays use ONLY entries in
# {0,+-1} (ALPHABET_D4.md Sec.4, `exact_rigidity.integer_rays_peres24`, re-cited not re-derived).
# The stable-graph's dot function, restricted to two PURE (X-free) rays, reduces to
# `conj_symbol('1'/'−1', mech) = literal +-1 (UNCHANGED, no mechanism dependence at all -- see
# `conj_symbol` above: only the 'X'/'-X' branches touch `mech`). So the pairwise/triad
# combinatorics among the pure rays is IDENTICAL for every mechanism (in fact identical to plain
# integer arithmetic) -- i.e. Peres-24's own internal structure is trivially "stable" for ANY
# mechanism, and it is already known KS-uncolorable on its own. Hence EVERY mechanism-stable graph
# is automatically KS-uncolorable. This is verified computationally below for one representative
# mechanism (the check is mechanism-INDEPENDENT by the argument above, so one witness suffices).
# ==================================================================================================
def peres24_check(mech_key="M9"):
    t0 = time.time()
    p24 = integer_rays_peres24()
    entries = set(c for v in p24 for c in v)
    assert entries <= {0, 1, -1}
    print(f"[peres24_check] Peres-24: {len(p24)} rays, entries {sorted(entries)} subset {{0,+-1}} "
          f"(re-cited from exact_rigidity.py / ALPHABET_D4.md Sec.4, not re-derived).")
    g = stable_graph(mech_key, verbose=False)
    rays = g["rays"]
    # map each PURE ({0,+-1}-only) symbolic ray to a concrete (int) vector for comparison
    sym2num = {"0": 0, "1": 1, "-1": -1}
    pure_idx = [i for i, r in enumerate(rays) if all(c in sym2num for c in r)]
    pure_vecs = [tuple(sym2num[c] for c in rays[i]) for i in pure_idx]
    print(f"[peres24_check] pure-{{0,+-1}} rays in the (mechanism-independent) raw pool: "
          f"{len(pure_idx)} of {len(rays)} total.")
    p24_pairs = [tuple((c, 0) for c in v) for v in p24]
    pure_pairs = [tuple((c, 0) for c in v) for v in pure_vecs]
    found = [any(proportional(pr, r, 0, 1) for r in pure_pairs) for pr in p24_pairs]
    n_in = sum(found)
    print(f"[peres24_check] Peres-24 rays found (exact proportionality, trivial ring) inside the "
          f"pure-{{0,+-1}} sub-pool: {n_in} / {len(p24)}.")
    assert n_in == len(p24), "Peres-24 NOT fully contained in the pure sub-pool -- STOP"
    # now confirm the STABLE graph (for this one representative mechanism) reproduces Peres-24's
    # own known-uncolorable combinatorics among those same pure rays: since conj_symbol is
    # mechanism-independent on pure entries, the induced sub-adjacency must match `bil_dot`
    # computed with the trivial ring (B=0,C=1, i.e. plain integers).
    pure_pos = {v: i for i, v in enumerate(pure_vecs)}
    trivial_uncolorable, _, pairs_triv, bases_triv = uncolorable_d(pure_pairs, bil_dot, 0, 1, 4)
    print(f"[peres24_check] the pure-{{0,+-1}} sub-pool ALONE (any mechanism, since it never "
          f"touches conj_X) is KS-uncolorable: {trivial_uncolorable} "
          f"({len(pure_pairs)} rays, {len(pairs_triv)} pairs, {len(bases_triv)} bases)")
    assert trivial_uncolorable
    print("[peres24_check] VERDICT: since the pure-{0,+-1} sub-pool is (a) present in EVERY "
          "mechanism-stable graph (its rays never use X, so membership is mechanism-free) and (b) "
          "ALONE already KS-uncolorable containing Peres-24, EVERY mechanism-stable graph, for "
          "EVERY one of the 10 positive-dimensional mechanisms, is automatically KS-uncolorable. "
          "The M3-style 'stable graph colorable => no family' exclusion used in d=3 "
          "(MECHANISM_MODULI.md Sec.4) THEREFORE CANNOT RUN IN d=4 -- it never produces a "
          "'colorable, so ruled out' verdict for ANY mechanism. This determines the whole "
          "branch's shape: mechanisms must be excluded (if at all) by a FLEX argument, not a "
          "colorability argument.")
    print(f"[peres24_check] done ({time.time()-t0:.2f}s)")
    return dict(n_in=n_in, trivial_uncolorable=trivial_uncolorable)


# ==================================================================================================
# STAGE 3: concrete_flex -- pick a concrete representative point on each mechanism's variety, build
# the CONCRETE pool via ks_flex_census's exact ring machinery (raw_vectors/collect_rays/
# build_structure_d), non-degeneration cross-check against the abstract stable graph (Stage 2)'s
# ray/pair/basis counts, greedy critical core, exact flex certificate.
# ==================================================================================================
CONCRETE_POINTS = {
    # name: (B, C, dotfn, mech_key, description)
    "M9_zeta6":  (1, -1, herm_dot, "M9",  "x=zeta_6=e^{i pi/3}=(1+i sqrt3)/2, t^2=t-1, |x|^2=1, Re(x)=1/2"),
    "M10_isqrt2":(0, -2, herm_dot, "M10", "x=i*sqrt(2), t^2=-2, Re(x)=0, |x|^2=2"),
    "M11a_1pi":  (2, -2, herm_dot, "M11a","x=1+i, t^2=2t-2, Re(x)=1, |x|^2=2"),
    "M8_sqrt3":  (0, 3,  bil_dot,  "M8",  "x=sqrt(3) (REAL point; cites ALPHABET_D4.md sqrt3_flex)"),
}

def _sym_ray_signature(rays_num, B, C, dotfn, dsize=4):
    """Build the pool/graph at a CONCRETE ring point, exactly mirroring alphabet_d4.py's
       _pool_common (unmodified logic, inlined here since alphabet_d4's version is hardcoded to
       its own 5 representative names)."""
    ONE = (1, 0); X = (0, 1)
    alph = [ZERO, ONE, qneg(ONE), X, qneg(X)]
    raws = raw_vectors(alph, dsize)
    rays = collect_rays(raws, B, C)
    pairs, bases, adj = build_structure_d(rays, dotfn, B, C, dsize)
    return rays, pairs, bases

def concrete_flex(name, trials=4, seed0=0):
    t0 = time.time()
    B, C, dotfn, mech_key, desc = CONCRETE_POINTS[name]
    print(f"[concrete_flex:{name}] {desc}  (ring B={B},C={C})")
    rays, pairs, bases = _sym_ray_signature(None, B, C, dotfn)
    (col,) = ks_colorable_generic(len(rays), pairs, [list(b) for b in bases])
    print(f"[concrete_flex:{name}] concrete pool: {len(rays)} rays, {len(pairs)} pairs, "
          f"{len(bases)} bases, KS-uncolorable={not col}")
    # non-degeneration cross-check vs the abstract mechanism-stable graph (Stage 2), when defined
    if mech_key in MECHS:
        g = stable_graph(mech_key, verbose=False)
        match = (len(rays) == len(g["rays"]) and len(pairs) == len(g["pairs"])
                 and len(bases) == len(g["bases"]))
        print(f"[concrete_flex:{name}] non-degeneration cross-check vs abstract {mech_key} stable "
              f"graph (rays={len(g['rays'])}, pairs={len(g['pairs'])}, bases={len(g['bases'])}): "
              f"EXACT MATCH={match}" + ("" if match else
              "  -- MISMATCH: concrete point has EXTRA (accidental) structure beyond what's "
              "stable for the whole variety, OR fewer (a degenerate/special point) -- see writeup."))
    assert not col
    bp = basis_participating(len(rays), bases)
    sub, _ = restrict(rays, bp)
    core_idx = greedy_critical_core_d(sub, dotfn, B, C, 4, trials=trials, seed0=seed0, verbose=True)
    core = [sub[i] for i in core_idx]
    cpairs, cbases, _ = build_structure_d(core, dotfn, B, C, 4)
    n_x = sum(1 for r in core if any(c[1] != 0 for c in r))
    print(f"[concrete_flex:{name}] critical core: {len(core)} rays ({n_x} use x), "
          f"{len(cpairs)} pairs, {len(cbases)} bases  ({time.time()-t0:.2f}s)")
    primes = find_primes_ring(B, C, count=2, below=200003)
    if dotfn is herm_dot:
        cert = exact_flex_hermitian_quadratic(core, B, C, primes)
    else:
        cert = exact_flex_real_quadratic(core, B, C, primes)
    print(f"[concrete_flex:{name}] n={cert['n']}, E={cert['E']}, "
          f"trivial-expected={cert['triv_expected']}, mod-p bound: 0 <= flex <= {cert['bound']}  "
          f"({time.time()-t0:.2f}s total)")
    cache_save(f"d4flex_{name}_core", core)
    return dict(name=name, rays=len(rays), pairs=len(pairs), bases=len(bases), core=core, cert=cert)


# ==================================================================================================
# STAGE 3b: symbolic critical core -- peel a CRITICAL KS-uncolorable core directly from the
# ABSTRACT stable graph (index-level only: pairs/bases, no ring arithmetic at all -- this is the
# key efficiency move that avoids the 272-ray full-pool blowup seen in concrete_flex). Any core
# found this way is uncolorable USING ONLY MECHANISM-STABLE edges, i.e. its KS-uncolorability
# certificate survives literally EVERY point of the mechanism variety (MECHANISM_MODULI.md Task 2's
# forward lemma, generalized): if additionally NO accidental extra structure/degeneration occurs at
# a generic point (checked below, on the SMALL core only -- cheap), this gives flex >= 1 directly.
# ==================================================================================================
class _TO(Exception): pass
def _handler(signum, frame): raise _TO()

def _colorable_guarded(nrays, pairs, bases, budget_s):
    """ks_colorable_generic with a wall-clock guard (SIGALRM): if the backtracking search takes
       longer than `budget_s` (occasionally slow on hard instances -- an already-disclosed
       property of this repo's exact backtracking colorability search, see BRANCH_W6/M3M2.md's
       'SAT grind' sections), treat the ray as NOT removable (conservatively keep it) rather than
       block. This is an honest, disclosed HEURISTIC SPEEDUP of the peel order only -- criticality
       of the FINAL core is always independently re-verified below with NO timeout."""
    import signal
    old = signal.signal(signal.SIGALRM, _handler)
    signal.setitimer(signal.ITIMER_REAL, budget_s)
    try:
        (col,) = ks_colorable_generic(nrays, pairs, [list(b) for b in bases])
        return col
    except _TO:
        return True  # conservative: ASSUME colorable (i.e. "removable is unsafe, keep it") on timeout
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)

def symbolic_greedy_core(V, pairs, bases, trials=1, seed0=0, verbose=True, budget_s=1.5,
                          wall_budget=32.0, resume=None):
    import random, time as _time
    idxs = list(range(V))
    (col0,) = ks_colorable_generic(V, pairs, [list(b) for b in bases])
    assert not col0, "stable graph itself is not KS-uncolorable -- cannot build a core"
    t_start = _time.time()
    best, complete = None, False
    for t in range(trials):
        order = idxs[:]
        random.Random(3000 + seed0 + t).shuffle(order)
        keep = set(idxs) if resume is None else set(resume)
        finished_pass = True
        for r in order:
            if r not in keep: continue
            if _time.time() - t_start > wall_budget:
                if verbose: print(f"    [wall-budget {wall_budget}s reached, stopping peel early]")
                finished_pass = False
                break
            cand = keep - {r}
            idxmap = {old: new for new, old in enumerate(sorted(cand))}
            sub_pairs = [(idxmap[i], idxmap[j]) for i, j in pairs if i in cand and j in cand]
            sub_bases = [tuple(idxmap[x] for x in b) for b in bases if all(x in cand for x in b)]
            col = _colorable_guarded(len(cand), sub_pairs, sub_bases, budget_s)
            if not col: keep = cand
        if best is None or len(keep) < len(best): best = keep
        complete = complete or finished_pass
        if verbose: print(f"    trial {t}: core size {len(keep)}  ({_time.time()-t_start:.1f}s)"
                           f"  {'[FULL PASS]' if finished_pass else '[TRUNCATED]'}")
    core = sorted(best)
    if not complete:
        return dict(core=core, converged=False)
    # FIXPOINT repair pass: a single greedy pass (esp. across resumed/checkpointed chunks) is NOT
    # guaranteed critical (an early-tested-and-kept ray can become removable after later removals
    # shrink the context) -- repeat full removal passes until ONE pass removes nothing, THEN this
    # is genuinely critical (every single remaining ray, tested against the FINAL set, is
    # necessary). No timeout guard in this repair stage -- exact, no shortcuts.
    changed = True
    while changed:
        changed = False
        if _time.time() - t_start > wall_budget:
            return dict(core=core, converged=False)
        for r in list(core):
            cand = [x for x in core if x != r]
            m2 = {old: new for new, old in enumerate(cand)}
            sp_ = [(m2[i], m2[j]) for i, j in pairs if i in cand and j in cand]
            sb_ = [tuple(m2[x] for x in b) for b in bases if all(x in cand for x in b)]
            (col,) = ks_colorable_generic(len(cand), sp_, [list(b) for b in sb_])
            if not col:
                core = cand
                changed = True
    return dict(core=core, converged=True)

def stable_core_and_test(mech_key, trials=1, ring_point=None, dsize=4, wall_budget=32.0, fresh=False,
                          graph_fn=None, seed0=0):
    """Peel a symbolic critical core from mechanism `mech_key`'s stable graph, then non-degeneration
       -check it at a concrete point (ring_point = (B,C,dotfn) triple) of the SAME mechanism.
       Checkpointed: if the peel doesn't converge within `wall_budget`, the partial `keep` set is
       cached and RESUMED on the next call (same mech_key) -- run this target repeatedly until it
       reports 'CONVERGED' to get the final critical core.
       `graph_fn`: OPTIONAL override for how the stable graph is built (default: unchanged,
       `stable_graph` i.e. the monomial/Laurent method) -- passing `stable_graph_moebius` (wrapped
       to drop the verbose kwarg) reuses this ENTIRE function, unmodified in its default-argument
       behavior, for the Moebius mechanisms (D4T Sec.6/triage-completion, no existing call site
       changes semantics: `graph_fn=None` reproduces the exact prior code path)."""
    t0 = time.time()
    g = (graph_fn or (lambda k: stable_graph(k, verbose=False)))(mech_key)
    rays, pairs, bases = g["rays"], g["pairs"], g["bases"]
    print(f"[stable_core:{mech_key}] stable graph: {len(rays)} rays, {len(pairs)} pairs, "
          f"{len(bases)} bases")
    done = None if fresh else cache_load(f"d4flex_{mech_key}_done")
    if done is not None:
        core_idx = done
        print(f"[stable_core:{mech_key}] using CACHED converged core: {len(core_idx)} rays.")
    else:
        resume = None if fresh else cache_load(f"d4flex_{mech_key}_partial")
        if resume: print(f"[stable_core:{mech_key}] resuming from cached partial core: {len(resume)} rays")
        res = symbolic_greedy_core(len(rays), pairs, bases, trials=trials, seed0=seed0, verbose=True,
                                    wall_budget=wall_budget, resume=resume)
        core_idx = res["core"]
        if not res["converged"]:
            cache_save(f"d4flex_{mech_key}_partial", core_idx)
            print(f"[stable_core:{mech_key}] NOT YET CONVERGED ({len(core_idx)} rays remain) -- "
                  f"partial state cached; re-run `core_{mech_key}` again to continue peeling.")
            return dict(mech=mech_key, converged=False, partial_size=len(core_idx))
        print(f"[stable_core:{mech_key}] PEEL CONVERGED (full pass with no timeout-skips).")
        cache_save(f"d4flex_{mech_key}_done", core_idx)
    core_syms = [rays[i] for i in core_idx]
    n_x = sum(1 for r in core_syms if any(c in ("X", "-X") for c in r))
    idxmap = {old: new for new, old in enumerate(core_idx)}
    core_pairs = [(idxmap[i], idxmap[j]) for i, j in pairs if i in idxmap and j in idxmap]
    core_bases = [tuple(idxmap[x] for x in b) for b in bases if all(x in idxmap for x in b)]
    print(f"[stable_core:{mech_key}] SYMBOLIC critical core: {len(core_syms)} rays ({n_x} use X), "
          f"{len(core_pairs)} pairs, {len(core_bases)} bases  ({time.time()-t0:.2f}s)")
    print(f"[stable_core:{mech_key}] core ray symbols: {core_syms}")
    result = dict(mech=mech_key, core_syms=core_syms, core_pairs=core_pairs, core_bases=core_bases)
    if ring_point is not None:
        B, C, dotfn, sym2ring = ring_point if len(ring_point) == 4 else (
            ring_point[0], ring_point[1], ring_point[2],
            {"0": ZERO, "1": (1, 0), "-1": (-1, 0), "X": (0, 1), "-X": (0, -1)})
        core_ring = [tuple(sym2ring[c] for c in r) for r in core_syms]
        pairs_r, bases_r, _ = build_structure_d(core_ring, dotfn, B, C, dsize)
        match = (sorted(pairs_r) == sorted(core_pairs) and
                 sorted(tuple(sorted(b)) for b in bases_r) == sorted(tuple(sorted(b)) for b in core_bases))
        print(f"[stable_core:{mech_key}] non-degeneration @ concrete ring point (B={B},C={C}): "
              f"induced pairs={len(pairs_r)} (expect {len(core_pairs)}), "
              f"bases={len(bases_r)} (expect {len(core_bases)})  EXACT MATCH={match}")
        (col,) = ks_colorable_generic(len(core_ring), pairs_r, [list(b) for b in bases_r])
        print(f"[stable_core:{mech_key}] concrete-point core KS-uncolorable (independent check): "
              f"{not col}")
        result["match"] = match
        result["core_ring"] = core_ring
        if match:
            print(f"[stable_core:{mech_key}] *** CANDIDATE FAMILY: the symbolic critical core is "
                  f"realized EXACTLY (no accidental extra/missing structure) at a generic concrete "
                  f"point, and is KS-uncolorable via ONLY mechanism-stable edges => this core "
                  f"realizes the SAME KS-uncolorable graph for EVERY point of the {mech_key} "
                  f"variety => flex >= 1 by the forward lemma (MECHANISM_MODULI.md Task 2, "
                  f"generalized), PENDING the exact-flex mod-p bound below. ***")
        primes = find_primes_ring(B, C, count=2, below=200003)
        if dotfn is herm_dot:
            cert = exact_flex_hermitian_quadratic(core_ring, B, C, primes)
        else:
            cert = exact_flex_real_quadratic(core_ring, B, C, primes)
        print(f"[stable_core:{mech_key}] EXACT FLEX at this concrete point: n={cert['n']}, "
              f"E={cert['E']}, trivial-expected={cert['triv_expected']}, "
              f"mod-p bound: 0 <= flex <= {cert['bound']}")
        result["cert"] = cert
    return result


# ==================================================================================================
# STAGE 6 (D4T triage completion): MOEBIUS stability for the 4 off-center circles M12a/M12b/M13a/
# M13b. On a GENERAL circle aN*|x|^2 + aRx*Re(x) + a1 = 0 (aN,aRx,a1 the exact integer coefficients
# read off `alphabet_d4.derive4()`, up to overall scalar -- checked against derive4()'s own raw
# witnesses below, not just re-typed from prose), writing N=x*x*, p=(x+x*)/2, the relation
#     aN*x*x* + aRx*(x+x*)/2 + a1 = 0
# is LINEAR in x* (x, the "raw" symbol, appears only to the first power), so it solves EXACTLY for
# x* as a MOEBIUS (linear-fractional) function of x -- clearing the /2:
#     2*aN*x*x* + aRx*x* + (aRx*x + 2*a1) = 0   =>   x* = -(aRx*x + 2*a1) / (2*aN*x + aRx)
# i.e.  x* = (n1*X + n0) / (d1*X + d0)   with   n1=-aRx, n0=-2*a1, d1=2*aN, d0=aRx.
# VERIFIED two independent ways (both done before writing this code, not after):
#   (a) sympy: aN*X*xstar(X) + aRx*(X+xstar(X))/2 + a1  simplifies to EXACTLY 0 for every one of
#       M12a/M12b/M13a/M13b (residual printed 0, no floats, `sp.simplify(sp.together(...))`).
#   (b) numerically at concrete Gaussian-integer points on each circle (x=(1+i)/2 for M12a etc.):
#       x* computed via the Moebius formula matches the literal complex conjugate exactly.
# This is the genuine Moebius generalization of the pure-monomial `circle_mech`/`line_mech` case
# (M8/M8'/M9 have a1=0,aRx=0 -- x*=R^2/X, no linear term; M10/M11a/M11b have aN=0 -- x*=2c-X,
# already linear) -- MECHS above; a1!=0 AND aRx!=0 AND aN!=0 simultaneously is exactly the NEW
# off-center-circle case that needed this extension.
# ==================================================================================================
MOEBIUS_MECHS = {
    "M12a": dict(name="M12a: |x|^2=Re(x) (center 1/2, R^2=1/4)",       aN=Fr(1), aRx=Fr(-1), a1=Fr(0)),
    "M12b": dict(name="M12b: |x|^2=-Re(x) (center -1/2, R^2=1/4)",     aN=Fr(1), aRx=Fr(1),  a1=Fr(0)),
    "M13a": dict(name="M13a: |x|^2-2Re(x)-1=0 (center 1, R^2=2)",      aN=Fr(1), aRx=Fr(-2), a1=Fr(-1)),
    "M13b": dict(name="M13b: |x|^2+2Re(x)-1=0 (center -1, R^2=2)",     aN=Fr(1), aRx=Fr(2),  a1=Fr(-1)),
}
for _mk, _m in MOEBIUS_MECHS.items():
    _aN, _aRx, _a1 = _m["aN"], _m["aRx"], _m["a1"]
    _m["n1"], _m["n0"], _m["d1"], _m["d0"] = -_aRx, -2 * _a1, 2 * _aN, _aRx

def _rf_D(mech):
    """D(X) = d1*X + d0, the SHARED Moebius denominator for this mechanism, as a plain
       (non-negative-exponent) Laurent dict -- reusing l1/lX unmodified."""
    return ladd(l1(mech["d0"]), lX(mech["d1"]))

def conj_symbol_moebius(sym, mech):
    """conj(u_c) as a rational-function pair (poly_dict, denom_power), denom_power in {0,1}
       (0 = no denominator; 1 = divided by D(X)). '0'/'1'/'-1' are UNCONDITIONAL literals (same
       argument as `conj_symbol`'s Peres-24 anchor -- untouched by which mechanism is in play)."""
    if sym == "0": return (lz(), 0)
    if sym == "1": return (l1(1), 0)
    if sym == "-1": return (l1(-1), 0)
    sgn = 1 if sym == "X" else -1
    poly = ladd(l1(sgn * mech["n0"]), lX(sgn * mech["n1"]))
    return (poly, 1)

SYM_RF = {"0": (lz(), 0), "1": (l1(1), 0), "-1": (l1(-1), 0), "X": (lX(1), 0), "-X": (lX(-1), 0)}

def rf_mul(rf1, rf2):
    (p1, f1), (p2, f2) = rf1, rf2
    return (lmul(p1, p2), f1 + f2)

def rf_add(rf1, rf2, mech):
    """Add two rational-function terms with (possibly) different denominator powers of the SAME
       D(X) by clearing to the higher power (multiplying the lower term's numerator by D(X) as
       many times as needed) -- exact Fraction arithmetic throughout, no floats. In every term
       that arises here at most ONE factor is ever `conj(u_c)` (denom power <=1 per raw symbol),
       so summing 4 coordinate-terms never needs more than denom power 1 overall."""
    (p1, f1), (p2, f2) = rf1, rf2
    if f1 == f2:
        return (ladd(p1, p2), f1)
    lo, hi = (rf1, rf2) if f1 < f2 else (rf2, rf1)
    lo_p, lo_f = lo
    hi_p, hi_f = hi
    D = _rf_D(mech)
    scaled = lo_p
    for _ in range(hi_f - lo_f):
        scaled = lmul(scaled, D)
    return (ladd(scaled, hi_p), hi_f)

def rf_is_zero(rf):
    """The rational function P(X)/D(X)^k is identically zero on the WHOLE circle iff its
       numerator polynomial is identically zero (D(X) is a nonzero polynomial -- d1,d0 not both
       0 -- so it never vanishes identically, hence never spuriously kills a nonzero numerator);
       this is the SAME identity-theorem argument as the monomial case, just with a numerator
       polynomial cleared of denominators instead of a bare Laurent dict."""
    return lzero(rf[0])

def stable_dot_moebius(u, v, mech):
    acc = (lz(), 0)
    for uc, vc in zip(u, v):
        term = rf_mul(conj_symbol_moebius(uc, mech), SYM_RF[vc])
        acc = rf_add(acc, term, mech)
    return acc

def stable_graph_moebius(mech_key, dsize=4, verbose=True):
    """Mirrors `stable_graph` exactly (same mechanism-independent ray set from
       `generic_symbolic_rays`, same dsize-clique enumeration) but tests edges via the Moebius
       rational-function stability test instead of the monomial Laurent-dict test."""
    t0 = time.time()
    mech = MOEBIUS_MECHS[mech_key]
    rays = generic_symbolic_rays(dsize)
    V = len(rays)
    pairs = [(i, j) for i, j in combinations(range(V), 2)
             if rf_is_zero(stable_dot_moebius(rays[i], rays[j], mech))]
    adj = [set() for _ in range(V)]
    for i, j in pairs: adj[i].add(j); adj[j].add(i)
    bases = []
    def extend(cands, cur):
        if len(cur) == dsize: bases.append(tuple(cur)); return
        if len(cur) + len(cands) < dsize: return
        cl = sorted(cands)
        for idx, v in enumerate(cl):
            rest = set(cl[idx + 1:]) & adj[v]
            extend(rest, cur + [v])
    for start in range(V):
        extend(set(x for x in adj[start] if x > start), [start])
    bp = basis_participating(V, bases)
    n_x_in_bp = sum(1 for i in bp if any(rays[i][c] in ("X", "-X") for c in range(dsize)))
    if verbose:
        print(f"[stable_graph_moebius:{mech_key}] {mech['name']}: rays={V} pairs={len(pairs)} "
              f"bases={len(bases)} basis-participating={len(bp)} X-using rays IN a basis: "
              f"{n_x_in_bp} ({time.time()-t0:.2f}s)")
    return dict(mech=mech_key, rays=rays, pairs=pairs, bases=bases, bp=bp, adj=adj)

def _moebius_sympy_crosscheck(mech_key, n_pairs=40, seed=0):
    """Independent SYMPY cross-check (per the brief: 'sympy handles this exactly') of the exact
       Fraction/Laurent-dict Moebius machinery above, on a random sample of ray pairs: builds the
       SAME Hermitian dot as a genuine sympy rational function of X (via the Moebius xstar(X)
       directly, no Laurent-dict trick at all) and compares cancellation-to-zero against
       `rf_is_zero`. Uses sympy's exact `cancel`/`fraction` (no floats)."""
    import sympy as sp
    import random
    Xs = sp.symbols("Xs")
    mech = MOEBIUS_MECHS[mech_key]
    xstar = (mech["n1"] * Xs + mech["n0"]) / (mech["d1"] * Xs + mech["d0"])
    sym2sp = {"0": sp.Integer(0), "1": sp.Integer(1), "-1": sp.Integer(-1), "X": Xs, "-X": -Xs}
    conjsp = {"0": sp.Integer(0), "1": sp.Integer(1), "-1": sp.Integer(-1), "X": xstar, "-X": -xstar}
    rays = generic_symbolic_rays(4)
    V = len(rays)
    rng = random.Random(seed)
    idxs = [(i, j) for i in range(V) for j in range(i + 1, V)]
    sample = rng.sample(idxs, min(n_pairs, len(idxs)))
    mismatches = 0
    for i, j in sample:
        u, v = rays[i], rays[j]
        expr = sum((conjsp[uc] * sym2sp[vc] for uc, vc in zip(u, v)), sp.Integer(0))
        num, den = sp.fraction(sp.cancel(sp.together(expr)))
        sp_zero = sp.expand(num) == 0
        my_zero = rf_is_zero(stable_dot_moebius(u, v, mech))
        if sp_zero != my_zero:
            mismatches += 1
            print(f"  MISMATCH pair ({i},{j}): sympy_zero={sp_zero} rf_zero={my_zero}")
    print(f"[moebius_sympy_crosscheck:{mech_key}] {len(sample)} random pairs: "
          f"{len(sample)-mismatches}/{len(sample)} agree with sympy's exact cancel/fraction "
          f"computation (mismatches={mismatches}).")
    return mismatches == 0

def stable_graph_moebius_nv(mech_key):
    return stable_graph_moebius(mech_key, verbose=False)


# ==================================================================================================
# STAGE 7 (D4T triage completion): M10 EXACT resolution -- push past the earlier `0<=flex<=2`
# mod-p bound with (a) more/independent primes AND (b) a genuinely EXACT (non-modular) rank, via
# sympy, at a CONCRETE Gaussian-integer point (x=2i, x=3i -- both already exact rational multiples
# of i, so the ring collapses to literal Q(i): no algebraic extension is needed at all, unlike the
# earlier x=i*sqrt(2) point (B=0,C=-2, genuinely irrational ring) -- this sidesteps the mod-p
# machinery's only real caveat (an unlucky prime could in principle under-report the true rank) by
# computing the ACTUAL rank of the ACTUAL integer Jacobian directly, exactly the same "EXACT over
# Q" style `exact_rigidity.py` already uses for real rays, generalized here to genuinely complex
# (Re,Im) ray coordinates via the standard Hermitian tangent equation
#   conj(w_i)*v_j + conj(v_i)*w_j = 0   (per edge, v_i/v_j COMPLEX constants, w_i/w_j the unknowns)
# -- derived by hand (real/imag split) and cross-checked to match `exact_flex_hermitian_quadratic`'s
# own row pattern (same coefficients, minus that function's abstract sqrt(D)/doubling bookkeeping,
# which collapses to the identity once (a,b) already means literal (Re,Im) -- true here because the
# ring generator is exactly i, Re(i)=0, Im(i)=1).
# ==================================================================================================
def _exact_rank_qq(rows, ncols):
    """Exact rank over Q of an integer/rational matrix given as a list of row-lists, via sympy's
       DomainMatrix/QQ (fraction-free field algorithm) -- MUCH faster than plain sympy
       Matrix.rank() (which uses generic symbolic simplification per entry: verified directly,
       >40s/timeout vs ~16s for a 955x712 case) for matrices of this size, and still EXACT (no
       floats, no mod-p reduction: this is the TRUE rank over Q, not a bound)."""
    import sympy as sp
    from sympy.polys.matrices import DomainMatrix
    dm = DomainMatrix.from_list_sympy(len(rows), ncols, rows).convert_to(sp.QQ)
    return dm.rank()

def exact_flex_hermitian_at_point(rays_ri):
    """rays_ri: list of tuples of (Re, Im) Fraction/int pairs -- ray coordinates at ONE CONCRETE
       numeric point (ring generator t=i, i.e. B=0,C=-1: (a,b) IS ALREADY (Re,Im), no conversion).
       Returns EXACT (sympy, rank over Q, no mod-p reduction anywhere) flex certificate, mirroring
       exact_rigidity.exact_flex's own 'EXACT over Q' style, generalized to complex v_i."""
    import sympy as sp
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
    # edge rows: conj(w_i)*v_j + conj(v_i)*w_j = 0, w_i=x_i+i*y_i unknown, v real/imag KNOWN
    #   Re part: x_i*Re_j + y_i*Im_j + Re_i*x_j + Im_i*y_j = 0
    #   Im part: x_i*Im_j - y_i*Re_j - Im_i*x_j + Re_i*y_j = 0
    for i, j in E:
        re = [sp.Integer(0)] * n; im = [sp.Integer(0)] * n
        for c in range(d):
            re[coord(i, c, True)] += Re[j][c]; re[coord(i, c, False)] += Im[j][c]
            re[coord(j, c, True)] += Re[i][c]; re[coord(j, c, False)] += Im[i][c]
            im[coord(i, c, True)] += Im[j][c]; im[coord(i, c, False)] -= Re[j][c]
            im[coord(j, c, True)] -= Im[i][c]; im[coord(j, c, False)] += Re[i][c]
        rows.append(re); rows.append(im)
    # norm rows: Re(conj(v_i)*w_i) = Re_i.x_i + Im_i.y_i = 0
    for i in range(V):
        r = [sp.Integer(0)] * n
        for c in range(d): r[coord(i, c, True)] = Re[i][c]; r[coord(i, c, False)] = Im[i][c]
        rows.append(r)
    rankJ = _exact_rank_qq(rows, n)
    ker = n - rankJ
    # trivial: per-vertex phase w_i = i*v_i (x=-Im_i, y=Re_i, restricted to vertex i)
    triv = []
    for i in range(V):
        t = [sp.Integer(0)] * n
        for c in range(d): t[coord(i, c, True)] = -Im[i][c]; t[coord(i, c, False)] = Re[i][c]
        triv.append(t)
    # u(d): iE_aa (diag phase on coord a, all i), i(E_ab+E_ba), (E_ab-E_ba) -- same construction
    # as exact_rigidity.exact_flex, generalized from real v_i to complex (Re,Im) v_i.
    for a in range(d):
        t = [sp.Integer(0)] * n
        for i in range(V): t[coord(i, a, True)] = -Im[i][a]; t[coord(i, a, False)] = Re[i][a]
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
    from sympy.polys.matrices import DomainMatrix
    dmJ = DomainMatrix.from_list_sympy(len(rows), n, rows).convert_to(sp.QQ)
    dmT = DomainMatrix.from_list_sympy(len(triv), n, triv).convert_to(sp.QQ)
    resid = (dmJ * dmT.transpose()).to_Matrix()
    ok0 = all(x == 0 for x in resid)
    flex = ker - rankT
    return dict(rankJ=rankJ, ker=ker, rankT=rankT, resid_ok=ok0, flex=flex, n=n, E=len(E), V=V, d=d)


SECTIONS_S12 = {"triage": mechanism_triage, "peres24": peres24_check}

if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "triage"
    if which in SECTIONS_S12:
        SECTIONS_S12[which]()
    elif which == "stable_all":
        for k in MECHS:
            stable_graph(k)
    elif which == "stable_moebius_all":
        for k in MOEBIUS_MECHS:
            stable_graph_moebius(k)
    elif which.startswith("stable_") and which[len("stable_"):] in MOEBIUS_MECHS:
        stable_graph_moebius(which[len("stable_"):])
    elif which.startswith("stable_"):
        stable_graph(which[len("stable_"):])
    elif which.startswith("flex_"):
        concrete_flex(which[len("flex_"):])
    elif which.startswith("moebius_crosscheck_"):
        _moebius_sympy_crosscheck(which[len("moebius_crosscheck_"):])
    elif which.startswith("core_") and which[len("core_"):] in MOEBIUS_MECHS:
        # D4T triage completion: off-center circles. Each mechanism's own Gaussian-integer point
        # ON its circle (derived by the rational-parametrization-through-a-known-rational-point
        # method, D4_FLEX_HUNT.md Sec.6) -- ring Z[i], B=0,C=-1 throughout.
        mech_key = which[len("core_"):]
        rp_moebius = {
            "M12a": (0, -1, herm_dot, {"0": ZERO, "1": (2, 0), "-1": (-2, 0), "X": (1, 2), "-X": (-1, -2)}),
            "M12b": (0, -1, herm_dot, {"0": ZERO, "1": (2, 0), "-1": (-2, 0), "X": (-1, -2), "-X": (1, 2)}),
            "M13a": (0, -1, herm_dot, {"0": ZERO, "1": (1, 0), "-1": (-1, 0), "X": (2, 1), "-X": (-2, -1)}),
            "M13b": (0, -1, herm_dot, {"0": ZERO, "1": (1, 0), "-1": (-1, 0), "X": (-2, -1), "-X": (2, 1)}),
        }[mech_key]
        stable_core_and_test(mech_key, ring_point=rp_moebius, graph_fn=stable_graph_moebius_nv)
    elif which.startswith("core_"):
        mech_key = which[len("core_"):]
        # M9_generic: x=(3+4i)/5, GENERIC (non-root-of-unity, Kronecker-forced) point on the unit
        # circle, realized with ALL-INTEGER ray coordinates by uniformly scaling the whole raw
        # alphabet by 5 (a global projective rescale -- provably identical rays/pairs/bases to the
        # true x=(3+4i)/5 point, verified: 272/2704/460, EXACT match to the abstract stable graph).
        sym2ring_M9g = {"0": ZERO, "1": (5, 0), "-1": (-5, 0), "X": (3, 4), "-X": (-3, -4)}
        # M10: x=2i (Z[i], B=0,C=-1) -- CHANGED from the earlier x=i*sqrt2 point (B=0,C=-2): D4T
        # triage completion found i*sqrt2 is a SPECIAL point for the FULL 208-ray basis-
        # participating M10 subgraph (extra accidental pairs/bases there -- 2152/272 vs the
        # abstract 2032/236), even though it happened to still match for one particular SMALL
        # critical core; x=2i (and x=3i, x=5i as independent cross-checks) verified EXACT MATCH
        # against the abstract graph at the FULL 208-ray level, not just a small core -- see
        # D4_FLEX_HUNT.md Sec.6 "Triage completion".
        sym2ring_M10g = {"0": ZERO, "1": (1, 0), "-1": (-1, 0), "X": (0, 2), "-X": (0, -2)}
        rp = {
            "M9": (0, -1, herm_dot, sym2ring_M9g),
            "M10": (0, -1, herm_dot, sym2ring_M10g), "M11a": (2, -2, herm_dot),
            "M11b": (-2, -2, herm_dot), "M8": (0, 3, bil_dot), "M8p": (0, Fr(1, 3), bil_dot),
        }.get(mech_key)
        # M10: seed0=1 reproduces the TIGHT 21-ray/2-X-ray critical core (D4_FLEX_HUNT.md Sec.6)
        # -- a fast-converging, densely-enough-connected core found to have EXACT flex=2 (not just
        # a mod-p bound); seed0=0's core (26 rays, 8 X-rays) is a valid but looser witness (exact
        # flex=8 there -- see the writeup's discussion of core-dependence of the raw mod-p/exact
        # bound for M10, unlike M9's dense 89-ray core which has no such ambiguity).
        seed0 = 1 if mech_key == "M10" else 0
        stable_core_and_test(mech_key, ring_point=rp, seed0=seed0)
    else:
        print(f"unknown section {which!r}")
        sys.exit(1)
