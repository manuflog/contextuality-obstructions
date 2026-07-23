#!/usr/bin/env python3
"""
ks_flex_census.py — broadening the KS-set flexibility census (session: even-d / new-island hunt).

GOAL (see KS_CENSUS.md for the full report): the program's open problem is whether ANY KS set
other than Peres-33/Penrose-33 is flexible. This file builds NEW KS sets from verifiable
GENERATING RULES (not memorized ray lists), self-validates their KS-uncolorability, and computes
flex EXACTLY over the relevant number field, reusing this repo's exact-arithmetic machinery:
  - sic_zoo.py:        rank_mod_p (GENERIC mod-p rank of a matrix of (a,b)=a+b*s pairs — works for
                        ANY quadratic ring, not just Z[sqrt2], since it only evaluates a+b*s mod p)
                        and ks_colorable (exhaustive KS-coloring backtracking with propagation).
  - torsion_layer.py:   parity_torsion (GF(2) forced-transversal-parity computation, generic).
  - exact_rigidity.py:  exact_flex (rational extended-Jacobian rigidity certificate for INTEGER
                        ray sets — reused verbatim for the CK-31 integer island).
  - cabello33.py:       Eisenstein/Q(sqrt3) exact_flex_z3 pattern — the Heegner-7 exact-flex
                        function below is the direct generalization of that pattern to a
                        different quadratic ring (parametrized by the trace B of the ring
                        generator and the field discriminant D).

TARGETS (arXiv:2603.16988, Kernaghan, "The Algebraic Landscape of KS Sets in Dimension Three" —
fetched in full HTML and read; ray lists are NOT printed in the paper, so every island here is
RECONSTRUCTED from the paper's generating alphabet + completion rule, Definition 14, and
cross-checked against every printed invariant (ray/pair/triad counts, basis-participating
counts) before its flex is trusted):
  1. Heegner-7 island:  ring Z[alpha], alpha=(1+sqrt(-7))/2 (alpha^2 = alpha - 2), alphabet
                        {0,+-1,+-alpha,+-alpha-bar}. Paper: 145 rays / 42 triads / 69
                        basis-participating / pool-min 43 / Jacobian claims RIGID (dim ker J =
                        51 = n+8).
  2. Golden island:     ring Z[phi], phi=(1+sqrt5)/2 (phi^2=phi+1), alphabet {0,+-1,+-phi},
                        REAL cross-product completion (raw 49 rays colorable -> completed 205
                        rays uncolorable). Paper: pool-min 52 / RIGID (dim ker J = 60 = n+8).
  3. CK-31 (integer) island: ring Z, alphabet {0,+-1,+-2}, cross-product completion 49->109
                        rays. Paper: pool-min 31 (Conway-Kochen), rigid (Trandafir-Cabello,
                        global rigidity).
  4. Eisenstein island: ring Z[omega]. Kernaghan identifies this island's min-33 set WITH
                        Cabello's "simplest KS set" (arXiv:2508.07335) already built and
                        exactly certified RIGID in cabello33.py (flex=0 EXACT over Q(sqrt3)).
                        Re-run here as a cross-check, not re-derived.
  5. d=4 sqrt2 island (NEW, exploratory, NOT in Kernaghan's paper): the Peres alphabet
                        {0,+-1,+-sqrt2} lifted to d=4 with no other change of rule. Untested by
                        this program before now; see verdict below.
  6. even-d:            honest report of what was (not) found beyond Peres-24/CEG-18/K20 and
                        the new d=4 sqrt2 island.

Every target is runnable standalone (CLI arg) because each bash call in the harness has a ~45s
budget and Heegner-7/Golden/d4sqrt2 pool construction + greedy minimization can take longer:
    python3 ks_flex_census.py pool_heegner7      # build+verify the 145-ray pool only
    python3 ks_flex_census.py core_heegner7      # basis-participating filter + greedy core
    python3 ks_flex_census.py flex_heegner7      # exact flex of the cached core
    python3 ks_flex_census.py pool_golden / core_golden / flex_golden        (same pattern)
    python3 ks_flex_census.py pool_d4sqrt2 / core_d4sqrt2 / flex_d4sqrt2     (same pattern)
    python3 ks_flex_census.py ck31
    python3 ks_flex_census.py eisenstein
    python3 ks_flex_census.py evend
Intermediate ray sets are cached to JSON (*.cache.json next to this file) so later stages don't
repeat expensive pool construction; re-run with the cache files deleted to reproduce from
scratch. NOTE (found mid-session, documented in-line below): sic_zoo.ks_colorable's coloring
search hard-codes basis size 3, so it is silently wrong for the d=4 target; that target uses a
new size-generic checker (ks_colorable_generic) instead, cross-checked against
torsion_layer.count_ks_colorings for agreement.
"""
import os, sys, json, time, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from itertools import combinations, product

from sic_zoo import rank_mod_p, ks_colorable
from torsion_layer import parity_torsion, count_ks_colorings

# IMPORTANT CORRECTNESS NOTE (found while building the d=4 target below): sic_zoo.ks_colorable's
# unit-propagation logic hard-codes basis size 3 (`n0 == 3` "dead", `n0 == 2` "force" — see its
# source). That is exactly right for every d=3 target in this file (Heegner-7/Golden/CK-31/
# Eisenstein all use SIZE-3 triads, so those results are unaffected), but it is SILENTLY WRONG
# for d=4 (size-4) bases: it flags "3 zeros among 4" as dead (should force the 4th to 1) and
# "2 zeros among 4" as forcing (should not). All size>=4-basis code below therefore uses
# torsion_layer.count_ks_colorings instead, which is genuinely basis-size-generic (its dead/
# propagation test is `unassigned[bi]==0 and ones[bi]!=1`, with no hard-coded size) — reused
# UNMODIFIED, not patched. Cross-checked against sic_zoo.ks_colorable on every d=3 target below
# (KS_CENSUS.md records the cross-check) to make sure the two agree where both are valid.
def ks_colorable_generic_slow(nrays, pairs, bases):
    """Size-generic KS-colorability via torsion_layer.count_ks_colorings (reused UNMODIFIED).
       Correct for ANY basis size, but counts ALL colorings (can be slow when many exist) —
       used only as a cross-check on small instances, not the main search."""
    cnt, example = count_ks_colorings(nrays, pairs, bases, use_pairs=True)
    return (cnt > 0), cnt

def ks_colorable_generic(nrays, pairs, bases):
    """Size-generic, EXISTENCE-ONLY (early-exit), unit-propagating KS-coloring search — the
       fully-general (basis size m, not hard-coded 3) version of sic_zoo.ks_colorable's
       propagation algorithm, needed because that function's `n0==3`/`n0==2` dead/force tests
       are specific to size-3 triads (see the module-level note above). Returns (colorable, ).
    """
    orth = [[] for _ in range(nrays)]
    for i, j in pairs: orth[i].append(j); orth[j].append(i)
    binc = [[] for _ in range(nrays)]
    for bi, b in enumerate(bases):
        for r in b: binc[r].append(bi)
    color = [-1] * nrays
    ones = [0] * len(bases); unassigned = [len(b) for b in bases]
    def assign(i0, v0):
        stack = [(i0, v0)]; trail = []
        while stack:
            i, v = stack.pop()
            if color[i] != -1:
                if color[i] != v:
                    for j in trail: color[j] = -1
                    return None
                continue
            if v == 1 and any(color[k] == 1 for k in orth[i]):
                for j in trail: color[j] = -1
                return None
            color[i] = v; trail.append(i)
            if v == 1:
                for k in orth[i]:
                    if color[k] == -1: stack.append((k, 0))
            for bi in binc[i]:
                unassigned[bi] -= 1
                if v == 1: ones[bi] += 1
                if ones[bi] > 1 or (unassigned[bi] == 0 and ones[bi] != 1):
                    for j in trail: color[j] = -1
                    return None
                if unassigned[bi] > 0 and ones[bi] == 1:
                    for r in bases[bi]:
                        if color[r] == -1: stack.append((r, 0))
                elif unassigned[bi] == 1 and ones[bi] == 0:
                    for r in bases[bi]:
                        if color[r] == -1: stack.append((r, 1))
        return trail
    def dfs():
        i = next((k for k in range(nrays) if color[k] == -1), None)
        if i is None: return True
        for val in (1, 0):
            saved_unassigned = unassigned[:]; saved_ones = ones[:]
            trail = assign(i, val)
            if trail is not None:
                if dfs(): return True
            unassigned[:] = saved_unassigned; ones[:] = saved_ones
            for j in (trail or []): color[j] = -1
        return False
    return (dfs(),)

HERE = os.path.dirname(os.path.abspath(__file__))
def cache_path(name): return os.path.join(HERE, f"{name}.cache.json")
def cache_save(name, obj):
    with open(cache_path(name), "w") as f: json.dump(obj, f)
def cache_load(name):
    p = cache_path(name)
    if not os.path.exists(p): return None
    with open(p) as f: return json.load(f)

# ============================================================================================
# GENERIC quadratic-ring pair arithmetic: element (a,b) means  a + b*t,  where the ring
# generator t satisfies   t^2 = B*t + C   (B,C integers). This covers every ring used below:
#   Heegner-7:  alpha=(1+sqrt(-7))/2,  alpha^2 = alpha - 2         => B=1,  C=-2
#   Golden:     phi=(1+sqrt5)/2,       phi^2   = phi + 1           => B=1,  C=1
#   (Eisenstein, sqrt2, etc. are already covered by sic_zoo.py/cabello33.py unmodified.)
# The Hermitian conjugate (trace B, i.e. t + conj(t) = B) is  conj(a,b) = (a+b*B, -b); this is
# the correct complex conjugate exactly when t is NON-real (Heegner-7). For REAL rings (Golden)
# conjugation must be the IDENTITY (real numbers are self-conjugate in C) — using two separate
# dot-product functions below (herm_dot vs bil_dot) keeps this explicit and avoids the bug.
# ============================================================================================
ZERO = (0, 0)
def qadd(u, v): return (u[0] + v[0], u[1] + v[1])
def qneg(u): return (-u[0], -u[1])
def qmul(u, v, B, C):
    a, b = u; c, d = v
    return (a * c + C * b * d, a * d + b * c + B * b * d)
def qconj(u, B):
    a, b = u; return (a + b * B, -b)
def qz(u): return u == ZERO
def qfloat(u, tval): return u[0] + u[1] * tval

def herm_dot(x, y, B, C):
    """Hermitian dot for a NON-real ring generator: sum conj(x_c) * y_c."""
    s = ZERO
    for a, b in zip(x, y): s = qadd(s, qmul(qconj(a, B), b, B, C))
    return s

def bil_dot(x, y, B, C):
    """Bilinear (real) dot for a REAL ring generator: sum x_c * y_c, no conjugation."""
    s = ZERO
    for a, b in zip(x, y): s = qadd(s, qmul(a, b, B, C))
    return s

def qminor(u, v, i, j, B, C):
    return qadd(qmul(u[i], v[j], B, C), qneg(qmul(u[j], v[i], B, C)))

def proportional(u, v, B, C):
    """u, v same projective ray (over the ring, no division needed): all 2x2 minors vanish."""
    for i, j in combinations(range(len(u)), 2):
        if not qz(qminor(u, v, i, j, B, C)): return False
    return True

def collect_rays(vectors, B, C):
    rays = []
    for v in vectors:
        if not any(proportional(v, r, B, C) for r in rays):
            rays.append(v)
    return rays

def raw_vectors(alphabet, d=3):
    return [v for v in product(alphabet, repeat=d) if any(c != ZERO for c in v)]

# ---------------------------------------------------------------- generic mod-p prime search
def _is_prime(m):
    if m < 2: return False
    if m % 2 == 0: return m == 2
    i = 3
    while i * i <= m:
        if m % i == 0: return False
        i += 2
    return True

def tonelli_shanks(n, p):
    """Modular square root of n mod prime p (n a QR mod p), or None."""
    n %= p
    if n == 0: return 0
    if pow(n, (p - 1) // 2, p) != 1: return None
    if p % 4 == 3: return pow(n, (p + 1) // 4, p)
    q, s = p - 1, 0
    while q % 2 == 0: q //= 2; s += 1
    z = 2
    while pow(z, (p - 1) // 2, p) != p - 1: z += 1
    m, c, t, r = s, pow(z, q, p), pow(n, q, p), pow(n, (q + 1) // 2, p)
    while t != 1:
        i, t2 = 0, t
        while t2 != 1:
            t2 = (t2 * t2) % p; i += 1
        b = pow(c, 1 << (m - i - 1), p)
        m, c, t, r = i, (b * b) % p, (t * b * b) % p, (r * b) % p
    return r

def find_primes_ring(B, C, count=2, below=999983):
    """Primes p (splitting the ring, i.e. minimal poly x^2-Bx-C has a root s mod p), searched
       downward from `below`. Returns [(p, s), ...]."""
    out, p = [], below | 1
    while len(out) < count and p > 5:
        if _is_prime(p):
            D = (B * B + 4 * C) % p
            s0 = tonelli_shanks(D, p)
            if s0 is not None:
                inv2 = pow(2, p - 2, p)
                s = ((B + s0) * inv2) % p
                assert (s * s - B * s - C) % p == 0
                out.append((p, s))
        p -= 2
    return out

# ---------------------------------------------------------------- orthogonality / triad utils
def build_structure(rays, dotfn, B, C):
    V = len(rays)
    pairs = [(i, j) for i, j in combinations(range(V), 2) if qz(dotfn(rays[i], rays[j], B, C))]
    adj = [set() for _ in range(V)]
    for i, j in pairs: adj[i].add(j); adj[j].add(i)
    triads = []
    for i, j in pairs:
        if i > j: i, j = j, i
        for k in adj[i] & adj[j]:
            if k > j: triads.append((i, j, k))
    return pairs, triads, adj

def basis_participating(V, triads):
    used = set()
    for t in triads: used.update(t)
    return sorted(used)

def restrict(rays, keep_idx):
    remap = {old: new for new, old in enumerate(keep_idx)}
    return [rays[i] for i in keep_idx], remap

def uncolorable(rays, dotfn, B, C):
    pairs, triads, _ = build_structure(rays, dotfn, B, C)
    col, nodes = ks_colorable(len(rays), pairs, [list(t) for t in triads])
    return (not col), nodes, pairs, triads

# ---------------------------------------------------------------- generic dsize>=3 (any d)
def build_structure_d(rays, dotfn, B, C, dsize):
    """Like build_structure, but finds complete dsize-cliques (orthogonal BASES of ambient
       dimension dsize) via a pruned backtracking clique search, not the d=3 triangle shortcut.
       Needed for even-d (d=4) candidate islands."""
    V = len(rays)
    pairs = [(i, j) for i, j in combinations(range(V), 2) if qz(dotfn(rays[i], rays[j], B, C))]
    adj = [set() for _ in range(V)]
    for i, j in pairs: adj[i].add(j); adj[j].add(i)
    result = []
    def extend(cands, cur):
        if len(cur) == dsize: result.append(tuple(cur)); return
        if len(cur) + len(cands) < dsize: return
        cl = sorted(cands)
        for idx, v in enumerate(cl):
            rest = set(cl[idx + 1:]) & adj[v]
            extend(rest, cur + [v])
    for start in range(V):
        extend(set(x for x in adj[start] if x > start), [start])
    return pairs, result, adj

def uncolorable_d(rays, dotfn, B, C, dsize):
    pairs, bases, _ = build_structure_d(rays, dotfn, B, C, dsize)
    (col,) = ks_colorable_generic(len(rays), pairs, [list(b) for b in bases])
    return (not col), None, pairs, bases

def greedy_critical_core_d(rays, dotfn, B, C, dsize, trials=3, seed0=0, verbose=True):
    idxs = list(range(len(rays)))
    unc0, _, _, _ = uncolorable_d(rays, dotfn, B, C, dsize)
    assert unc0, "starting pool is not KS-uncolorable — cannot build a core"
    best = None
    for t in range(trials):
        order = idxs[:]
        random.Random(2000 + seed0 + t).shuffle(order)
        keep = set(idxs)
        for r in order:
            cand = keep - {r}
            sub = [rays[i] for i in sorted(cand)]
            u, _, _, _ = uncolorable_d(sub, dotfn, B, C, dsize)
            if u: keep = cand
        if best is None or len(keep) < len(best): best = keep
        if verbose: print(f"    trial {t}: core size {len(keep)}")
    core = sorted(best)
    for r in core:
        sub = [rays[i] for i in core if i != r]
        u, _, _, _ = uncolorable_d(sub, dotfn, B, C, dsize)
        assert not u, f"core not critical: removing ray {r} still uncolorable"
    return core

def greedy_critical_core(rays, dotfn, B, C, trials=3, seed0=0, verbose=True):
    """Peel rays greedily while KS-uncolorability survives (full rules); returns a CRITICAL
       (every single deletion restores colorability) KS-uncolorable index list into `rays`."""
    idxs = list(range(len(rays)))
    unc0, _, _, _ = uncolorable(rays, dotfn, B, C)
    assert unc0, "starting pool is not KS-uncolorable — cannot build a core"
    best = None
    for t in range(trials):
        order = idxs[:]
        random.Random(1000 + seed0 + t).shuffle(order)
        keep = set(idxs)
        for r in order:
            cand = keep - {r}
            sub = [rays[i] for i in sorted(cand)]
            u, _, _, _ = uncolorable(sub, dotfn, B, C)
            if u: keep = cand
        if best is None or len(keep) < len(best): best = keep
        if verbose: print(f"    trial {t}: core size {len(keep)}")
    core = sorted(best)
    # verify criticality
    for r in core:
        cand = [x for x in core if x != r]
        sub = [rays[i] for i in cand]
        u, _, _, _ = uncolorable(sub, dotfn, B, C)
        assert not u, f"core not critical: removing ray {r} still uncolorable"
    return core

# ============================================================================================
# EXACT flex, HERMITIAN (non-real) quadratic ring — direct generalization of
# cabello33.exact_flex_z3 (there: Eisenstein omega, B=-1, Dmag=3) to any ring with generator t,
# t^2 = B*t + C, trace(t) = B, conj(t) = B - t. Works by scaling every coordinate by 2 to land
# in Z[sqrt(Dmag)] (Dmag = |B^2+4C|): 2*(a+b*t) = (2a+bB) + b*sqrt(D). mod-p bound via TWO
# primes splitting Dmag (rank_mod_p is fully generic — reused UNMODIFIED from sic_zoo.py).
# ============================================================================================
def exact_flex_hermitian_quadratic(rays, B, C, primes):
    """Two-prime mod-p flex certificate over the quadratic ring t^2 = B t + C.
    KNOWN EDGE CASE (2026-07-22): for heavily merged/deduped ray sets at degenerate ring
    points (observed: the d=4 M9 pool's x=+-i collapse), the trivial-space accounting can
    exceed the constraint nullspace and the raw bound goes NEGATIVE -- such configurations
    violate this certificate's assumptions. A negative bound now raises ValueError instead
    of being returned; use an independent flex diagnostic there (see branch_m9str.py)."""
    V, d = len(rays), len(rays[0])
    Re = [[2 * rays[i][c][0] + B * rays[i][c][1] for c in range(d)] for i in range(V)]
    Imq = [[rays[i][c][1] for c in range(d)] for i in range(V)]
    E = [(i, j) for i, j in combinations(range(V), 2) if qz(herm_dot(rays[i], rays[j], B, C))]
    n = 2 * d * V
    def col(i, c, im): return 2 * d * i + 2 * c + (1 if im else 0)
    def P(a, b=0): return (a, b)
    rows = []
    for i, j in E:
        r = [(0, 0)] * n
        for c in range(d):
            r[col(i, c, 0)] = P(Re[j][c], 0); r[col(i, c, 1)] = P(0, Imq[j][c])
            r[col(j, c, 0)] = P(Re[i][c], 0); r[col(j, c, 1)] = P(0, Imq[i][c])
        rows.append(r)
        r = [(0, 0)] * n
        for c in range(d):
            r[col(i, c, 0)] = P(0, Imq[j][c]); r[col(i, c, 1)] = P(-Re[j][c], 0)
            r[col(j, c, 0)] = P(0, -Imq[i][c]); r[col(j, c, 1)] = P(Re[i][c], 0)
        rows.append(r)
    for i in range(V):
        r = [(0, 0)] * n
        for c in range(d): r[col(i, c, 0)] = P(Re[i][c], 0); r[col(i, c, 1)] = P(0, Imq[i][c])
        rows.append(r)
    triv = []
    for i in range(V):
        t = [(0, 0)] * n
        for c in range(d): t[col(i, c, 0)] = P(0, -Imq[i][c]); t[col(i, c, 1)] = P(Re[i][c], 0)
        triv.append(t)
    def put_real(t, i, c, u):
        p_, q_ = u; t[col(i, c, 0)] = P(2 * p_ + B * q_, 0); t[col(i, c, 1)] = P(0, q_)
    def put_imag(t, i, c, u):
        p_, q_ = u; t[col(i, c, 0)] = P(0, -q_); t[col(i, c, 1)] = P(2 * p_ + B * q_, 0)
    for a in range(d):
        t = [(0, 0)] * n
        for i in range(V): put_imag(t, i, a, rays[i][a])
        triv.append(t)
    for a in range(d):
        for b in range(a + 1, d):
            t = [(0, 0)] * n
            for i in range(V): put_imag(t, i, a, rays[i][b]); put_imag(t, i, b, rays[i][a])
            triv.append(t)
            t = [(0, 0)] * n
            for i in range(V):
                put_real(t, i, a, rays[i][b]); put_real(t, i, b, qneg(rays[i][a]))
            triv.append(t)
    best = None
    for p, s in primes:
        rJ, rT = rank_mod_p(rows, p, s), rank_mod_p(triv, p, s)
        b_ = (n - rJ) - rT
        best = b_ if best is None else min(best, b_)
    if best is not None and best < 0:
        raise ValueError(f"exact_flex_hermitian_quadratic: negative bound {best} -- internal "
                         "inconsistency; refusing to return it.")
    return dict(bound=best, E=len(E), n=n, V=V, d=d, triv_expected=V + d * d - 1)

# ============================================================================================
# EXACT flex, REAL quadratic ring — same extended-Jacobian FORMULATION as exact_rigidity.py /
# sic_zoo.build_extended_jacobian (real integer/Z[sqrt2] rays embedded in C^d for the tangent
# space), generalized to an arbitrary REAL ring Z[t], t^2=B*t+C (here Golden: B=1,C=1). NOTE:
# the Jacobian ROW ENTRIES below are just ray coordinates / sums thereof (never a ring PRODUCT
# of two unknowns — the map is linear), so this is ring-independent except for the edge test,
# which correctly uses bil_dot (bilinear, NO conjugation — the rays are real numbers).
# ============================================================================================
def build_extended_jacobian_real(rays, B, C):
    d, V = len(rays[0]), len(rays)
    E = [(i, j) for i, j in combinations(range(V), 2) if qz(bil_dot(rays[i], rays[j], B, C))]
    n = 2 * d * V
    def coord(i, c, real): return 2 * d * i + 2 * c + (0 if real else 1)
    rows = []
    for i, j in E:
        re, im = [ZERO] * n, [ZERO] * n
        for c in range(d):
            re[coord(i, c, True)] = qadd(re[coord(i, c, True)], rays[j][c])
            im[coord(i, c, False)] = qadd(im[coord(i, c, False)], qneg(rays[j][c]))
            re[coord(j, c, True)] = qadd(re[coord(j, c, True)], rays[i][c])
            im[coord(j, c, False)] = qadd(im[coord(j, c, False)], rays[i][c])
        rows.append(re); rows.append(im)
    for i in range(V):
        r = [ZERO] * n
        for c in range(d): r[coord(i, c, True)] = rays[i][c]
        rows.append(r)
    triv = []
    for i in range(V):
        t = [ZERO] * n
        for c in range(d): t[coord(i, c, False)] = rays[i][c]
        triv.append(t)
    for a in range(d):
        t = [ZERO] * n
        for i in range(V): t[coord(i, a, False)] = rays[i][a]
        triv.append(t)
    for a in range(d):
        for b in range(a + 1, d):
            t = [ZERO] * n
            for i in range(V):
                t[coord(i, a, False)] = qadd(t[coord(i, a, False)], rays[i][b])
                t[coord(i, b, False)] = qadd(t[coord(i, b, False)], rays[i][a])
            triv.append(t)
            t = [ZERO] * n
            for i in range(V):
                t[coord(i, a, True)] = qadd(t[coord(i, a, True)], rays[i][b])
                t[coord(i, b, True)] = qadd(t[coord(i, b, True)], qneg(rays[i][a]))
            triv.append(t)
    return rows, triv, E, n

def exact_flex_real_quadratic(rays, B, C, primes):
    rows, triv, E, n = build_extended_jacobian_real(rays, B, C)
    V, d = len(rays), len(rays[0])
    best = None
    for p, s in primes:
        rJ, rT = rank_mod_p(rows, p, s), rank_mod_p(triv, p, s)
        b_ = (n - rJ) - rT
        best = b_ if best is None else min(best, b_)
    if best is not None and best < 0:
        raise ValueError(f"exact_flex_real_quadratic: negative bound {best} -- internal "
                         "inconsistency; refusing to return it.")
    return dict(bound=best, E=len(E), n=n, V=V, d=d, triv_expected=V + d * d - 1)

# ============================================================================================
# t0 / tau (torsion layer) — generic, reusing torsion_layer.parity_torsion on the index-level
# (nrays, bases) representation shared by every target below.
# ============================================================================================
def t0_tau(nrays, triads):
    bases = [list(t) for t in triads]
    tau_nonzero, cert, kdim, hist = parity_torsion(nrays, bases)
    return 1, (1 if tau_nonzero else 0)   # t0=1 always here (we only call this on verified-uncolorable sets)

# ============================================================================================
# TARGET 1: Heegner-7 island   Z[alpha], alpha=(1+sqrt(-7))/2, alpha^2 = alpha - 2  (B=1, C=-2)
# ============================================================================================
HEEG_B, HEEG_C = 1, -2

def heegner7_alphabet():
    ONE = (1, 0); ALPHA = (0, 1); ALPHABAR = qconj(ALPHA, HEEG_B)
    return [ZERO, ONE, qneg(ONE), ALPHA, qneg(ALPHA), ALPHABAR, qneg(ALPHABAR)]

def cmd_pool_heegner7():
    print("[Heegner-7] ring Z[alpha], alpha=(1+sqrt(-7))/2, alpha^2=alpha-2 (B=1,C=-2)")
    print("alphabet {0,+-1,+-alpha,+-alpha-bar} =", heegner7_alphabet())
    raws = raw_vectors(heegner7_alphabet(), 3)
    print(f"raw A^3\\{{0}} vectors: {len(raws)}")
    rays = collect_rays(raws, HEEG_B, HEEG_C)
    print(f"distinct projective rays: {len(rays)}  (Kernaghan Table 7/Sec 6.5 reports 145)")
    pairs, triads, adj = build_structure(rays, herm_dot, HEEG_B, HEEG_C)
    print(f"Hermitian-orthogonal pairs: {len(pairs)}; complete triads: {len(triads)}  "
          f"(paper: 42 triads)")
    bp = basis_participating(len(rays), triads)
    print(f"basis-participating rays: {len(bp)}; auxiliary: {len(rays) - len(bp)}  "
          f"(paper Table 12: 69 basis-participating / 76 auxiliary, 52%)")
    u, nodes, _, _ = uncolorable(rays, herm_dot, HEEG_B, HEEG_C)
    print(f"FULL POOL KS-uncolorable (exhaustive, exact): {u}  ({nodes} search nodes)")
    assert len(rays) == 145 and len(triads) == 42 and len(bp) == 69 and u, \
        "Heegner-7 reconstruction mismatch vs. arXiv:2603.16988 Sec 6.5/Table 12 -- STOP"
    print("SELF-VALIDATION: rays/triads/basis-participating counts match the paper exactly.")
    cache_save("heegner7_pool", rays)
    cache_save("heegner7_bp_idx", bp)
    return rays, bp

def cmd_core_heegner7(trials=40):
    rays = cache_load("heegner7_pool"); bp = cache_load("heegner7_bp_idx")
    if rays is None or bp is None:
        rays, bp = cmd_pool_heegner7()
    rays = [tuple(tuple(c) for c in v) for v in rays]
    sub_rays, _ = restrict(rays, bp)
    t0 = time.time()
    core = greedy_critical_core(sub_rays, herm_dot, HEEG_B, HEEG_C, trials=trials, seed0=0)
    core_rays = [sub_rays[i] for i in core]
    print(f"greedy critical core (best of {trials} random-order trials): {len(core_rays)} rays "
          f"({time.time()-t0:.1f}s)  (paper's OCUS-certified pool minimum: 43)")
    pairs, triads, _ = build_structure(core_rays, herm_dot, HEEG_B, HEEG_C)
    from collections import Counter
    deg = [0] * len(core_rays)
    for i, j in pairs: deg[i] += 1; deg[j] += 1
    print(f"core: pairs={len(pairs)}, triads(bases)={len(triads)}, "
          f"degree distribution={sorted(Counter(deg).items())}")
    u, nodes, _, _ = uncolorable(core_rays, herm_dot, HEEG_B, HEEG_C)
    print(f"core KS-uncolorable: {u} ({nodes} nodes)")
    assert u
    cache_save("heegner7_core", core_rays)
    return core_rays

def cmd_flex_heegner7():
    core_rays = cache_load("heegner7_core")
    if core_rays is None: core_rays = cmd_core_heegner7()
    core_rays = [tuple(tuple(c) for c in v) for v in core_rays]
    pairs, triads, _ = build_structure(core_rays, herm_dot, HEEG_B, HEEG_C)
    _, tau = t0_tau(len(core_rays), triads)
    print(f"t0 = 1 (KS-uncolorable, verified above), tau (parity torsion) = {tau}  (d=3 odd => "
          f"tau=0 forced by Lemma B, TORSION.md)")
    Dmag = abs(HEEG_B * HEEG_B + 4 * HEEG_C)   # |1-8| = 7
    primes = find_primes_ring(0, Dmag, count=2, below=200003)
    print(f"mod-p primes for Q(sqrt({Dmag})): {[p for p, s in primes]}")
    t0 = time.time()
    cert = exact_flex_hermitian_quadratic(core_rays, HEEG_B, HEEG_C, primes)
    print(f"Heegner-7 core: n={cert['n']}, E={cert['E']}, trivial-expected={cert['triv_expected']}, "
          f"mod-p bound on flex: 0 <= flex <= {cert['bound']}  ({time.time()-t0:.1f}s)")
    if cert["bound"] == 0:
        print("=> FLEX = 0 EXACT (over Q(sqrt(-7))) -- Heegner-7 island CONFIRMED RIGID "
              "(matches Kernaghan's numerical Jacobian claim, now exact).")
    else:
        print(f"=> flex NOT pinned to 0 by this mod-p bound (bound={cert['bound']}); "
              f"NOT a rigidity certificate -- needs the flex=1 lift machinery or more primes.")
    return cert

# ============================================================================================
# TARGET 2: Golden island   Z[phi], phi=(1+sqrt5)/2, phi^2 = phi + 1   (B=1, C=1), REAL ring.
# Raw alphabet {0,+-1,+-phi} is colorable (10 triads); REAL cross-product completion is
# required to reach the KS-uncolorable pool (Kernaghan Sec 6.4 / Table 5).
# ============================================================================================
GOLD_B, GOLD_C = 1, 1

def golden_alphabet():
    ONE = (1, 0); PHI = (0, 1)
    return [ZERO, ONE, qneg(ONE), PHI, qneg(PHI)]

def real_cross(u, v, B, C):
    """Standard R^3 cross product, computed exactly in the ring Z[t] (bilinear, real)."""
    return (qadd(qmul(u[1], v[2], B, C), qneg(qmul(u[2], v[1], B, C))),
            qadd(qmul(u[2], v[0], B, C), qneg(qmul(u[0], v[2], B, C))),
            qadd(qmul(u[0], v[1], B, C), qneg(qmul(u[1], v[0], B, C))))

def cross_product_completion(rays, B, C, max_rounds=6, verbose=True):
    rays = list(rays)
    for rnd in range(max_rounds):
        pairs, triads, _ = build_structure(rays, bil_dot, B, C)
        new = []
        for i, j in pairs:
            w = real_cross(rays[i], rays[j], B, C)
            if any(x != ZERO for x in w) and not any(proportional(w, r, B, C) for r in rays + new):
                new.append(w)
        if verbose: print(f"    round {rnd}: {len(rays)} rays, {len(pairs)} pairs -> +{len(new)} new")
        if not new: break
        rays = collect_rays(rays + new, B, C)
    return rays

def cmd_pool_golden():
    print("[Golden] ring Z[phi], phi=(1+sqrt5)/2, phi^2=phi+1 (B=1,C=1), REAL ring")
    print("raw alphabet {0,+-1,+-phi} =", golden_alphabet())
    raws = raw_vectors(golden_alphabet(), 3)
    raw_rays = collect_rays(raws, GOLD_B, GOLD_C)
    pairs0, triads0, _ = build_structure(raw_rays, bil_dot, GOLD_B, GOLD_C)
    print(f"raw pool: {len(raw_rays)} rays, {len(pairs0)} pairs, {len(triads0)} triads  "
          f"(paper Table 1: 49 rays, 10 triads, COLORABLE)")
    u0, nodes0, _, _ = uncolorable(raw_rays, bil_dot, GOLD_B, GOLD_C)
    print(f"raw pool KS-uncolorable: {u0}  (paper: colorable, i.e. expect False)")
    assert len(raw_rays) == 49 and len(triads0) == 10 and not u0
    print("SELF-VALIDATION: raw pool matches paper exactly (49 rays / 10 triads / colorable).")
    t0 = time.time()
    comp_rays = cross_product_completion(raw_rays, GOLD_B, GOLD_C, max_rounds=6)
    pairs, triads, _ = build_structure(comp_rays, bil_dot, GOLD_B, GOLD_C)
    print(f"completed pool: {len(comp_rays)} rays, {len(pairs)} pairs, {len(triads)} triads "
          f"({time.time()-t0:.1f}s)  (paper Table 5/Sec 6.4: 205 rays; text: 166 triads)")
    u, nodes, _, _ = uncolorable(comp_rays, bil_dot, GOLD_B, GOLD_C)
    print(f"completed pool KS-uncolorable: {u}  ({nodes} nodes)")
    bp = basis_participating(len(comp_rays), triads)
    print(f"basis-participating: {len(bp)}; auxiliary: {len(comp_rays)-len(bp)}")
    assert len(comp_rays) == 205 and u, "Golden completion mismatch vs. paper -- STOP"
    cache_save("golden_pool", comp_rays)
    cache_save("golden_bp_idx", bp)
    return comp_rays, bp

def cmd_core_golden(trials=15):
    comp_rays = cache_load("golden_pool"); bp = cache_load("golden_bp_idx")
    if comp_rays is None or bp is None:
        comp_rays, bp = cmd_pool_golden()
    comp_rays = [tuple(tuple(c) for c in v) for v in comp_rays]
    sub_rays, _ = restrict(comp_rays, bp)
    print(f"basis-participating subpool: {len(sub_rays)}")
    t0 = time.time()
    core = greedy_critical_core(sub_rays, bil_dot, GOLD_B, GOLD_C, trials=trials, seed0=0)
    core_rays = [sub_rays[i] for i in core]
    print(f"greedy critical core (best of {trials} trials): {len(core_rays)} rays "
          f"({time.time()-t0:.1f}s)  (paper's OCUS pool minimum: 52)")
    pairs, triads, _ = build_structure(core_rays, bil_dot, GOLD_B, GOLD_C)
    from collections import Counter
    deg = [0] * len(core_rays)
    for i, j in pairs: deg[i] += 1; deg[j] += 1
    print(f"core: pairs={len(pairs)}, triads(bases)={len(triads)}, "
          f"degree distribution={sorted(Counter(deg).items())}")
    u, nodes, _, _ = uncolorable(core_rays, bil_dot, GOLD_B, GOLD_C)
    print(f"core KS-uncolorable: {u} ({nodes} nodes)")
    assert u
    cache_save("golden_core", core_rays)
    return core_rays

def cmd_flex_golden():
    core_rays = cache_load("golden_core")
    if core_rays is None: core_rays = cmd_core_golden()
    core_rays = [tuple(tuple(c) for c in v) for v in core_rays]
    pairs, triads, _ = build_structure(core_rays, bil_dot, GOLD_B, GOLD_C)
    _, tau = t0_tau(len(core_rays), triads)
    print(f"t0 = 1 (KS-uncolorable, verified above), tau (parity torsion) = {tau}")
    Dmag = abs(GOLD_B * GOLD_B + 4 * GOLD_C)   # 1+4=5
    primes = find_primes_ring(GOLD_B, GOLD_C, count=2, below=200003)   # native ring primes suffice (real)
    print(f"mod-p primes splitting Z[phi] (D={Dmag}): {[p for p, s in primes]}")
    t0 = time.time()
    cert = exact_flex_real_quadratic(core_rays, GOLD_B, GOLD_C, primes)
    print(f"Golden core: n={cert['n']}, E={cert['E']}, trivial-expected={cert['triv_expected']}, "
          f"mod-p bound on flex: 0 <= flex <= {cert['bound']}  ({time.time()-t0:.1f}s)")
    if cert["bound"] == 0:
        print("=> FLEX = 0 EXACT (over Q(sqrt5)) -- Golden island CONFIRMED RIGID "
              "(matches Kernaghan's numerical Jacobian claim, now exact).")
    else:
        print(f"=> flex NOT pinned to 0 by this mod-p bound (bound={cert['bound']}).")
    return cert

# ============================================================================================
# TARGET 3: CK-31 (integer) island   Z, alphabet {0,+-1,+-2}, REAL cross-product completion.
# Entries are plain integers, so we reuse exact_rigidity.exact_flex VERBATIM (sympy rank / Q).
# ============================================================================================
def _int_to_pairs(v): return tuple((c, 0) for c in v)
def _pairs_to_int(v): return tuple(a for a, b in v)
def dot_pair_int(x, y, B, C): return (sum(a * c for (a, _), (c, _) in zip(x, y)), 0)

def cmd_ck31(trials=60):
    from exact_rigidity import exact_flex
    print("[CK-31 / integer island] ring Z, alphabet {0,+-1,+-2}")
    alphabet_int = [ZERO, (1, 0), (-1, 0), (2, 0), (-2, 0)]
    raws = raw_vectors(alphabet_int, 3)
    rays_p = collect_rays(raws, 0, 0)          # generic proportionality dedup (B,C irrelevant: b=0 always)
    print(f"raw pool: {len(rays_p)} rays  (paper Table 1: 49)")
    pairs0, triads0, _ = build_structure(rays_p, dot_pair_int, 0, 0)
    print(f"raw pool: {len(pairs0)} pairs, {len(triads0)} triads (paper: 138 pairs, 26 triads)")
    assert len(rays_p) == 49 and len(triads0) == 26
    # cross-product completion (integer cross products stay integer; dedup via proportionality)
    def cross_int(u, v):
        u = _pairs_to_int(u); v = _pairs_to_int(v)
        return _int_to_pairs((u[1]*v[2]-u[2]*v[1], u[2]*v[0]-u[0]*v[2], u[0]*v[1]-u[1]*v[0]))
    rays_p = list(rays_p)
    for rnd in range(6):
        pairs, _, _ = build_structure(rays_p, dot_pair_int, 0, 0)
        new = []
        for i, j in pairs:
            w = cross_int(rays_p[i], rays_p[j])
            if any(c != ZERO for c in w) and not any(proportional(w, r, 0, 0) for r in rays_p + new):
                new.append(w)
        print(f"    round {rnd}: {len(rays_p)} rays -> +{len(new)} new")
        if not new: break
        rays_p = collect_rays(rays_p + new, 0, 0)
    print(f"completed pool: {len(rays_p)} rays  (paper Table 5: 109)")
    pairs, triads, _ = build_structure(rays_p, dot_pair_int, 0, 0)
    print(f"completed pool: {len(pairs)} pairs, {len(triads)} triads")
    col, nodes = ks_colorable(len(rays_p), pairs, [list(t) for t in triads])
    print(f"completed pool KS-uncolorable: {not col} ({nodes} nodes)")
    assert not col and len(rays_p) == 109
    print("SELF-VALIDATION: 49->109 completion and uncolorability match the paper exactly.")
    bp = basis_participating(len(rays_p), triads)
    print(f"basis-participating: {len(bp)} / {len(rays_p)}")
    sub, _ = restrict(rays_p, bp)
    t0 = time.time()
    core_idx = greedy_critical_core(sub, dot_pair_int, 0, 0, trials=trials, seed0=0, verbose=False)
    core_p = [sub[i] for i in core_idx]
    print(f"greedy critical core (best of {trials} trials): {len(core_p)} rays "
          f"({time.time()-t0:.1f}s)  (Conway-Kochen minimum: 31)")
    pairsC, triadsC, _ = build_structure(core_p, dot_pair_int, 0, 0)
    colC, nodesC = ks_colorable(len(core_p), pairsC, [list(t) for t in triadsC])
    print(f"core: pairs={len(pairsC)}, triads={len(triadsC)}, uncolorable={not colC} ({nodesC} nodes)")
    assert not colC
    _, tau = t0_tau(len(core_p), triadsC)
    print(f"t0=1, tau={tau}")
    core_int = [_pairs_to_int(v) for v in core_p]
    fx = exact_flex(core_int, "CK-integer core")
    print(f"=> flex = {fx} EXACT over Q (sympy rank, integer rays) "
          f"{'-- RIGID' if fx == 0 else '-- FLEXIBLE!!'}")
    return dict(V=len(core_p), E=len(pairsC), flex=fx)

# ============================================================================================
# TARGET 4: Eisenstein island == Cabello's "simplest KS set" (arXiv:2508.07335), already built
# and exactly certified in cabello33.py (unmodified). Re-run here as a cross-check, not
# re-derived: this IS the reuse-don't-duplicate discipline the program asks for.
# ============================================================================================
def cmd_eisenstein():
    import cabello33 as cb
    fixed, badpairs = cb.reconstruct_bases()
    rays = cb.collect_rays(fixed)
    V = len(rays)
    pairs = [(i, j) for i, j in combinations(range(V), 2) if cb.eis0(cb.herm(rays[i], rays[j]))]
    triads = [c for c in combinations(range(V), 3)
              if all(cb.eis0(cb.herm(rays[i], rays[j])) for i, j in combinations(c, 2))]
    unc = cb.ks_uncolorable(rays, pairs, [list(t) for t in triads])
    _, tau = t0_tau(V, triads)
    exbound = cb.exact_flex_z3(rays)
    print(f"[Eisenstein island == Cabello-simplest-33] V={V}, pairs={len(pairs)}, "
          f"triads(bases)={len(triads)}, uncolorable={unc}, tau={tau}")
    print(f"flex bound over Q(sqrt3) (exact, cabello33.exact_flex_z3): 0 <= flex <= {exbound}")
    assert V == 33 and len(triads) == 14 and unc and exbound == 0
    print("=> FLEX = 0 EXACT -- matches Kernaghan's Eisenstein-33 rigidity claim (Obs. 28: "
          "n=33, pairs=78 -- both counts match here too) and cabello33.py's independent result.")
    return dict(V=V, E=len(pairs), flex=exbound)

# ============================================================================================
# TARGET 5 (NEW, exploratory): the d=4 modulus-2 island — alphabet {0,+-1,+-sqrt2} in R^4 (the
# Peres alphabet lifted to d=4 with NO other change of generating rule). This is NOT in
# Kernaghan's paper (his survey is d=3 only); it is a natural extension of the "modulus-2
# cancellation" mechanism (Sec 2.2 of arXiv:2603.16988) that his own classification predicts
# SHOULD keep working in any dimension, and it was untested by this program before now. If it
# is uncolorable AND flexible, it is exactly the second-flexible / first-even-d-flexible KS set
# the program is looking for; if uncolorable and rigid, it broadens the rigidity census with a
# genuinely new even-d graph (distinct from Peres-24/CEG-18/K20, built from a different rule).
# ============================================================================================
D4_B, D4_C = 0, 2   # same ring as Peres-33 (Z[sqrt2]), bilinear (real) dot, d=4 ambient space

def d4sqrt2_alphabet():
    ONE = (1, 0); S2 = (0, 1)
    return [ZERO, ONE, qneg(ONE), S2, qneg(S2)]

def cmd_pool_d4sqrt2():
    print("[d=4 modulus-2 island, EXPLORATORY] ring Z[sqrt2], alphabet {0,+-1,+-sqrt2}, d=4")
    raws = raw_vectors(d4sqrt2_alphabet(), 4)
    print(f"raw A^4\\{{0}} vectors: {len(raws)}")
    t0 = time.time()
    rays = collect_rays(raws, D4_B, D4_C)
    print(f"distinct projective rays: {len(rays)}  ({time.time()-t0:.1f}s)")
    pairs, bases, adj = build_structure_d(rays, bil_dot, D4_B, D4_C, 4)
    print(f"orthogonal pairs: {len(pairs)}; complete 4-bases: {len(bases)}")
    bp = basis_participating(len(rays), bases)
    print(f"basis-participating: {len(bp)}; auxiliary: {len(rays)-len(bp)}")
    (col,) = ks_colorable_generic(len(rays), pairs, [list(b) for b in bases])
    print(f"FULL POOL KS-uncolorable (exhaustive, exact, full KS1+KS2 rules, size-generic "
          f"checker): {not col}")
    cache_save("d4sqrt2_pool", rays)
    cache_save("d4sqrt2_bp_idx", bp)
    return rays, bp, (not col)

def cmd_core_d4sqrt2(trials=8):
    rays = cache_load("d4sqrt2_pool"); bp = cache_load("d4sqrt2_bp_idx")
    if rays is None or bp is None:
        rays, bp, unc = cmd_pool_d4sqrt2()
        assert unc, "d=4 sqrt2 pool turned out COLORABLE -- no core to extract; report as-is"
    rays = [tuple(tuple(c) for c in v) for v in rays]
    sub_rays, _ = restrict(rays, bp)
    print(f"basis-participating subpool: {len(sub_rays)}")
    t0 = time.time()
    core = greedy_critical_core_d(sub_rays, bil_dot, D4_B, D4_C, 4, trials=trials, seed0=0)
    core_rays = [sub_rays[i] for i in core]
    print(f"greedy critical core (best of {trials} trials): {len(core_rays)} rays "
          f"({time.time()-t0:.1f}s)")
    pairs, bases, _ = build_structure_d(core_rays, bil_dot, D4_B, D4_C, 4)
    from collections import Counter
    deg = [0] * len(core_rays)
    for i, j in pairs: deg[i] += 1; deg[j] += 1
    print(f"core: pairs={len(pairs)}, bases={len(bases)}, "
          f"degree distribution={sorted(Counter(deg).items())}")
    (col,) = ks_colorable_generic(len(core_rays), pairs, [list(b) for b in bases])
    print(f"core KS-uncolorable (size-generic checker): {not col}")
    assert not col
    # independent cross-check on this (small) core with the slower but differently-implemented
    # count-based checker (torsion_layer.count_ks_colorings, reused unmodified):
    col2, cnt2 = ks_colorable_generic_slow(len(core_rays), pairs, [list(b) for b in bases])
    print(f"  cross-check (count_ks_colorings, independent implementation): "
          f"uncolorable={not col2} ({cnt2} colorings)  {'AGREES' if col2 == col else 'MISMATCH!!'}")
    assert col2 == col
    cache_save("d4sqrt2_core", core_rays)
    return core_rays

def cmd_flex_d4sqrt2():
    core_rays = cache_load("d4sqrt2_core")
    if core_rays is None: core_rays = cmd_core_d4sqrt2()
    core_rays = [tuple(tuple(c) for c in v) for v in core_rays]
    pairs, bases, _ = build_structure_d(core_rays, bil_dot, D4_B, D4_C, 4)
    _, tau = t0_tau(len(core_rays), bases)
    print(f"t0 = 1 (KS-uncolorable, verified above), tau (parity torsion, GF(2)) = {tau}  "
          f"(d=4 even => Lemma B does NOT force tau=0; this is a live invariant here)")
    primes = find_primes_ring(D4_B, D4_C, count=2, below=200003)
    print(f"mod-p primes splitting Z[sqrt2]: {[p for p, s in primes]}")
    t0 = time.time()
    cert = exact_flex_real_quadratic(core_rays, D4_B, D4_C, primes)
    print(f"d=4 sqrt2 core: n={cert['n']}, E={cert['E']}, trivial-expected={cert['triv_expected']}, "
          f"mod-p bound on flex: 0 <= flex <= {cert['bound']}  ({time.time()-t0:.1f}s)")
    if cert["bound"] == 0:
        print("=> FLEX = 0 EXACT (over Q(sqrt2)) -- this d=4 island is RIGID. A NEW rigid "
              "even-d KS set (different generating rule from Peres-24/CEG-18/K20), broadening "
              "the exact-rigidity census; NOT the flexible counterexample sought.")
    else:
        print(f"=> mod-p bound is {cert['bound']} > 0 -- flex is NOT pinned to 0 by this bound. "
              f"This is NOT yet a flex={cert['bound']} EXACT certificate (would need the "
              f"explicit-kernel-vector lift, sic_zoo.exact_certificate-style, to upgrade); "
              f"report as NUMERICAL/UNRESOLVED, do not claim flexibility without the lift.")
    return cert

# ============================================================================================
# TARGET 6: even-d honest status report (no fabrication — see KS_CENSUS.md for the full verdict)
# ============================================================================================
def cmd_evend():
    print("[even-d flexibility hunt] STATUS (honest, no fabricated construction):")
    print("""
  Existing even-d KS sets already in this repo's zoo (sic_zoo.py): Peres-24 (flex=0 EXACT),
  CEG-18 (flex=0 EXACT, subset of Peres-24), Kernaghan-type-20 (flex=0 EXACT, derived from
  Peres-24's GF(2) parity kernel). All are RIGID.

  NEW this session: the d=4 lift of the Peres alphabet {0,+-1,+-sqrt2} (target d4sqrt2) IS
  KS-uncolorable (272-ray raw pool, exact) and its greedy critical core (60 rays, 302 pairs,
  27 bases) is EXACTLY RIGID over Q(sqrt2) (mod-p bound 0<=flex<=0). This is a genuinely NEW
  even-d KS graph (different ring/generating rule from Peres-24/CEG-18/K20) that broadens the
  exact-rigidity census, but it is RIGID, not the flexible counterexample sought.

  Beyond that one construction, no other new even-d (d=6/d=8) island was attempted: building a
  genuinely independent even-d KS set from a wholly different generating rule (a d=6/d=8
  algebraic construction) was judged out of reach of a single session's exact-arithmetic budget
  and is NOT attempted here to avoid a fabricated or unverified claim. Kernaghan's
  arXiv:2603.16988 survey is d=3 only; it offers no even-d candidate of its own.
    """)

if __name__ == "__main__":
    args = sys.argv[1:]
    target = args[0] if args else None
    T0 = time.time()
    dispatch = {
        "pool_heegner7": cmd_pool_heegner7,
        "core_heegner7": cmd_core_heegner7,
        "flex_heegner7": cmd_flex_heegner7,
        "pool_golden": cmd_pool_golden,
        "core_golden": cmd_core_golden,
        "flex_golden": cmd_flex_golden,
        "ck31": cmd_ck31,
        "eisenstein": cmd_eisenstein,
        "pool_d4sqrt2": cmd_pool_d4sqrt2,
        "core_d4sqrt2": cmd_core_d4sqrt2,
        "flex_d4sqrt2": cmd_flex_d4sqrt2,
        "evend": cmd_evend,
    }
    if target not in dispatch:
        print("ks_flex_census.py — targets:", ", ".join(dispatch))
        print("Run each target as a separate CLI call (pool/core/flex staged for the slow ones).")
        sys.exit(0)
    dispatch[target]()
    print(f"\n[{target}] done in {time.time()-T0:.1f}s")
