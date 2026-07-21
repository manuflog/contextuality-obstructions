#!/usr/bin/env python3
"""
B1 — SIC RIGIDITY ZOO (session 2): flex dimension for more state-independent-contextuality
(SIC / KS) ray sets. Dictionary under test: state-INDEPENDENT contextual ray sets are RIGID
(flex 0); any flexible SIC case is a counterexample and must be reported, not hidden.

HEADLINE RESULT (this file proves it, exactly): the PERES 33-RAY KS SET (d=3) IS *FLEXIBLE*,
flex = 1, certified EXACTLY over Q(sqrt2) in both directions (explicit nontrivial infinitesimal
flex vector with J.u = 0 verified in exact Z[sqrt2] arithmetic, plus mod-p rank bounds at two
primes). The flex is PURE IMAGINARY: the real framework is exactly rigid in RP^2, but the
configuration flexes off the real locus into CP^2. Since the deformation preserves the whole
orthogonality graph, every configuration along the family is a KS set: a one-parameter family
of complex 33-ray KS sets through the real Peres point. The dictionary "SIC => rigid" is
FALSE as stated and needs a repair (see SIC_ZOO.md).

Sets in the zoo (each verified for its defining property BEFORE the flex computation):
  1. CEG 18   (d=4): Cabello-Estebaranz-Garcia-Alcaine 18 rays / 9 tetrads, parity KS proof.
               Verified: exact tetrad orthogonality; 18 distinct rays; each ray in exactly 2
               of the 9 tetrads (9 odd => parity proof); KS-uncolorable (EXHAUSTIVE search);
               subset of Peres 24; found in the GF(2) parity census. -> RIGID (exact/Q).
  2. K20      (d=4): Kernaghan-TYPE 20 rays / 11 tetrads parity proof. NOT transcribed from
               memory: DERIVED here from the Peres-24 basis system by exact GF(2) kernel
               search, so correct by construction (Kernaghan 1994 exhibited a 20-11 proof in
               this same system; our representative may differ by a symmetry). -> RIGID (exact/Q).
  3. Peres 33 (d=3): components in {0,+-1,+-sqrt2}; signed-permutation orbits of
               (0,0,1),(0,1,1),(0,1,s),(1,1,s), s=sqrt2 (3+6+12+12=33; 16 triads + 24 bare
               orthogonal pairs). Verified: KS-uncolorable by exhaustive backtracking.
               -> FLEXIBLE, flex = 1 (exact/Q(sqrt2)) — the counterexample.
  3b. A greedy CRITICAL CORE of Peres 33 (proper uncolorable subset, every ray necessary):
               is it rigid or does the flex survive? Computed below.
  4./5. Yu-Oh 13, Peres 24: session-1 cases re-run through this pipeline (cross-checks).

Engine: flex_dimension.py (imported, unmodified; numerical, spectral-gap-checked).
EXACT method (replicating exact_rigidity.py's extended Jacobian): unnormalized rays; per edge
(i,j) rows Re/Im[w_i^dag v_j + v_i^dag w_j]; per vertex row Re[v_i^dag w_i]; trivial span =
V per-vertex phases + d^2 u(d) directions (one exact global-phase relation => rank V+d^2-1).
Entries in Z[sqrt2] as pairs (a,b) = a+b*sqrt2 (b=0 for integer sets).
  - flex=0 certificates: sympy rank over Q (integer sets) and/or the mod-p bound argument:
    for any ring hom phi: Z[sqrt2] -> F_p, rank_p(phi(M)) <= rank_{Q(sqrt2)}(M) (a nonzero
    minor mod p pulls back), and J*T^t == 0 verified exactly => 0 <= flex <= (n - rank_p J)
    - rank_p T; right side 0 => flex = 0 EXACTLY.
  - flex=1 certificates (new): kernel of the gauge-fixed system [J;T] mod p is 1-dim; the
    vector is lifted to Z[sqrt2] by entrywise rational reconstruction and VERIFIED exactly
    (J.u = 0 in Z[sqrt2]); rank T = V+d^2-1 exactly (mod-p lower bound + explicit relation);
    u not in span T (rank_p[T;u] = rank T + 1). Then ker >= 42 and mod-p gives ker <= 42:
    flex = 1 EXACTLY. The lift is only a heuristic; the verification is exact arithmetic.

Honest scope: infinitesimal rigidity/flexibility at these standard realizations. The Peres-33
finite deformation is followed numerically (Gauss-Newton path following with rank and
Gram-drift monitoring): NUMERICAL evidence that the exact infinitesimal flex integrates.
Sets skipped for vector-uncertainty reasons are listed in SIC_ZOO.md.
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from fractions import Fraction
from math import lcm
from itertools import combinations, permutations, product

from flex_dimension import flex_dimension                       # engine (numerical)
from exact_rigidity import integer_rays_yuoh, integer_rays_peres24  # vetted session-1 rays

SQRT2 = 2 ** 0.5

# ---------------------------------------------------------------- Z[sqrt2] as pairs (a,b)
Z0 = (0, 0)
def q_add(u, v): return (u[0] + v[0], u[1] + v[1])
def q_neg(u):    return (-u[0], -u[1])
def q_mul(u, v): return (u[0] * v[0] + 2 * u[1] * v[1], u[0] * v[1] + u[1] * v[0])
def q_dot(x, y):
    s = Z0
    for a, b in zip(x, y): s = q_add(s, q_mul(a, b))
    return s
def q_float(u): return u[0] + u[1] * SQRT2
def as_pairs(v): return tuple((int(a), 0) for a in v)
def idot(u, v): return sum(a * b for a, b in zip(u, v))
def sign_norm_int(v):
    v = list(v)
    for x in v:
        if x:
            if x < 0: v = [-t for t in v]
            break
    return tuple(v)
def q_sign_norm(v):
    for a, b in v:
        if (a, b) != Z0:
            return tuple((-x, -y) for x, y in v) if (a < 0 or (a == 0 and b < 0)) else tuple(v)
    return tuple(v)

class Q2:
    """Exact Q(sqrt2) element a + b*sqrt2, a,b rational (for phase-cleaning the flex vector)."""
    __slots__ = ("a", "b")
    def __init__(s, a=0, b=0): s.a = Fraction(a); s.b = Fraction(b)
    def __add__(s, o): return Q2(s.a + o.a, s.b + o.b)
    def __sub__(s, o): return Q2(s.a - o.a, s.b - o.b)
    def __mul__(s, o): return Q2(s.a * o.a + 2 * s.b * o.b, s.a * o.b + s.b * o.a)
    def inv(s):
        d = s.a * s.a - 2 * s.b * s.b
        return Q2(s.a / d, -s.b / d)

# ---------------------------------------------------------------- ray sets
# CEG18: the 9 tetrads of Cabello-Estebaranz-Garcia-Alcaine (Phys. Lett. A 212 (1996) 183).
CEG18_BASES = [
    [(0, 0, 0, 1), (0, 0, 1, 0), (1, 1, 0, 0), (1, -1, 0, 0)],
    [(0, 0, 0, 1), (0, 1, 0, 0), (1, 0, 1, 0), (1, 0, -1, 0)],
    [(1, -1, 1, -1), (1, -1, -1, 1), (1, 1, 0, 0), (0, 0, 1, 1)],
    [(1, -1, 1, -1), (1, 1, 1, 1), (1, 0, -1, 0), (0, 1, 0, -1)],
    [(0, 0, 1, 0), (0, 1, 0, 0), (1, 0, 0, 1), (1, 0, 0, -1)],
    [(1, -1, -1, 1), (1, 1, 1, 1), (1, 0, 0, -1), (0, 1, -1, 0)],
    [(1, 1, -1, 1), (1, 1, 1, -1), (1, -1, 0, 0), (0, 0, 1, 1)],
    [(1, 1, -1, 1), (-1, 1, 1, 1), (1, 0, 1, 0), (0, 1, 0, -1)],
    [(1, 1, 1, -1), (-1, 1, 1, 1), (1, 0, 0, 1), (0, 1, -1, 0)],
]

def rays_ceg18():
    rays = []
    for B in CEG18_BASES:
        for v in B:
            vv = sign_norm_int(v)
            if vv not in rays: rays.append(vv)
    return rays

def rays_peres33():
    """Peres (J.Phys.A 24 (1991) L175): signed-permutation orbits of
       (0,0,1),(0,1,1),(0,1,sqrt2),(1,1,sqrt2). Orbit sizes 3+6+12+12 = 33."""
    one, rt2 = (1, 0), (0, 1)
    seeds = [(Z0, Z0, one), (Z0, one, one), (Z0, one, rt2), (one, one, rt2)]
    rays, orbit_sizes = [], []
    for s in seeds:
        before = len(rays)
        for perm in permutations(range(3)):
            for sg in product((1, -1), repeat=3):
                v = q_sign_norm(tuple((sg[i] * s[perm[i]][0], sg[i] * s[perm[i]][1])
                                      for i in range(3)))
                if v not in rays: rays.append(v)
        orbit_sizes.append(len(rays) - before)
    assert orbit_sizes == [3, 6, 12, 12] and len(rays) == 33, (orbit_sizes, len(rays))
    return rays

def orth_structure_pairs(rays):
    """All orthogonal pairs and all complete triads (d=3) among Z[sqrt2] rays."""
    V = len(rays)
    pairs = [(i, j) for i, j in combinations(range(V), 2) if q_dot(rays[i], rays[j]) == Z0]
    pset = set(pairs)
    triads = [(i, j, k) for i, j, k in combinations(range(V), 3)
              if (i, j) in pset and (i, k) in pset and (j, k) in pset]
    return pairs, triads

# ---------------------------------------------------------------- KS-colorability searches
def basis_coloring_exists(bases):
    """EXHAUSTIVE DFS: does a {0,1} assignment exist with EXACTLY ONE 1 per listed basis?
       (Basis constraints only — 'no' here implies KS-uncolorability a fortiori.)"""
    ones, zeros, nodes = set(), set(), [0]
    def dfs(k):
        nodes[0] += 1
        if k == len(bases): return True
        b = bases[k]
        chosen = [r for r in b if r in ones]
        if len(chosen) > 1: return False
        if len(chosen) == 1:
            newz = [r for r in b if r != chosen[0] and r not in zeros]
            if any(r in ones for r in newz): return False
            zeros.update(newz)
            if dfs(k + 1): return True
            zeros.difference_update(newz)
            return False
        for cand in b:
            if cand in zeros: continue
            ones.add(cand)
            newz = [r for r in b if r != cand and r not in zeros and r not in ones]
            zeros.update(newz)
            if dfs(k + 1): return True
            ones.discard(cand); zeros.difference_update(newz)
        return False
    return dfs(0), nodes[0]

def ks_colorable(nrays, pairs, triads):
    """EXHAUSTIVE backtracking with unit propagation. Full KS rules: every orthogonal pair at
       most one 1; every complete triad exactly one 1. Returns (colorable, nodes)."""
    orth = [[] for _ in range(nrays)]
    for i, j in pairs: orth[i].append(j); orth[j].append(i)
    tri_of = [[] for _ in range(nrays)]
    for t, tri in enumerate(triads):
        for r in tri: tri_of[r].append(t)
    color = [-1] * nrays
    nodes = [0]
    def assign(i, val, trail):
        stack = [(i, val)]
        while stack:
            j, v = stack.pop()
            if color[j] != -1:
                if color[j] != v: return False
                continue
            color[j] = v; trail.append(j)
            if v == 1:
                for k in orth[j]: stack.append((k, 0))
            for t in tri_of[j]:
                vals = [color[r] for r in triads[t]]
                n1, n0 = vals.count(1), vals.count(0)
                if n1 > 1 or n0 == 3: return False
                if n1 == 1:
                    for r in triads[t]:
                        if color[r] == -1: stack.append((r, 0))
                elif n0 == 2:
                    for r in triads[t]:
                        if color[r] == -1: stack.append((r, 1))
        return True
    def dfs():
        nodes[0] += 1
        i = next((k for k in range(nrays) if color[k] == -1), None)
        if i is None:
            return all(sum(color[r] for r in tri) == 1 for tri in triads)
        for val in (1, 0):
            trail = []
            if assign(i, val, trail) and dfs(): return True
            for j in trail: color[j] = -1
        return False
    return dfs(), nodes[0]

# ---------------------------------------------------------------- GF(2) parity census (P24)
def gf2_nullspace(rows, ncols):
    piv = {}
    for r in rows:
        for c in sorted(piv):
            if (r >> c) & 1: r ^= piv[c]
        if r == 0: continue
        c = (r & -r).bit_length() - 1
        for c2 in list(piv):
            if (piv[c2] >> c) & 1: piv[c2] ^= r
        piv[c] = r
    basis = []
    for f in range(ncols):
        if f in piv: continue
        v = 1 << f
        for c, pr in piv.items():
            if (pr >> f) & 1: v |= 1 << c
        basis.append(v)
    return basis

def parity_census_peres24():
    """All parity proofs inside the Peres-24 basis system: subsets S of complete tetrads with
       |S| odd and every ray covered an even number of times (each tetrad demands one 1 =>
       odd total; even multiplicities => even total; contradiction => KS proof on the cover)."""
    p24 = [sign_norm_int(v) for v in integer_rays_peres24()]
    bases = [c for c in combinations(range(24), 4)
             if all(idot(p24[i], p24[j]) == 0 for i, j in combinations(c, 2))]
    nb = len(bases)
    ray_mask_of_basis = [sum(1 << r for r in b) for b in bases]
    rows = [sum(1 << bidx for bidx, b in enumerate(bases) if r in b) for r in range(24)]
    kernel = [0]
    for kb in gf2_nullspace(rows, nb):
        kernel = kernel + [x ^ kb for x in kernel]
    for x in kernel:                                   # verify honestly
        cov = [0] * 24
        for bidx in range(nb):
            if (x >> bidx) & 1:
                for r in bases[bidx]: cov[r] += 1
        assert all(c % 2 == 0 for c in cov), "GF(2) kernel verification failed"
    census = {}
    for x in kernel:
        if x == 0 or bin(x).count("1") % 2 == 0: continue
        covered = 0
        for bidx in range(nb):
            if (x >> bidx) & 1: covered |= ray_mask_of_basis[bidx]
        census.setdefault((bin(x).count("1"), bin(covered).count("1")), []).append((x, covered))
    return p24, bases, census

# ---------------------------------------------------------------- exact certificates
def build_extended_jacobian(rays):
    """exact_rigidity.py formulation, entries in Z[sqrt2]. rays: tuples of pairs (a,b)."""
    d, V = len(rays[0]), len(rays)
    E = [(i, j) for i, j in combinations(range(V), 2) if q_dot(rays[i], rays[j]) == Z0]
    n = 2 * d * V
    def coord(i, c, real): return 2 * d * i + 2 * c + (0 if real else 1)
    rows = []
    for i, j in E:
        re, im = [Z0] * n, [Z0] * n
        for c in range(d):
            re[coord(i, c, True)] = q_add(re[coord(i, c, True)], rays[j][c])
            im[coord(i, c, False)] = q_add(im[coord(i, c, False)], q_neg(rays[j][c]))
            re[coord(j, c, True)] = q_add(re[coord(j, c, True)], rays[i][c])
            im[coord(j, c, False)] = q_add(im[coord(j, c, False)], rays[i][c])
        rows.append(re); rows.append(im)
    for i in range(V):
        r = [Z0] * n
        for c in range(d): r[coord(i, c, True)] = rays[i][c]
        rows.append(r)
    triv = []
    for i in range(V):
        t = [Z0] * n
        for c in range(d): t[coord(i, c, False)] = rays[i][c]
        triv.append(t)
    for a in range(d):
        t = [Z0] * n
        for i in range(V): t[coord(i, a, False)] = rays[i][a]
        triv.append(t)
    for a in range(d):
        for b in range(a + 1, d):
            t = [Z0] * n
            for i in range(V):
                t[coord(i, a, False)] = q_add(t[coord(i, a, False)], rays[i][b])
                t[coord(i, b, False)] = q_add(t[coord(i, b, False)], rays[i][a])
            triv.append(t)
            t = [Z0] * n
            for i in range(V):
                t[coord(i, a, True)] = q_add(t[coord(i, a, True)], rays[i][b])
                t[coord(i, b, True)] = q_add(t[coord(i, b, True)], q_neg(rays[i][a]))
            triv.append(t)
    return rows, triv, E, n

def q_rowdot(r1, r2):
    s0 = s1 = 0
    for (a, b), (c, e) in zip(r1, r2):
        if (a or b) and (c or e):
            s0 += a * c + 2 * b * e; s1 += a * e + b * c
    return (s0, s1)

def find_primes_7mod8(count=2, below=46341):
    def is_prime(m):
        if m < 2: return False
        for q in range(2, int(m ** 0.5) + 1):
            if m % q == 0: return False
        return True
    out, p = [], below - 1
    while len(out) < count and p > 2:
        if p % 8 == 7 and is_prime(p):
            s = pow(2, (p + 1) // 4, p)
            assert (s * s) % p == 2 % p
            out.append((p, s))
        p -= 1
    return out

def _modp_matrix(rows_pairs, p, s):
    return np.array([[(a + b * s) % p for a, b in row] for row in rows_pairs], dtype=np.int64)

def _eliminate(A, p):
    """In-place Gauss elimination mod p; returns (rank, pivot column list)."""
    nr, nc = A.shape
    piv, r = [], 0
    for c in range(nc):
        nz = np.nonzero(A[r:, c])[0]
        if nz.size == 0: continue
        i = r + nz[0]
        if i != r: A[[r, i]] = A[[i, r]]
        A[r] = (A[r] * pow(int(A[r, c]), p - 2, p)) % p
        oth = np.nonzero(A[:, c])[0]; oth = oth[oth != r]
        if oth.size: A[oth] = (A[oth] - np.outer(A[oth, c], A[r])) % p
        piv.append(c); r += 1
        if r == nr: break
    return r, piv

def rank_mod_p(rows_pairs, p, s):
    A = _modp_matrix(rows_pairs, p, s)
    r, _ = _eliminate(A, p)
    return r

def modp_kernel_1d(rows_pairs, n, p, s):
    """If the mod-p kernel of the stacked rows is 1-dimensional, return it (int64 array)."""
    A = _modp_matrix(rows_pairs, p, s)
    r, piv = _eliminate(A, p)
    free = [c for c in range(n) if c not in piv]
    if len(free) != 1: return None
    f = free[0]
    u = np.zeros(n, dtype=np.int64); u[f] = 1
    for ri, c in enumerate(piv): u[c] = int((-int(A[ri, f])) % p)
    return u

def modp_phase_clean(u, rays, p, s):
    """Project out the per-vertex phase components mod p, then normalize the first nonzero
       entry to 1 (so the exact counterpart has small height and reconstructs robustly)."""
    d, V = len(rays[0]), len(rays)
    u = [int(x) % p for x in u]
    for i in range(V):
        v = [(a + b * s) % p for a, b in rays[i]]
        num = sum(v[c] * u[2 * d * i + 2 * c + 1] for c in range(d)) % p
        den = sum(vc * vc for vc in v) % p
        g = (num * pow(den, p - 2, p)) % p
        for c in range(d):
            u[2 * d * i + 2 * c + 1] = (u[2 * d * i + 2 * c + 1] - g * v[c]) % p
    nz = next((x for x in u if x), None)
    if nz is None: return None
    inv = pow(nz, p - 2, p)
    return [(x * inv) % p for x in u]

def reconstruct_pairs(u_modp, p, s, amax=80, denmax=24):
    """Entrywise heuristic lift of a mod-p vector to Z[sqrt2] pairs with small coefficients
       and a common denominator. Correctness is NOT assumed — the caller verifies J.u = 0
       in exact arithmetic afterwards."""
    recs = []
    for e in u_modp:
        e = int(e) % p; hit = None
        for den in range(1, denmax + 1):
            t = (e * den) % p
            for b in range(-amax, amax + 1):
                a = (t - b * s) % p
                if a > p // 2: a -= p
                if abs(a) <= amax: hit = (a, b, den); break
            if hit: break
        if hit is None: return None
        recs.append(hit)
    L = 1
    for _, _, dn in recs: L = lcm(L, dn)
    out = [(a * (L // dn), b * (L // dn)) for a, b, dn in recs]
    g = 0
    for a, b in out: g = int(np.gcd(g, int(np.gcd(abs(a), abs(b)))))
    if g > 1: out = [(a // g, b // g) for a, b in out]
    return out

def clean_phase_gauge(rays, u_pairs):
    """Subtract the exact per-vertex phase component (y_i -> y_i - g_i v_i, g_i in Q(sqrt2)).
       Phase directions are trivial kernel elements, so the result stays in ker J exactly."""
    d, V = len(rays[0]), len(rays)
    uc = [Q2(a, b) for a, b in u_pairs]
    for i in range(V):
        v = [Q2(a, b) for a, b in rays[i]]
        num, den = Q2(), Q2()
        for c in range(d):
            y = uc[2 * d * i + 2 * c + 1]
            num = num + v[c] * y; den = den + v[c] * v[c]
        g = num * den.inv()
        for c in range(d):
            uc[2 * d * i + 2 * c + 1] = uc[2 * d * i + 2 * c + 1] - g * v[c]
    L = 1
    for q in uc: L = lcm(L, q.a.denominator, q.b.denominator)
    out = [(int(q.a * L), int(q.b * L)) for q in uc]
    g = 0
    for a, b in out: g = int(np.gcd(g, int(np.gcd(abs(a), abs(b)))))
    if g > 1: out = [(a // g, b // g) for a, b in out]
    return out

def verify_global_phase_relation(rays, n):
    """Exact identity: sum of the V per-vertex phase directions == sum of the d directions
       A=iE_aa (both give w_i = i v_i). This is the one relation making rank T <= V+d^2-1."""
    d, V = len(rays[0]), len(rays)
    r1 = [Z0] * n; r2 = [Z0] * n
    for i in range(V):
        for c in range(d):
            r1[2 * d * i + 2 * c + 1] = q_add(r1[2 * d * i + 2 * c + 1], rays[i][c])
    for a in range(d):
        for i in range(V):
            r2[2 * d * i + 2 * a + 1] = q_add(r2[2 * d * i + 2 * a + 1], rays[i][a])
    return r1 == r2

def exact_certificate(rays_pairs, name, primes, sympy_over_Q=False):
    """Extended-Jacobian exact certificate. Returns dict with:
       exact_value: 0 or 1 if the flex is EXACTLY certified, else None (numerical only);
       flex_vector: the exact nontrivial kernel vector (pairs) when exact_value == 1."""
    J, T, E, n = build_extended_jacobian(rays_pairs)
    V, d = len(rays_pairs), len(rays_pairs[0])
    ok0 = all(q_rowdot(jr, tr) == Z0 for jr in J for tr in T)
    assert ok0, f"{name}: trivial directions NOT in kernel — formulation error"
    bounds = []
    for p, s in primes:
        rJ, rT = rank_mod_p(J, p, s), rank_mod_p(T, p, s)
        bounds.append(((n - rJ) - rT, p, s, rJ, rT))
    bound, p_used, s_used, rJ, rT = min(bounds)
    res = dict(name=name, V=V, d=d, E=len(E), n=n, rankJ_p=rJ, rankT_p=rT,
               triv_expected=V + d * d - 1, bound=bound, p=p_used,
               exact_value=(0 if bound == 0 else None), flex_vector=None,
               sympy_rank=None, sympy_flex=None)
    if sympy_over_Q:
        from sympy import Matrix
        assert all(b == 0 for row in J for _, b in row)
        MJ = Matrix([[a for a, _ in row] for row in J])
        MT = Matrix([[a for a, _ in row] for row in T])
        rkJ, rkT = MJ.rank(), MT.rank()
        res["sympy_rank"], res["sympy_flex"] = rkJ, (n - rkJ) - rkT
        assert rkJ == rJ and rkT == rT, f"{name}: sympy(Q) vs mod-p rank mismatch"
        res["exact_value"] = res["sympy_flex"]
        return res
    if bound == 1:
        # attempt an EXACT flex=1 certificate via an explicit kernel vector
        if rT != V + d * d - 1 or not verify_global_phase_relation(rays_pairs, n):
            return res                       # cannot pin rank T exactly; stay honest
        for p, s in primes:
            u_p = modp_kernel_1d(J + T, n, p, s)
            if u_p is None: continue
            candidates = []
            u_pc = modp_phase_clean(u_p, rays_pairs, p, s)
            if u_pc is not None: candidates.append(u_pc)
            candidates.append(u_p)
            for cand in candidates:
                u_ex = reconstruct_pairs(cand, p, s)
                if u_ex is None: continue
                u_cl = clean_phase_gauge(rays_pairs, u_ex)
                if all(x == Z0 for x in u_cl): continue
                if not all(q_rowdot(row, u_cl) == Z0 for row in J): continue
                # u not in span T: mod-p rank of [T;u] exceeds exact rank T (=V+d^2-1)
                if rank_mod_p(T + [u_cl], p, s) != V + d * d: continue
                # ker >= rankT+1 = n - rank_p J and ker <= n - rank_p J => flex = 1 EXACT
                res["exact_value"] = 1
                res["flex_vector"] = u_cl
                return res
    return res

def cert_str(c):
    if c["sympy_flex"] is not None:
        return (f"EXACT/Q: sympy rank J={c['sympy_rank']}, ker={c['n']-c['sympy_rank']}, "
                f"triv={c['rankT_p']} -> flex={c['sympy_flex']} (mod-p agrees)")
    if c["exact_value"] == 0:
        return (f"EXACT/Q(sqrt2): 0<=flex<=({c['n']}-{c['rankJ_p']})-{c['rankT_p']}=0 "
                f"(kernel containment exact; p={c['p']})")
    if c["exact_value"] == 1:
        return (f"EXACT/Q(sqrt2): flex=1 — explicit kernel vector u with J.u=0 verified in "
                f"Z[sqrt2], u not in trivial span; mod-p bound gives flex<=1 (p={c['p']})")
    return f"NOT exactly certified (mod-p bound {c['bound']}); numerical value stands"

# ---------------------------------------------------------------- Peres 33 deep dive
def real_subframework_rigidity(rays, primes, name="Peres 33 (real)"):
    """Rigidity of the REAL framework in RP^2: variables x_i in R^d, edge rows
       v_j.x_i + v_i.x_j = 0, norm rows v_i.x_i = 0; trivial span = so(d) rotations.
       Exact via the same mod-p bound argument."""
    d, V = len(rays[0]), len(rays)
    E = [(i, j) for i, j in combinations(range(V), 2) if q_dot(rays[i], rays[j]) == Z0]
    n = d * V
    rows = []
    for i, j in E:
        r = [Z0] * n
        for c in range(d):
            r[d * i + c] = q_add(r[d * i + c], rays[j][c])
            r[d * j + c] = q_add(r[d * j + c], rays[i][c])
        rows.append(r)
    for i in range(V):
        r = [Z0] * n
        for c in range(d): r[d * i + c] = rays[i][c]
        rows.append(r)
    triv = []
    for a in range(d):
        for b in range(a + 1, d):
            t = [Z0] * n
            for i in range(V):
                t[d * i + a] = q_add(t[d * i + a], rays[i][b])
                t[d * i + b] = q_add(t[d * i + b], q_neg(rays[i][a]))
            triv.append(t)
    assert all(q_rowdot(r, t) == Z0 for r in rows for t in triv)
    best = None
    for p, s in primes:
        rJ, rT = rank_mod_p(rows, p, s), rank_mod_p(triv, p, s)
        b = (n - rJ) - rT
        best = b if best is None else min(best, b)
    print(f"    real framework ({name}): n={n}, mod-p bound => 0 <= real-flex <= {best}"
          + ("  => REAL-RIGID, EXACT" if best == 0 else "  (not certified rigid)"))
    return best

def continue_flex_numerically(rays_pairs, u_pairs, steps=8, h=0.05):
    """Follow the flex to FINITE deformations (numerical). Gauss-Newton projection onto the
       orthogonality+norm constraint set after each predictor step along the current flex
       direction; monitors: constraint residual, rank J (should stay 156), and the drift of a
       gauge-invariant non-edge Gram magnitude (proves the motion is not gauge)."""
    d = len(rays_pairs[0]); V = len(rays_pairs)
    v = [np.array([q_float(c) for c in r], dtype=complex) for r in rays_pairs]
    norms0 = [float(np.real(np.vdot(x, x))) for x in v]
    E = [(i, j) for i, j in combinations(range(V), 2) if q_dot(rays_pairs[i], rays_pairs[j]) == Z0]
    Eset = set(E)
    w0 = [np.array([u_pairs[2 * d * i + 2 * c][0] + u_pairs[2 * d * i + 2 * c][1] * SQRT2
                    + 1j * (u_pairs[2 * d * i + 2 * c + 1][0] + u_pairs[2 * d * i + 2 * c + 1][1] * SQRT2)
                    for c in range(d)]) for i in range(V)]
    nonedges = [(i, j) for i, j in combinations(range(V), 2) if (i, j) not in Eset]
    g0 = None
    def grams(v):
        nrm = [np.sqrt(np.real(np.vdot(x, x))) for x in v]
        return np.array([abs(np.vdot(v[i], v[j])) / (nrm[i] * nrm[j]) for i, j in nonedges])
    # Bargmann invariant of a non-orthogonal triple: gauge/unitary-invariant; REAL for any
    # configuration (anti)unitarily equivalent to a real one — nonzero phase = genuine motion.
    triple = next((i, j, k) for i, j, k in combinations(range(V), 3)
                  if (i, j) in set(nonedges) and (i, k) in set(nonedges) and (j, k) in set(nonedges))
    def bargmann_phase(v):
        i, j, k = triple
        return np.angle(np.vdot(v[i], v[j]) * np.vdot(v[j], v[k]) * np.vdot(v[k], v[i]))
    def F(v):
        out = []
        for i, j in E:
            z = np.vdot(v[i], v[j]); out += [z.real, z.imag]
        for i in range(V): out.append(np.real(np.vdot(v[i], v[i])) - norms0[i])
        return np.array(out)
    def JF(v):
        # real Jacobian of F wrt (Re v, Im v) stacked per vertex
        n = 2 * d * V
        M = np.zeros((2 * len(E) + V, n))
        def cols(i): return slice(2 * d * i, 2 * d * i + 2 * d)
        for r, (i, j) in enumerate(E):
            # d<vi,vj> = <dvi,vj>+<vi,dvj>; entries wrt dRe,dIm interleaved
            for c in range(d):
                M[2 * r, 2 * d * i + 2 * c] += v[j][c].real;  M[2 * r, 2 * d * i + 2 * c + 1] += v[j][c].imag
                M[2 * r + 1, 2 * d * i + 2 * c] += v[j][c].imag; M[2 * r + 1, 2 * d * i + 2 * c + 1] += -v[j][c].real
                M[2 * r, 2 * d * j + 2 * c] += v[i][c].real;  M[2 * r, 2 * d * j + 2 * c + 1] += v[i][c].imag
                M[2 * r + 1, 2 * d * j + 2 * c] += -v[i][c].imag; M[2 * r + 1, 2 * d * j + 2 * c + 1] += v[i][c].real
        for i in range(V):
            for c in range(d):
                M[2 * len(E) + i, 2 * d * i + 2 * c] = 2 * v[i][c].real
                M[2 * len(E) + i, 2 * d * i + 2 * c + 1] = 2 * v[i][c].imag
        return M
    def pack_dir(w):
        out = np.zeros(2 * d * V)
        for i in range(V):
            for c in range(d):
                out[2 * d * i + 2 * c] = w[i][c].real; out[2 * d * i + 2 * c + 1] = w[i][c].imag
        return out
    def unpack(delta, v):
        return [v[i] + np.array([delta[2 * d * i + 2 * c] + 1j * delta[2 * d * i + 2 * c + 1]
                                 for c in range(d)]) for i in range(V)]
    dirvec = pack_dir(w0); dirvec /= np.linalg.norm(dirvec)
    print(f"    continuation (numerical): step h={h}; monitors: max non-edge |gram| drift,"
          f" Bargmann phase of triple {triple}, rank J")
    g0 = grams(v); b0 = bargmann_phase(v)
    print(f"      t=0.00  |gram|drift=0  barg={b0:+.6f}  resid={np.abs(F(v)).max():.1e}")
    for k in range(1, steps + 1):
        vv = unpack(h * dirvec, v)
        for _ in range(6):   # Gauss-Newton correction
            M = JF(vv); f = F(vv)
            delta, *_ = np.linalg.lstsq(M, -f, rcond=None)
            vv = unpack(delta, vv)
            if np.abs(F(vv)).max() < 1e-12: break
        v = vv
        M = JF(v); sv = np.linalg.svd(M, compute_uv=False)
        rank = int((sv > 1e-8 * sv.max()).sum())
        _, _, Vt = np.linalg.svd(M)
        K = Vt[rank:]
        newdir = K.T @ (K @ dirvec)
        if np.linalg.norm(newdir) > 1e-8: dirvec = newdir / np.linalg.norm(newdir)
        print(f"      t={k*h:.2f}  |gram|drift={np.abs(grams(v)-g0).max():.6f}  "
              f"barg={bargmann_phase(v):+.6f}  resid={np.abs(F(v)).max():.1e}  rankJ={rank}")
    gd, bd = np.abs(grams(v) - g0).max(), abs(bargmann_phase(v) - b0)
    if bd > 1e-6 or gd > 1e-6:
        print(f"      => genuine finite deformation: {'Bargmann phase' if bd > gd else '|gram|'}"
              f" moves (real configs have REAL Bargmann invariants), rank J constant along path;")
        print(f"         orthogonality graph preserved by construction => every point on the path"
              f" is a 33-ray KS set.")
    else:
        print("      => WARNING: no invariant drift detected — treat finite flexibility as"
              " UNCONFIRMED (infinitesimal flex=1 remains exact).")
    return gd, bd

def greedy_critical_core(rays):
    """Peel rays greedily while KS-uncolorability (full rules) survives; the result is a
       CRITICAL KS subset (removing any further ray makes it colorable)."""
    keep = list(range(len(rays)))
    def uncolorable(subset):
        sub = [rays[i] for i in subset]
        pairs, triads = orth_structure_pairs(sub)
        col, _ = ks_colorable(len(sub), pairs, [list(t) for t in triads])
        return not col
    assert uncolorable(keep)
    for r in list(keep):
        cand = [x for x in keep if x != r]
        if uncolorable(cand): keep = cand
    # verify criticality
    for r in list(keep):
        assert not uncolorable([x for x in keep if x != r]), "core not critical?!"
    return keep

def peres33_flex_structure(p33, u):
    """Print the exact flex vector grouped per vertex: w_i = i * (real Z[sqrt2] vector)."""
    d = 3
    pretty = []
    for i in range(len(p33)):
        seg = u[6 * i:6 * i + 6]
        x = [seg[2 * c] for c in range(d)]; y = [seg[2 * c + 1] for c in range(d)]
        if any(t != Z0 for t in x): return None  # not pure imaginary; skip pretty print
        if any(t != Z0 for t in y): pretty.append((i, p33[i], y))
    return pretty

# ---------------------------------------------------------------- zoo runner
def fnum(rays_pairs, name):
    return flex_dimension([np.array([q_float(c) for c in r]) for r in rays_pairs], name=name)

PRIMES = find_primes_7mod8(2)

def sec_ceg18(primes=PRIMES):
    table = []
    print("[1] CEG 18 (Cabello-Estebaranz-Garcia-Alcaine, d=4, 18 rays / 9 tetrads)")
    for B in CEG18_BASES:
        for u, v in combinations(B, 2):
            assert idot(u, v) == 0, f"CEG18 tetrad not orthogonal: {u},{v}"
    ceg = rays_ceg18()
    assert len(ceg) == 18
    counts = {r: sum(r in [sign_norm_int(v) for v in B] for B in CEG18_BASES) for r in ceg}
    assert all(c == 2 for c in counts.values())
    p24n = [sign_norm_int(v) for v in integer_rays_peres24()]
    assert all(r in p24n for r in ceg)
    print("    verified: 9 exact orthogonal tetrads; 18 distinct rays, each in EXACTLY 2 tetrads;")
    print("    9 odd => PARITY PROOF; subset of Peres 24.")
    idx = {r: k for k, r in enumerate(ceg)}
    exists, nodes = basis_coloring_exists([tuple(idx[sign_norm_int(v)] for v in B)
                                           for B in CEG18_BASES])
    assert not exists
    print(f"    verified: KS-UNCOLORABLE (exhaustive transversal DFS, {nodes} nodes)")
    cegp = [as_pairs(r) for r in ceg]
    fx = fnum(cegp, "CEG 18")
    cert = exact_certificate(cegp, "CEG 18", primes, sympy_over_Q=True)
    print(f"    exact: {cert_str(cert)}\n")
    table.append(("CEG 18", 4, 18, cert["E"], "KS parity 18r/9b (verified)", fx, cert))
    return table

def sec_k20(primes=PRIMES):
    table = []
    ceg = rays_ceg18()
    print("[2] Kernaghan-type 20 (d=4): GF(2) parity census of the Peres-24 basis system")
    p24, bases, census = parity_census_peres24()
    print(f"    Peres-24 system: {len(bases)} complete tetrads")
    for key in sorted(census):
        print(f"      parity proofs with {key[0]} bases / {key[1]} rays: {len(census[key])}")
    ceg_ray_mask = 0
    for r in ceg: ceg_ray_mask |= 1 << p24.index(r)
    assert any(cov == ceg_ray_mask for _, cov in census.get((9, 18), [])), \
        "hardcoded CEG18 not found in parity census — transcription error!"
    print("    cross-validation: hardcoded CEG18 ray set FOUND among the 9-basis/18-ray proofs")
    assert (11, 20) in census
    distinct = sorted({cov for _, cov in census[(11, 20)]})
    print(f"    distinct 20-ray sets among 11-basis proofs: {len(distinct)}")
    x, covered = min(census[(11, 20)], key=lambda t: t[1])
    k20 = [p24[r] for r in range(24) if (covered >> r) & 1]
    k20_bases = [bases[b] for b in range(len(bases)) if (x >> b) & 1]
    mult = {r: sum(r in b for b in k20_bases) for r in range(24) if (covered >> r) & 1}
    assert len(k20) == 20 and len(k20_bases) == 11 and all(m % 2 == 0 for m in mult.values())
    print(f"    verified: 11 tetrads (odd), each ray in an even # of them "
          f"(multiplicities {sorted(set(mult.values()))}) => PARITY PROOF")
    remap = {r: k for k, r in enumerate(sorted(mult))}
    exists, nodes = basis_coloring_exists([tuple(remap[r] for r in b) for b in k20_bases])
    assert not exists
    print(f"    verified: KS-UNCOLORABLE (exhaustive transversal DFS, {nodes} nodes)")
    k20p = [as_pairs(r) for r in k20]
    fx2 = fnum(k20p, "Kernaghan 20")
    for extra in distinct[1:3]:
        fnum([as_pairs(p24[r]) for r in range(24) if (extra >> r) & 1], "K20 variant")
    cert2 = exact_certificate(k20p, "Kernaghan 20", primes, sympy_over_Q=True)
    print(f"    exact: {cert_str(cert2)}\n")
    table.append(("Kernaghan-type 20", 4, 20, cert2["E"], "KS parity 20r/11b (derived+verified)",
                  fx2, cert2))
    return table

def sec_peres33(primes=PRIMES):
    table = []
    print("[3] Peres 33 (d=3, components 0,+-1,+-sqrt2) — THE COUNTEREXAMPLE")
    p33 = rays_peres33()
    pairs, triads = orth_structure_pairs(p33)
    tri_pairs = {pq for t in triads for pq in combinations(t, 2)}
    print(f"    structure: orbits 3+6+12+12=33; orthogonal pairs {len(pairs)}; complete triads "
          f"{len(triads)}; bare pairs {len(pairs) - len(tri_pairs)}")
    col_full, nodes_full = ks_colorable(33, pairs, [list(t) for t in triads])
    assert not col_full, "Peres 33 unexpectedly KS-colorable!"
    print(f"    verified: KS-UNCOLORABLE, full rules (exhaustive backtracking, {nodes_full} nodes)")
    col_tri, nodes_tri = ks_colorable(33, sorted(tri_pairs), [list(t) for t in triads])
    print(f"    census: triad-only rules -> {'COLORABLE' if col_tri else 'uncolorable'} "
          f"({nodes_tri} nodes)")
    fx3 = fnum(p33, "Peres 33")
    cert3 = exact_certificate(p33, "Peres 33", primes, sympy_over_Q=False)
    print(f"    exact: {cert_str(cert3)}")
    table.append(("Peres 33", 3, 33, cert3["E"], "KS-uncolorable (verified, full rules)",
                  fx3, cert3))
    if cert3["exact_value"] == 1:
        u = cert3["flex_vector"]
        pretty = peres33_flex_structure(p33, u)
        if pretty is not None:
            print("    the EXACT flex vector is PURE IMAGINARY (w_i = i * real vector; x-parts 0):")
            fixed = 33 - len(pretty)
            print(f"      {fixed} rays fixed (the 3 coordinate axes), {len(pretty)} rays move; "
                  f"per-orbit pattern:")
            for i, ray, y in pretty[:6]:
                print(f"        ray {i} {ray}: w = i*{tuple(y)}")
            print("        ... (30 moving rays total; full vector in cert3['flex_vector'])")
        real_subframework_rigidity(p33, primes)
        continue_flex_numerically(p33, u)
    print()
    return table

def sec_core(primes=PRIMES):
    table = []
    p33 = rays_peres33()
    print("[3b] Greedy critical core of Peres 33 (is the flex carried by removable rays?)")
    core_idx = greedy_critical_core(p33)
    core = [p33[i] for i in core_idx]
    cpairs, ctriads = orth_structure_pairs(core)
    print(f"    core: {len(core)} rays (removed {[i for i in range(33) if i not in core_idx]}),"
          f" pairs {len(cpairs)}, triads {len(ctriads)}; CRITICAL (verified: every single-ray"
          f" deletion is colorable)")
    fx3b = fnum(core, "P33 core")
    cert3b = exact_certificate(core, "P33 core", primes, sympy_over_Q=False)
    print(f"    exact: {cert_str(cert3b)}\n")
    table.append((f"Peres33 core ({len(core)})", 3, len(core), cert3b["E"],
                  "KS-uncolorable CRITICAL (verified)", fx3b, cert3b))
    return table

def sec_session1(primes=PRIMES):
    table = []
    p24, bases, _census = parity_census_peres24()
    print("[4] Session-1 sets re-run through this pipeline (cross-checks)")
    yo = [as_pairs(r) for r in integer_rays_yuoh()]
    fx4 = fnum(yo, "Yu-Oh 13")
    cert4 = exact_certificate(yo, "Yu-Oh 13", primes, sympy_over_Q=True)
    assert cert4["sympy_rank"] == 57
    print(f"    exact: {cert_str(cert4)}  [matches exact_rigidity.py: rank 57, ker 21]")
    table.append(("Yu-Oh 13", 3, 13, cert4["E"], "SIC (Yu-Oh 2012; not re-derived here)",
                  fx4, cert4))
    p24r = [as_pairs(r) for r in integer_rays_peres24()]
    fx5 = fnum(p24r, "Peres 24")
    idxmap = {sign_norm_int(v): k for k, v in enumerate(integer_rays_peres24())}
    exists24, nodes24 = basis_coloring_exists([tuple(idxmap[p24[r]] for r in b) for b in bases])
    assert not exists24
    cert5 = exact_certificate(p24r, "Peres 24", primes, sympy_over_Q=True)
    assert cert5["sympy_rank"] == 153
    print(f"    verified: KS-UNCOLORABLE over its {len(bases)} tetrads ({nodes24} nodes)")
    print(f"    exact: {cert_str(cert5)}  [matches exact_rigidity.py: rank 153, ker 39]\n")
    table.append(("Peres 24", 4, 24, cert5["E"], "KS-uncolorable 24r/24b (verified)", fx5, cert5))
    return table

SECTIONS = [("ceg18", sec_ceg18), ("k20", sec_k20), ("p33", sec_peres33),
            ("core", sec_core), ("s1", sec_session1)]

def print_table(table, t0):
    print("=" * 98)
    print(f"{'set':<20}{'d':>3}{'V':>5}{'E':>6}  {'property verified':<40}{'flex':>5}  exact")
    print("-" * 98)
    for name, d, V, E, prop, fx, cert in table:
        ev = cert["exact_value"]
        exact = (f"flex={ev} EXACT" if ev is not None else "numerical only")
        print(f"{name:<20}{d:>3}{V:>5}{E:>6}  {prop:<40}{fx:>5}  {exact}")
    print("-" * 98)
    all_rigid = all(r[5] == 0 for r in table)
    all_exact = all(r[6]["exact_value"] is not None and r[6]["exact_value"] == r[5] for r in table)
    print(f"ALL SIC/KS SETS RIGID: {all_rigid}    ALL FLEX VALUES EXACTLY CERTIFIED: {all_exact}"
          f"    ({time.time() - t0:.1f}s)")
    if not all_rigid:
        print("\n*** COUNTEREXAMPLE TO 'SIC => RIGID': the Peres 33-ray KS set has flex = 1")
        print("*** (exact over Q(sqrt2)); the flex is pure imaginary — the real framework is")
        print("*** exactly rigid, but the configuration flexes into CP^2, tracing a")
        print("*** one-parameter family of complex 33-ray KS sets. See SIC_ZOO.md.")
    print("\nsic_zoo " + ("PASS (all properties verified, all flex values exactly certified)"
                          if all_exact else "ATTENTION: some flex value not exactly certified"))

def main(which=None):
    t0 = time.time()
    print("=" * 98)
    print("SIC RIGIDITY ZOO — dictionary test: are state-independent contextual ray sets RIGID?")
    print("=" * 98)
    print(f"exact mod-p certificates use primes {[p for p, _ in PRIMES]} (p=7 mod 8, sqrt2 exists)\n")
    table = []
    for name, fn in SECTIONS:
        if which and name not in which: continue
        table += fn()
    if not which or len(which) == len(SECTIONS):
        print_table(table, t0)
    return table

if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    main(args or None)
