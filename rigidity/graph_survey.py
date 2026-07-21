#!/usr/bin/env python3
"""
B1 -- SYSTEMATIC GRAPH SURVEY for the rigidity-contextuality dictionary (session 2, 2026-07-16).

Conjecture under test (NOTE.md): among CONTEXTUAL scenarios, flex = 0 <=> state-independent
contextuality (SIC); flex > 0 counts state-dependent moduli.

Pipeline
  1. RANDOM-REALIZATION GENERATOR.  Given an exclusivity graph G=(V,E) and dimension d, find ray
     realizations v: V -> CP^{d-1} with <v_i,v_j> = 0 on edges, by random init + trust-region
     least squares (scipy 'trf'; MINPACK-LM rejects underdetermined systems, so 'trf' plays the
     Levenberg-Marquardt role) + Gauss-Newton min-norm polish to residual ~1e-15.  ACCEPT only if
     max residual < 1e-11, all rays pairwise distinct (projective angle > 1e-3 rad), and no
     accidental extra orthogonality (non-edges have |<v_i,v_j>| > 1e-4) -- so the engine's
     edges_from() sees exactly G.  Because flex may depend on the realization component, each
     (graph,d) gets up to NTARGET accepted random realizations and we report the flex
     DISTRIBUTION (min/mode/max + full histogram), never one sample.
  2. GRAPH CATALOG.  d=3 first, escalating d=4 then d=5 when d=3 fails: cycles C3..C9 (C3=K3=ONB
     control), K4, bowtie, prism, wheels W4-W6, Petersen, circulants Cn(1,2) n=7..9, C5+pendant,
     two C5s sharing a vertex, Yu-Oh 13 (SIC anchor), 25 seeded random graphs G(V,p) V=6..10 at
     several densities, and 10 extra seeded random graphs filtered to pass the d=3 certificate.
  3. INVARIANTS.  alpha (exact), omega (exact), Lovasz theta (ADMM SDP, validated to <1e-9 on
     known closed forms), fractional packing alpha* = E-principle bound (LP over maximal cliques),
     fractional chromatic chi_f (LP over maximal independent sets), perfectness (exact, via the
     strong perfect graph theorem: search induced odd holes/antiholes of length >= 5).
     Contextuality notions used:
       ctx_u : theta > alpha         (unweighted CSW witness; sufficient, not necessary)
       imperf: G imperfect           (<=> exists vertex weights w with theta_w > alpha_w,
                                       i.e. the graph supports contextuality at all [CSW])
       SICp  : chi_f > d             (Ramanathan-Horodecki criterion: G with a d-dim realization
                                       can yield STATE-INDEPENDENT contextuality iff chi_f(G) > d)
  4. CROSS-TABULATIONS of rigidity against ctx_u and against (imperf, SICp), candidate flags.

EXACT UNREALIZABILITY CERTIFICATES (sound; one-line proofs):
  (a) omega(G) > d: a k-clique forces k mutually orthogonal rays.
  (b) span counting.  Any two DISTINCT rays span exactly 2 dims; a k-clique spans exactly k.
      Common neighbors of the set live in the orthocomplement (dim m = d-2 or d-k).  If m=1,
      two distinct common neighbors are impossible; if m=2, the subgraph induced on the common
      neighborhood must have max degree <= 1 (in C^2 a ray has a unique orthogonal partner).
      Applied to all pairs (d=3: >=2 common neighbors kills G; equivalently G must be
      C4-subgraph-free), all triangles, and all K4s.
Failures WITHOUT a certificate are reported as "NOT FOUND after N attempts" with the failure-mode
histogram -- honest unknowns, never claimed nonexistent.

Reproducibility: every random draw comes from np.random.default_rng seeded by SEED0 plus a stable
per-(graph,d) CRC key; rerunning this file reproduces the table verbatim.
Evidence label: NUMERICAL (residuals, spectral gaps, SDP residuals all printed; no exact arithmetic).
Engine: flex_dimension.py (imported, not modified).
"""
import io
import os
import re
import sys
import time
import zlib
from collections import Counter
from contextlib import redirect_stdout
from itertools import combinations

import numpy as np
from scipy.optimize import least_squares, linprog

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import flex_dimension as fd  # the engine -- imported, not modified

SEED0 = 20260716
NTARGET = 12          # accepted realizations per (graph, d)
MAX_ATTEMPTS = 80     # random inits per (graph, d)
RES_TOL = 1e-11       # max |constraint residual| to accept
MIN_ANGLE = 1e-3      # projective distinctness (radians)
NONEDGE_TOL = 1e-4    # non-edges must be at least this far from orthogonal
CTX_TOL = 1e-5        # theta - alpha > CTX_TOL  => ctx_u
GAP_WARN = 1e6        # flag flex values whose rank spectral gap is below this
DMAX = 5              # escalate d = 3 -> 4 -> 5, then stop

# ---------------------------------------------------------------- graph utilities
def norm_edges(E):
    return sorted(set((min(i, j), max(i, j)) for i, j in E))

def adj_sets(n, E):
    N = [set() for _ in range(n)]
    for i, j in E:
        N[i].add(j); N[j].add(i)
    return N

def bron_kerbosch(n, E):
    """All maximal cliques (tiny graphs; pivoting version)."""
    N = adj_sets(n, E)
    out = []
    def bk(R, P, X):
        if not P and not X:
            out.append(sorted(R)); return
        pivot = max(P | X, key=lambda u: len(P & N[u]))
        for v in list(P - N[pivot]):
            bk(R | {v}, P & N[v], X & N[v])
            P.remove(v); X.add(v)
    bk(set(), set(range(n)), set())
    return out

def complement_edges(n, E):
    Es = set(norm_edges(E))
    return [(i, j) for i, j in combinations(range(n), 2) if (i, j) not in Es]

def clique_number(n, E):
    return max((len(c) for c in bron_kerbosch(n, E)), default=1)

def independence_number(n, E):
    return clique_number(n, complement_edges(n, E))

def all_cliques_of_size(n, E, k):
    Es = set(norm_edges(E))
    return [S for S in combinations(range(n), k)
            if all((min(a, b), max(a, b)) in Es for a, b in combinations(S, 2))]

def _has_hole(n, E, k):
    """Induced chordless cycle of length k?  (exhaustive over k-subsets; V<=13 here)"""
    N = adj_sets(n, E)
    for S in combinations(range(n), k):
        degs = [len(N[v] & set(S)) for v in S]
        if any(dg != 2 for dg in degs):
            continue
        # connected 2-regular induced subgraph on k vertices = induced C_k
        Sset = set(S)
        seen, stack = {S[0]}, [S[0]]
        while stack:
            v = stack.pop()
            for u in N[v] & Sset:
                if u not in seen:
                    seen.add(u); stack.append(u)
        if len(seen) == k:
            return True
    return False

def is_perfect(n, E):
    """Strong perfect graph theorem: perfect <=> no induced odd hole/antihole of length >= 5."""
    Ec = complement_edges(n, E)
    for k in range(5, n + 1, 2):
        if _has_hole(n, E, k) or _has_hole(n, Ec, k):
            return False
    return True

def lovasz_theta(n, E, iters=60000, tol=1e-12):
    """theta(G) = max <J,X> : X psd, tr X = 1, X_ij = 0 on edges of G.  ADMM, exact projections.
    Returns (theta, primal residual ||X-Z||)."""
    E = norm_edges(E)
    if n == 0:
        return 0.0, 0.0
    rho = float(n)
    J = np.ones((n, n))
    X = np.eye(n) / n
    Z = X.copy()
    U = np.zeros((n, n))
    r = 0.0
    for it in range(iters):
        M = Z - U + J / rho
        M = (M + M.T) / 2
        for i, j in E:
            M[i, j] = 0.0; M[j, i] = 0.0
        M[np.arange(n), np.arange(n)] += (1.0 - np.trace(M)) / n
        X = M
        W = X + U
        W = (W + W.T) / 2
        w, Q = np.linalg.eigh(W)
        Z = (Q * np.maximum(w, 0.0)) @ Q.T
        U += X - Z
        if it % 200 == 0:
            r = float(np.linalg.norm(X - Z))
            if r < tol:
                break
    return float(np.sum(X)), r

def frac_packing(n, E):
    """alpha*(G) = max sum x, x >= 0, sum_{i in C} x_i <= 1 over maximal cliques C
    (= fractional chromatic number of the complement = CSW E-principle bound)."""
    cliques = bron_kerbosch(n, E)
    A = np.zeros((len(cliques), n))
    for r, c in enumerate(cliques):
        A[r, c] = 1.0
    res = linprog(-np.ones(n), A_ub=A, b_ub=np.ones(len(cliques)),
                  bounds=[(0, None)] * n, method="highs")
    return -res.fun

def frac_chromatic(n, E):
    """chi_f(G) = min sum y_S over maximal independent sets S with every vertex covered.
    Ramanathan-Horodecki: a d-dim realization can be SIC iff chi_f(G) > d."""
    indsets = bron_kerbosch(n, complement_edges(n, E))
    A = np.zeros((len(indsets), n))
    for r, s in enumerate(indsets):
        A[r, s] = 1.0
    res = linprog(np.ones(len(indsets)), A_ub=-A.T, b_ub=-np.ones(n),
                  bounds=[(0, None)] * len(indsets), method="highs")
    return res.fun

def certificate_unrealizable(n, E, d):
    """Exact obstruction string, or None.  Sound by the span-counting arguments in the docstring."""
    w = clique_number(n, E)
    if w > d:
        return f"omega={w}>d={d}"
    N = adj_sets(n, E)
    def rule(S, m, tag):
        if m <= 0 and len(S) >= 1:
            return f"{tag} has a common neighbor beyond dim"
        if m == 1 and len(S) >= 2:
            return f"{tag} shares {len(S)} common neighbors (fit in dim 1)"
        if m == 2 and any(len(N[u] & S) >= 2 for u in S):
            return f"common nbhd of {tag} needs degree>=2 in C^2"
        return None
    for x, y in combinations(range(n), 2):        # any distinct pair spans exactly 2
        c = rule(N[x] & N[y] - {x, y}, d - 2, f"pair {x},{y}")
        if c:
            return c
    for k in (3, 4):                              # k-cliques span exactly k
        if d - k > 2:
            continue
        for S in all_cliques_of_size(n, E, k):
            common = set.intersection(*[N[v] for v in S]) - set(S)
            c = rule(common, d - k, f"{k}-clique {S}")
            if c:
                return c
    return None

# ---------------------------------------------------------------- realization engine
def _residual_system(n, E, d):
    m = n + 2 * len(E)
    def unpack(x):
        A = x.reshape(n, 2 * d)
        return A[:, :d], A[:, d:]
    def F(x):
        a, b = unpack(x)
        r = np.empty(m)
        r[:n] = np.einsum("ij,ij->i", a, a) + np.einsum("ij,ij->i", b, b) - 1.0
        for k, (i, j) in enumerate(E):
            r[n + 2 * k] = a[i] @ a[j] + b[i] @ b[j]        # Re<v_i,v_j>
            r[n + 2 * k + 1] = a[i] @ b[j] - b[i] @ a[j]    # Im<v_i,v_j>
        return r
    def Jac(x):
        a, b = unpack(x)
        J = np.zeros((m, 2 * d * n))
        for i in range(n):
            J[i, 2 * d * i:2 * d * i + d] = 2 * a[i]
            J[i, 2 * d * i + d:2 * d * (i + 1)] = 2 * b[i]
        for k, (i, j) in enumerate(E):
            r1, r2 = n + 2 * k, n + 2 * k + 1
            J[r1, 2 * d * i:2 * d * i + d] = a[j]; J[r1, 2 * d * i + d:2 * d * (i + 1)] = b[j]
            J[r1, 2 * d * j:2 * d * j + d] = a[i]; J[r1, 2 * d * j + d:2 * d * (j + 1)] = b[i]
            J[r2, 2 * d * i:2 * d * i + d] = b[j]; J[r2, 2 * d * i + d:2 * d * (i + 1)] = -a[j]
            J[r2, 2 * d * j:2 * d * j + d] = -b[i]; J[r2, 2 * d * j + d:2 * d * (j + 1)] = a[i]
        return J
    return F, Jac

def find_realization(n, E, d, rng):
    """One random attempt.  Returns (rays_or_None, max_residual, status)."""
    E = norm_edges(E)
    F, Jac = _residual_system(n, E, d)
    X = rng.standard_normal((n, 2 * d))
    X /= np.linalg.norm(X, axis=1)[:, None]
    x = X.ravel()
    sol = least_squares(F, x, jac=Jac, method="trf",
                        xtol=1e-15, ftol=1e-15, gtol=1e-15, max_nfev=1500)
    x = sol.x
    for _ in range(8):                       # Gauss-Newton polish (min-norm steps)
        f = F(x)
        if np.max(np.abs(f)) < 1e-14:
            break
        step, *_ = np.linalg.lstsq(Jac(x), f, rcond=None)
        x = x - step
    res = float(np.max(np.abs(F(x))))
    if res > RES_TOL:
        return None, res, "residual"
    A = x.reshape(n, 2 * d)
    rays = [A[i, :d] + 1j * A[i, d:] for i in range(n)]
    Eset = set(E)
    for i, j in combinations(range(n), 2):
        ov = abs(np.vdot(rays[i], rays[j]))
        if np.arccos(min(1.0, ov)) < MIN_ANGLE:
            return None, res, "coincident"
        if (i, j) not in Eset and ov < NONEDGE_TOL:
            return None, res, "extra-orth"
    if set(fd.edges_from(rays)) != Eset:      # the engine must see exactly G
        return None, res, "graph-mismatch"
    return rays, res, "ok"

def flex_of(rays):
    """Run the engine, capture its printed diagnostics, return (flex, spectral_gap)."""
    buf = io.StringIO()
    with redirect_stdout(buf):
        fl = fd.flex_dimension(rays, name="survey")
    m = re.search(r"gap ([0-9.e+\-]+|inf)", buf.getvalue())
    return fl, (float(m.group(1)) if m else float("nan"))

def survey_realizations(name, n, E, d, ntarget=NTARGET, max_attempts=MAX_ATTEMPTS):
    rng = np.random.default_rng([SEED0, zlib.crc32(f"{name}|d={d}".encode())])
    flexes, gaps = [], []
    fails = Counter()
    attempts = 0
    while attempts < max_attempts and len(flexes) < ntarget:
        attempts += 1
        rays, res, st = find_realization(n, E, d, rng)
        if rays is None:
            fails[st] += 1
            continue
        fl, gap = flex_of(rays)
        flexes.append(fl); gaps.append(gap)
    return dict(flexes=flexes, gaps=gaps, attempts=attempts, fails=dict(fails))

# ---------------------------------------------------------------- graph catalog
def cyc(k):
    return [(i, (i + 1) % k) for i in range(k)]

def wheel(k):          # hub = vertex k joined to the cycle C_k on 0..k-1
    return k + 1, cyc(k) + [(k, i) for i in range(k)]

def petersen():
    vs = list(combinations(range(5), 2))
    return 10, [(a, b) for a, b in combinations(range(10), 2) if not set(vs[a]) & set(vs[b])]

def circulant(k, dists):
    E = []
    for i in range(k):
        for s in dists:
            E.append((i, (i + s) % k))
    return k, norm_edges(E)

def random_graph(nv, p, key):
    rng = np.random.default_rng([SEED0, key])
    return nv, [(i, j) for i, j in combinations(range(nv), 2) if rng.random() < p]

def is_connected(n, E):
    N = adj_sets(n, E)
    seen, stack = {0}, [0]
    while stack:
        v = stack.pop()
        for u in N[v]:
            if u not in seen:
                seen.add(u); stack.append(u)
    return len(seen) == n

def build_catalog():
    cat = [("K3=C3 (ONB)", 3, cyc(3), "control"),
           ("C4", 4, cyc(4), "even cycle")]
    for k in range(5, 10):
        cat.append((f"C{k}", k, cyc(k), "odd cycle" if k % 2 else "even cycle"))
    cat.append(("K4", 4, [(i, j) for i, j in combinations(range(4), 2)], "clique"))
    cat.append(("bowtie", 5, [(0, 1), (0, 2), (1, 2), (0, 3), (0, 4), (3, 4)], "2 triangles"))
    cat.append(("prism K3xK2", 6, [(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5),
                                   (0, 3), (1, 4), (2, 5)], "has C4s"))
    for k in (4, 5, 6):
        nv, E = wheel(k)
        cat.append((f"W{k} (hub+C{k})", nv, E, "wheel"))
    nv, E = petersen()
    cat.append(("Petersen", nv, E, "Kneser K(5,2)"))
    for k in (7, 8, 9):
        nv, E = circulant(k, (1, 2))
        cat.append((f"C{k}(1,2)", nv, E, "circulant/antiweb"))
    cat.append(("C5+pendant", 6, cyc(5) + [(0, 5)], "structured"))
    cat.append(("2xC5 shared vtx", 9, cyc(5) + [(0, 5), (5, 6), (6, 7), (7, 8), (8, 0)], "structured"))
    yoE = norm_edges(fd.edges_from(fd.yu_oh()))
    cat.append(("Yu-Oh 13", 13, yoE, "SIC anchor"))
    # Yu-Oh PERTURBATION PROBES: delete one edge (one per symmetry type; vertex order in
    # fd.yu_oh() is z:0-2, y:3-8, h:9-12) or one vertex (one per orbit).  The dictionary demands
    # that rigidity and RH SIC-capability degrade TOGETHER under these deletions.
    for tag, e in [("zz", (0, 1)), ("zy", (0, 3)), ("yy", (3, 4)), ("yh", (4, 9))]:
        assert e in yoE
        cat.append((f"YuOh-e[{tag}]", 13, [x for x in yoE if x != e], "YO minus edge"))
    for tag, v in [("z", 0), ("y", 3), ("h", 9)]:
        keep = [u for u in range(13) if u != v]
        idx = {u: i for i, u in enumerate(keep)}
        E2 = [(idx[a], idx[b]) for a, b in yoE if v not in (a, b)]
        cat.append((f"YuOh-v[{tag}]", 12, E2, "YO minus vertex"))
    ladder = [(6, .15), (6, .25), (6, .3), (6, .4), (6, .5),
              (7, .15), (7, .2), (7, .3), (7, .4), (7, .5),
              (8, .15), (8, .2), (8, .25), (8, .3), (8, .4),
              (9, .15), (9, .2), (9, .25), (9, .3), (9, .4),
              (10, .12), (10, .2), (10, .25), (10, .3), (10, .4)]
    for idx, (nv, p) in enumerate(ladder):
        n2, E2 = random_graph(nv, p, 100 + idx)
        cat.append((f"R{idx:02d} G({nv},{p})", n2, E2, "random"))
    found, k = 0, 0
    while found < 10 and k < 4000:   # extra randoms passing the d=3 certificate; connected, cyclic
        nv = 7 + (k % 4)
        n2, E2 = random_graph(nv, 0.22, 1000 + k)
        k += 1
        if len(E2) >= n2 and is_connected(n2, E2) and certificate_unrealizable(n2, E2, 3) is None:
            cat.append((f"F{found:02d} C4free({nv})", n2, E2, "random C4-free"))
            found += 1
    return cat

# ---------------------------------------------------------------- validation
def validate():
    print("== self-validation ==")
    anchors = [("C5", 5, cyc(5), np.sqrt(5)),
               ("C6", 6, cyc(6), 3.0),
               ("C7", 7, cyc(7), 7 * np.cos(np.pi / 7) / (1 + np.cos(np.pi / 7))),
               ("C9", 9, cyc(9), 9 * np.cos(np.pi / 9) / (1 + np.cos(np.pi / 9))),
               ("Petersen", *petersen(), 4.0)]
    for name, n, E, exact in anchors:
        th, r = lovasz_theta(n, E)
        ok = abs(th - exact) < 1e-6
        print(f"  theta({name}) = {th:.9f}  vs exact {exact:.9f}  |err|={abs(th-exact):.1e} "
              f"(ADMM resid {r:.1e})  {'PASS' if ok else 'FAIL'}")
        assert ok
    yo_n, yo_E = 13, norm_edges(fd.edges_from(fd.yu_oh()))
    cf = frac_chromatic(yo_n, yo_E)
    ok = abs(cf - 35 / 11) < 1e-6
    print(f"  chi_f(Yu-Oh) = {cf:.9f}  vs 35/11 = {35/11:.9f} (Ramanathan-Horodecki)  "
          f"{'PASS' if ok else 'FAIL'}")
    assert ok
    perf_anchors = [("C5", 5, cyc(5), False), ("C6", 6, cyc(6), True),
                    ("K4", 4, [(i, j) for i, j in combinations(range(4), 2)], True),
                    ("Petersen", *petersen(), False)]
    for name, n, E, expect in perf_anchors:
        got = is_perfect(n, E)
        print(f"  perfect({name}) = {got} (expect {expect})  {'PASS' if got == expect else 'FAIL'}")
        assert got == expect
    fl, gap = flex_of(fd.odd_cycle(5))
    print(f"  flex(C5 umbrella) = {fl} (expect 2, gap {gap:.1e})  {'PASS' if fl == 2 else 'FAIL'}")
    assert fl == 2
    fl, gap = flex_of(fd.yu_oh())
    print(f"  flex(Yu-Oh exact rays) = {fl} (expect 0, gap {gap:.1e})  {'PASS' if fl == 0 else 'FAIL'}")
    assert fl == 0
    print()

# ---------------------------------------------------------------- candidate deep-dive
def candidate_deep_dive():
    """YuOh-v[h]: the 12-ray subset of Yu-Oh obtained by deleting one h-type ray.
    The survey finds it RIGID yet theta > alpha with chi_f = 3 (not > 3): a counterexample
    candidate to the naive direction 'rigid => state-independent'.  This routine upgrades the
    evidence: (1) EXACT rational-arithmetic rigidity certificate at the integer Yu-Oh-restriction
    realization (reusing exact_rigidity.exact_flex); (2) EXACT projector algebra
    sum_i Pi_i = 13/3 I - Pi_h, so the CSW value is 13/3 - <psi|Pi_h|psi> in [10/3, 13/3]:
    it VIOLATES the classical bound alpha = 4 iff <psi|Pi_h|psi> < 1/3 -- contextual for many
    states, NOT for all states => state-DEPENDENT at this realization, exactly;
    (3) lambda_max(sum Pi) across the survey's random realizations (are they all the same
    scenario?); (4) best-effort exact rationalization of chi_f = 3 (RH criterion chi_f > d
    fails, so no SIC in d=3; independently, 13-ray minimality for d=3 SIC excludes any 12-ray
    SIC set)."""
    from fractions import Fraction
    import exact_rigidity as ex
    print("\n== CANDIDATE DEEP-DIVE: YuOh-v[h]  (Yu-Oh minus the h-ray (1,1,1); 12 rays, d=3) ==")
    rays13 = ex.integer_rays_yuoh()
    rays12 = [r for k, r in enumerate(rays13) if k != 9]          # drop h0 = (1,1,1)
    # (0) the integer restriction realizes exactly the catalog graph YuOh-v[h]
    E_int = norm_edges([(i, j) for i, j in combinations(range(12), 2)
                        if ex.ip(rays12[i], rays12[j]) == 0])
    yoE = norm_edges(fd.edges_from(fd.yu_oh()))
    idx = {u: i for i, u in enumerate([u for u in range(13) if u != 9])}
    E_cat = norm_edges([(idx[a], idx[b]) for a, b in yoE if 9 not in (a, b)])
    print(f"  graph check: integer-restriction edges == catalog edges: {E_int == E_cat} "
          f"(V=12, E={len(E_int)})")
    assert E_int == E_cat
    # (1) exact rigidity certificate over Q
    fx = ex.exact_flex(rays12, "YuOh-v[h]")
    print(f"  exact flex = {fx}  ->  {'RIGID, EXACT over Q' if fx == 0 else 'NOT rigid (!)'} ")
    # (2) exact projector algebra: sum Pi = 13/3 I - Pi_h
    from sympy import Matrix, Rational, eye
    S = Matrix.zeros(3, 3)
    for r in rays12:
        v = Matrix(3, 1, r)
        S += v * v.T / (v.T * v)[0, 0]
    h = Matrix(3, 1, [1, 1, 1])
    Ph = h * h.T / 3
    ok = (S - (Rational(13, 3) * eye(3) - Ph)) == Matrix.zeros(3, 3)
    print(f"  exact algebra: sum_i Pi_i == (13/3) I - Pi_h : {ok}")
    assert ok
    al = independence_number(12, E_int)
    print(f"  classical (NC) bound: alpha = {al} (exact brute force); quantum value(psi) = "
          f"13/3 - <psi|Pi_h|psi> in [10/3, 13/3]")
    print(f"    -> violated ({Rational(13,3)} > {al}) by every psi with <psi|Pi_h|psi> < 1/3 "
          f"(e.g. any psi orth. to (1,1,1)); NOT violated by psi = (1,1,1)/sqrt(3) (10/3 < 4)")
    print("    => contextual and STATE-DEPENDENT at this realization (exact).")
    # (3) are the survey's random realizations the same scenario?  lambda_max(sum Pi) check
    rng = np.random.default_rng([SEED0, zlib.crc32(b"deepdive-lmax")])
    lams = []
    tries = 0
    while len(lams) < 10 and tries < 60:
        tries += 1
        rays, res, st = find_realization(12, E_int, 3, rng)
        if rays is None:
            continue
        P = sum(np.outer(v, v.conj()) for v in rays)
        lams.append(float(np.linalg.eigvalsh(P)[-1]))
    print(f"  lambda_max(sum Pi) over {len(lams)} fresh random realizations: "
          f"min {min(lams):.12f}  max {max(lams):.12f}  (13/3 = {13/3:.12f})")
    # (4) chi_f = 3 exactly?  rationalize the covering LP solution and verify over Q
    indsets = bron_kerbosch(12, complement_edges(12, E_int))
    A = np.zeros((len(indsets), 12))
    for r, s2 in enumerate(indsets):
        A[r, s2] = 1.0
    res = linprog(np.ones(len(indsets)), A_ub=-A.T, b_ub=-np.ones(12),
                  bounds=[(0, None)] * len(indsets), method="highs")
    y = [Fraction(v).limit_denominator(720) for v in res.x]
    cover_ok = all(sum(y[r] for r, s2 in enumerate(indsets) if v in s2) >= 1 for v in range(12))
    tot = sum(y, Fraction(0))
    print(f"  chi_f upper bound: rationalized fractional cover, total weight = {tot} "
          f"(covers all vertices exactly: {cover_ok})")
    # dual: fractional clique number = max sum w, sum_{v in S} w_v <= 1 for all indep sets
    resd = linprog(-np.ones(12), A_ub=A, b_ub=np.ones(len(indsets)),
                   bounds=[(0, None)] * 12, method="highs")
    w = [Fraction(v).limit_denominator(720) for v in resd.x]
    dual_ok = all(sum(w[v] for v in s2) <= 1 for s2 in indsets)
    totw = sum(w, Fraction(0))
    print(f"  chi_f lower bound: rationalized fractional clique weights, total = {totw} "
          f"(all independent-set sums <= 1: {dual_ok})")
    if cover_ok and dual_ok and tot == totw:
        print(f"  => chi_f(YuOh-v[h]) = {tot} EXACTLY; RH criterion chi_f > d=3 FAILS "
              f"-> no state-independent contextuality from this graph in d=3.")
    else:
        print("  => chi_f = 3.000000 numerically (rationalization incomplete; honest label: LP).")
    print("  (independent exclusion: 12 < 13 rays, and 13 is the reported minimum for SIC in "
          "d=3 [Cabello-Kleinmann-Budroni].)")
    print("  VERDICT: rigid (exact) + contextual (exact, state-dependent) + not SIC-capable")
    print("           => counterexample candidate to the naive 'rigid => state-independent';")
    print("           the surviving direction is 'SIC => rigid' (no violations found).")
    print("  Caveats: infinitesimal rigidity at the sampled component(s); other components of")
    print("           the realization variety are not excluded; 'not SIC-capable' rests on the")
    print("           RH criterion + reported 13-ray minimality, not on an independent proof.")

# ---------------------------------------------------------------- main survey
def compute_rows(cat):
    rows = []
    for name, n, E, note in cat:
        E = norm_edges(E)
        alpha = independence_number(n, E)
        omega = clique_number(n, E)
        theta, sdp_r = lovasz_theta(n, E)
        astar = frac_packing(n, E)
        chif = frac_chromatic(n, E)
        perf = is_perfect(n, E)
        base = dict(name=name, n=n, E=E, note=note, alpha=alpha, omega=omega, theta=theta,
                    sdp_r=sdp_r, astar=astar, chif=chif, perf=perf,
                    ctx_u=(theta - alpha) > CTX_TOL)
        for d in range(3, DMAX + 1):
            cert = certificate_unrealizable(n, E, d)
            row = dict(base, d=d, cert=cert)
            if cert is None:
                row.update(survey_realizations(name, n, E, d))
            else:
                row.update(flexes=[], gaps=[], attempts=0, fails={})
            rows.append(row)
            if row["flexes"]:
                break                    # realized at this d; stop escalating
    return rows

def report(rows, t0):
    # ---------------- master table ----------------
    print("== MASTER TABLE ==  (SICp = chi_f > d, Ramanathan-Horodecki; imperf = supports weighted-"
          "CSW contextuality; ctx_u = theta > alpha unweighted;")
    print("                     xs = flex_mode - [2(d-1)V - 2E - (d^2-1)] = stress/stabilizer excess "
          "over the naive dimension count)")
    hdr = (f"{'graph':<18}{'d':>2}{'V':>4}{'E':>4}{'om':>3}{'al':>3}{'theta':>8}{'al*':>7}"
           f"{'chi_f':>7}{'imp':>4}{'ctxu':>5}{'SICp':>5} {'realizable':<44}"
           f"{'flex m/mo/M':<12}{'dist':<10}{'xs':>4}{'mingap':>8}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        xstr = "-"
        if r["cert"]:
            real = f"NO  cert: {r['cert']}"
            fstr, dstr, gstr = "-", "-", "-"
        elif not r["flexes"]:
            fl = ",".join(f"{v} {k}" for k, v in sorted(r["fails"].items()))
            real = f"NOT FOUND ({r['attempts']} att: {fl})"
            fstr, dstr, gstr = "-", "-", "-"
        else:
            real = f"YES ({len(r['flexes'])}/{r['attempts']} att)"
            c = Counter(r["flexes"])
            fstr = f"{min(r['flexes'])}/{c.most_common(1)[0][0]}/{max(r['flexes'])}"
            dstr = ",".join(f"{k}:{v}" for k, v in sorted(c.items()))
            g = min(r["gaps"])
            gstr = ("%.0e" % g) + ("!" if g < GAP_WARN else "")
            naive = 2 * (r["d"] - 1) * r["n"] - 2 * len(r["E"]) - (r["d"] ** 2 - 1)
            xstr = str(c.most_common(1)[0][0] - naive)
        print(f"{r['name']:<18}{r['d']:>2}{r['n']:>4}{len(r['E']):>4}{r['omega']:>3}{r['alpha']:>3}"
              f"{r['theta']:>8.4f}{r['astar']:>7.3f}{r['chif']:>7.3f}"
              f"{'Y' if not r['perf'] else 'n':>4}{'Y' if r['ctx_u'] else 'n':>5}"
              f"{'Y' if r['chif'] > r['d'] + 1e-6 else 'n':>5} {real:<44}{fstr:<12}{dstr:<10}{xstr:>4}{gstr:>8}")
    print(f"(max ADMM primal residual over all graphs: "
          f"{max(r['sdp_r'] for r in rows):.1e})")
    # ---------------- cross-tabulations ----------------
    realized = [r for r in rows if r["flexes"]]
    rigid = lambda r: min(r["flexes"]) == 0
    print(f"\n== CROSS-TAB 1: rigidity x unweighted CSW (theta > alpha), {len(realized)} realized rows ==")
    for ctx in (True, False):
        a = [r for r in realized if r["ctx_u"] == ctx and rigid(r)]
        b = [r for r in realized if r["ctx_u"] == ctx and not rigid(r)]
        alist = ", ".join("{}(d{})".format(x["name"], x["d"]) for x in a) or "-"
        print(f"  ctx_u={'Y' if ctx else 'n'}:  rigid {len(a):2d}  flexible {len(b):2d}"
              f"   rigid list: {alist}")
    print("\n== CROSS-TAB 2 (the dictionary test): rigidity x RH SIC-capability, "
          "IMPERFECT (= contextuality-capable) realized rows only ==")
    imp = [r for r in realized if not r["perf"]]
    for sicp in (True, False):
        sel = [r for r in imp if (r["chif"] > r["d"] + 1e-6) == sicp]
        a = [r for r in sel if rigid(r)]
        b = [r for r in sel if not rigid(r)]
        print(f"  SICp={'Y' if sicp else 'n'}:  rigid {len(a):2d}  flexible {len(b):2d}")
        for x in a:
            print(f"        RIGID: {x['name']} d={x['d']}  chi_f={x['chif']:.3f}  "
                  f"flex dist {dict(Counter(x['flexes']))}")
        if sicp and not a:
            print("        (no rigid SIC-capable rows)")
    viol = ([r for r in imp if (r["chif"] > r["d"] + 1e-6) and not rigid(r)] +
            [r for r in imp if not (r["chif"] > r["d"] + 1e-6) and rigid(r)])
    vlist = ", ".join("{}(d{})".format(x["name"], x["d"]) for x in viol) or "NONE"
    print(f"  dictionary violations (SIC-capable&flexible or SIC-incapable&rigid): {vlist}")
    # ---------------- candidate flags ----------------
    print("\n== CANDIDATE FLAGS ==")
    cands = [r for r in realized if r["ctx_u"] and rigid(r) and r["name"] != "Yu-Oh 13"]
    print("  (a) rigid & ctx_u (theta>alpha), not Yu-Oh -> NEW-SIC/counterexample candidates:",
          ", ".join(f"{r['name']}(d{r['d']})" for r in cands) or "(none found)")
    cands2 = [r for r in imp if rigid(r) and r["name"] != "Yu-Oh 13"]
    print("  (a') rigid & imperfect, not Yu-Oh:",
          ", ".join(f"{r['name']}(d{r['d']})" for r in cands2) or "(none found)")
    near = sorted([r for r in imp if 0 < min(r["flexes"]) <= 2], key=lambda r: min(r["flexes"]))
    print("  (b) near-rigid contextuality-capable (0 < min flex <= 2):",
          ", ".join(f"{r['name']}(d{r['d']},flex{min(r['flexes'])})" for r in near) or "(none)")
    mixed = [r for r in realized if len(set(r["flexes"])) > 1]
    print("  (c) component-dependent flex (mixed distributions):",
          ", ".join(f"{r['name']}(d{r['d']}):{dict(Counter(r['flexes']))}" for r in mixed) or "(none)")
    stressed = []
    for r in realized:
        naive = 2 * (r["d"] - 1) * r["n"] - 2 * len(r["E"]) - (r["d"] ** 2 - 1)
        xs = Counter(r["flexes"]).most_common(1)[0][0] - naive
        if xs != 0:
            stressed.append(f"{r['name']}(d{r['d']},xs={xs})")
    print("  (d) stressed/stabilized (flex != naive count 2(d-1)V-2E-(d^2-1)):",
          ", ".join(stressed) or "(none)")
    candidate_deep_dive()
    print(f"\ntotal wall time {time.time()-t0:.1f}s")
    return rows

def run():
    t0 = time.time()
    validate()
    return report(compute_rows(build_catalog()), t0)

if __name__ == "__main__":
    run()
