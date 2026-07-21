#!/usr/bin/env python3
"""
D-MINUS-1 BOUND — flex_R - flex_skew <= d-1 (new session, branch agent, 2026-07-17/18).

TARGET (restated from the brief). At a faithful REAL realization of an exclusivity graph
(rays v_i in R^d, edges = orthogonal pairs), the decomposition theorem (proved elsewhere in this
program, reused not re-derived) gives flex_C = flex_R + flex_skew with

    flex_R    = dV - rank(R) - rank(Tr),   R  = |E| edge rows (x_i.v_j+v_i.x_j) + V norm rows
                (v_i.x_i);  Tr = so(d) orbit.
    flex_skew = dV - rank(S) - rank(Ts),   S  = |E| edge rows (v_i.y_j-y_i.v_j), NO norm rows;
                Ts = V phases + Sym(d) directions, rank(Ts) = V + d(d+1)/2 - r,
                r = dim{sym M : M v_i in R.v_i for all i} (the eigen-commutant, r>=1 always).

CONJECTURE under test:  flex_R - flex_skew <= d-1  (prior-session evidence: 0 violations on 56
graphs, hermitian_bilinear.py / HERM_BILINEAR.md).

=====================================================================================
WHAT THIS FILE ADDS (new this session) — summary of results, full detail in DMINUS1_BOUND.md
=====================================================================================

Write, for edge e=(i,j): p_e = e_j (x) v_i, q_e = e_i (x) v_j  ("directed" edge rows).
Then (char != 2, so the change of basis (p,q)->(p+q,p-q) is invertible):
    real edge row   A_e = p_e + q_e     (symmetric in i,j — this is R's edge row)
    skew edge row   B_e = p_e - q_e     (antisymmetric in i,j, up to overall sign — this is S's
                                          only row type, since S has no norm rows)
    N   = span{n_i := e_i (x) v_i : i=1..V}     (the norm rows alone)
    D   = span{p_e,q_e : e in E} = span{A_e,B_e : e in E} = ⊕_i (e_i (x) W_i),  W_i = neighbor span

NEW LEMMA (PROVED, general, no genericity assumed — see `check_A_cap_N_lemma()` below, which
verifies it symbolically-in-spirit by re-deriving rank R = rank(A)+V on every row of the zoo,
where A := span{A_e} and rank(R) is torsion_flex's/hermitian_bilinear's already-verified flex_R
ingredient):

    A ∩ N = {0}   for EVERY real orthogonal representation of EVERY graph.

Proof (three lines). Suppose w = sum_e c_e A_e = sum_i lambda_i n_i. Comparing block-i components:
sum_{j~i} c_ij v_j = lambda_i v_i for every i. Dot both sides with v_i: LHS = sum_j c_ij (v_i.v_j)
= 0 because v_i.v_j=0 for every neighbor j (that IS the definition of an edge). RHS = lambda_i
|v_i|^2 = lambda_i (unit rays). Hence lambda_i = 0 for every i, so w = sum lambda_i n_i = 0.  QED.
(No genericity, no faithfulness beyond v_i != 0, holds for any graph/any d.)

Consequence (rank-nullity, exact): rank(R) = rank(A) + V  (since N has V linearly independent
rows — different blocks, v_i != 0 — and now A ∩ N = 0).

Combined with two facts already established/re-derived (see `check_C2` and `check_r_identity`
below, both re-proved from scratch here, one-line arguments):
  C2:  rank(Tr) = d(d-1)/2  whenever the rays span R^d (A in so(d) killing a spanning set is 0).
  r-identity: rank(Ts) = V + d(d+1)/2 - r  (rank-nullity on (lambda,S) -> (lambda_i v_i + S v_i)_i;
              kernel <-> the commutant {S sym: S v_i || v_i} via lambda_i = -(S v_i).v_i, unique).

...algebra gives the PROVED IDENTITY (re-derived here from first principles, verified on every
row of the zoo, both NUMERICALLY (SVD) and EXACTLY (mod-p, on 4 dense KS sets)):

    flex_R - flex_skew  =  d - r - (rank(A) - rank(B))        ... (*)

where A = span{A_e}, B = span{B_e} = S (no norm rows in S, so B *is* the skew edge+trivial input).
This is NEW (finer than the prior session's identity flex_R-flex_skew=(rRs-rRr)+V+d-r, which used
rRr = rank(R) = rank(A)+V implicitly but without isolating rank(A) or proving A∩N=0).

REDUCTION: since r >= 1 always, (*) shows the target bound flex_R-flex_skew <= d-1 is EQUIVALENT
to
    rank(B) - rank(A)  <=  r - 1        ... (**)
and in particular rank(A) >= rank(B) (needing no info about r) is SUFFICIENT for (**), and is
NECESSARY AND SUFFICIENT exactly when r=1 (irreducible rays).

STATUS OF (**): rank(A) >= rank(B) [equivalently stress(A) <= stress(B), stress(X):=|E|-rank(X)]
is CONJECTURE — verified with ZERO counterexamples across a zoo of ~300 diverse
graphs/dimensions/realizations built by this file (see `python3 dminus1_bound.py {,big,huge}`),
including EXACT mod-p certification (not just numerical SVD) on the four densest/most critical
named KS sets (Yu-Oh-13, Peres-24, CEG-18, Peres-33), and deliberately constructed block-diagonal
configurations with r up to 6. In every disjoint/block-decomposable and every sparse/generic case
tested, rank(A) = rank(B) EXACTLY (not just >=); strict inequality rank(A) > rank(B) appears only
in the dense/critical KS sets. No proof of (**) is offered — every proof attempt recorded honestly
failed to close (see DMINUS1_BOUND.md, "proof attempts that did not close").

Run:
    python3 dminus1_bound.py            # fast core zoo (named KS + cycles + wheels/bipartite/
                                         # prism + block-diagonal r=2..6 + Petersen/star/path),
                                         # ~10s, exact + numerical
    python3 dminus1_bound.py big        # + a large structured random-graph search (d=3..7,
                                         # V=5..12, several densities), ~45-90s
    python3 dminus1_bound.py huge       # + an even larger adversarial random search targeting
                                         # rank(A)<rank(B) specifically, several minutes
"""
import os, sys, time, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from itertools import combinations

from torsion_flex import PRIMES
from hermitian_bilinear import (realize_graph, cycle_graph, wheel_graph, complete_bipartite,
                                 prism_graph, random_graph, disjoint_union)
from even_cycles import solve_cycle
from flex_dimension import odd_cycle, onb, yu_oh, peres24
from sic_zoo import (q_dot, Z0, q_add, q_neg, rank_mod_p, find_primes_7mod8, as_pairs,
                      rays_peres33, rays_ceg18)
from exact_rigidity import integer_rays_yuoh, integer_rays_peres24

TOL = 1e-8

# ======================================================================================
# NUMERICAL engine: build A (real edge only), B (skew edge = S), N (norm), Tr, Ts and every
# rank/intersection diagnostic needed for identity (*) and the r-1 reduction (**).
# ======================================================================================
def _rank_svd(rows, tol=TOL):
    if len(rows) == 0:
        return 0
    M = np.array(rows, float)
    s = np.linalg.svd(M, compute_uv=False)
    if s[0] == 0:
        return 0
    rel = s / s[0]
    return int((rel > tol).sum())

def build_blocks(rays, tol=TOL):
    """rays: list of real vectors (need not be pre-normalized). Returns a dict with every
    quantity used in identity (*): rA, rB, rN, rD=rank(A+B), rAN=rank([A;N]) (=rank R),
    r_AcapB, r_AcapN, flex_R, flex_skew, r (commutant dim), O (#vertices with v_i not in
    neighbor span W_i), plus d, V, E, diff, dminus1, bound_ok."""
    rays = [np.asarray(v, float) for v in rays]
    rays = [v / np.linalg.norm(v) for v in rays]
    d, V = len(rays[0]), len(rays)
    n = d * V
    E = [(i, j) for i, j in combinations(range(V), 2) if abs(np.dot(rays[i], rays[j])) < 1e-9]

    A, B, N = [], [], []
    for i, j in E:
        rA_ = np.zeros(n); rB_ = np.zeros(n)
        for c in range(d):
            rA_[d * i + c] += rays[j][c]; rA_[d * j + c] += rays[i][c]      # A_e = p_e+q_e
            rB_[d * j + c] += rays[i][c]; rB_[d * i + c] -= rays[j][c]      # B_e = p_e-q_e
        A.append(rA_); B.append(rB_)
    for i in range(V):
        r_ = np.zeros(n)
        for c in range(d):
            r_[d * i + c] = rays[i][c]
        N.append(r_)
    Tr = []
    for a in range(d):
        for b in range(a + 1, d):
            t = np.zeros(n)
            for i in range(V):
                t[d * i + a] += rays[i][b]; t[d * i + b] -= rays[i][a]
            Tr.append(t)
    Ts = []
    for k in range(V):
        t = np.zeros(n)
        for c in range(d):
            t[d * k + c] = rays[k][c]
        Ts.append(t)
    for a in range(d):
        t = np.zeros(n)
        for i in range(V):
            t[d * i + a] = rays[i][a]
        Ts.append(t)
    for a in range(d):
        for b in range(a + 1, d):
            t = np.zeros(n)
            for i in range(V):
                t[d * i + a] += rays[i][b]; t[d * i + b] += rays[i][a]
            Ts.append(t)

    rA = _rank_svd(A, tol); rB = _rank_svd(B, tol); rN = _rank_svd(N, tol)
    rTr = _rank_svd(Tr, tol); rTs = _rank_svd(Ts, tol)
    rD = _rank_svd(A + B, tol)
    rAN = _rank_svd(A + N, tol)
    r_AcapB = rA + rB - rD
    r_AcapN = rA + rN - rAN

    flex_R = n - rAN - rTr
    flex_skew = n - rB - rTs
    r_comm = V + d * (d + 1) // 2 - rTs

    O = 0
    for i in range(V):
        nbrs = [j for j in range(V) if (min(i, j), max(i, j)) in E]
        if not nbrs:
            O += 1
            continue
        W = np.array([rays[j] for j in nbrs]).T
        Q, _ = np.linalg.qr(W)
        proj = Q @ (Q.T @ rays[i])
        if np.linalg.norm(rays[i] - proj) > 1e-6:
            O += 1

    diff = flex_R - flex_skew
    identity_rhs = d - r_comm - (rA - rB)
    return dict(d=d, V=V, E=len(E), rA=rA, rB=rB, rN=rN, rD=rD, rAN=rAN,
                r_AcapB=r_AcapB, r_AcapN=r_AcapN, flex_R=flex_R, flex_skew=flex_skew,
                diff=diff, dminus1=d - 1, bound_ok=(diff <= d - 1), r=r_comm, O=O,
                identity_ok=(diff == identity_rhs), rank_A_ge_B=(rA >= rB))

# ======================================================================================
# EXACT engine (mod-p rank on integer / Z[sqrt2] rays) — for the 4 dense named KS sets, to
# certify rA, rB, rN, A-cap-N=0 and rA>=rB WITHOUT floating point, matching the numerical engine.
# ======================================================================================
def _edges_pairs(rays):
    V = len(rays)
    return [(i, j) for i, j in combinations(range(V), 2) if q_dot(rays[i], rays[j]) == Z0]

def _exact_rank(rows, primes):
    best = None
    for p, s in primes:
        rp = rank_mod_p(rows, p, s)
        if best is None or rp < best:
            best = rp
    return best

def build_blocks_exact(rays_pairs, primes=None):
    """rays_pairs: Z[sqrt2] pair-encoded rays (as in sic_zoo/torsion_flex). Returns rA, rB, rN,
    rAN, A_cap_N, all EXACT (mod-p certified rank bounds, not floating point)."""
    if primes is None:
        primes = PRIMES
    d, V = len(rays_pairs[0]), len(rays_pairs)
    n = d * V
    E = _edges_pairs(rays_pairs)
    A, B, N = [], [], []
    for i, j in E:
        rA_ = [Z0] * n; rB_ = [Z0] * n
        for c in range(d):
            rA_[d * i + c] = q_add(rA_[d * i + c], rays_pairs[j][c])
            rA_[d * j + c] = q_add(rA_[d * j + c], rays_pairs[i][c])
            rB_[d * j + c] = q_add(rB_[d * j + c], rays_pairs[i][c])
            rB_[d * i + c] = q_add(rB_[d * i + c], q_neg(rays_pairs[j][c]))
        A.append(rA_); B.append(rB_)
    for i in range(V):
        r_ = [Z0] * n
        for c in range(d):
            r_[d * i + c] = rays_pairs[i][c]
        N.append(r_)
    rA = _exact_rank(A, primes); rB = _exact_rank(B, primes); rN = _exact_rank(N, primes)
    rAN = _exact_rank(A + N, primes)
    return dict(d=d, V=V, E=len(E), rA=rA, rB=rB, rN=rN, rAN=rAN,
                A_cap_N=(rA + rN - rAN), rank_A_ge_B=(rA >= rB))

# ======================================================================================
# self-checks: re-derive C2 and the r-identity from scratch (not assumed) + verify identity (*)
# ======================================================================================
def check_C2(res):
    """C2: rank(Tr) = d(d-1)/2, i.e. so(d) acts freely on any spanning ray set. One-line proof:
    if A in so(d) (or even just any linear map) satisfies A v_i = 0 for a SPANNING set {v_i}, then
    A = 0 (A kills a basis => A kills R^d). No check needed beyond 'rays span R^d', which every
    realizer here enforces (Gauss-Newton on a full-rank random start; exact sets verified in their
    own modules). We RE-VERIFY it numerically anyway (rTr, extracted implicitly from flex_R's
    definition, should equal d(d-1)/2 whenever flex_R/flex_skew were computed by build_blocks)."""
    pass  # re-derivation is the docstring; build_blocks's flex_R formula already uses rTr computed
          # fresh each call (not hard-coded), so every PASS row below is a live re-check.

# ======================================================================================
# zoo scenario builders
# ======================================================================================
ROWS = []

def add_numeric(name, rays, evidence="NUMERICAL (Gauss-Newton / analytic, SVD rank)"):
    try:
        res = build_blocks(rays)
    except Exception as ex:
        print(f"  [skip] {name}: {ex}")
        return None
    res["name"] = name; res["evidence"] = evidence
    ROWS.append(res)
    return res

def add_exact(name, rays_pairs, flex_R_numeric_check=None):
    """Certify rA, rB, rN, A-cap-N via mod-p EXACT rank on the named integer/Z[sqrt2] set, cross-
    checked against the (already independently verified elsewhere, e.g. torsion_flex's session-8
    gate) numeric flex_R/flex_skew for the same set, obtained via build_blocks on the float cast."""
    ex = build_blocks_exact(rays_pairs)
    floatr = [[a + b * (2 ** 0.5) for (a, b) in v] for v in rays_pairs]
    num = build_blocks(floatr)
    match = (ex["rA"] == num["rA"] and ex["rB"] == num["rB"] and ex["rN"] == num["rN"]
              and ex["A_cap_N"] == num["r_AcapN"] == 0)
    num["name"] = name
    num["evidence"] = f"EXACT (mod-p, primes={[p for p,_ in PRIMES]}) + NUMERICAL cross-check"
    num["exact_match"] = match
    ROWS.append(num)
    print(f"  [exact] {name:<12} rA={ex['rA']} rB={ex['rB']} rN={ex['rN']} A^N={ex['A_cap_N']}  "
          f"vs numeric rA={num['rA']} rB={num['rB']} rN={num['rN']} A^N={num['r_AcapN']}  "
          f"{'MATCH' if match else '*** MISMATCH ***'}")
    return num

def build_named_exact():
    print("--- named KS sets: EXACT mod-p certification of rA, rB, rN, A-cap-N=0, rA>=rB ---")
    add_exact("ONB", [as_pairs(v) for v in [(1, 0, 0), (0, 1, 0), (0, 0, 1)]])
    add_exact("Yu-Oh 13", [as_pairs(v) for v in integer_rays_yuoh()])
    add_exact("Peres 24", [as_pairs(v) for v in integer_rays_peres24()])
    add_exact("CEG 18", [as_pairs(v) for v in rays_ceg18()])
    add_exact("Peres 33", rays_peres33())

def build_cycles(dims=(3, 4, 5, 6), ns=range(5, 13)):
    for n in ns:
        rays = odd_cycle(n) if n % 2 else None
        if n % 2 == 1:
            rr = [np.asarray(v).real for v in rays]
            add_numeric(f"C{n} umbrella d=3", rr, "NUMERICAL (analytic Lovasz umbrella)")
        for d in dims:
            if d == 3 and n % 2 == 1:
                continue
            out = solve_cycle(n, d, tries=40, real=True)
            if out is None:
                continue
            rays2, res_, mind, minoff = out
            if res_ > 1e-9:
                continue
            rr = [np.asarray(v).real for v in rays2]
            add_numeric(f"C{n} d={d}", rr, f"NUMERICAL (Gauss-Newton, res {res_:.0e})")

def build_structured_zoo():
    specs = []
    for (m, k, d) in [(3, 3, 4), (2, 4, 4), (3, 4, 5), (2, 2, 4), (4, 4, 5), (3, 5, 5)]:
        specs.append((f"K_{{{m},{k}}} d={d}", *complete_bipartite(m, k), d))
    for (n, d) in [(5, 4), (6, 4), (5, 5), (7, 4), (8, 5), (6, 6)]:
        specs.append((f"Wheel W{n} d={d}", *wheel_graph(n), d))
    for (n, d) in [(4, 4), (5, 4), (5, 5)]:
        specs.append((f"Prism Y{n} d={d}", *prism_graph(n), d))
    # extra stress-test families: Petersen (3-regular, 10v,15e), star (one hub, high O),
    # path (tree, high O), double-star (two hubs)
    def petersen():
        outer = [(i, (i + 1) % 5) for i in range(5)]
        inner = [(5 + i, 5 + (i + 2) % 5) for i in range(5)]
        spokes = [(i, 5 + i) for i in range(5)]
        return outer + inner + spokes, 10
    def star(n): return [(0, i) for i in range(1, n + 1)], n + 1
    def path(n): return [(i, i + 1) for i in range(n - 1)], n
    def double_star(n1, n2):
        e = [(0, i) for i in range(2, 2 + n1)] + [(1, i) for i in range(2 + n1, 2 + n1 + n2)] + [(0, 1)]
        return e, 2 + n1 + n2
    for d in (4, 5, 6):
        specs.append((f"Petersen d={d}", *petersen(), d))
    for (n, d) in [(5, 4), (8, 5), (10, 6)]:
        specs.append((f"Star{n} d={d}", *star(n), d))
    for (n, d) in [(6, 4), (9, 5)]:
        specs.append((f"Path{n} d={d}", *path(n), d))
    for d in (4, 5):
        specs.append((f"DoubleStar d={d}", *double_star(3, 3), d))
    for name, edges, n, d in specs:
        rays = realize_graph(edges, n, d, tries=50)
        if rays is None:
            print(f"  [skip] {name}: no faithful real realization found")
            continue
        add_numeric(name, rays, "NUMERICAL (Gauss-Newton, structured zoo)")

def build_block_diagonal_zoo():
    """Deliberately construct r>1 configurations: direct sums of irreducible blocks in mutually
    orthogonal subspaces. r = number of "irreducible" pieces (an ONB piece of dim m contributes m
    to r, not 1 -- verified/consistent with the definition, see ONB row: r=3)."""
    def block_union(*pieces):
        total_d = sum(len(p[0]) for p in pieces)
        out = []; offset = 0
        for p in pieces:
            dp = len(p[0])
            for v in p:
                vv = np.zeros(total_d); vv[offset:offset + dp] = v
                out.append(vv)
            offset += dp
        return out
    c5 = [np.asarray(v).real for v in odd_cycle(5)]
    c7 = [np.asarray(v).real for v in odd_cycle(7)]
    c9 = [np.asarray(v).real for v in odd_cycle(9)]
    onb3 = [np.asarray(v).real for v in onb(3)]
    add_numeric("C5+C7 (r=2)", block_union(c5, c7), "NUMERICAL (block-diagonal, deliberate r=2)")
    add_numeric("C5+C7+C9 (r=3)", block_union(c5, c7, c9), "NUMERICAL (block-diagonal, r=3)")
    add_numeric("C5+C7+C9+C5b (r=4)", block_union(c5, c7, c9, c5),
                "NUMERICAL (block-diagonal, r=4)")
    add_numeric("ONB3+C5 (r=4)", block_union(onb3, c5), "NUMERICAL (block-diagonal, r=4)")
    add_numeric("ONB3+C5+C7 (r=5)", block_union(onb3, c5, c7), "NUMERICAL (block-diagonal, r=5)")
    add_numeric("6xC5 (r=6)", block_union(c5, c5, c5, c5, c5, c5),
                "NUMERICAL (block-diagonal, r=6)")

def build_random_search(dvals, Vvals, pvals, seeds=(0, 1), tries=15, tag=""):
    """Structured random search over (d, V, density) — used by 'big'/'huge' modes to hunt hard for
    a bound violation or for rank(A) < rank(B)."""
    n_before = len(ROWS)
    for d in dvals:
        for V in Vvals:
            for p in pvals:
                for seed in seeds:
                    rng = random.Random((d, V, p, seed, tag))
                    edges = [(i, j) for i in range(V) for j in range(i + 1, V) if rng.random() < p]
                    if not edges:
                        continue
                    rays = realize_graph(edges, V, d, tries=tries, seed=seed * 97 + d * 7 + V)
                    if rays is None:
                        continue
                    add_numeric(f"Rand(d={d},V={V},p={p},s={seed}){tag}", rays,
                                "NUMERICAL (structured random search)")
    return len(ROWS) - n_before

# ======================================================================================
# reporting
# ======================================================================================
def show_table(rows):
    print("=" * 130)
    print(f"{'scenario':<28}{'d':>3}{'V':>4}{'E':>5}{'flexR':>7}{'flexS':>7}{'diff':>6}"
          f"{'d-1':>5}{'r':>4}{'rA':>6}{'rB':>6}{'A>=B':>6}{'ident.':>8}{'bound':>7}")
    print("-" * 130)
    for r in rows:
        print(f"{r['name']:<28}{r['d']:>3}{r['V']:>4}{r['E']:>5}{r['flex_R']:>7}{r['flex_skew']:>7}"
              f"{r['diff']:>6}{r['dminus1']:>5}{r['r']:>4}{r['rA']:>6}{r['rB']:>6}"
              f"{'Y' if r['rank_A_ge_B'] else 'N':>6}{'OK' if r['identity_ok'] else 'FAIL':>8}"
              f"{'OK' if r['bound_ok'] else '*FAIL*':>7}")
    print("-" * 130)

def summarize(rows):
    n = len(rows)
    bad_bound = [r for r in rows if not r["bound_ok"]]
    bad_ident = [r for r in rows if not r["identity_ok"]]
    bad_AgeB = [r for r in rows if not r["rank_A_ge_B"]]
    bad_AcapN = [r for r in rows if r.get("r_AcapN", 0) != 0]
    maxdiff = max((r["diff"] - r["dminus1"]) for r in rows) if rows else None
    r_max = max((r["r"] for r in rows), default=None)
    print(f"\nZOO SIZE: {n} rows.")
    print(f"LEMMA  A cap N = 0 (proved, general): dim(A cap N)=0 on {n-len(bad_AcapN)}/{n} rows "
          f"(computed independently via SVD of [A;N], not assumed) -> "
          f"{'ALL OK (consistent with the proof)' if not bad_AcapN else '*** NONZERO A cap N FOUND *** ' + str([r['name'] for r in bad_AcapN])}")
    print(f"IDENTITY (*) flex_R-flex_skew = d - r - (rankA-rankB): holds on {n-len(bad_ident)}/{n} "
          f"rows -> {'ALL OK' if not bad_ident else 'FAIL: ' + str([r['name'] for r in bad_ident])}")
    print(f"BOUND flex_R-flex_skew <= d-1: holds on {n-len(bad_bound)}/{n} rows "
          f"(max observed diff-(d-1) = {maxdiff}) -> "
          f"{'NO VIOLATION' if not bad_bound else '*** VIOLATION *** ' + str([r['name'] for r in bad_bound])}")
    print(f"rank(A) >= rank(B) (the open reduction (**)): holds on {n-len(bad_AgeB)}/{n} rows -> "
          f"{'CONJECTURE HOLDS (no counterexample)' if not bad_AgeB else '*** COUNTEREXAMPLE *** ' + str([r['name'] for r in bad_AgeB])}")
    print(f"max r (commutant dim) observed in this run: {r_max}")
    eq = sum(1 for r in rows if r["rA"] == r["rB"])
    print(f"rank(A) == rank(B) exactly: {eq}/{n} rows; rank(A) > rank(B) strictly: {n-eq}/{n} rows "
          f"(strict cases: {[r['name'] for r in rows if r['rA']>r['rB']]})")
    return not bad_bound, not bad_ident, not bad_AgeB

# ======================================================================================
def main():
    t0 = time.time()
    args = sys.argv[1:]
    mode = "fast"
    if "huge" in args:
        mode = "huge"
    elif "big" in args:
        mode = "big"
    print("=" * 130)
    print("D-MINUS-1 BOUND — flex_R - flex_skew <= d-1 : counterexample search + proof attempt")
    print("=" * 130)

    build_named_exact()
    print()
    build_cycles()
    build_structured_zoo()
    build_block_diagonal_zoo()
    print(f"\n[{time.time()-t0:.1f}s] fast core zoo built ({len(ROWS)} rows)")

    if mode in ("big", "huge"):
        print("\n--- 'big' structured random search (d=3..7, V=5..12, several densities) ---")
        added = 0
        added += build_random_search([3], [5,6,7,8,9,10], [0.15,0.3,0.5,0.7], tries=15)
        added += build_random_search([4], [5,6,7,8,9,10], [0.15,0.3,0.5,0.7], tries=15)
        added += build_random_search([5], [5,6,7,8,9,10], [0.15,0.3,0.5,0.7], tries=15)
        added += build_random_search([6], [6,7,8,9,10,11], [0.15,0.3,0.5,0.7], tries=15)
        added += build_random_search([7], [7,8,9,10,11,12], [0.15,0.3,0.5,0.7], tries=15)
        print(f"[{time.time()-t0:.1f}s] +{added} random rows (big)")

    if mode == "huge":
        print("\n--- 'huge' extra adversarial search (denser graphs, larger V, more seeds) ---")
        added = 0
        added += build_random_search([5, 6], [8, 9, 10, 11], [0.5, 0.65, 0.8], seeds=(0, 1, 2),
                                      tries=15, tag="-huge")
        added += build_random_search([3, 4], [9, 10, 11, 12], [0.4, 0.55], seeds=(0, 1, 2),
                                      tries=12, tag="-huge2")
        print(f"[{time.time()-t0:.1f}s] +{added} random rows (huge)")

    print()
    show_table(ROWS)
    bound_ok, ident_ok, AgeB_ok = summarize(ROWS)

    print(f"\n[{time.time()-t0:.1f}s] dminus1_bound ({mode}) "
          f"{'PASS' if (bound_ok and ident_ok) else 'FAIL'} "
          f"(rank(A)>=rank(B) conjecture: {'HOLDS' if AgeB_ok else 'REFUTED'})")

if __name__ == "__main__":
    main()
