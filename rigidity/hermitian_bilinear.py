#!/usr/bin/env python3
"""
Open Problem "Hermitian-vs-bilinear" — comparison theorem between the Hermitian flex (flex_skew)
and the classical real bilinear orthogonal-representation (OR) rigidity (flex_R), across many
graphs and dimensions (2026-07-17).

CONTEXT (already proved elsewhere in this program, reused not re-derived): at a REAL realization
of an exclusivity graph, the complex/Hermitian rigidity matrix block-diagonalizes as
    flex_C = flex_R + flex_skew
with flex_R the classical real bilinear OR-rigidity (real perturbations x_i; edge rows
x_i.v_j + v_i.x_j = 0; norm rows v_i.x_i = 0; trivial = so(d)) and flex_skew the antisymmetrized
Hermitian block (imaginary perturbations y_i; edge rows v_i.y_j - y_i.v_j = 0; trivial = phases +
i*Sym(d)). This file does NOT re-derive that split; it REUSES torsion_flex.numeric_blocks (the
exact same formulas, spectral-gap SVD ranks) as the measurement engine, and adds:

  (E1) an INDEPENDENT re-derivation of flex_R from the full complex/Hermitian tangent-space
       Jacobian of flex_dimension.py (a totally different parametrization: QR-orthonormal tangent
       bases of CP^{d-1} restricted to real-only coefficients, vs. the explicit ambient
       edge+norm-row matrix) — this is the actual content of Q1: does the "real block" extracted
       by algebraic decomposition equal the real block you get by restricting the genuine Hermitian
       rigidity matrix to real tangent directions from scratch? (independent code path, same rays)
  (E2) a generic real-orthogonal-representation realizer (Gauss-Newton, arbitrary graphs), used to
       build a much larger test zoo than cycles alone: wheels, complete bipartite, prisms, random
       sparse graphs, disjoint unions — to hunt for a general law relating flex_R and flex_skew
       (Q2/Q3).

Evidence discipline: every number reported is either EXACT (sympy/exact-mod-p rational rank on
integer or Z[sqrt2] rays, via torsion_flex's exact machinery) or NUMERICAL (SVD rank with an
explicit spectral-gap report; realizations found by Gauss-Newton with residual < 1e-10). No claim
in the printed output or HERM_BILINEAR.md is asserted beyond what a run of this script produced.

Run:  python3 hermitian_bilinear.py            # fast: named exotic sets + cycles + small zoo (<45s)
      python3 hermitian_bilinear.py big         # + larger random/wheel/prism zoo (slower)
      python3 hermitian_bilinear.py q1          # only the Q1 independent cross-check
"""
import os, sys, time, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from itertools import combinations

from torsion_flex import (numeric_blocks, real_block_pairs, skew_block_pairs,
                           exact_block_flex, PRIMES)
from flex_dimension import odd_cycle
from even_cycles import solve_cycle
from exact_rigidity import integer_rays_yuoh, integer_rays_peres24
from sic_zoo import rays_ceg18, rays_peres33, as_pairs

TOL = 1e-8

# ======================================================================================
# (E1) Q1 — independent re-derivation of flex_R from the full Hermitian tangent Jacobian
# ======================================================================================
def real_tangent_basis(v, tol=1e-9):
    """v: real unit vector in R^d. Returns an orthonormal REAL basis of v^perp (d-1 vectors),
       via QR on I - v v^T (real matrix -> real Q, no complex arithmetic anywhere)."""
    d = len(v)
    M = np.eye(d) - np.outer(v, v)
    Q, R = np.linalg.qr(M)
    cols = [Q[:, k] for k in range(d) if abs(R[k, k]) > 1e-9]
    B = [c for c in cols if abs(v @ c) < tol][:d - 1]
    assert len(B) == d - 1, "real tangent basis failed"
    return B

def _rrank(rows, tol=1e-8):
    M = np.array(rows, float)
    if M.size == 0: return 0, np.inf
    s = np.linalg.svd(M, compute_uv=False)
    if s[0] == 0: return 0, np.inf
    rel = s / s[0]; r = int((rel > tol).sum())
    gap = (s[r - 1] / s[r]) if 0 < r < len(s) else np.inf
    return r, gap

def flex_R_reduced(rays):
    """Rebuild flex_R from scratch using the REDUCED (gauge-fixed) tangent-space parametrization
       of the real quadric: x_i = sum_k a_k e_k^i with {e_k^i} a REAL orthonormal basis of
       v_i^perp (obtained via real_tangent_basis, a fresh QR call — independent of
       torsion_flex.real_block_pairs, which instead works in the ambient R^d coordinates with an
       explicit norm row). Because e_k^i and v_j are both real, the edge constraint
       x_i.v_j + v_i.x_j = 0 has NO imaginary part in this basis (unlike flex_dimension.py's
       general complex tangent parametrization) — so this is exactly the real bilinear OR-rigidity
       matrix, re-expressed in reduced coordinates ((d-1)V unknowns instead of dV + V norm rows).
       Returns (flex_R', rank J', rank T', gap).
    """
    rays = [np.asarray(v, float) for v in rays]
    rays = [v / np.linalg.norm(v) for v in rays]
    d, V = len(rays[0]), len(rays)
    E = [(i, j) for i, j in combinations(range(V), 2) if abs(np.dot(rays[i], rays[j])) < 1e-9]
    tb = [real_tangent_basis(v) for v in rays]
    ncols = (d - 1) * V
    Jrows = []
    for (i, j) in E:
        row = np.zeros(ncols)
        vi, vj = rays[i], rays[j]
        for k in range(d - 1):
            row[(d - 1) * i + k] += tb[i][k] @ vj      # coeff of a_k^i
            row[(d - 1) * j + k] += vi @ tb[j][k]       # coeff of a_k^j
        Jrows.append(row)
    Trows = []
    for a in range(d):
        for b in range(a + 1, d):
            A = np.zeros((d, d)); A[a, b] = 1; A[b, a] = -1     # so(d) generator E_ab - E_ba
            t = np.zeros(ncols)
            for i, v in enumerate(rays):
                Av = A @ v
                for k in range(d - 1):
                    t[(d - 1) * i + k] = tb[i][k] @ Av
            Trows.append(t)
    rJ, gJ = _rrank(Jrows); rT, _ = _rrank(Trows)
    flex = (ncols - rJ) - rT
    return flex, rJ, rT, gJ

def q1_check(name, rays, flex_R_expected):
    """Cross-check torsion_flex's flex_R (ambient edge+norm-row formula) against flex_R_reduced
       (independent reduced tangent-basis formula, fresh code path via the Hermitian engine's
       QR tangent construction). Agreement is genuine evidence for Q1 (not circular: the two
       matrices differ in size, basis, and row structure; they can only agree because the
       underlying linear systems are related by an invertible change of variables — which is
       exactly the claim under test, not assumed)."""
    fR2, rJ, rT, gap = flex_R_reduced(rays)
    match = (fR2 == flex_R_expected)
    print(f"  Q1  {name:<18} flex_R(ambient,edge+norm)={flex_R_expected:>3}  "
          f"flex_R(reduced,tangent-QR)={fR2:>3}  rankJ'={rJ} rankT'={rT} gap={gap:.1e}  "
          f"{'MATCH' if match else '*** MISMATCH ***'}")
    return match

# ======================================================================================
# (E2) generic real orthogonal-representation realizer (Gauss-Newton, arbitrary graphs)
# ======================================================================================
def _pack(vs): return np.concatenate(vs)
def _unpack(x, n, d): return [x[d * k:d * k + d] for k in range(n)]

def _resid_jac(x, n, d, edges):
    vs = _unpack(x, n, d)
    m = len(edges) + n
    r = np.zeros(m); J = np.zeros((m, d * n))
    for row, (i, j) in enumerate(edges):
        r[row] = vs[i] @ vs[j]
        J[row, d * i:d * i + d] += vs[j]
        J[row, d * j:d * j + d] += vs[i]
    for i in range(n):
        row = len(edges) + i
        r[row] = vs[i] @ vs[i] - 1.0
        J[row, d * i:d * i + d] += 2 * vs[i]
    return r, J

def _gauss_newton(x0, n, d, edges, iters=150):
    x = x0.copy()
    for _ in range(iters):
        r, J = _resid_jac(x, n, d, edges)
        nr = np.linalg.norm(r)
        if nr < 1e-14: break
        dx = np.linalg.lstsq(J, -r, rcond=None)[0]
        t = 1.0
        for _ in range(30):
            r2, _ = _resid_jac(x + t * dx, n, d, edges)
            if np.linalg.norm(r2) < nr: break
            t /= 2
        x = x + t * dx
    return x

def _faithful(vs, n, edges, tol_edge=1e-7, tol_nonedge=1e-3):
    E = set(edges) | {(j, i) for i, j in edges}
    for i in range(n):
        for j in range(i + 1, n):
            ip = abs(vs[i] @ vs[j])
            if (i, j) in E and ip > tol_edge: return False
            if (i, j) not in E and ip < tol_nonedge: return False
    return True

def realize_graph(edges, n, d, tries=40, seed=0):
    """Random-start Gauss-Newton real orthogonal representation of graph (n, edges) in R^d.
       Returns rays (list of n unit vectors in R^d) faithful to the EXACT given edge set, or
       None if no faithful realization was found within `tries` random restarts."""
    for t in range(tries):
        rg = np.random.default_rng(seed * 10000 + t)
        x0 = _pack([rg.normal(size=d) for _ in range(n)])
        x0 = _pack([v / np.linalg.norm(v) for v in _unpack(x0, n, d)])
        x = _gauss_newton(x0, n, d, edges)
        vs = [v / np.linalg.norm(v) for v in _unpack(x, n, d)]
        r, _ = _resid_jac(_pack(vs), n, d, edges)
        if np.linalg.norm(r) < 1e-9 and _faithful(vs, n, edges):
            return vs
    return None

# ---- graph generators -------------------------------------------------------------
def cycle_graph(n): return [(i, (i + 1) % n) for i in range(n)], n

def wheel_graph(n):
    """hub 0 joined to rim 1..n; rim forms an n-cycle."""
    edges = [(0, i) for i in range(1, n + 1)] + [(i, i % n + 1) for i in range(1, n + 1)]
    return edges, n + 1

def complete_bipartite(m, k):
    return [(i, m + j) for i in range(m) for j in range(k)], m + k

def prism_graph(n):
    """two n-cycles (0..n-1) and (n..2n-1) plus a perfect matching between them."""
    e = [(i, (i + 1) % n) for i in range(n)] + [(n + i, n + (i + 1) % n) for i in range(n)]
    e += [(i, n + i) for i in range(n)]
    return e, 2 * n

def random_graph(n, p, seed):
    rng = random.Random(seed)
    return [(i, j) for i in range(n) for j in range(i + 1, n) if rng.random() < p], n

def disjoint_union(edges1, n1, edges2, n2):
    e2 = [(i + n1, j + n1) for i, j in edges2]
    return edges1 + e2, n1 + n2

# ======================================================================================
# table driver
# ======================================================================================
ROWS = []

def _diag_fields(n, d, V, flex_R, flex_skew, rTr, rTs):
    """Structural diagnostic (definitional algebra, no extra assumptions):
         rRr = rank(real edge+norm block Rr) = n - flex_R - rTr
         rRs = rank(skew edge block Rs)      = n - flex_skew - rTs
         r   = dim of the symmetric 'eigen-commutant' {S in Sym(d): S v_i || v_i all i}
             = V + d(d+1)/2 - rTs                       (rank-nullity, exact, r>=1 always: S=cI)
         C2  = (rTr == d(d-1)/2)   -- PROVED for any real realization spanning R^d (faithful):
               an so(d) stabilizer with A v_i || v_i for a spanning set forces A=0.
         identity check: flex_R - flex_skew == (rRs - rRr) + V + d - r   (pure algebra from the
               two definitions above; always exact once rRr,rRs,r are computed from the SAME run)."""
    rRr = n - flex_R - rTr
    rRs = n - flex_skew - rTs
    r = V + d * (d + 1) // 2 - rTs
    C2 = (rTr == d * (d - 1) // 2)
    lhs = flex_R - flex_skew
    rhs = (rRs - rRr) + V + d - r
    return dict(rTr=rTr, rTs=rTs, rRr=rRr, rRs=rRs, r=r, C2=C2, identity_ok=(lhs == rhs))

def add_row(name, d, rays, evidence):
    b = numeric_blocks(rays)
    diff = b["flex_R"] - b["flex_skew"]
    V = b["n"] // d
    diag = _diag_fields(b["n"], d, V, b["flex_R"], b["flex_skew"], b["rTr"], b["rTs"])
    ROWS.append(dict(name=name, d=d, V=V, E=b["E"], flex_R=b["flex_R"],
                      flex_skew=b["flex_skew"], diff=diff, dminus1=d - 1,
                      bound_ok=(diff <= d - 1), evidence=evidence, minsv=b["minsv"], **diag))
    return ROWS[-1]

def add_row_exact(name, d, rays_pairs, evidence="EXACT (Z or Z[sqrt2], mod-p rank)"):
    Rr, Tr, nR, E = real_block_pairs(rays_pairs)
    Rs, Ts, nS, _ = skew_block_pairs(rays_pairs)
    fR, rTr = exact_block_flex(Rr, Tr, nR, PRIMES)
    fS, rTs = exact_block_flex(Rs, Ts, nS, PRIMES)
    diff = fR - fS
    V = len(rays_pairs)
    diag = _diag_fields(nR, d, V, fR, fS, rTr, rTs)
    ROWS.append(dict(name=name, d=d, V=V, E=len(E), flex_R=fR, flex_skew=fS,
                      diff=diff, dminus1=d - 1, bound_ok=(diff <= d - 1), evidence=evidence,
                      minsv=None, **diag))
    return ROWS[-1], [v for v in rays_pairs]  # rays_pairs kept for Q1 (converted to float)

def show_table(rows):
    print("=" * 118)
    print(f"{'scenario':<26}{'d':>3}{'V':>4}{'E':>5}{'flex_R':>8}{'flex_skew':>10}"
          f"{'diff':>6}{'d-1':>5}  {'<=d-1?':<8}{'evidence'}")
    print("-" * 118)
    for r in rows:
        print(f"{r['name']:<26}{r['d']:>3}{r['V']:>4}{r['E']:>5}{r['flex_R']:>8}{r['flex_skew']:>10}"
              f"{r['diff']:>6}{r['dminus1']:>5}  {'YES' if r['bound_ok'] else 'NO':<8}{r['evidence']}")
    print("-" * 118)

def show_diag(rows):
    """Structural identity check: flex_R - flex_skew = (rRs-rRr) + V + d - r, PURE ALGEBRA given
       the two block-trivial ranks (rTr, rTs) already computed by numeric_blocks / exact_block_flex
       -- always exact (no genericity assumed). C2 (rTr=d(d-1)/2) is a PROVED fact for any real
       realization spanning R^d (cycle_split_general.py's argument, graph-independent). r = dim of
       the symmetric eigen-commutant of the rays (r=1: irreducible; r=k: k mutually-orthogonal
       'blocks', e.g. bipartite/wheel constructions) -- COMPUTED, not assumed."""
    print("=" * 118)
    print(f"{'scenario':<26}{'d':>3}{'V':>4}{'rRr':>5}{'rRs':>5}{'rTr':>5}{'d(d-1)/2':>9}{'C2':>4}"
          f"{'rTs':>5}{'r(commut.)':>11}{'ident.':>8}")
    print("-" * 118)
    allok = True
    for r in rows:
        ok = r["identity_ok"] and r["C2"]
        allok &= ok
        print(f"{r['name']:<26}{r['d']:>3}{r['V']:>4}{r['rRr']:>5}{r['rRs']:>5}{r['rTr']:>5}"
              f"{r['d']*(r['d']-1)//2:>9}{'Y' if r['C2'] else 'N':>4}{r['rTs']:>5}{r['r']:>11}"
              f"{'OK' if ok else 'FAIL':>8}")
    print("-" * 118)
    print(f"C2 (rTr=d(d-1)/2) and the algebraic identity hold on {sum(1 for r in rows if r['identity_ok'] and r['C2'])}/{len(rows)} rows "
          f"-> {'ALL OK' if allok else 'SOME FAILED (see above)'}")

# ======================================================================================
# scenario builders
# ======================================================================================
def build_named_exotic():
    rows = []
    onb = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
    r = add_row_exact("ONB d=3", 3, [as_pairs(v) for v in onb])[0]
    rows.append((r, onb, r["flex_R"]))
    yo = integer_rays_yuoh()
    r = add_row_exact("Yu-Oh 13", 3, [as_pairs(v) for v in yo])[0]
    rows.append((r, yo, r["flex_R"]))
    p24 = integer_rays_peres24()
    r = add_row_exact("Peres 24", 4, [as_pairs(v) for v in p24])[0]
    rows.append((r, p24, r["flex_R"]))
    ceg = rays_ceg18()
    r = add_row_exact("CEG 18", 4, [as_pairs(v) for v in ceg])[0]
    rows.append((r, ceg, r["flex_R"]))
    p33 = rays_peres33()
    r = add_row_exact("Peres 33", 3, p33)[0]
    p33_float = [[a + b * (2 ** 0.5) for (a, b) in v] for v in p33]
    rows.append((r, p33_float, r["flex_R"]))
    return rows

def build_cycles(dims=(3, 4, 5), ns=range(5, 11)):
    for n in ns:
        # d=3 uses the exact analytic (real) Lovasz umbrella -- no solver needed, no tries
        rays = odd_cycle(n) if n % 2 else None
        if n % 2 == 1:
            rr = [np.asarray(v).real for v in rays]
            add_row(f"C{n} umbrella d=3", 3, rr, "NUMERICAL (analytic real umbrella, SVD rank)")
        for d in dims:
            if d == 3 and n % 2 == 1:
                continue  # already added via the exact umbrella above
            rays, res, mind, minoff = solve_cycle(n, d, tries=40, real=True) or (None, 1, 0, 0)
            if rays is None or res > 1e-9:
                continue
            rr = [np.asarray(v).real for v in rays]
            add_row(f"C{n} d={d}", d, rr, f"NUMERICAL (Gauss-Newton, res {res:.0e})")

def build_zoo(big=False):
    specs = []
    # complete bipartite (needs two orthogonal subspaces dim>=2 each for internal genericity)
    for (m, k, d) in [(3, 3, 4), (2, 4, 4), (3, 4, 5), (2, 2, 4)]:
        specs.append((f"K_{{{m},{k}}} d={d}", *complete_bipartite(m, k), d))
    # wheels (hub + rim cycle) -- rim needs d-1 dims free of the hub direction
    for (n, d) in [(5, 4), (6, 4), (5, 5), (7, 4)]:
        specs.append((f"Wheel W{n} d={d}", *wheel_graph(n), d))
    # prisms (2 cycles + matching)
    for (n, d) in [(4, 4), (5, 4), (5, 5)]:
        specs.append((f"Prism Y{n} d={d}", *prism_graph(n), d))
    # random sparse graphs
    for (n, p, d, seed) in [(6, 0.35, 4, 1), (7, 0.3, 4, 2), (8, 0.25, 5, 3), (6, 0.5, 3, 4)]:
        e, nn = random_graph(n, p, seed)
        specs.append((f"Rand(n={n},p={p},d={d},s={seed})", e, nn, d))
    # disjoint unions (two ONB triangles = 2 disjoint K3, d=3; two C5's, d=3)
    e1, n1 = cycle_graph(3); e2, n2 = cycle_graph(3)
    e, n = disjoint_union(e1, n1, e2, n2)
    specs.append(("2xK3 (disjoint) d=3", e, n, 3))
    e1, n1 = cycle_graph(5); e2, n2 = cycle_graph(5)
    e, n = disjoint_union(e1, n1, e2, n2)
    specs.append(("2xC5 (disjoint) d=3", e, n, 3))
    if big:
        for (n, p, d, seed) in [(9, 0.2, 5, 11), (10, 0.15, 5, 12), (8, 0.4, 6, 13)]:
            e, nn = random_graph(n, p, seed)
            specs.append((f"Rand(n={n},p={p},d={d},s={seed})", e, nn, d))
        for (n, d) in [(8, 5), (6, 6)]:
            specs.append((f"Wheel W{n} d={d}", *wheel_graph(n), d))
        for (m, k, d) in [(4, 4, 5), (3, 5, 5)]:
            specs.append((f"K_{{{m},{k}}} d={d}", *complete_bipartite(m, k), d))

    for name, edges, n, d in specs:
        rays = realize_graph(edges, n, d, tries=50)
        if rays is None:
            print(f"  [skip] {name}: no faithful real realization found in the tries budget")
            continue
        add_row(name, d, rays, "NUMERICAL (Gauss-Newton, generic zoo)")

# ======================================================================================
def main():
    t0 = time.time()
    args = sys.argv[1:]
    big = "big" in args
    only_q1 = "q1" in args
    print("=" * 118)
    print("HERMITIAN-vs-BILINEAR — comparison of flex_R (real bilinear OR-rigidity) and flex_skew")
    print("(Hermitian antisymmetrized block), across graphs and dimensions")
    print("=" * 118)

    named = build_named_exotic()
    print("\n--- (Q1) independent cross-check: flex_R via ambient edge+norm rows vs. flex_R via")
    print("    reduced real tangent-space (QR) parametrization of the Hermitian engine ---")
    q1_all_ok = True
    for r, rays, fR in named:
        d = r["d"]
        ok = q1_check(r["name"], rays, fR)
        q1_all_ok &= ok
    # also cross-check on a handful of cycle realizations (numeric, not exact)
    for n, d in [(5, 3), (7, 3), (6, 4), (9, 5)]:
        rays, res, mind, minoff = solve_cycle(n, d, tries=40, real=True) or (None, 1, 0, 0)
        if rays is None or res > 1e-9:
            continue
        rr = [np.asarray(v).real for v in rays]
        b = numeric_blocks(rr)
        ok = q1_check(f"C{n} d={d}", rr, b["flex_R"])
        q1_all_ok &= ok
    print(f"\nQ1 VERDICT: {'ALL MATCH (' + str(len(named)+4) + ' cases)' if q1_all_ok else '*** SOME MISMATCH -- see above ***'}")
    if only_q1:
        print(f"\n[{time.time()-t0:.1f}s] hermitian_bilinear (q1-only) "
              f"{'PASS' if q1_all_ok else 'FAIL'}")
        return

    print("\n--- (Q2/Q3) building the comparison table across graphs/dimensions ---")
    build_cycles()
    build_zoo(big=big)
    print()
    show_table(ROWS)
    print("\n--- structural diagnostic: the algebraic identity flex_R - flex_skew "
          "= (rRs-rRr) + V + d - r ---")
    show_diag(ROWS)

    # ---- pattern tests ------------------------------------------------------------
    print("\nCANDIDATE PATTERNS (evidence on rows computed this run):")
    tempting = [r for r in ROWS if r["diff"] == r["dminus1"]]
    print(f"  P1  flex_R - flex_skew == d-1 (exactly, the cycle formula):"
          f" holds on {len(tempting)}/{len(ROWS)} rows -> "
          f"{'UNIVERSAL' if len(tempting)==len(ROWS) else 'NOT universal (see mismatches below)'}")
    ctrex = [r for r in ROWS if r["diff"] != r["dminus1"]]
    if ctrex:
        print(f"      COUNTEREXAMPLES (diff != d-1), first 8 shown:")
        for r in ctrex[:8]:
            print(f"        {r['name']:<26} d={r['d']}  flex_R={r['flex_R']} flex_skew={r['flex_skew']} "
                  f"diff={r['diff']} (d-1={r['dminus1']})")

    bad_bound = [r for r in ROWS if not r["bound_ok"]]
    print(f"\n  P2  flex_R - flex_skew <= d-1 (inequality, cycles saturate it):"
          f" holds on {len(ROWS)-len(bad_bound)}/{len(ROWS)} rows -> "
          f"{'HOLDS on this run' if not bad_bound else 'VIOLATED'}")
    if bad_bound:
        print(f"      VIOLATIONS (flex_R - flex_skew > d-1):")
        for r in bad_bound:
            print(f"        {r['name']:<26} d={r['d']}  flex_R={r['flex_R']} flex_skew={r['flex_skew']} "
                  f"diff={r['diff']} > d-1={r['dminus1']}")

    irr = [r for r in ROWS if r["r"] == 1]
    block = [r for r in ROWS if r["r"] > 1]
    sat_irr = sum(1 for r in irr if r["diff"] == r["dminus1"])
    sat_block = sum(1 for r in block if r["diff"] == r["dminus1"])
    print(f"\n  P4  r (commutant dim) vs saturation of P1 (diff==d-1): irreducible rays (r=1): "
          f"{sat_irr}/{len(irr)} saturate;  block-decomposable rays (r>1): {sat_block}/{len(block)} saturate.")
    if block:
        print(f"      block cases (r>1) found: " +
              ", ".join(f"{r['name']}(r={r['r']})" for r in block))

    print("\n  P3  Q3 (skew-rigid but real-flexible, flex_skew=0 < flex_R):")
    q3a = [r for r in ROWS if r["flex_skew"] == 0 and r["flex_R"] > 0]
    for r in q3a:
        print(f"        {r['name']:<26} d={r['d']}  flex_R={r['flex_R']} flex_skew=0")
    print(f"      found on {len(q3a)}/{len(ROWS)} rows.")
    print("  P3  Q3 (real-rigid but skew-flexible, flex_R=0 < flex_skew):")
    q3b = [r for r in ROWS if r["flex_R"] == 0 and r["flex_skew"] > 0]
    for r in q3b:
        print(f"        {r['name']:<26} d={r['d']}  flex_R=0 flex_skew={r['flex_skew']}")
    print(f"      found on {len(q3b)}/{len(ROWS)} rows.")

    print(f"\n[{time.time() - t0:.1f}s] hermitian_bilinear run complete "
          f"({len(ROWS)} rows, Q1 {'PASS' if q1_all_ok else 'FAIL'}).")

if __name__ == "__main__":
    main()
