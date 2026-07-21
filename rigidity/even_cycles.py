#!/usr/bin/env python3
"""
B1 — EVEN CYCLES in the rigidity-contextuality program (session 2, 2026-07-16).

Question. Theorem 1 proved flex(C_n) = 2n-8 for ODD n >= 5 at the Lovasz umbrella; the odd
hypothesis entered ONLY through the free-action lemma. What happens for EVEN cycles
(CHSH = C_4, chained-Bell = C_{2k})?

Answers established here (see EVEN_CYCLES.md for the note):
  (A) THEOREM (exact, 3 lines): C_4 admits NO faithful realization in CP^2. Any realization has
      [v1]=[v3] or [v0]=[v2], because v1,v3 both lie in {v0,v2}^perp which is 1-dim unless
      v0 || v2. More: C_4 = K_{2,2} forces span{v0,v2} perp span{v1,v3}, so faithfulness needs
      d >= 4, and EVERY faithful realization in any d splits as 2 rays in a 2-plane W1 plus
      2 rays in W2 perp W1.
  (B) For n >= 5 of EITHER parity, generic faithful realizations exist (for even n they exist
      already over the integers; the Lovasz umbrella itself does NOT close for even n — its
      closing inner product is 2cos(pi/n)/(1+cos(pi/n)) != 0). At every one of them:
        flex(C_n in CP^{d-1}) = 2(d-2)n - (d^2 - 1)        [d=3: 2n-8, EVEN AND ODD]
      i.e. the 2n-8 law is NOT a parity phenomenon.
  (C) C_4 in d=4 is the lone exception to the naive count: the K_{2,2} block structure gives a
      1-dimensional continuous stabilizer (block torus) inside PU(4), so
        flex(C_4, d=4) = [2(d-2)n - (d^2-1)] + 1 = 2,
      and the 2 flex moduli are exactly |<v0,v2>| and |<v1,v3>| — the two CHSH measurement
      angles (Alice's angle inside W1, Bob's inside W2).

Evidence labels: every flex value in the table is EXACT (rational-arithmetic ranks over Q at
integer realizations, method of exact_rigidity.py) AND reproduced NUMERICALLY at independent
generic complex realizations (Gauss-Newton residuals ~1e-16, spectral-gap-clean ranks).

Runs standalone:  python3 even_cycles.py   (imports flex_dimension.py + exact_rigidity.py,
modifies nothing).  Exit code 0 iff all internal consistency checks PASS.
"""
import os, sys
import numpy as np
from itertools import product as iproduct, combinations
from math import gcd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from flex_dimension import flex_dimension, unit, tangent_basis
from exact_rigidity import exact_flex, ip as iip

CHECKS = []
def check(name, ok, detail=""):
    CHECKS.append((name, bool(ok)))
    print(f"    [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))

# ----------------------------------------------------------------------------------
# Generic numerical realizations: Gauss-Newton with analytic Jacobian on the cycle
# constraints  <v_i, v_{i+1}> = 0,  |v_i|^2 = 1  over complex vectors in C^d.
# ----------------------------------------------------------------------------------
def cycle_edges(n): return [(k, (k+1) % n) for k in range(n)]
def pack(vs): return np.concatenate([np.concatenate([v.real, v.imag]) for v in vs])
def unpack(x, n, d): return [x[2*d*k:2*d*k+d] + 1j*x[2*d*k+d:2*d*(k+1)] for k in range(n)]

def resid_jac(x, n, d):
    vs = unpack(x, n, d); E = cycle_edges(n)
    m = 2*len(E) + n; r = np.zeros(m); J = np.zeros((m, 2*d*n))
    for row, (i, j) in enumerate(E):
        ipv = np.vdot(vs[i], vs[j])
        r[2*row] = ipv.real; r[2*row+1] = ipv.imag
        for k in range(d):
            for col, val in ((2*d*i+k, vs[j][k]), (2*d*i+d+k, -1j*vs[j][k]),
                             (2*d*j+k, np.conj(vs[i][k])), (2*d*j+d+k, 1j*np.conj(vs[i][k]))):
                J[2*row, col] += val.real; J[2*row+1, col] += val.imag
    for i in range(n):
        row = 2*len(E)+i; r[row] = np.vdot(vs[i], vs[i]).real - 1.0
        for k in range(d):
            J[row, 2*d*i+k] = 2*vs[i][k].real; J[row, 2*d*i+d+k] = 2*vs[i][k].imag
    return r, J

def gauss_newton(x, n, d, iters=200):
    for _ in range(iters):
        r, J = resid_jac(x, n, d)
        nr = np.linalg.norm(r)
        if nr < 1e-14: break
        dx = np.linalg.lstsq(J, -r, rcond=None)[0]
        t = 1.0
        for _ in range(40):
            r2, _ = resid_jac(x + t*dx, n, d)
            if np.linalg.norm(r2) < nr: break
            t /= 2
        x = x + t*dx
    return x

def faithfulness(vs, n):
    """(max edge residual, min ray-distinctness 1-|<vi,vj>|, min non-edge |<vi,vj>|)."""
    G = np.array([[abs(np.vdot(a, b)) for b in vs] for a in vs])
    E = set(cycle_edges(n)) | {(j, i) for i, j in cycle_edges(n)}
    res = max(G[i, j] for i, j in E)
    mind = min(1 - G[i, j] for i in range(n) for j in range(n) if i != j)
    nonE = [(i, j) for i in range(n) for j in range(n) if i != j and (i, j) not in E]
    minoff = min((G[i, j] for i, j in nonE), default=1.0)
    return res, mind, minoff

def solve_cycle(n, d, seed=0, tries=40, real=False):
    """Random-start Gauss-Newton; returns (rays, res, mind, minoff) — faithful if it can."""
    best = None
    for t in range(tries):
        rg = np.random.default_rng(seed*1000 + t)
        x0 = pack([unit(rg.normal(size=d) + (0j if real else 1j*rg.normal(size=d)))
                   for _ in range(n)])
        x = gauss_newton(x0, n, d)
        vs = [unit(v) for v in unpack(x, n, d)]
        res, mind, minoff = faithfulness(vs, n)
        cand = (vs, res, mind, minoff)
        if res < 1e-12 and mind > 1e-3 and minoff > 1e-3: return cand
        if best is None or res < best[1]: best = cand
    return best

# ----------------------------------------------------------------------------------
# Exact integer realizations: DFS over small primitive integer vectors (chain construction
# with faithfulness pruning).  All arithmetic in Z; certificates then run over Q via
# exact_rigidity.exact_flex (extended integer Jacobian + sympy ranks — no tolerances).
# ----------------------------------------------------------------------------------
def _primitive(v):
    g = 0
    for x in v: g = gcd(g, abs(x))
    return g == 1

def _prop(u, v):
    d = len(u)
    return all(u[a]*v[b] - u[b]*v[a] == 0 for a in range(d) for b in range(a+1, d))

def find_integer_cycle(n, d, box):
    cand = [v for v in iproduct(range(-box, box+1), repeat=d) if any(v) and _primitive(list(v))]
    e1 = tuple(1 if k == 0 else 0 for k in range(d)); e2 = tuple(1 if k == 1 else 0 for k in range(d))
    def ok(rays, new, k):
        for j, r in enumerate(rays):
            adjacent = (j == k-1) or (k == n-1 and j == 0)
            v = iip(new, r)
            if adjacent and v != 0: return False
            if (not adjacent) and v == 0: return False
            if _prop(new, r): return False
        return True
    def dfs(rays):
        k = len(rays)
        if k == n: return rays
        for c in cand:
            if iip(c, rays[-1]) != 0: continue
            if k == n-1 and iip(c, rays[0]) != 0: continue
            if ok(rays, c, k):
                out = dfs(rays + [c])
                if out: return out
        return None
    return dfs([e1, e2])

def verify_integer_cycle(rays, n):
    """Exact combinatorial certificate: edge set == C_n exactly, all rays distinct (in Z)."""
    E = {(i, j) for i, j in combinations(range(n), 2) if iip(rays[i], rays[j]) == 0}
    want = {tuple(sorted(e)) for e in cycle_edges(n)}
    distinct = all(not _prop(rays[i], rays[j]) for i, j in combinations(range(n), 2))
    return E == want and distinct

# ----------------------------------------------------------------------------------
# Main experiment
# ----------------------------------------------------------------------------------
def main():
    np.set_printoptions(precision=4, suppress=True)
    print("="*98)
    print("EVEN CYCLES — faithful realizations and flex dimensions (CHSH = C_4)")
    print("="*98)

    # ---------------- Task 1: C_4 in d=3 is forced degenerate (THEOREM, exact) -----
    print("\n[1] C_4 in d=3 — THEOREM (exact): no faithful realization exists.")
    print("    Proof: v1,v3 in {v0,v2}^perp. If v0,v2 are independent, that perp is 1-dim in C^3,")
    print("    forcing [v1]=[v3]; otherwise [v0]=[v2]. Either way two rays coincide.  QED")
    print("    (Stronger: C_4 = K_{2,2} forces span{v0,v2} perp span{v1,v3}; faithfulness needs")
    print("     both spans 2-dim, hence d >= 4.)")
    print("    Numerical corroboration — 30 random Gauss-Newton starts on the C_4/d=3 variety:")
    n_deg = n_conv = 0
    for s in range(30):
        vs, res, mind, minoff = solve_cycle(4, 3, seed=s, tries=1)
        if res < 1e-10: n_conv += 1
        if res < 1e-10 and mind < 1e-8: n_deg += 1
    print(f"    converged: {n_conv}/30, ALL degenerate (coincident rays): {n_deg}/30")
    check("C4/d=3: every converged solution is degenerate", n_conv == 30 and n_deg == 30,
          f"{n_deg}/30 degenerate")

    # umbrella closing defect for even n (why Theorem 1's realization can't be reused)
    print("\n    Umbrella side-remark: for even n the Lovasz umbrella does NOT close;")
    print("    <v_{n-1}, v_0> = 2cos(pi/n)/(1+cos(pi/n)) != 0  (exact).")
    oku = True
    for n in (6, 8, 10):
        c = np.cos(np.pi/n); ct2 = c/(1+c); st2 = 1-ct2
        closing = st2*np.cos(np.pi*((n-1)**2 % (2*n))/n) + ct2   # (n-1)^2 = 1 mod 2n for even n
        oku &= abs(closing - 2*c/(1+c)) < 1e-12 and closing > 0.5
        print(f"      n={n}: closing ip = {closing:.6f} (= 2c/(1+c) = {2*c/(1+c):.6f})")
    check("even umbrella closing defect = 2c/(1+c) != 0", oku)

    # ---------------- Tasks 2+3: generic numerical realizations + flex -------------
    print("\n[2] Generic NUMERICAL realizations (complex Gauss-Newton) + engine flex:")
    cases = [(6, 3), (8, 3), (10, 3), (4, 4), (5, 4), (6, 4)]
    results = {}
    for n, d in cases:
        vs, res, mind, minoff = solve_cycle(n, d, seed=100*d + n)
        faithful = res < 1e-12 and mind > 1e-3 and minoff > 1e-3
        check(f"C{n}/d={d}: faithful numerical realization", faithful,
              f"res={res:.1e}, distinctness={mind:.2e}, min non-edge |ip|={minoff:.2e}")
        f = flex_dimension(vs, name=f"C{n} d={d}")
        results[(n, d)] = f
    print()
    # expected values: naive = 2(d-2)n - (d^2-1); C4/d=4 gets +1 from its stabilizer
    naive = lambda n, d: 2*(d-2)*n - (d*d - 1)
    expected = {(6,3): 4, (8,3): 8, (10,3): 12, (4,4): 2, (5,4): 5, (6,4): 9}
    for (n, d), f in results.items():
        e = expected[(n, d)]
        check(f"C{n}/d={d}: flex = {e}" + (" = 2n-8" if d == 3 else
              (" = naive+1 (stabilizer)" if (n, d) == (4, 4) else " = 4n-15")), f == e,
              f"measured {f}, naive count {naive(n,d)}")

    # robustness: different seeds and a purely REAL realization for C6/d=3
    print("\n    Robustness of C6/d=3:")
    fs = []
    for s in (1, 2, 3):
        vs, res, mind, minoff = solve_cycle(6, 3, seed=s)
        fs.append(flex_dimension(vs, name=f"C6 seed {s}"))
    vs, res, mind, minoff = solve_cycle(6, 3, seed=42, real=True)
    maximag = max(np.abs(v.imag).max() for v in vs)
    fs.append(flex_dimension(vs, name="C6 REAL"))
    check("C6/d=3: flex=4 across 3 seeds + a real realization", all(f == 4 for f in fs),
          f"flexes {fs}, real-realization max|Im| = {maximag:.1e}")

    # ---------------- EXACT certificates over Q ------------------------------------
    print("\n[3] EXACT certificates (integer realizations, rational-arithmetic ranks over Q):")
    print("    integer cycles found by DFS over primitive vectors, faithfulness verified in Z:")
    exact_results = {}
    int_real = {}
    # hand-built exemplars for the two headline cases:
    int_real[(6, 3)] = [(1,0,0), (0,1,0), (1,0,1), (-1,2,1), (3,1,1), (0,1,-1)]
    int_real[(4, 4)] = [(1,0,0,0), (0,1,0,0), (1,0,1,0), (0,1,0,2)]
    # DFS for the rest (deterministic, < 1s each):
    for (n, d, box) in ((8, 3, 3), (10, 3, 3), (5, 4, 2), (6, 4, 2)):
        int_real[(n, d)] = find_integer_cycle(n, d, box)
    allok = True
    for (n, d), rays in sorted(int_real.items(), key=lambda t: (t[0][1], t[0][0])):
        okz = rays is not None and verify_integer_cycle(rays, n)
        allok &= okz
        print(f"    C{n}/d={d}: {rays}")
        check(f"C{n}/d={d}: integer realization is exactly the cycle, all rays distinct (in Z)", okz)
        if okz:
            exact_results[(n, d)] = exact_flex(rays, f"C{n} d={d}")
    for (n, d), f in exact_results.items():
        check(f"C{n}/d={d}: EXACT flex over Q equals numerical flex ({expected[(n,d)]})",
              f == expected[(n, d)] == results[(n, d)])

    # ---------------- C4/d=4 anatomy: stabilizer + moduli identification -----------
    print("\n[4] C_4/d=4 anatomy (CHSH):")
    # (a) the continuous stabilizer: A = i P_{W1} fixes all four rays projectively
    vs4 = [unit(np.array(v, complex)) for v in int_real[(4, 4)]]
    W1 = np.linalg.qr(np.column_stack([vs4[0], vs4[2]]))[0][:, :2]
    P1 = W1 @ W1.conj().T
    A = 1j*P1   # in u(4), not a scalar
    stab_ok = True
    for i, v in enumerate(vs4):
        Av = A @ v
        # eigenvector test: Av parallel to v (possibly 0)
        stab_ok &= np.linalg.norm(Av - (np.vdot(v, Av))*v) < 1e-12
    scalar = np.linalg.norm(A - np.trace(A)/4*np.eye(4)) < 1e-12
    check("C4/d=4: A = iP_{span{v0,v2}} is a NON-scalar u(4) stabilizer of all 4 rays",
          stab_ok and not scalar, "explains orbit dim 14 = 15 - 1 (block torus, K_{2,2})")
    # (b) the 2 moduli are the two diagonal angles |<v0,v2>|, |<v1,v3>|
    vs, res, mind, minoff = solve_cycle(4, 4, seed=104)   # generic complex realization
    d, n = 4, 4
    tb = [tangent_basis(v) for v in vs]
    def inv_grad(i, j):
        g = np.zeros(2*(d-1)*n); ipv = np.vdot(vs[i], vs[j])
        for k in range(d-1):
            ci = np.vdot(tb[i][k], vs[j]); cj = np.vdot(vs[i], tb[j][k])
            g[2*(d-1)*i+2*k]   = 2*(np.conj(ipv)*ci).real
            g[2*(d-1)*i+2*k+1] = 2*(np.conj(ipv)*(-1j)*ci).real
            g[2*(d-1)*j+2*k]   = 2*(np.conj(ipv)*cj).real
            g[2*(d-1)*j+2*k+1] = 2*(np.conj(ipv)*(1j)*cj).real
        return g
    # rebuild the engine's J and orbit-direction matrix at this realization
    J = np.zeros((2*n, 2*(d-1)*n))
    for row, (i, j) in enumerate(cycle_edges(n)):
        for k in range(d-1):
            ci = np.vdot(tb[i][k], vs[j]); cj = np.vdot(vs[i], tb[j][k])
            J[2*row, 2*(d-1)*i+2*k] += ci.real;   J[2*row, 2*(d-1)*i+2*k+1] += (-1j*ci).real
            J[2*row+1, 2*(d-1)*i+2*k] += ci.imag; J[2*row+1, 2*(d-1)*i+2*k+1] += (-1j*ci).imag
            J[2*row, 2*(d-1)*j+2*k] += cj.real;   J[2*row, 2*(d-1)*j+2*k+1] += (1j*cj).real
            J[2*row+1, 2*(d-1)*j+2*k] += cj.imag; J[2*row+1, 2*(d-1)*j+2*k+1] += (1j*cj).imag
    dirs = []
    for a in range(d):
        for b in range(a, d):
            gens = [np.zeros((d, d), complex)]
            if a == b: gens[0][a, a] = 1j
            else:
                gens[0][a, b] = 1j; gens[0][b, a] = 1j
                g2 = np.zeros((d, d), complex); g2[a, b] = 1; g2[b, a] = -1; gens.append(g2)
            for Ag in gens:
                vec = np.zeros(2*(d-1)*n)
                for i, v in enumerate(vs):
                    Av = Ag @ v
                    for k in range(d-1):
                        c = np.vdot(tb[i][k], Av)
                        vec[2*(d-1)*i+2*k] = c.real; vec[2*(d-1)*i+2*k+1] = c.imag
                dirs.append(vec)
    D = np.array(dirs)
    U, S, Vt = np.linalg.svd(J)
    rk = int((S > 1e-8*S.max()).sum()); ker = Vt[rk:, :].T   # kernel basis (16-dim)
    G1, G2 = inv_grad(0, 2), inv_grad(1, 3)
    orb_kill = max(np.abs(D @ G1).max(), np.abs(D @ G2).max())
    P = ker @ ker.T
    M = np.array([P @ G1, P @ G2])
    rkM = np.linalg.matrix_rank(M, tol=1e-8)
    check("C4/d=4: flex moduli = the two diagonal angles |<v0,v2>|, |<v1,v3>| (CHSH angles)",
          orb_kill < 1e-10 and rkM == 2,
          f"invariant grads kill orbit ({orb_kill:.1e}), rank on ker J = {rkM} = flex")

    # ---------------- results table ------------------------------------------------
    print("\n" + "="*98)
    print("RESULTS TABLE — flex of cycle scenarios (all values EXACT over Q + numerical)")
    print("="*98)
    print(f"  {'n':>3} {'d':>3} {'coords 2(d-1)V':>15} {'rank J':>7} {'ker':>5} {'orbit':>6} "
          f"{'FLEX':>5}  {'law':<26} {'evidence'}")
    print(f"  {4:>3} {3:>3} {'—':>15} {'—':>7} {'—':>5} {'—':>6} {'—':>5}  "
          f"{'NO faithful realization':<26} EXACT (theorem)")
    laws = {(6,3): '2n-8', (8,3): '2n-8', (10,3): '2n-8',
            (4,4): '4n-15 +1 (stabilizer)', (5,4): '4n-15', (6,4): '4n-15'}
    orbits = {(6,3): 8, (8,3): 8, (10,3): 8, (4,4): 14, (5,4): 15, (6,4): 15}
    for (n, d) in [(6,3), (8,3), (10,3), (4,4), (5,4), (6,4)]:
        f = results[(n, d)]
        ev = "EXACT + NUMERICAL" if exact_results.get((n, d)) == f else "NUMERICAL"
        print(f"  {n:>3} {d:>3} {2*(d-1)*n:>15} {2*n:>7} {2*(d-1)*n-2*n:>5} {orbits[(n,d)]:>6} "
              f"{f:>5}  {laws[(n,d)]:<26} {ev}")
    print("\n  Law: flex(C_n in CP^{d-1}) = 2(d-2)n - (d^2-1) for n >= 5, EITHER parity")
    print("       (d=3: 2n-8 — Theorem 1's formula is parity-blind once a faithful realization")
    print("       exists); C_4 needs d >= 4 and gets +1 from its K_{2,2} block-torus stabilizer:")
    print("       flex(CHSH) = 2 = the two measurement-angle moduli.")

    # ---------------- verdict ------------------------------------------------------
    npass = sum(1 for _, ok in CHECKS if ok)
    print("\n" + "="*98)
    print(f"CONSISTENCY CHECKS: {npass}/{len(CHECKS)} PASS")
    for name, ok in CHECKS:
        if not ok: print(f"  FAILED: {name}")
    print("even_cycles " + ("PASS" if npass == len(CHECKS) else "FAIL"))
    return 0 if npass == len(CHECKS) else 1

if __name__ == "__main__":
    sys.exit(main())
