#!/usr/bin/env python3
"""
B1 — SESSION 11: the graded cycle split, GENERAL d and BOTH parities (extends thm:cyclesplit).

Predicted (from the decomposition + the general cycle law flex_C = 2(d-2)n - (d^2-1)):
    flex_R(C_n, d)    = (d-2) n - d(d-1)/2
    flex_skew(C_n, d) = (d-2) n - (d+2)(d-1)/2
    sum               = 2(d-2) n - (d^2 - 1)   [= general cycle law, thm:generallaw]

Proof (conditional; the three conditions are the natural generalizations of Lemmas I/II):
  Let J_ext be the extended real Jacobian at a faithful real realization. The general cycle law
  is equivalent to rank J_ext = 3n (stress-freeness, Lemma I' of thm:generallaw). By the
  decomposition J_ext = R (>= 2n rows) (+) S (<= n rows); rank R + rank S = 3n forces
  rank R = 2n, rank S = n.  [C1]
  Real trivial = so(d): a stabilizer A=-A^T with A v_i in R v_i has v_i.Av_i=0 => A v_i=0; the rays
  span R^d (faithful) => A=0, so rank(real trivial) = d(d-1)/2.  [C2]  => flex_R = (d-2)n - d(d-1)/2.
  Skew trivial = {n phases} + {Sym(d)}: a relation is a symmetric M with every v_i an eigenvector;
  when the only such M is scalar (irreducible realization), there is exactly ONE relation, so
  rank(skew trivial) = n + d(d+1)/2 - 1.  [C3]  => flex_skew = (d-2)n - (d+2)(d-1)/2.

This script builds a faithful REAL realization for each (n,d) and checks the two formulas AND the
three conditions C1 (block ranks 2n, n), C2 (so(d) free), C3 (exactly one skew relation), so the
conditional theorem's hypotheses are machine-verified at explicit realizations. Evidence: NUMERICAL
(Gauss-Newton realizations, spectral-gap-clean ranks); the odd d=3 umbrella is the EXACT anchor
(cycle_split_theorem.py). C_4 is excluded (no faithful d=3 realization; the CHSH exception).
"""
import os, sys, io, contextlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from itertools import combinations
from even_cycles import solve_cycle

def rrank(rows, tol=1e-8):
    M = np.array(rows, float)
    if M.size == 0: return 0
    s = np.linalg.svd(M, compute_uv=False)
    return int((s / s[0] > tol).sum()) if s[0] > 0 else 0

def blocks(rays):
    rays = [np.asarray(v, float) for v in rays]
    d, V = len(rays[0]), len(rays); n = d * V
    E = [(i, j) for i, j in combinations(range(V), 2) if abs(np.dot(rays[i], rays[j])) < 1e-9]
    R, S, TR, TS = [], [], [], []
    for i, j in E:
        r = np.zeros(n)
        for c in range(d): r[d*i+c] += rays[j][c]; r[d*j+c] += rays[i][c]
        R.append(r)
        r = np.zeros(n)
        for c in range(d): r[d*j+c] += rays[i][c]; r[d*i+c] -= rays[j][c]
        S.append(r)
    for i in range(V):
        r = np.zeros(n)
        for c in range(d): r[d*i+c] = rays[i][c]
        R.append(r)
    for a in range(d):
        for b in range(a+1, d):
            t = np.zeros(n)
            for i in range(V): t[d*i+a] += rays[i][b]; t[d*i+b] -= rays[i][a]
            TR.append(t)
    for k in range(V):
        t = np.zeros(n)
        for cc in range(d): t[d*k+cc] = rays[k][cc]
        TS.append(t)
    for a in range(d):
        t = np.zeros(n)
        for i in range(V): t[d*i+a] = rays[i][a]
        TS.append(t)
    for a in range(d):
        for b in range(a+1, d):
            t = np.zeros(n)
            for i in range(V): t[d*i+a] += rays[i][b]; t[d*i+b] += rays[i][a]
            TS.append(t)
    return R, S, TR, TS, n, len(E)

def pred_R(n, d): return (d-2)*n - d*(d-1)//2
def pred_S(n, d): return (d-2)*n - (d+2)*(d-1)//2

def check(n, d, tries=80):
    with contextlib.redirect_stdout(io.StringIO()):
        rays, res, mind, minoff = solve_cycle(n, d, tries=tries, real=True)
    if rays is None or res > 1e-9:
        print(f"  C{n:<2} d={d}: no real faithful realization found (res={res})"); return None
    rr = [np.asarray(v).real for v in rays]
    R, S, TR, TS, N, E = blocks(rr)
    rR, rS, rTR, rTS = rrank(R), rrank(S), rrank(TR), rrank(TS)
    flexR = (N - rR) - rTR; flexS = (N - rS) - rTS
    C1 = (rR == 2*n and rS == n)
    C2 = (rTR == d*(d-1)//2)
    C3 = (rTS == n + d*(d+1)//2 - 1)          # exactly one skew relation
    form = (flexR == pred_R(n, d) and flexS == pred_S(n, d))
    ok = C1 and C2 and C3 and form
    par = "odd " if n % 2 else "even"
    print(f"  C{n:<2} d={d} ({par}): flexR={flexR}(pred {pred_R(n,d)}) flexS={flexS}(pred {pred_S(n,d)}) "
          f"| C1(ranks {rR},{rS})={C1} C2(so{d}={rTR})={C2} C3(skew-rel=1)={C3}  {'OK' if ok else 'FAIL'}")
    return ok

if __name__ == "__main__":
    print("GRADED CYCLE SPLIT — general d, both parities (conditional theorem + machine-checked)")
    print("  flex_R=(d-2)n-d(d-1)/2,  flex_skew=(d-2)n-(d+2)(d-1)/2")
    grid = [(5,3),(7,3),(9,3),(6,3),(8,3),(10,3),(5,4),(7,4),(6,4),(8,4),(5,5),(6,5),(7,5)]
    allok = True; done = 0
    for n, d in grid:
        r = check(n, d)
        if r is not None: allok &= r; done += 1
    print(f"\ncycle_split_general {'PASS' if allok and done>=10 else 'CHECK'} "
          f"({done} realizations verified; odd d=3 is proved exactly in cycle_split_theorem.py)")
