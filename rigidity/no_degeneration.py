#!/usr/bin/env python3
"""
B1 — SESSION 11: EXACT no-degeneration of the Peres-Penrose circle (upgrades the grid scan).

The circle v_j(theta) is an exact one-parameter family (peres_penrose.SLICE, Laurent polynomials
in z=e^{i theta} over Z[sqrt2]). The 72 orthogonality polynomials vanish identically (proved,
thm:circle); a "degeneration" would be a NON-edge pair (i,j) whose Hermitian inner product
<v_i(theta), v_j(theta)> vanishes at some real theta, adding an accidental edge. Numerically the
minimum non-edge |Gram| stays ~0.12>0; here we prove NON-VANISHING EXACTLY for all real theta and
all 456 non-edge pairs.

Method (exact, rational root counting only — NO algebraic-extension solves):
  L(z) = <v_i, v_j>(theta) = sum_k A_k z^k,  A_k in Z[sqrt2], k in {-2..2}.
  |L|^2(theta) = L * conj(L) is a real Laurent poly with Z[sqrt2] coeffs, symmetric M_{-k}=M_k, so
     g(c) := |L|^2 = M_0 + sum_{k>0} 2 M_k T_k(c),   c = cos theta,  T_k = Chebyshev, deg g <= 4.
  Split g(c) = p(c) + sqrt2 * q(c) with p, q in Z[c]. If g(c*)=0 for some real c* then
  sqrt2 = -p(c*)/q(c*) (or p=q=0), so c* is a root of the RATIONAL polynomial
     h(c) := p(c)^2 - 2 q(c)^2  in Z[c].
  Hence: if h has NO real root in [-1,1] and g(1) > 0 (the Peres point theta=0, a genuine non-edge),
  then g(c) > 0 for all c in [-1,1], i.e. the pair never becomes orthogonal. Root counting for h is
  done over Q with sympy's real_root_isolation/count_roots (exact, fast).

A single surviving root in [-1,1] for any pair would be a real degeneration and must be reported.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from itertools import combinations
from fractions import Fraction
from sympy import Poly, symbols, Integer
from peres_penrose import SLICE, L_herm, L_conj, L_mul, T1_EDGES

c = symbols('c')
# Chebyshev T_0..T_4 in c
T = [Poly(1, c), Poly(c, c), Poly(2*c**2 - 1, c), Poly(4*c**3 - 3*c, c), Poly(8*c**4 - 8*c**2 + 1, c)]

def laurent_abs2(L):
    """|L|^2 as a Laurent dict (real Z[sqrt2] coeffs), symmetric."""
    return L_mul(L, L_conj(L))

def g_pq(M):
    """From symmetric Laurent |L|^2 dict M (exp -> (a,b)=a+b sqrt2) build p(c), q(c) in Z[c]
       with g(c) = p + sqrt2 q,  g = sum_k M_k z^k = M_0 + sum_{k>0} 2 M_k T_k(c)."""
    p = Poly(0, c); q = Poly(0, c)
    seen = set()
    for e, (a, b) in M.items():
        k = e[0]
        if k < 0: continue
        ak = a if k == 0 else 2 * a
        bk = b if k == 0 else 2 * b
        p = p + Integer(ak) * T[abs(k)]
        q = q + Integer(bk) * T[abs(k)]
        seen.add(k)
    return p, q

def g_at_one(M):
    """g(1) = |L|^2 at theta=0 (z=1) = sum_k M_k  as (a,b)=a+b sqrt2; return float sign check value."""
    a = sum(v[0] for v in M.values()); b = sum(v[1] for v in M.values())
    return a, b   # value a + b*sqrt2

def positive_q2(a, b):
    """exact test a + b*sqrt2 > 0 for rational a,b."""
    if a == 0 and b == 0: return False
    if a >= 0 and b >= 0: return True
    if a <= 0 and b <= 0: return False
    # mixed sign: compare a^2 vs 2 b^2 with the sign of the sqrt2 term
    if a > 0:   # a>0, b<0: positive iff a^2 > 2 b^2
        return a * a > 2 * b * b
    else:       # a<0, b>0: positive iff 2 b^2 > a^2
        return 2 * b * b > a * a

def main():
    edges = {tuple(sorted(e)) for e in T1_EDGES}
    V = len(SLICE)
    nonedges = [(i, j) for i, j in combinations(range(V), 2) if (i, j) not in edges]
    print(f"Peres-Penrose circle: {V} rays, {len(edges)} edges, {len(nonedges)} non-edge pairs")
    print("Proving each non-edge Gram is strictly positive for ALL real theta (exact)...")
    bad = []; checked = 0; peres_pos = 0
    for (i, j) in nonedges:
        L = L_herm(SLICE[i], SLICE[j])
        if not L:                       # identically zero would be a hidden edge — cannot happen
            bad.append((i, j, "identically zero!")); continue
        M = laurent_abs2(L)
        p, q = g_pq(M)
        h = (p * p - 2 * q * q)          # in Z[c]
        # exact real-root count of h on [-1,1]
        if h.is_zero:
            bad.append((i, j, "h==0 (|L|^2 has a real-circle root family)")); continue
        nroots = h.count_roots(-1, 1)    # closed interval, exact
        a1, b1 = g_at_one(M)
        if positive_q2(Fraction(a1), Fraction(b1)): peres_pos += 1
        if nroots != 0:
            # a root of h is only a CANDIDATE; confirm it is an actual root of g by checking there
            bad.append((i, j, f"h has {nroots} root(s) in [-1,1] — candidate degeneration"))
        checked += 1
    print(f"  non-edges checked: {checked}; Peres-point (theta=0) positive: {peres_pos}/{len(nonedges)}")
    if not bad:
        print("  h(c)=p^2-2q^2 has NO root in [-1,1] for every non-edge pair, and g(1)>0 throughout")
        print("  => every non-edge Gram is strictly positive for all real theta.")
        print("\nNO-DEGENERATION PROVED (exact): the orthogonality graph is constant along the entire")
        print("circle; no accidental edge ever appears. The family is a genuine 1-parameter loop of")
        print("33-ray KS sets with constant flex. no_degeneration PASS")
    else:
        print(f"\n  {len(bad)} candidate degeneration(s):")
        for i, j, why in bad[:20]: print(f"    ({i},{j}): {why}")
        print("no_degeneration ATTENTION — candidates must be resolved (check if h-root is a true g-root)")

if __name__ == "__main__":
    main()
