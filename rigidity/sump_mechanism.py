#!/usr/bin/env python3
"""
B1 — SESSION 11: the mechanism behind  sum_j P_j = 11 I  for Peres-33 (Open Problem 1).

Claim (Schur/orbit mechanism, EXACT over Q(sqrt2)):
  Peres-33 is the disjoint union of FOUR orbits of the hyperoctahedral group B_3 (signed 3x3
  permutation matrices, |B_3|=48), from seeds (0,0,1),(0,1,1),(0,1,s),(1,1,s), s=sqrt2. B_3 acts
  on R^3 by its standard reflection representation, which is IRREDUCIBLE (no signed-permutation-
  invariant line: sign flips kill the all-ones vector). By Schur's lemma every single B_3-orbit of
  unit vectors is a TIGHT FRAME: sum over the orbit of |v><v| = (|orbit as rays| / 3) I. Hence
     orbit sizes (as rays) 3, 6, 12, 12  ->  frame constants 1, 2, 4, 4  ->  sum = 11,
  so sum_j P_j = 11 I is FORCED by the orbit structure, not a numerical coincidence. Along the
  Gould-Aravind circle the identity persists exactly (Theorem `witness invariance`, proved
  separately as a Laurent identity), so 11 is the value everywhere on the family.

This script proves the mechanism exactly: it rebuilds the four orbits, checks each is a single
closed B_3-orbit, and computes each orbit's frame operator in EXACT Q(sqrt2) arithmetic, verifying
it equals (|orbit|/3) I. It also re-verifies the total = 11 I and exhibits the refutation-proofing:
the constants 1,2,4,4 are exactly |orbit|/3 (a Schur prediction), distinguishing this mechanism
from the two refuted ones (11-basis partition; excess=basis-union).
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fractions import Fraction
from itertools import permutations, product
from sic_zoo import rays_peres33, q_sign_norm, Z0

# ---- exact Q(sqrt2) scalars as (a,b) = a + b*sqrt2, a,b in Q ----
def qadd(u, v): return (u[0] + v[0], u[1] + v[1])
def qmul(u, v): return (u[0]*v[0] + 2*u[1]*v[1], u[0]*v[1] + u[1]*v[0])
def qfrompair(p): return (Fraction(p[0]), Fraction(p[1]))
def qdotpair(x, y):
    s = (Fraction(0), Fraction(0))
    for a, b in zip(x, y): s = qadd(s, qmul(qfrompair(a), qfrompair(b)))
    return s

SEEDS = [((0,0),(0,0),(1,0)), ((0,0),(1,0),(1,0)), ((0,0),(1,0),(0,1)), ((1,0),(1,0),(0,1))]

def b3_orbit(seed):
    """All distinct rays in the B_3 orbit of seed (as sign-normalised Z[sqrt2] tuples)."""
    orb = []
    for perm in permutations(range(3)):
        for sg in product((1, -1), repeat=3):
            v = q_sign_norm(tuple((sg[i]*seed[perm[i]][0], sg[i]*seed[perm[i]][1]) for i in range(3)))
            if v not in orb: orb.append(v)
    return orb

def frame_operator(orbit):
    """Exact 3x3 frame operator  sum_i |v_i><v_i| / <v_i|v_i>  in Q(sqrt2); return matrix + is-scalar."""
    M = [[(Fraction(0), Fraction(0)) for _ in range(3)] for _ in range(3)]
    for v in orbit:
        n2 = qdotpair(v, v)
        # 1/n2 in Q(sqrt2): (a - b sqrt2)/(a^2 - 2 b^2)
        a, b = n2; den = a*a - 2*b*b; inv = (a/den, -b/den)
        for p in range(3):
            for q in range(3):
                term = qmul(qmul(qfrompair(v[p]), qfrompair(v[q])), inv)
                M[p][q] = qadd(M[p][q], term)
    c = M[0][0]
    is_scalar = all(M[p][q] == (c if p == q else (Fraction(0), Fraction(0)))
                    for p in range(3) for q in range(3))
    return M, c, is_scalar

def qstr(c): return f"{c[0]}" + ("" if c[1] == 0 else f"+{c[1]}*sqrt2")

if __name__ == "__main__":
    print("Sum P = 11 I for Peres-33 — the orbit/Schur mechanism (EXACT over Q(sqrt2))")
    print("="*84)
    p33 = rays_peres33()
    p33set = set(p33)
    total = [[(Fraction(0), Fraction(0)) for _ in range(3)] for _ in range(3)]
    grand_const = (Fraction(0), Fraction(0))
    allok = True
    covered = []
    for seed in SEEDS:
        orb = b3_orbit(seed)
        # orbit must be a subset of Peres-33 and closed
        subset = all(v in p33set for v in orb)
        M, c, is_scalar = frame_operator(orb)
        pred = Fraction(len(orb), 3)
        ok = subset and is_scalar and c == (pred, Fraction(0))
        allok &= ok
        covered += orb
        grand_const = qadd(grand_const, c)
        for p in range(3):
            for q in range(3): total[p][q] = qadd(total[p][q], M[p][q])
        print(f"  seed {tuple(qstr(x) for x in seed)}: |orbit|={len(orb):2d} rays  "
              f"frame op = ({qstr(c)}) I   Schur predicts |orbit|/3 = {pred}   "
              f"{'OK' if ok else 'FAIL'}")
    # the four orbits partition the 33 rays
    partition = (len(covered) == 33 and set(covered) == p33set and len(set(covered)) == 33)
    # total frame operator
    tc = total[0][0]
    total_scalar = all(total[p][q] == (tc if p == q else (Fraction(0), Fraction(0)))
                       for p in range(3) for q in range(3))
    print("-"*84)
    print(f"  orbits partition Peres-33: {partition}  (union has {len(set(covered))} distinct rays)")
    print(f"  sum of the four constants: {qstr(grand_const)}   (1+2+4+4 = 11)")
    print(f"  total frame operator = ({qstr(tc)}) I : scalar={total_scalar}, equals 11*I: {tc==(Fraction(11),Fraction(0))}")
    print("-"*84)
    verdict = allok and partition and total_scalar and tc == (Fraction(11), Fraction(0)) \
              and grand_const == (Fraction(11), Fraction(0))
    if verdict:
        print("MECHANISM CONFIRMED: each B_3 orbit is a tight frame with constant |orbit|/3 (Schur,")
        print("B_3 irreducible on R^3); 3/3+6/3+12/3+12/3 = 11 => sum P = 11 I is FORCED, not")
        print("coincidental. This resolves Open Problem 1 at theta=0; the exact Laurent identity")
        print("(witness invariance) carries 11 I along the whole Gould-Aravind circle.")
    print("\nsump_mechanism " + ("PASS" if verdict else "FAIL"))
