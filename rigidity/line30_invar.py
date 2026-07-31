#!/usr/bin/env python3
# line30_invar.py -- TASK 3a: unitary invariants at two moduli, exact in Q(sqrt3).
#
# For a configuration of rays {rho_i} the multiset { |<rho_i, rho_j>|^2 } over all
# unordered pairs is invariant under global unitaries, per-ray phases, relabeling, and
# antiunitaries.  We compute it EXACTLY (K = Q(sqrt3), Fractions) at t = 2 and t = 3 for
#   (a) the FULL 145-ray stable pool  -> if multisets differ, the two realizations are
#       unitarily INEQUIVALENT: the t-line is a genuine curve of inequivalent
#       realizations of the constant stable graph;
#   (b) the 43-ray critical core      -> its rays are projectively t-constant, so the
#       multiset must be IDENTICAL at all t (checked exactly);
# plus one Bargmann triple invariant on mixed rays as a second separating invariant.
import json, os, sys
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from itertools import combinations
from fractions import Fraction

from line30_ring import kadd, ksub, kmul, kneg, kzero, kinv, KZERO

HERE = os.path.dirname(os.path.abspath(__file__))
pool = json.load(open(os.path.join(HERE, "line30_stable_pool.cache.json")))
core = json.load(open(os.path.join(HERE, "line30_core.cache.json")))
RAYS = [tuple(r) for r in pool["rays"]]
CORE = [tuple(s) for s in core["core_syms"]]


def entry(sym, t):
    """(Re, Im) in K of the alphabet entry at x = t*zeta."""
    th = Fraction(t) / 2
    base = {"0": ((Fraction(0), Fraction(0)), (Fraction(0), Fraction(0))),
            "1": ((Fraction(1), Fraction(0)), (Fraction(0), Fraction(0))),
            "X": ((Fraction(0), th), (th, Fraction(0))),
            "Y": ((Fraction(0), th), (-th, Fraction(0)))}
    if sym.startswith("-"):
        re, im = base[sym[1:]]
        return (kneg(re), kneg(im))
    return base[sym]


def hdot(u, v, t):
    """<u, v> = sum conj(u_c) v_c as (Re, Im) K-pair."""
    A, B = KZERO, KZERO
    for su, sv in zip(u, v):
        (ru, iu), (rv, iv) = entry(su, t), entry(sv, t)
        # conj(u)*v = (ru - i iu)(rv + i iv) = (ru rv + iu iv) + i(ru iv - iu rv)
        A = kadd(A, kadd(kmul(ru, rv), kmul(iu, iv)))
        B = kadd(B, ksub(kmul(ru, iv), kmul(iu, rv)))
    return A, B


def norm2(u, t):
    A, _ = hdot(u, u, t)
    return A


def gram2_multiset(rays, t):
    """multiset of |<rho_i, rho_j>|^2 (normalized), exact K values."""
    out = []
    n2 = [norm2(r, t) for r in rays]
    for i, j in combinations(range(len(rays)), 2):
        A, B = hdot(rays[i], rays[j], t)
        num = kadd(kmul(A, A), kmul(B, B))
        val = kmul(num, kinv(kmul(n2[i], n2[j])))
        out.append((val[0], val[1]))
    return sorted(out)


for label, rays in (("FULL POOL", RAYS), ("CORE", CORE)):
    m2 = gram2_multiset(rays, 2)
    m3 = gram2_multiset(rays, 3)
    same = (m2 == m3)
    print(f"[invar] {label}: Gram-moduli^2 multiset over {len(m2)} pairs: "
          f"t=2 vs t=3 -> {'IDENTICAL' if same else 'DIFFERENT'}")
    if not same:
        diff2 = [v for v in set(m2) - set(m3)][:4]
        diff3 = [v for v in set(m3) - set(m2)][:4]
        fmt = lambda v: f"{v[0]}+{v[1]}*sqrt3"
        print(f"        values at t=2 only (sample): {[fmt(v) for v in diff2]}")
        print(f"        values at t=3 only (sample): {[fmt(v) for v in diff3]}")

# one explicit separating pair, human-checkable: rho = (0,1,X) vs sigma = (0,1,0):
# |<sigma, rho>|^2 / (1 * (1+t^2)) = 1/(1+t^2): 1/5 at t=2, 1/10 at t=3.
u, v = ("0", "1", "0"), ("0", "1", "X")
for t in (2, 3):
    A, B = hdot(u, v, t)
    num = kadd(kmul(A, A), kmul(B, B))
    val = kmul(num, kinv(kmul(norm2(u, t), norm2(v, t))))
    print(f"[invar] |<(0,1,0),(0,1,X)>|^2 normalized at t={t}: {val[0]} + {val[1]}*sqrt3")

# Bargmann invariant of a mixed triple (third-order unitary invariant), exact:
tri = (("0", "1", "X"), ("1", "0", "0"), ("1", "1", "-X"))
for t in (2, 3):
    prod = ((Fraction(1), Fraction(0)), KZERO)   # complex K-pair (Re, Im)
    for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
        A, B = hdot(a, b, t)
        pr, pi = prod
        prod = (ksub(kmul(pr, A), kmul(pi, B)), kadd(kmul(pr, B), kmul(pi, A)))
    n = kmul(norm2(tri[0], t), kmul(norm2(tri[1], t), norm2(tri[2], t)))
    prod = (kmul(prod[0], kinv(n)), kmul(prod[1], kinv(n)))
    print(f"[invar] Bargmann <1,2><2,3><3,1> (normalized) of {tri} at t={t}: "
          f"Re = {prod[0][0]}+{prod[0][1]}*sqrt3, Im = {prod[1][0]}+{prod[1][1]}*sqrt3")
