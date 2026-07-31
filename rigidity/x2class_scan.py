#!/usr/bin/env python3
"""
x2class_scan.py -- KS-colorability of the Hermitian-CLOSED pool at exact test points on
every locus produced by x2class_derive.py, old and new.

Every point is exact (its own number field, built by x2class_pools.make_ring); every
verdict is decided TWICE (pysat Glucose3 + an independent triad-driven backtracker) and
the two must agree.  Results cached in x2class_scan.cache.json.

Loci covered
  L30   x^2+(x*)^2 =  N     arg x = +-30 deg (mod 180)          [NEW, 1-dimensional]
  L60   x^2+(x*)^2 = -N     arg x = +-60 deg (mod 180)          [NEW, 1-dimensional]
  H+    x^2+(x*)^2 =  1     rect. hyperbola p^2-q^2 =  1/2      [NEW, 1-dimensional]
  H-    x^2+(x*)^2 = -1     rect. hyperbola p^2-q^2 = -1/2      [NEW, 1-dimensional]
  U     N = 1               unit circle  (2-TERM identity 1-N=0; missed by the published
                            'two-term sums are trivial' claim, already for OPEN pools)
  IM    x + x* = 0          imaginary axis (2-TERM; also missed, also OPEN-relevant)
  D45   x^2+(x*)^2 = 0      arg x = +-45 deg (2-TERM, closed-only)
  CTRL  generic points avoiding everything
Usage:  python3 x2class_scan.py [group ...]      (default: all)
"""
import json, os, sys, time
import sympy as sp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from x2class_pools import analyse, make_ring, build_pool, decide   # noqa: E402

CACHE = os.path.join(HERE, "x2class_scan.cache.json")
S3, S2, S5, S7, S14, S6, S15, S23, S11 = (sp.sqrt(3), sp.sqrt(2), sp.sqrt(5), sp.sqrt(7),
                                          sp.sqrt(14), sp.sqrt(6), sp.sqrt(15), sp.sqrt(23),
                                          sp.sqrt(11))
R = sp.Rational


def L30(t):
    """x = t e^{i30}"""
    return (t * S3 / 2, t / 2)


def L60(t):
    """x = t e^{i60}"""
    return (t / 2, t * S3 / 2)


GROUPS = {
    # ---------------- NEW locus L30: x^2 + (x*)^2 = N, arg x = 30 deg, modulus t free
    'L30': [("L30 t=2   x=sqrt3+i",            L30(2)),
            ("L30 t=4   x=2sqrt3+2i",          L30(4)),
            ("L30 t=6   x=3sqrt3+3i",          L30(6)),
            ("L30 t=2/3 x=(sqrt3+i)/3",        L30(R(2, 3))),
            ("L30 t=5   x=5(sqrt3+i)/2",       L30(5)),
            ("L30 t=sqrt3 x=2+omega",          L30(S3)),
            ("L30 t=sqrt5 x=sqrt15/2+isqrt5/2", L30(S5)),
            ("L30 t=sqrt2 (also on M2)",       L30(S2)),
            ("L30 t=1   x=e^{i30} (also |x|=1)", L30(1))],
    # ---------------- NEW locus L60: x^2 + (x*)^2 = -N, arg x = 60 deg
    'L60': [("L60 t=2   x=1+isqrt3  [special]", L60(2)),
            ("L60 t=1/2 x=(1+isqrt3)/4 [special]", L60(R(1, 2))),
            ("L60 t=4   x=2+2isqrt3",          L60(4)),
            ("L60 t=6   x=3+3isqrt3",          L60(6)),
            ("L60 t=2/3 x=(1+isqrt3)/3",       L60(R(2, 3))),
            ("L60 t=5   x=5(1+isqrt3)/2",      L60(5)),
            ("L60 t=sqrt3 x=sqrt3/2+3i/2",     L60(S3)),
            ("L60 t=sqrt5 x=sqrt5/2+isqrt15/2", L60(S5)),
            ("L60 t=1   x=e^{i60} (also M3)",  L60(1))],
    # ---------------- NEW locus H+ : x^2+(x*)^2 = 1  (p^2-q^2 = 1/2)
    'H+': [("H+ x=3/4+i/4      N=5/8",   (R(3, 4), R(1, 4))),
           ("H+ x=9/8+7i/8     N=65/32", (R(9, 8), R(7, 8))),
           ("H+ x=1+isqrt2/2   N=3/2",   (sp.Integer(1), S2 / 2)),
           ("H+ x=3/2+isqrt7/2 N=4",     (R(3, 2), S7 / 2)),
           ("H+ x=2+isqrt14/2  N=15/2",  (sp.Integer(2), S14 / 2)),
           ("H+ x=sqrt2+isqrt6/2 N=7/2", (S2, S6 / 2)),
           ("H+ x=5/2+isqrt23/2 N=12",   (R(5, 2), S23 / 2)),
           ("H+ x=sqrt2/2 (real, =M2)",  (S2 / 2, sp.Integer(0)))],
    # ---------------- NEW locus H- : x^2+(x*)^2 = -1  (p^2-q^2 = -1/2)
    'H-': [("H- x=1/4+3i/4      N=5/8",   (R(1, 4), R(3, 4))),
           ("H- x=7/8+9i/8      N=65/32", (R(7, 8), R(9, 8))),
           ("H- x=sqrt2/2+i     N=3/2",   (S2 / 2, sp.Integer(1))),
           ("H- x=sqrt7/2+3i/2  N=4",     (S7 / 2, R(3, 2))),
           ("H- x=sqrt14/2+2i   N=15/2",  (S14 / 2, sp.Integer(2))),
           ("H- x=sqrt6/2+isqrt2 N=7/2",  (S6 / 2, S2)),
           ("H- x=isqrt2/2 (=M2)",        (sp.Integer(0), S2 / 2)),
           ("H- x=1/2+isqrt3/2 (=M3,omega6)", (R(1, 2), S3 / 2))],
    # ---------------- 2-TERM locus U : |x| = 1
    'U': [("U x=(3+4i)/5",   (R(3, 5), R(4, 5))),
          ("U x=(5+12i)/13", (R(5, 13), R(12, 13))),
          ("U x=(8+15i)/17", (R(8, 17), R(15, 17))),
          ("U x=(7+24i)/25", (R(7, 25), R(24, 25))),
          ("U x=(20+21i)/29", (R(20, 29), R(21, 29)))],
    # ---------------- 2-TERM locus IM : Re x = 0
    'IM': [("IM x=2i",     (sp.Integer(0), sp.Integer(2))),
           ("IM x=3i",     (sp.Integer(0), sp.Integer(3))),
           ("IM x=5i/2",   (sp.Integer(0), R(5, 2))),
           ("IM x=isqrt3", (sp.Integer(0), S3)),
           ("IM x=isqrt5", (sp.Integer(0), S5)),
           ("IM x=i/3",    (sp.Integer(0), R(1, 3)))],
    # ---------------- 2-TERM locus D45 : x^2 + (x*)^2 = 0
    'D45': [("D45 x=1+i",         (sp.Integer(1), sp.Integer(1))),
            ("D45 x=2+2i",        (sp.Integer(2), sp.Integer(2))),
            ("D45 x=3+3i",        (sp.Integer(3), sp.Integer(3))),
            ("D45 x=sqrt2+isqrt2", (S2, S2)),
            ("D45 x=5/2+5i/2",    (R(5, 2), R(5, 2)))],
    # ---------------- the PUBLISHED loci M1..M5 at generic (non-intersection) points,
    #                  plus the degenerate / root-of-unity points
    'KNOWN': [("M2 x=(1+7i)/5    N=2",   (R(1, 5), R(7, 5))),
              ("M2 x=(7+i)/5     N=2",   (R(7, 5), R(1, 5))),
              ("M2 x=(1+7i)/10   N=1/2", (R(1, 10), R(7, 10))),
              ("M2 x=sqrt2 (real)",      (S2, sp.Integer(0))),
              ("M2 x=1+i (Gaussian)",    (sp.Integer(1), sp.Integer(1))),
              ("M2 x=(1+isqrt7)/2 Heegner", (R(1, 2), S7 / 2)),
              ("M2 x=isqrt2",            (sp.Integer(0), S2)),
              ("M3 x=1/2+3i",            (R(1, 2), sp.Integer(3))),
              ("M3 x=1/2+2i",            (R(1, 2), sp.Integer(2))),
              ("M3 x=1/2+i/3",           (R(1, 2), R(1, 3))),
              ("M3 x=-1/2+2i",           (R(-1, 2), sp.Integer(2))),
              ("M5 x=2/5+4i/5",          (R(2, 5), R(4, 5))),
              ("M5 x=1/5+3i/5",          (R(1, 5), R(3, 5))),
              ("M5 x=8/5+4i/5",          (R(8, 5), R(4, 5))),
              ("M4 x=(1+sqrt5)/2 golden", ((1 + S5) / 2, sp.Integer(0))),
              ("M1 x=2",                 (sp.Integer(2), sp.Integer(0))),
              ("M1 x=1/2",               (R(1, 2), sp.Integer(0))),
              ("deg x=i",                (sp.Integer(0), sp.Integer(1))),
              ("deg x=1",                (sp.Integer(1), sp.Integer(0))),
              ("deg x=omega",            (R(-1, 2), S3 / 2))],
    # ---------------- controls: avoid every locus
    'CTRL': [("CTRL x=3",     (sp.Integer(3), sp.Integer(0))),
             ("CTRL x=1+2i",  (sp.Integer(1), sp.Integer(2))),
             ("CTRL x=2+3i",  (sp.Integer(2), sp.Integer(3))),
             ("CTRL x=4+i",   (sp.Integer(4), sp.Integer(1))),
             ("CTRL x=1+3i",  (sp.Integer(1), sp.Integer(3))),
             ("CTRL x=sqrt5+2i", (S5, sp.Integer(2)))],
}

p_, q_ = sp.symbols('p q', real=True)
LOCI = {
    'M1': lambda p, q: q == 0 and p in (2, -2, R(1, 2), R(-1, 2)),
    'M2': lambda p, q: sp.simplify(p ** 2 + q ** 2 - 2) == 0 or sp.simplify(p ** 2 + q ** 2 - R(1, 2)) == 0,
    'M3': lambda p, q: sp.simplify(p - R(1, 2)) == 0 or sp.simplify(p + R(1, 2)) == 0,
    'M4': lambda p, q: q == 0 and (sp.simplify(p ** 2 - p - 1) == 0 or sp.simplify(p ** 2 + p - 1) == 0),
    'M5': lambda p, q: sp.simplify(2 * p - (p ** 2 + q ** 2)) == 0 or sp.simplify(2 * p + (p ** 2 + q ** 2)) == 0,
    'L30': lambda p, q: sp.simplify(p ** 2 - 3 * q ** 2) == 0,
    'L60': lambda p, q: sp.simplify(3 * p ** 2 - q ** 2) == 0,
    'H+': lambda p, q: sp.simplify(2 * p ** 2 - 2 * q ** 2 - 1) == 0,
    'H-': lambda p, q: sp.simplify(2 * p ** 2 - 2 * q ** 2 + 1) == 0,
    'U': lambda p, q: sp.simplify(p ** 2 + q ** 2 - 1) == 0,
    'IM': lambda p, q: sp.simplify(p) == 0,
    'RE': lambda p, q: sp.simplify(q) == 0,
    'D45': lambda p, q: sp.simplify(p ** 2 - q ** 2) == 0,
}


def loci_of(p, q):
    return [k for k, f in LOCI.items() if f(sp.nsimplify(p), sp.nsimplify(q))]


def run(groups):
    cache = {}
    if os.path.exists(CACHE):
        cache = json.load(open(CACHE))
    for g in groups:
        print("=" * 108)
        print("GROUP %s" % g)
        print("=" * 108)
        for name, (pe, qe) in GROUPS[g]:
            for closed in (True, False):
                key = "%s|%s" % (name, closed)
                if key in cache:
                    r = cache[key]
                    print("  [cached] %-36s %s %3d/%4d/%3d -> %s"
                          % (name, "closed" if closed else "open  ",
                             r['rays'], r['pairs'], r['triads'],
                             "COLORABLE" if r['colorable'] else "UNCOLORABLE"))
                    continue
                r = analyse(name, pe, qe, closed=closed, verbose=False)
                r['loci'] = loci_of(pe, qe)
                r['N'] = str(sp.nsimplify(sp.simplify(pe ** 2 + qe ** 2)))
                cache[key] = r
                print("  %-36s %s %3d/%4d/%3d -> %-11s  N=%-8s loci=%s"
                      % (name, "closed" if closed else "open  ",
                         r['rays'], r['pairs'], r['triads'],
                         "COLORABLE" if r['colorable'] else "UNCOLORABLE",
                         r['N'], ",".join(r['loci'])))
                json.dump(cache, open(CACHE, 'w'), indent=1)
    return cache


if __name__ == '__main__':
    gs = sys.argv[1:] or list(GROUPS)
    run(gs)
    print("\ncache -> %s" % CACHE)
