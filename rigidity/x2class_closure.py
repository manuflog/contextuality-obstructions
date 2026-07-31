#!/usr/bin/env python3
"""
x2class_closure.py -- closing the corrected closed-pool dichotomy.

THE ARGUMENT.  x2class_derive.py shows that the complete list of loci on which SOME
(<= 3 term) vanishing sum over the closed term set holds is

   curves   M2a p^2+q^2=2   M2b p^2+q^2=1/2   M3a p=1/2   M3b p=-1/2
            M5a p^2+q^2=2p  M5b p^2+q^2=-2p   L30 p^2=3q^2  L60 q^2=3p^2
            H+  2(p^2-q^2)=1   H- 2(p^2-q^2)=-1
            U   p^2+q^2=1   IM p=0   RE q=0   D45a p=q   D45b p=-q
   points   M1 {+-2,+-1/2}   M4 {golden}   {+-1}  {+-i}  {primitive 3rd/6th roots}
            and 8 extra points, all of which lie ON L60.

Because the classification is EXHAUSTIVE, whether a given pair of alphabet vectors is
orthogonal depends only on WHICH of these loci contain x.  Hence the index-labelled
orthogonality graph of the closed pool is a FUNCTION OF THE LOCUS-MEMBERSHIP VECTOR
alone.  So it suffices to test one representative of every realizable membership class:
  * the generic class (no locus),
  * one generic point of each curve,
  * every point lying on >= 2 loci  --  i.e. every pairwise intersection point, since
    any higher intersection is in particular a pairwise one.

WHY THE MEMBERSHIP-GRAPH PRINCIPLE IS A THEOREM AND NOT A HEURISTIC.  Two things can vary
with x: which symbol patterns give the same RAY, and which ray pairs are ORTHOGONAL.
  * orthogonality: <u,v> = sum_i conj(u_i) v_i, at most three nonzero terms, each of one of
    the six closed types -> covered by the <=3-term classification;
  * ray identity: u ~ v iff u x v = 0, whose entries are u_a v_b - u_b v_a, i.e. TWO-term
    sums of products u_a v_b with u_a, v_b in {+-1,+-x,+-x*}.  Those products are
    1, x, x*, x^2, x x* = N, (x*)^2 -- THE SAME six types.  So ray identity is covered by
    the 2-term classification (x2class_derive.py stage 2), whose loci are U, IM, RE, D45
    and isolated points lying on them, all of which are in the curve list below.
Hence both the ray set and the edge set are determined by the locus-membership vector.

Stage `lemma`   : verify the membership-graph principle at label level (two different
                  representatives of the same class give the LITERALLY identical labelled
                  graph, over different number fields).
Stage `audit`   : exact pairwise intersections of all 15 curves and all point loci.
Stage `verdict` : colorability at every representative; assert the dichotomy
                     UNCOLORABLE  <=>  x in M1 u M2 u L30 u L60.
"""
import json, os, sys, time
from itertools import combinations

import sympy as sp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from x2class_pools import make_ring, build_pool, labelled_pool, decide      # noqa: E402
from x2class_derive import finite_solve, norm_poly                          # noqa: E402

CACHE = os.path.join(HERE, "x2class_closure.cache.json")
p, q = sp.symbols('p q', real=True)
R = sp.Rational
S3, S5 = sp.sqrt(3), sp.sqrt(5)

CURVES = [
    ('M2a', p ** 2 + q ** 2 - 2),
    ('M2b', p ** 2 + q ** 2 - R(1, 2)),
    ('M3a', p - R(1, 2)),
    ('M3b', p + R(1, 2)),
    ('M5a', p ** 2 + q ** 2 - 2 * p),
    ('M5b', p ** 2 + q ** 2 + 2 * p),
    ('L30', p ** 2 - 3 * q ** 2),
    ('L60', 3 * p ** 2 - q ** 2),
    ('H+', 2 * p ** 2 - 2 * q ** 2 - 1),
    ('H-', 2 * p ** 2 - 2 * q ** 2 + 1),
    ('U', p ** 2 + q ** 2 - 1),
    ('IM', p),
    ('RE', q),
    ('D45a', p - q),
    ('D45b', p + q),
]
GOLD = sp.solve(p ** 2 - p - 1, p) + sp.solve(p ** 2 + p - 1, p)
POINTS = ([('M1', (sp.Integer(s), sp.Integer(0))) for s in (2, -2)] +
          [('M1', (R(s, 2), sp.Integer(0))) for s in (1, -1)] +
          [('M4', (g, sp.Integer(0))) for g in GOLD] +
          # the 8 extra ISOLATED points of the closed classification (2x+-(x*)^2 = 0 and
          # x+-2(x*)^2 = 0).  All of them lie ON L60; they are listed separately because
          # they carry EXTRA edges beyond the generic L60 graph.
          [('P2', (sp.Integer(a), b * S3)) for a in (1, -1) for b in (1, -1)] +
          [('Ph', (R(a, 4), R(b, 4) * S3)) for a in (1, -1) for b in (1, -1)])
UNCOL = {'M2a', 'M2b', 'L30', 'L60', 'M1'}      # the predicted uncolorable union


def membership(pv, qv):
    return tuple(sorted([nm for nm, f in CURVES if sp.simplify(f.subs({p: pv, q: qv})) == 0] +
                        [nm for nm, (a, b) in POINTS
                         if sp.simplify(pv - a) == 0 and sp.simplify(qv - b) == 0]))


def predicted_uncolorable(mem):
    return any(m in UNCOL for m in mem)


# --------------------------------------------------------------------- stage lemma
def lemma():
    print("=" * 104)
    print("STAGE lemma -- the labelled closed graph depends only on the locus-membership class")
    print("=" * 104)
    S7 = sp.sqrt(7)
    tests = [   # every POSITIVE-DIMENSIONAL membership class, >= 2 representatives each
        ("generic", [(sp.Integer(1), sp.Integer(2)), (sp.Integer(2), sp.Integer(3)),
                     (S5, sp.Integer(2))]),
        ("L30", [(S3, sp.Integer(1)), (2 * S3, sp.Integer(2)), (sp.sqrt(15) / 2, S5 / 2)]),
        ("L60", [(sp.Integer(2), 2 * S3), (S5 / 2, sp.sqrt(15) / 2), (R(1, 3), S3 / 3)]),
        ("M2a", [(R(1, 5), R(7, 5)), (R(7, 5), R(1, 5))]),
        ("M2b", [(R(1, 10), R(7, 10)), (R(7, 10), R(1, 10))]),
        ("M3a", [(R(1, 2), sp.Integer(3)), (R(1, 2), sp.Integer(2)), (R(1, 2), R(1, 3))]),
        ("M3b", [(R(-1, 2), sp.Integer(3)), (R(-1, 2), sp.Integer(2)), (R(-1, 2), R(1, 3))]),
        ("M5a", [(R(2, 5), R(4, 5)), (R(1, 5), R(3, 5)), (R(8, 5), R(4, 5))]),
        ("M5b", [(R(-2, 5), R(4, 5)), (R(-1, 5), R(3, 5)), (R(-8, 5), R(4, 5))]),
        ("H+", [(R(3, 4), R(1, 4)), (R(3, 2), S7 / 2), (R(9, 8), R(7, 8))]),
        ("H-", [(R(1, 4), R(3, 4)), (S7 / 2, R(3, 2)), (R(7, 8), R(9, 8))]),
        ("U", [(R(3, 5), R(4, 5)), (R(5, 13), R(12, 13)), (R(8, 17), R(15, 17))]),
        ("IM", [(sp.Integer(0), sp.Integer(3)), (sp.Integer(0), sp.Integer(2)),
                (sp.Integer(0), R(5, 2))]),
        ("RE", [(sp.Integer(3), sp.Integer(0)), (sp.Integer(5), sp.Integer(0)),
                (sp.Integer(7), sp.Integer(0))]),
        ("D45a", [(sp.Integer(2), sp.Integer(2)), (sp.Integer(3), sp.Integer(3)),
                  (sp.sqrt(2), sp.sqrt(2))]),
        ("D45b", [(sp.Integer(2), sp.Integer(-2)), (sp.Integer(3), sp.Integer(-3)),
                  (sp.sqrt(2), -sp.sqrt(2))]),
    ]
    for cls, pts in tests:
        base = None
        mem0 = None
        for (pe, qe) in pts:
            mem = membership(sp.nsimplify(pe), sp.nsimplify(qe))
            C, x, xs = make_ring(pe, qe)
            lab, E, T = labelled_pool(C, [C.zero, C.one, C.neg(C.one), x, C.neg(x),
                                          xs, C.neg(xs)])
            if base is None:
                base, mem0 = (lab, E, T), mem
            else:
                assert mem == mem0, "%s: representatives in different classes %s vs %s" % (
                    cls, mem0, mem)
                assert lab == base[0] and E == base[1] and T == base[2], \
                    "%s: labelled graphs DIFFER between representatives of one class" % cls
        print("  %-8s class %-24s : %d reps, labelled graph identical, %d rays / %d pairs / %d triads"
              % (cls, str(mem0), len(pts), len(base[0]), len(base[1]), len(base[2])))
    print("  -> membership-graph principle confirmed on all %d positive-dimensional classes "
          "/ %d representatives.\n" % (len(tests), sum(len(t[1]) for t in tests)))


# --------------------------------------------------------------------- stage audit
def audit():
    print("=" * 104)
    print("STAGE audit -- all pairwise intersections of the 15 curves (exact)")
    print("=" * 104)
    reps = {}          # membership class -> representative (pv,qv)
    # generic representative
    reps[()] = (sp.Integer(1), sp.Integer(2))
    # one generic point per curve (found by scanning a parameter until it hits only that curve)
    generic_on = {
        'M2a': (R(1, 5), R(7, 5)), 'M2b': (R(1, 10), R(7, 10)),
        'M3a': (R(1, 2), sp.Integer(3)), 'M3b': (R(-1, 2), sp.Integer(3)),
        'M5a': (R(2, 5), R(4, 5)), 'M5b': (R(-2, 5), R(4, 5)),
        'L30': (S3, sp.Integer(1)), 'L60': (sp.Integer(2), 2 * S3),   # t=4: avoids P2/Ph
        'H+': (R(3, 4), R(1, 4)), 'H-': (R(1, 4), R(3, 4)),
        'U': (R(3, 5), R(4, 5)), 'IM': (sp.Integer(0), sp.Integer(3)),
        'RE': (sp.Integer(3), sp.Integer(0)),
        'D45a': (sp.Integer(2), sp.Integer(2)), 'D45b': (sp.Integer(2), sp.Integer(-2)),
    }
    for nm, pt in generic_on.items():
        mem = membership(*pt)
        assert nm in mem, "%s: chosen generic point is not on the curve" % nm
        reps.setdefault(mem, pt)
    for nm, pt in POINTS:
        reps.setdefault(membership(*pt), pt)
    npairs = 0
    for (n1, f1), (n2, f2) in combinations(CURVES, 2):
        g = sp.gcd(f1, f2)
        if g.free_symbols:
            print("  !! %s and %s share a component -- unexpected" % (n1, n2))
            continue
        pts = finite_solve(sp.expand(f1), sp.expand(f2))
        npairs += 1
        for (pv, qv) in pts:
            if pv == 0 and qv == 0:
                continue
            mem = membership(pv, qv)
            reps.setdefault(mem, (pv, qv))
    print("  %d curve pairs solved exactly; %d distinct membership classes to test."
          % (npairs, len(reps)))
    return reps


# --------------------------------------------------------------------- stage verdict
def verdict(reps):
    print("\n" + "=" * 104)
    print("STAGE verdict -- colorability of the closed pool at one representative of each class")
    print("=" * 104)
    cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
    bad = []
    rows = []
    for mem in sorted(reps, key=lambda m: (len(m), m)):
        pv, qv = reps[mem]
        key = "%s|%s" % (sp.srepr(pv), sp.srepr(qv))
        if key in cache:
            r = cache[key]
        else:
            C, x, xs = make_ring(pv, qv)
            rays, E, T = build_pool(C, x, xs, closed=True)
            col, _ = decide(len(rays), E, T, str(mem))
            r = {'rays': len(rays), 'pairs': len(E), 'triads': len(T), 'colorable': bool(col)}
            cache[key] = r
            json.dump(cache, open(CACHE, 'w'), indent=1)
        pred = predicted_uncolorable(mem)
        ok = (pred == (not r['colorable']))
        rows.append((mem, pv, qv, r, pred, ok))
        print("  x = %-30s %-46s %3d/%4d/%3d -> %-11s  predicted %-11s %s"
              % ("%s + %s i" % (pv, qv), ",".join(mem) if mem else "(generic)",
                 r['rays'], r['pairs'], r['triads'],
                 "COLORABLE" if r['colorable'] else "UNCOLORABLE",
                 "UNCOLORABLE" if pred else "COLORABLE",
                 "ok" if ok else "*** MISMATCH ***"))
        if not ok:
            bad.append((mem, pv, qv))
    print("\n  classes tested: %d ; mismatches: %d" % (len(rows), len(bad)))
    assert not bad, "DICHOTOMY FALSIFIED at %s" % bad
    print("  DICHOTOMY CONFIRMED on every realizable membership class:")
    print("     closed pool over {0,+-1,+-x,+-x*} in C^3 is KS-UNCOLORABLE")
    print("       <=>  |x|^2 in {2,1/2}  or  x in {+-2,+-1/2}   (mechanism A on x)")
    print("       or   arg x = +-30 or +-60 deg (mod 180)       (mechanism B on x/x*)")
    return rows


def openaudit():
    """A second, smaller scope error in thm:structure, in the OPEN-pool statement it kept.

    The theorem asserts that for x avoiding M1-M4 the open pool's graph is 'identical, as an
    index-labelled graph', to the one at x'=3.  It is not: at a real avoiding point x*=x, so
    the term types x and x* coincide and 54 extra two-term cancellations appear.  What IS
    true (and all the proof needs) is the INCLUSION."""
    print("\n" + "=" * 104)
    print("STAGE openaudit -- thm:structure's 'index-identical to x'=3' claim, OPEN pools")
    print("=" * 104)
    def op(pe, qe):
        C, x, xs = make_ring(pe, qe)
        return labelled_pool(C, [C.zero, C.one, C.neg(C.one), x, C.neg(x)])
    a = op(sp.Integer(3), sp.Integer(0))
    d = op(sp.Integer(5), sp.Integer(0))
    b = op(sp.Integer(1), sp.Integer(2))
    c = op(sp.Integer(2), sp.Integer(3))
    print("  x'=3   open pool %d/%d/%d   (the theorem's reference point)"
          % (len(a[0]), len(a[1]), len(a[2])))
    print("  x=5    open pool %d/%d/%d   labelled graph == x'=3 : %s"
          % (len(d[0]), len(d[1]), len(d[2]), d == a))
    print("  x=1+2i open pool %d/%d/%d   labelled graph == x'=3 : %s   EDGES SUBSET of x'=3 : %s"
          % (len(b[0]), len(b[1]), len(b[2]), b == a, b[1] <= a[1]))
    print("  x=2+3i open pool %d/%d/%d   labelled graph == x=1+2i : %s"
          % (len(c[0]), len(c[1]), len(c[2]), c == b))
    assert b[0] == a[0] and b[1] < a[1] and b == c and d == a
    print("  -> the two avoiding classes (real / non-real) give DIFFERENT graphs: %d edges of the"
          % len(a[1] - b[1]))
    print("     x'=3 graph are absent at a non-real avoiding x.  The claim 'identical' is false;")
    print("     the true and sufficient statement is 'subgraph of the x'=3 graph on the same")
    print("     ray labels', which still gives colorability.  (Scope repair, conclusion intact.)")
    # M3 vs M5: is the 1/x duality still visible in the CLOSED graph?
    def cl(pe, qe):
        C, x, xs = make_ring(pe, qe)
        return labelled_pool(C, [C.zero, C.one, C.neg(C.one), x, C.neg(x), xs, C.neg(xs)])
    m3 = cl(R(1, 2), sp.Integer(3))
    m5 = cl(R(2, 5), R(4, 5))
    print("  [aside] closed graphs on M3 (%d/%d/%d) and on M5 (%d/%d/%d): labelled-identical: %s"
          % (len(m3[0]), len(m3[1]), len(m3[2]), len(m5[0]), len(m5[1]), len(m5[2]), m3 == m5))


if __name__ == '__main__':
    t0 = time.time()
    lemma()
    openaudit()
    reps = audit()
    verdict(reps)
    print("\ntotal %.1fs ; cache -> %s" % (time.time() - t0, CACHE))
