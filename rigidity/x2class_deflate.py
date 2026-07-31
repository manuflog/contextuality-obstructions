#!/usr/bin/env python3
"""
x2class_deflate.py -- THE DEFLATIONARY TEST.

Question: is the uncolorable content of each NEW locus a genuinely new mechanism, or an
already-known island reached through the RATIO  r = x/x*  (the route by which
alphabet_paper.tex rem:ratioB deflated the arg=30deg line)?

THE GENERAL LEMMA (proved here computationally, and it is two lines by hand).
Inside the Hermitian-closed pool over A = {0,+-1,+-x,+-x*} sits the DEGREE-ONE SUBPOOL

        D(x) = { rays with all entries in {0, +-x, +-x*} }.

Projective rescaling by 1/x* maps D(x) bijectively onto the pool over {0,+-1,+-r},
r = x/x*, and multiplies every Hermitian inner product by N = |x|^2 -- so it is an
ISOMORPHISM OF ORTHOGONALITY GRAPHS.  Hence:

    closed pool at x is KS-uncolorable  WHENEVER  the UNCLOSED pool over {0,+-1,+-r},
    r = x/x*, is KS-uncolorable.

|r| = 1 always, so only phase can act here; the unclosed unimodular pools that are
uncolorable are exactly those with r a primitive 3rd or 6th root of unity (r = +-omega,
+-omega-bar), i.e.

        2 arg x  =  +-60 or +-120 deg  (mod 360)   <=>   arg x = +-30 or +-60 (mod 180)
                                                   <=>   x on L30 or L60.

So BOTH new uncolorable loci are the Eisenstein island in disguise, at every modulus,
with no stable-graph argument and no exceptional-modulus caveat needed.

Stages:
  ratio     -- symbolic r = x/x* on every locus; is it a root of unity?
  embed     -- exact per-point verification of the lemma on L30, L60 and the
               colorable comparison loci (H+, H-, U, D45): build D(x) inside the closed
               pool, check it is a literal subgraph, and check its SYMBOL-PATTERN
               labelled graph is identical to the labelled {0,+-1,+-omega} pool.
"""
import json, os, sys, time
from itertools import combinations, product as iproduct

import sympy as sp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from x2class_pools import (make_ring, build_pool, labelled_pool, decide, CF)   # noqa: E402

CACHE = os.path.join(HERE, "x2class_deflate.cache.json")
R = sp.Rational
S3, S2, S5, S7 = sp.sqrt(3), sp.sqrt(2), sp.sqrt(5), sp.sqrt(7)


# ------------------------------------------------------------------ stage `ratio`
def ratio_stage():
    print("=" * 100)
    print("STAGE ratio -- r = x/x* on each locus, and whether r is a root of unity")
    print("=" * 100)
    t = sp.Symbol('t', positive=True)
    rows = []
    for name, xexpr, desc in [
        ("L30", t * sp.exp(sp.I * sp.pi / 6), "arg x = 30 deg, any modulus t"),
        ("L60", t * sp.exp(sp.I * sp.pi / 3), "arg x = 60 deg, any modulus t"),
        ("D45", t * sp.exp(sp.I * sp.pi / 4), "arg x = 45 deg, any modulus t"),
        ("IM ", t * sp.I, "arg x = 90 deg"),
        ("RE ", t, "x real"),
    ]:
        r = sp.simplify(xexpr / sp.conjugate(xexpr))
        rows.append((name, desc, sp.nsimplify(sp.simplify(r)), None))
    for name, desc, r, _ in rows:
        rr = sp.simplify(r)
        order = None
        for k in range(1, 25):
            if sp.simplify(rr ** k - 1) == 0:
                order = k
                break
        print("  %-4s %-32s r = %-24s root of unity of order %s"
              % (name, desc, rr, order if order else "NO (or > 24)"))
    # H+ / H- : r depends on the modulus
    print("\n  For H+ (x^2+(x*)^2 = 1) and H- (= -1):  r = x^2/N is unimodular with")
    print("     Re(r) = Re(x^2)/N = (+-1/2)/N,  N = |x|^2 free along the hyperbola.")
    print("     r is a root of unity only when Re(r) in {0,+-1/2,+-1}, i.e.")
    for sgn, lab in ((1, 'H+'), (-1, 'H-')):
        sols = []
        Nv = sp.Symbol('N', positive=True)
        for c in (0, R(1, 2), R(-1, 2), 1, -1):
            s = sp.solve(sp.Eq(sp.Rational(sgn, 2) / Nv, c), Nv)
            for v in s:
                if v.is_positive:
                    sols.append((c, v))
        print("     %s : cos(arg r) = %s/(2N) is special only at %s"
              % (lab, '1' if sgn > 0 else '-1',
                 ", ".join("N=%s (cos=%s)" % (v, c) for c, v in sols)))
    print("     -> on H+ the only root-of-unity points are N=1 (r=e^{i60}, i.e. x=e^{i30},")
    print("        which is L30 itself) and N=1/2 (r=1, x real, the M2 point x=1/sqrt2);")
    print("        on H- : N=1 (r=e^{i120} = L60) and N=1/2 (r=-1, x=i/sqrt2, M2).")
    print("     Everywhere ELSE on H+ / H- the ratio has infinite order -> nothing to deflate,")
    print("     and the scan indeed finds those points COLORABLE.")
    print("\n  For U (|x|=1): r = x^2, a root of unity iff x is; the tested points")
    print("     (3+4i)/5 etc. have infinite order (Niven) -> COLORABLE, as found.")


# ------------------------------------------------------------------ stage `embed`
def degree_one_subpool(C, x, xs, rays):
    """Indices (into `rays`) of the rays all of whose entries lie in {0,+-x,+-x*}."""
    syms = [C.zero, x, C.neg(x), xs, C.neg(xs)]
    keys = {}
    for i, v in enumerate(rays):
        k = next(j for j in range(3) if not C.iszero(v[j]))
        inv = C.inv(v[k])
        keys[tuple(C.mul(e, inv) for e in v)] = i
    idxs = []
    for pat in iproduct(range(5), repeat=3):
        v = tuple(syms[k] for k in pat)
        if all(C.iszero(e) for e in v):
            continue
        k = next(j for j in range(3) if not C.iszero(v[j]))
        inv = C.inv(v[k])
        key = tuple(C.mul(e, inv) for e in v)
        assert key in keys, "degree-one ray missing from the closed pool -- impossible"
        idxs.append(keys[key])
    return sorted(set(idxs))


def embed_stage(points):
    print("\n" + "=" * 100)
    print("STAGE embed -- the degree-one subpool D(x), and its identification with the")
    print("               UNCLOSED Eisenstein pool over {0,+-1,+-omega}")
    print("=" * 100)
    # reference: the labelled unclosed omega-pool
    Com, om, oms = make_ring(R(-1, 2), S3 / 2)                       # omega, order 3
    ref3 = labelled_pool(Com, [Com.zero, Com.one, Com.neg(Com.one), om, Com.neg(om)])
    C6, e6, _ = make_ring(R(1, 2), S3 / 2)                           # e^{i pi/3}, order 6
    # symbol layout for the order-6 case puts -r (= omega^2) in the generator slot, because
    # {0,+-1,+-e^{i pi/3}} = {0,+-1,+-omega^2} = conj{0,+-1,+-omega} AS A SET
    ref6 = labelled_pool(C6, [C6.zero, C6.one, C6.neg(C6.one), C6.neg(e6), e6])
    REFS = {3: ref3, 6: ref6}
    for k, (lb, LE, LT) in sorted(REFS.items()):
        print("  reference  {0,+-1,+-r}, ord(r)=%d, unclosed : %d rays / %d pairs / %d triads"
              % (k, len(lb), len(LE), len(LT)), end="  ")
        ok, _ = decide(len(lb), [(lb.index(a), lb.index(b)) for a, b in LE],
                       [tuple(lb.index(z) for z in t) for t in LT], "ref%d" % k)
        print("-> %s" % ("COLORABLE" if ok else "UNCOLORABLE"))
        assert not ok
    # the two alphabets are the SAME SET: {0,+-1,+-e^{i pi/3}} = {0,+-1,+-omega^2}
    e6v = R(1, 2) + S3 * sp.I / 2                      # e^{i pi/3}
    omv = R(-1, 2) + S3 * sp.I / 2                     # omega = e^{2 i pi/3}
    assert sp.simplify(-e6v - omv ** 2) == 0, "-e^{i pi/3} != omega^2"
    print("  (-e^{i pi/3} = omega^2, so the order-6 alphabet {0,+-1,+-e^{i pi/3}} is the")
    print("   SAME SET as {0,+-1,+-omega^2} = conj{0,+-1,+-omega}: one island, not two.)")
    assert ref3[0] == ref6[0] and ref3[1] == ref6[1], \
        "order-3 and order-6 labelled pools differ -- check symbol layout"
    print("  the order-3 and order-6 labelled pools are LITERALLY EQUAL (same labels, same edges).")
    ref_lab, ref_E, ref_T = ref3
    out = []
    for name, pe, qe in points:
        C, x, xs = make_ring(pe, qe)
        rays, E, Tri = build_pool(C, x, xs, closed=True)
        idxs = degree_one_subpool(C, x, xs, rays)
        pos = {v: k for k, v in enumerate(idxs)}
        Ed = [(pos[i], pos[j]) for (i, j) in E if i in pos and j in pos]
        Es = set(Ed)
        Td = [t for t in combinations(range(len(idxs)), 3)
              if all((min(a, b), max(a, b)) in Es for a, b in combinations(t, 2))]
        col, _ = decide(len(idxs), Ed, Td, name + " D(x)")
        # labelled comparison: rescale by 1/x*, symbols [0,+x*,-x*,+x,-x] -> [0,+1,-1,+r,-r]
        # (the sign convention on the generator slot is exactly the omega vs omega^2 choice)
        A = labelled_pool(C, [C.zero, xs, C.neg(xs), x, C.neg(x)])
        B = labelled_pool(C, [C.zero, xs, C.neg(xs), C.neg(x), x])
        same = (A[0] == ref_lab and A[1] == ref_E and A[2] == ref_T) or \
               (B[0] == ref_lab and B[1] == ref_E and B[2] == ref_T)
        r = C.mul(x, C.inv(xs))
        # order of r in the ring
        order = None
        pw = C.one
        for k in range(1, 25):
            pw = C.mul(pw, r)
            if pw == C.one:
                order = k
                break
        print("  %-30s closed %3d/%4d/%3d | D(x) %3d/%4d/%3d -> %-11s | ord(x/x*) = %-5s | "
              "labelled graph == omega-pool: %s"
              % (name, len(rays), len(E), len(Tri), len(idxs), len(Ed), len(Td),
                 "COLORABLE" if col else "UNCOLORABLE", order, same))
        out.append({'name': name, 'closed': [len(rays), len(E), len(Tri)],
                    'D': [len(idxs), len(Ed), len(Td)], 'D_colorable': bool(col),
                    'order_ratio': order, 'labelled_equals_omega': bool(same)})
    return out


if __name__ == '__main__':
    ratio_stage()
    pts = [
        # ---- L30, several moduli
        ("L30 t=2   x=sqrt3+i",       S3, sp.Integer(1)),
        ("L30 t=4   x=2sqrt3+2i",     2 * S3, sp.Integer(2)),
        ("L30 t=2/3 x=(sqrt3+i)/3",   S3 / 3, R(1, 3)),
        ("L30 t=sqrt5",               sp.sqrt(15) / 2, S5 / 2),
        ("L30 t=1   x=e^{i30}",       S3 / 2, R(1, 2)),
        ("L30 t=sqrt3 x=2+omega",     R(3, 2), S3 / 2),
        # ---- L60, several moduli (incl. the two special moduli t=2 and t=1/2)
        ("L60 t=2   x=1+isqrt3",      sp.Integer(1), S3),
        ("L60 t=1/2 x=(1+isqrt3)/4",  R(1, 4), S3 / 4),
        ("L60 t=4   x=2+2isqrt3",     sp.Integer(2), 2 * S3),
        ("L60 t=2/3 x=(1+isqrt3)/3",  R(1, 3), S3 / 3),
        ("L60 t=sqrt5",               S5 / 2, sp.sqrt(15) / 2),
        ("L60 t=1   x=e^{i60}",       R(1, 2), S3 / 2),
        # ---- colorable comparison loci
        ("H+ x=3/2+isqrt7/2",         R(3, 2), S7 / 2),
        ("H- x=1/4+3i/4",             R(1, 4), R(3, 4)),
        ("U  x=(3+4i)/5",             R(3, 5), R(4, 5)),
        ("D45 x=2+2i",                sp.Integer(2), sp.Integer(2)),
        ("M2 x=(1+7i)/5",             R(1, 5), R(7, 5)),
        ("CTRL x=1+2i",               sp.Integer(1), sp.Integer(2)),
    ]
    res = embed_stage(pts)
    json.dump(res, open(CACHE, 'w'), indent=1)
    print("\ncache -> %s" % CACHE)
