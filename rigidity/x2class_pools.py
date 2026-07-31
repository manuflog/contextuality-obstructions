#!/usr/bin/env python3
"""
x2class_pools.py -- exact-arithmetic engine for Hermitian-CLOSED two-symbol pools
A = {0,+-1,+-x,+-x*} in C^3, plus KS-colorability by two independent deciders.

RING CONSTRUCTION (generic, one ring per test point, built automatically).
For x = p + i q with p,q real algebraic, put K = Q(p,q) (a REAL number field; sympy's
primitive_element gives a primitive element theta with p,q as exact Q-polynomials in
theta) and work in K[i] = K + K i.  Complex conjugation is then the literal sign flip on
the i-part -- no floating point anywhere.  This is the generic version of the hand-built
Z[sqrt3,i] and Z[omega] rings used in x2closed_sqrt3i.py / x2closed_omega2.py, and it is
cross-checked against them (stage `selfcheck`).

Rays are deduplicated by exact projective NORMALISATION (divide by the first nonzero
entry, using exact inversion in K[i]) rather than by pairwise cross products -- much
faster, and cross-checked against the cross-product test on the first pool built.

COLORABILITY.  A KS colouring assigns 0/1 to rays so that (a) no orthogonal pair is
both-1 and (b) every orthogonal triad carries a 1.  Decided twice:
  * pysat Glucose3 on the CNF  (-i,-j) for pairs, (i,j,k) for triads;
  * an independent triad-driven backtracker (choose the unsatisfied triad with fewest
    live rays, branch on which of its rays is 1, propagate 0 to all its neighbours).
Both must agree or the script aborts.
"""
import json, os, sys, time
from fractions import Fraction as F
from itertools import combinations, product as iproduct

import sympy as sp
from sympy.polys.numberfields import primitive_element

HERE = os.path.dirname(os.path.abspath(__file__))


# ================================================================= real number field
class RF:
    """Q[T]/(m), m monic irreducible, coefficients ASCENDING."""

    def __init__(self, masc):
        self.m = [F(c) for c in masc]
        assert self.m[-1] == 1, "minimal polynomial must be monic"
        self.d = len(masc) - 1
        self._one = tuple([F(1)] + [F(0)] * (self.d - 1))
        self._zero = tuple([F(0)] * self.d)

    def zero(self):
        return self._zero

    def one(self):
        return self._one

    def rat(self, r):
        v = [F(0)] * self.d
        v[0] = F(r)
        return tuple(v)

    def add(self, a, b):
        return tuple(x + y for x, y in zip(a, b))

    def sub(self, a, b):
        return tuple(x - y for x, y in zip(a, b))

    def neg(self, a):
        return tuple(-x for x in a)

    def mul(self, a, b):
        d = self.d
        pr = [F(0)] * (2 * d - 1)
        for i, ai in enumerate(a):
            if ai:
                for j, bj in enumerate(b):
                    if bj:
                        pr[i + j] += ai * bj
        for k in range(2 * d - 2, d - 1, -1):
            c = pr[k]
            if c:
                pr[k] = F(0)
                for t in range(d):
                    pr[k - d + t] -= c * self.m[t]
        return tuple(pr[:d])

    def inv(self, a):
        d = self.d
        cols = []
        for j in range(d):
            e = [F(0)] * d
            e[j] = F(1)
            cols.append(self.mul(a, tuple(e)))
        # solve  sum_j y_j * cols[j] = e_0
        M = [[cols[j][i] for j in range(d)] + [F(1) if i == 0 else F(0)] for i in range(d)]
        for c in range(d):
            piv = next((r for r in range(c, d) if M[r][c] != 0), None)
            assert piv is not None, "singular -> element is zero, cannot invert"
            M[c], M[piv] = M[piv], M[c]
            pv = M[c][c]
            M[c] = [v / pv for v in M[c]]
            for r in range(d):
                if r != c and M[r][c] != 0:
                    f = M[r][c]
                    M[r] = [vr - f * vc for vr, vc in zip(M[r], M[c])]
        return tuple(M[i][d] for i in range(d))


class CF:
    """K[i] = K + K i, with K = RF.  Elements are pairs (a, b) of K-elements."""

    def __init__(self, K):
        self.K = K
        self.zero = (K.zero(), K.zero())
        self.one = (K.one(), K.zero())

    def add(self, u, v):
        K = self.K
        return (K.add(u[0], v[0]), K.add(u[1], v[1]))

    def sub(self, u, v):
        K = self.K
        return (K.sub(u[0], v[0]), K.sub(u[1], v[1]))

    def neg(self, u):
        K = self.K
        return (K.neg(u[0]), K.neg(u[1]))

    def mul(self, u, v):
        K = self.K
        return (K.sub(K.mul(u[0], v[0]), K.mul(u[1], v[1])),
                K.add(K.mul(u[0], v[1]), K.mul(u[1], v[0])))

    def conj(self, u):
        return (u[0], self.K.neg(u[1]))

    def iszero(self, u):
        return u == self.zero

    def inv(self, u):
        K = self.K
        den = K.add(K.mul(u[0], u[0]), K.mul(u[1], u[1]))
        di = K.inv(den)
        return (K.mul(u[0], di), K.neg(K.mul(u[1], di)))


def make_ring(pexpr, qexpr, verbose=False):
    """Build (CF, x, xstar) for x = pexpr + i qexpr, both real algebraic sympy numbers."""
    T = sp.Symbol('T')
    pexpr, qexpr = sp.nsimplify(pexpr), sp.nsimplify(qexpr)
    f, coeffs, reps = primitive_element([pexpr, qexpr], T, ex=True)
    fc = sp.Poly(f, T).all_coeffs()
    lead = sp.Rational(fc[0])
    masc = list(reversed([sp.Rational(c) / lead for c in fc]))   # force monic
    K = RF([sp.Rational(c) for c in masc])
    def rep2vec(rep):
        asc = list(reversed(rep))
        asc = [sp.Rational(c) for c in asc] + [sp.Integer(0)] * (K.d - len(asc))
        assert len(asc) == K.d, "rep longer than field degree"
        return tuple(F(sp.Rational(c)) for c in asc)
    pv, qv = rep2vec(reps[0]), rep2vec(reps[1])
    # ---- self-check: numerically re-evaluate p and q from the representation
    th = sum(sp.Rational(c) * e for c, e in zip(coeffs, [pexpr, qexpr]))
    thn = sp.N(th, 40)
    for vec, tgt in ((pv, pexpr), (qv, qexpr)):
        val = sum(sp.Rational(c.numerator, c.denominator) * thn ** k for k, c in enumerate(vec))
        assert abs(sp.N(val - tgt, 40)) < sp.Float('1e-25'), \
            "field representation failed for %s (got %s)" % (tgt, val)
    C = CF(K)
    x = (pv, qv)
    xs = C.conj(x)
    if verbose:
        print("     ring: Q(theta)[i], minpoly(theta) = %s, deg = %d" % (sp.factor(f), K.d))
    return C, x, xs


# ================================================================= pool construction
def build_pool(C, x, xs, closed=True):
    alpha = [C.zero, C.one, C.neg(C.one), x, C.neg(x)]
    if closed:
        alpha += [xs, C.neg(xs)]
    rays, seen = [], {}
    for v in iproduct(alpha, repeat=3):
        if all(C.iszero(e) for e in v):
            continue
        k = next(i for i in range(3) if not C.iszero(v[i]))
        inv = C.inv(v[k])
        key = tuple(C.mul(e, inv) for e in v)
        if key not in seen:
            seen[key] = len(rays)
            rays.append(v)
    n = len(rays)

    def hdot(u, v):
        s = C.zero
        for a, b in zip(u, v):
            s = C.add(s, C.mul(C.conj(a), b))
        return s

    E = [(i, j) for i in range(n) for j in range(i + 1, n) if C.iszero(hdot(rays[i], rays[j]))]
    Es = set(E)
    Tri = [t for t in combinations(range(n), 3)
           if all((a, b) in Es for a, b in combinations(t, 2))]
    return rays, E, Tri


def labelled_pool(C, syms):
    """Pool over an alphabet given as a list of ring elements `syms` (syms[0] must be 0).

    Rays are labelled INDEX-INDEPENDENTLY by the lexicographically smallest SYMBOL PATTERN
    (a triple of indices into `syms`) in their projective class.  Two pools built over
    different rings but the same symbol layout can therefore be compared literally.
    Returns (labels sorted, edge set of label pairs, triad set)."""
    assert C.iszero(syms[0])
    cls = {}
    for pat in iproduct(range(len(syms)), repeat=3):
        v = tuple(syms[k] for k in pat)
        if all(C.iszero(e) for e in v):
            continue
        k = next(i for i in range(3) if not C.iszero(v[i]))
        inv = C.inv(v[k])
        key = tuple(C.mul(e, inv) for e in v)
        if key not in cls or pat < cls[key][0]:
            cls[key] = (pat, v)
    labels = sorted(p for p, _ in cls.values())
    vecs = {p: v for p, v in cls.values()}

    def hdot(u, v):
        s = C.zero
        for a, b in zip(u, v):
            s = C.add(s, C.mul(C.conj(a), b))
        return s

    E = set()
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            if C.iszero(hdot(vecs[labels[i]], vecs[labels[j]])):
                E.add((labels[i], labels[j]))
    idx = {l: i for i, l in enumerate(labels)}
    Ei = set((idx[a], idx[b]) for a, b in E)
    Tri = set()
    for t in combinations(range(len(labels)), 3):
        if all((min(a, b), max(a, b)) in Ei for a, b in combinations(t, 2)):
            Tri.add(tuple(labels[k] for k in t))
    return labels, E, Tri


def cross_check_rays(C, rays):
    """Independent projective-distinctness check by cross products (the reference method)."""
    def cross(u, v):
        return (C.sub(C.mul(u[1], v[2]), C.mul(u[2], v[1])),
                C.sub(C.mul(u[2], v[0]), C.mul(u[0], v[2])),
                C.sub(C.mul(u[0], v[1]), C.mul(u[1], v[0])))
    n = len(rays)
    for i in range(n):
        for j in range(i + 1, n):
            if all(C.iszero(e) for e in cross(rays[i], rays[j])):
                return False, (i, j)
    return True, None


# ================================================================= colorability
def sat_colorable(n, E, Tri):
    from pysat.solvers import Glucose3
    s = Glucose3()
    for i, j in E:
        s.add_clause([-(i + 1), -(j + 1)])
    for (i, j, k) in Tri:
        s.add_clause([i + 1, j + 1, k + 1])
    r = s.solve()
    model = s.get_model() if r else None
    s.delete()
    return r, model


def bt_colorable(n, E, Tri, node_cap=4_000_000):
    """Independent backtracker: branch on which ray of an unsatisfied triad gets the 1."""
    nbr = [set() for _ in range(n)]
    for i, j in E:
        nbr[i].add(j)
        nbr[j].add(i)
    tri_of = [[] for _ in range(n)]
    for t_idx, t in enumerate(Tri):
        for r in t:
            tri_of[r].append(t_idx)
    state = [None] * n          # None unknown, 1 = coloured, 0 = not
    nodes = [0]

    def assign1(r, trail):
        """set r to 1 and force its neighbours to 0; False on contradiction."""
        stack = [r]
        while stack:
            u = stack.pop()
            if state[u] == 1:
                continue
            if state[u] == 0:
                return False
            state[u] = 1
            trail.append(u)
            for w in nbr[u]:
                if state[w] == 1:
                    return False
                if state[w] is None:
                    state[w] = 0
                    trail.append(w)
        return True

    def undo(trail, mark):
        while len(trail) > mark:
            state[trail.pop()] = None

    def pick():
        best, bestlen = None, 99
        for t_idx, t in enumerate(Tri):
            if any(state[r] == 1 for r in t):
                continue
            live = [r for r in t if state[r] is None]
            if not live:
                return 'DEAD', None
            if len(live) < bestlen:
                best, bestlen = (t_idx, live), len(live)
                if bestlen == 1:
                    break
        return ('OK', best) if best else ('SAT', None)

    def rec():
        nodes[0] += 1
        if nodes[0] > node_cap:
            raise RuntimeError("backtracker node cap exceeded")
        st, best = pick()
        if st == 'DEAD':
            return False
        if st == 'SAT':
            return True
        _t_idx, live = best
        for r in live:
            trail = []
            mark = 0
            ok = assign1(r, trail)
            if ok and rec():
                return True
            undo(trail, mark)
        return False

    return rec(), nodes[0]


def decide(n, E, Tri, label=""):
    a, _model = sat_colorable(n, E, Tri)
    b, nodes = bt_colorable(n, E, Tri)
    assert a == b, "SAT (%s) and backtracker (%s) DISAGREE at %s" % (a, b, label)
    return a, nodes


# ================================================================= driver helpers
def analyse(name, pexpr, qexpr, closed=True, verbose=True, xcheck=False):
    t0 = time.time()
    C, x, xs = make_ring(pexpr, qexpr, verbose=verbose)
    rays, E, Tri = build_pool(C, x, xs, closed=closed)
    if xcheck:
        ok, bad = cross_check_rays(C, rays)
        assert ok, "ray dedup disagrees with cross-product test at %s" % (bad,)
    col, nodes = decide(len(rays), E, Tri, label=name)
    dt = time.time() - t0
    if verbose:
        print("  %-34s %s  %3d rays / %4d pairs / %3d triads -> %s   (%.1fs, %d bt-nodes)"
              % (name, "closed" if closed else "open  ", len(rays), len(E), len(Tri),
                 "COLORABLE" if col else "UNCOLORABLE", dt, nodes))
    return {'name': name, 'p': str(pexpr), 'q': str(qexpr), 'closed': closed,
            'rays': len(rays), 'pairs': len(E), 'triads': len(Tri), 'colorable': bool(col)}


def selfcheck():
    print("=" * 96)
    print("STAGE 0 -- engine self-check against the two hand-built reference rings")
    print("=" * 96)
    # x = sqrt3 + i : reference x2closed_sqrt3i.py -> closed 145/390/30 UNCOLORABLE,
    #                                                 open   49/60/4  COLORABLE
    r = analyse("x = sqrt3 + i", sp.sqrt(3), sp.Integer(1), closed=True, xcheck=True)
    assert (r['rays'], r['pairs'], r['triads'], r['colorable']) == (145, 390, 30, False), r
    r = analyse("x = sqrt3 + i", sp.sqrt(3), sp.Integer(1), closed=False)
    assert (r['rays'], r['pairs'], r['triads'], r['colorable']) == (49, 60, 4, True), r
    # x = 2 + omega = 3/2 + i sqrt3/2 : reference x2closed_omega2.py -> 145/486/38 UNCOLORABLE
    r = analyse("x = 2 + omega", sp.Rational(3, 2), sp.sqrt(3) / 2, closed=True)
    assert (r['rays'], r['pairs'], r['triads'], r['colorable']) == (145, 486, 38, False), r
    # x = 2 + 2i : rem:closedfail records 127 rays
    r = analyse("x = 2 + 2i", sp.Integer(2), sp.Integer(2), closed=True)
    assert r['rays'] == 127, r
    # x = 3 (the paper's colorable reference point) : 49 rays, colorable
    r = analyse("x = 3 (real reference)", sp.Integer(3), sp.Integer(0), closed=True)
    assert (r['rays'], r['colorable']) == (49, True), r
    # x = omega : Eisenstein, closed 57/174/22 uncolorable (TWO_MECHANISM sec.3)
    r = analyse("x = omega", sp.Rational(-1, 2), sp.sqrt(3) / 2, closed=True)
    assert (r['rays'], r['pairs'], r['triads'], r['colorable']) == (57, 174, 22, False), r
    print("  ALL REFERENCE NUMBERS REPRODUCED -- engine trusted.\n")


if __name__ == '__main__':
    selfcheck()
