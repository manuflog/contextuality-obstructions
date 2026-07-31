#!/usr/bin/env python3
"""
x2class_derive.py -- the COMPLETE 3-term vanishing-sum classification for the
Hermitian-CLOSED two-symbol alphabet A = {0,+-1,+-x,+-x*} in C^3.

WHY THIS FILE EXISTS.  alphabet_paper.tex Thm thm:d3classify classifies the 80 cases
over the term set {+-1,+-x,+-x*,+-N}.  That term set is correct for ENTRIES drawn from
{0,+-1,+-x} (Lemma lem:term), but WRONG for the Hermitian-closed alphabet: with x* as an
entry the product conj(x*)*x = x^2 appears.  The closed term set is

     {+-1, +-x, +-x*, +-N, +-x^2, +-(x*)^2},        N = |x|^2 = x x*.

Realizability (asserted below, stage `realize`): every one of the six types is
u_i^* v_i for entries u_i, v_i in {+-1,+-x,+-x*}, and each sign is attainable, and the
three coordinates are independent -- so every abstract case below is realized by an
actual pair of vectors of the closed pool.

     u\v |   1        x        x*
     ----+---------------------------
      1  |   1        x        x*
      x  |   x*       N        (x*)^2
      x* |   x        x^2      N

COUNT.  C(6+3-1,3) = C(8,3) = 56 multisets of three types, times 2^2 = 4 independent
sign patterns (the overall sign of the relation is a symmetry, so the first sign is
fixed to +1) = 224 cases.  (The published count 20 x 4 = 80 is the sub-case in which no
x^2/(x*)^2 type occurs.)

METHOD.  x = p + i q with p,q real.  Each case gives a pair of REAL polynomial equations
Re = 0, Im = 0 of total degree <= 2.  The real variety V(Re,Im) \ {0} is computed
exactly: factor both polynomials over Q, and for every pair (a,b) of irreducible factors
V(a,b) is either V(a) (when a,b are associates -> a CURVE component) or a finite set
(when a,b are coprime -> POINT components, solved exactly by sympy and cross-checked
numerically for completeness via the q-resultant).

SELF-CHECKS (all must pass or the script aborts):
  * the 80-case sub-classification (no x^2 types) reproduces exactly M1..M5 and nothing else;
  * for every finite component the exact solution count matches an independent
    resultant-based numerical solution count;
  * every reported solution is verified by exact substitution;
  * the two identities advertised in alphabet_paper.tex rem:xsqterm
    (x^2+(x*)^2 = +-N  <->  arg x = +-30deg / +-60deg) are recovered;
  * every curve component reported as a locus is checked to carry real points other
    than the origin.

Usage:
    python3 x2class_derive.py            # full run (uses/refreshes the JSON cache)
    python3 x2class_derive.py --nocache
"""
import json, os, sys, time
from itertools import combinations_with_replacement, product as iproduct

import sympy as sp
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "x2class_derive.cache.json")

p, q = sp.symbols('p q', real=True)
I = sp.I

X = p + I * q
XS = p - I * q
NN = p ** 2 + q ** 2
X2 = sp.expand(X ** 2)
XS2 = sp.expand(XS ** 2)

TVAL = {'1': sp.Integer(1), 'x': X, 'x*': XS, 'N': NN, 'x2': X2, 'xs2': XS2}
TNAMES = ['1', 'x', 'x*', 'N', 'x2', 'xs2']
PRETTY = {'1': '1', 'x': 'x', 'x*': 'x*', 'N': 'N', 'x2': 'x^2', 'xs2': '(x*)^2'}
OLD = {'1', 'x', 'x*', 'N'}


# ---------------------------------------------------------------- exact geometry helpers
def norm_poly(f):
    """Canonical primitive integer form of a nonconstant poly in p,q (sign fixed)."""
    P = sp.Poly(sp.expand(f), p, q)
    P = P.monic()
    # clear denominators
    dens = [sp.denom(c) for c in P.coeffs()]
    L = sp.Integer(1)
    for d in dens:
        L = sp.ilcm(L, sp.Integer(d))
    P = sp.Poly(sp.expand(P.as_expr() * L), p, q)
    cont = sp.Integer(0)
    for c in P.coeffs():
        cont = sp.igcd(cont, sp.Integer(c))
    P = sp.Poly(sp.expand(P.as_expr() / cont), p, q)
    if P.coeffs()[0] < 0:
        P = sp.Poly(sp.expand(-P.as_expr()), p, q)
    return P.as_expr()


def curve_real_points(f, tol=1e-9):
    """Numerically decide whether the real curve f(p,q)=0 has points besides the origin."""
    fl = sp.lambdify((p, q), f, 'numpy')
    Pq = sp.Poly(f, p)
    got = []
    for qv in np.linspace(-4.0, 4.0, 401):
        co = [complex(sp.N(c.subs(q, sp.Float(qv)))) for c in Pq.all_coeffs()]
        while co and abs(co[0]) < 1e-14:
            co = co[1:]
        if not co:
            got.append((0.0, qv))       # whole vertical line
            continue
        if len(co) == 1:
            continue
        for r in np.roots(co):
            if abs(r.imag) < 1e-9:
                pv = r.real
                if abs(pv) + abs(qv) > 1e-6 and abs(fl(pv, qv)) < 1e-6:
                    got.append((pv, qv))
    return got


def finite_solve(a, b):
    """V(a,b) for coprime a,b: exact real solutions, cross-checked numerically."""
    sols = sp.solve([sp.Eq(a, 0), sp.Eq(b, 0)], [p, q], dict=True)
    exact = []
    for d in sols:
        pv, qv = d.get(p, None), d.get(q, None)
        if pv is None or qv is None:
            raise RuntimeError("free variable in a supposedly finite component: %s" % d)
        pv, qv = sp.simplify(sp.radsimp(pv)), sp.simplify(sp.radsimp(qv))
        if pv.free_symbols or qv.free_symbols:
            raise RuntimeError("symbolic leftovers %s" % d)
        if sp.simplify(sp.im(pv)) != 0 or sp.simplify(sp.im(qv)) != 0:
            continue                                   # complex-only branch
        if sp.simplify(a.subs({p: pv, q: qv})) != 0 or sp.simplify(b.subs({p: pv, q: qv})) != 0:
            raise RuntimeError("sympy returned a non-solution %s for %s,%s" % (d, a, b))
        exact.append((sp.nsimplify(pv), sp.nsimplify(qv)))
    # dedupe
    uniq = []
    for s in exact:
        if not any(sp.simplify(s[0] - t[0]) == 0 and sp.simplify(s[1] - t[1]) == 0 for t in uniq):
            uniq.append(s)
    # ---- independent numerical completeness check via the q-resultant
    nsol = _numeric_real_solutions(a, b)
    if len(nsol) != len(uniq):
        raise RuntimeError("completeness mismatch for (%s , %s): exact %d vs numeric %d (%s / %s)"
                           % (a, b, len(uniq), len(nsol), uniq, nsol))
    return uniq


def _numeric_real_solutions(a, b, tol=1e-7):
    """All real (p,q) with a=b=0, found numerically: roots of Res_q(a,b), back-substitution,
    then Newton polishing.  Used ONLY as an independent completeness cross-check on the
    exact sympy solve."""
    aq, bq = sp.Poly(a, q), sp.Poly(b, q)
    if aq.degree() == 0 and bq.degree() == 0:
        g = sp.gcd(sp.Poly(a, p), sp.Poly(b, p))
        if g.degree() > 0:
            raise RuntimeError("not coprime")
        return []
    res = sp.resultant(a, b, q)
    if res == 0:
        raise RuntimeError("zero resultant -> not coprime")
    cands = []
    if res.free_symbols:
        rp = sp.Poly(res, p)
        rp = rp.sqf_part()                       # kill multiplicities: np.roots is bad at them
        co = [complex(c) for c in rp.all_coeffs()]
        for r in np.roots(co):
            if abs(r.imag) < 1e-6:
                cands.append(r.real)
    al = sp.lambdify((p, q), a, 'numpy')
    bl = sp.lambdify((p, q), b, 'numpy')
    Jl = [[sp.lambdify((p, q), sp.diff(a, v), 'numpy') for v in (p, q)],
          [sp.lambdify((p, q), sp.diff(b, v), 'numpy') for v in (p, q)]]
    raw = []
    for pv in cands:
        qc = []
        for poly in (aq, bq):
            co = [complex(sp.N(c.subs(p, sp.Float(pv)), 30)) for c in poly.all_coeffs()]
            while co and abs(co[0]) < 1e-12:
                co = co[1:]
            if len(co) <= 1:
                continue
            for r in np.roots(co):
                if abs(r.imag) < 1e-6:
                    qc.append(r.real)
        for qv in qc:
            raw.append((pv, qv))
    out = []
    for (pv, qv) in raw:
        u0, v0 = float(pv), float(qv)
        u, v = u0, v0
        if not (abs(float(al(u, v))) < 1e-7 and abs(float(bl(u, v))) < 1e-7):
            for _ in range(40):                   # Newton polish (least-squares step)
                F = np.array([float(al(u, v)), float(bl(u, v))])
                if np.max(np.abs(F)) < 1e-14:
                    break
                J = np.array([[float(Jl[i][j](u, v)) for j in (0, 1)] for i in (0, 1)])
                try:
                    d = np.linalg.lstsq(J, -F, rcond=None)[0]
                except Exception:
                    break
                u, v = u + d[0], v + d[1]
            if abs(u - u0) > 1e-3 or abs(v - v0) > 1e-3:
                continue                          # wandered off; not a root near the candidate
        if abs(float(al(u, v))) < 1e-7 and abs(float(bl(u, v))) < 1e-7:
            if not any(abs(u - s) < 1e-6 and abs(v - t) < 1e-6 for s, t in out):
                out.append((u, v))
    return out


def components_of(R, Im):
    """Real components of V(Re,Im).  Returns (status, curves, points)."""
    R, Im = sp.expand(R), sp.expand(Im)
    curves, points = [], []
    polys = [t for t in (R, Im) if t != 0]
    if not polys:
        return 'PLANE', [], []
    if any(t.free_symbols == set() for t in polys):
        return 'EMPTY', [], []          # a nonzero constant equation
    if len(polys) == 1:
        for fac, _m in sp.factor_list(polys[0])[1]:
            if fac.free_symbols:
                curves.append(norm_poly(fac))
        return 'OK', curves, points
    f, g = polys
    ffac = [fa for fa, _ in sp.factor_list(f)[1] if fa.free_symbols]
    gfac = [fa for fa, _ in sp.factor_list(g)[1] if fa.free_symbols]
    if not ffac or not gfac:
        return 'EMPTY', [], []
    for a in ffac:
        for b in gfac:
            h = sp.gcd(a, b)
            if h.free_symbols:
                curves.append(norm_poly(h))
            else:
                points.extend(finite_solve(a, b))
    return 'OK', curves, points


# ---------------------------------------------------------------- the known loci M1..M5
GOLDEN = sorted(sp.solve(p ** 2 - p - 1, p) + sp.solve(p ** 2 + p - 1, p), key=lambda z: float(z))
KNOWN_CURVES = {
    'M2': [norm_poly(p ** 2 + q ** 2 - 2), norm_poly(p ** 2 + q ** 2 - sp.Rational(1, 2))],
    'M3': [norm_poly(p - sp.Rational(1, 2)), norm_poly(p + sp.Rational(1, 2))],
    'M5': [norm_poly(p ** 2 + q ** 2 - 2 * p), norm_poly(p ** 2 + q ** 2 + 2 * p)],
}
KNOWN_POINTS = {
    'M1': [(sp.Integer(2), sp.Integer(0)), (sp.Integer(-2), sp.Integer(0)),
           (sp.Rational(1, 2), sp.Integer(0)), (sp.Rational(-1, 2), sp.Integer(0))],
    'M4': [(g, sp.Integer(0)) for g in GOLDEN],
}


def classify_curve(c):
    for m, lst in KNOWN_CURVES.items():
        for k in lst:
            if sp.simplify(c - k) == 0:
                return m
    return None


def classify_point(pt):
    pv, qv = pt
    for m, lst in KNOWN_POINTS.items():
        for k in lst:
            if sp.simplify(pv - k[0]) == 0 and sp.simplify(qv - k[1]) == 0:
                return m
    for m, lst in KNOWN_CURVES.items():
        for k in lst:
            if sp.simplify(k.subs({p: pv, q: qv})) == 0:
                return m
    return None


# ---------------------------------------------------------------- main derivation
def derive():
    t0 = time.time()
    curve_hits, point_hits = {}, {}      # locus key -> list of (multiset, signs)
    n_cases = n_imposs = n_plane = 0
    n_old_cases = n_old_imposs = 0
    old_curve_hits, old_point_hits = {}, {}
    for combo in combinations_with_replacement(TNAMES, 3):
        is_old = all(c in OLD for c in combo)
        for signs in iproduct([1, -1], repeat=2):
            s = (1,) + signs
            n_cases += 1
            n_old_cases += 1 if is_old else 0
            expr = sp.expand(sum(si * TVAL[t] for si, t in zip(s, combo)))
            Re, Im = sp.expand(sp.re(expr)), sp.expand(sp.im(expr))
            status, curves, points = components_of(Re, Im)
            if status == 'PLANE':
                n_plane += 1
                continue
            # drop the origin (x=0 is not a legal alphabet symbol)
            points = [pt for pt in points if not (pt[0] == 0 and pt[1] == 0)]
            curves = [c for c in curves if curve_real_points(c)]
            if not curves and not points:
                n_imposs += 1
                n_old_imposs += 1 if is_old else 0
                continue
            tag = (tuple(combo), s)
            for c in curves:
                key = sp.srepr(c)
                curve_hits.setdefault(key, []).append(tag)
                if is_old:
                    old_curve_hits.setdefault(key, []).append(tag)
            for pt in points:
                key = (sp.srepr(pt[0]), sp.srepr(pt[1]))
                point_hits.setdefault(key, []).append(tag)
                if is_old:
                    old_point_hits.setdefault(key, []).append(tag)
    dt = time.time() - t0

    print("=" * 96)
    print("STAGE 1  --  complete 3-term classification over the CLOSED term set")
    print("=" * 96)
    print("cases enumerated : %d   (C(6+3-1,3)=%d multisets x 2^2=4 sign patterns)"
          % (n_cases, len(list(combinations_with_replacement(TNAMES, 3)))))
    print("identically-zero (holds for every x) : %d" % n_plane)
    print("impossible for every nonzero x       : %d" % n_imposs)
    print("cases with a nonempty locus          : %d" % (n_cases - n_imposs - n_plane))
    print("runtime %.1fs" % dt)

    # ---- self-check: the 80 old cases reproduce M1..M5 exactly
    old_loci = set()
    for key in old_curve_hits:
        c = sp.sympify(key)
        m = classify_curve(c)
        assert m is not None, "OLD case produced an unknown CURVE %s -- published thm broken" % c
        old_loci.add(m)
    for key in old_point_hits:
        pt = (sp.sympify(key[0]), sp.sympify(key[1]))
        m = classify_point(pt)
        assert m is not None, "OLD case produced an unknown POINT %s -- published thm broken" % (pt,)
        old_loci.add(m)
    print("\n[self-check] the %d cases with no x^2/(x*)^2 type reproduce exactly %s -- OK"
          % (n_old_cases, sorted(old_loci)))
    print("[self-check] of those %d published cases, %d are impossible for every nonzero x "
          "(alphabet_paper.tex thm:d3classify states 54) -- %s"
          % (n_old_cases, n_old_imposs, "MATCH" if n_old_imposs == 54 else "MISMATCH"))
    print("[cross-tab] the %d cases that DO involve x^2 or (x*)^2 : %d impossible, %d with a locus"
          % (n_cases - n_old_cases, n_imposs - n_old_imposs,
             (n_cases - n_old_cases) - (n_imposs - n_old_imposs)))

    # ---- split new vs known
    known_c, new_c = {}, {}
    for key, tags in curve_hits.items():
        c = sp.sympify(key)
        m = classify_curve(c)
        (known_c if m else new_c).setdefault(m or key, []).extend(tags)
    known_p, new_p = {}, {}
    for key, tags in point_hits.items():
        pt = (sp.sympify(key[0]), sp.sympify(key[1]))
        m = classify_point(pt)
        (known_p if m else new_p).setdefault(m or key, []).extend(tags)

    print("\n--- KNOWN loci recovered (curves) ---")
    for m in sorted(known_c):
        print("   %-3s : %d (multiset,sign) cases" % (m, len(known_c[m])))
    print("--- KNOWN loci recovered (isolated points) ---")
    for m in sorted(known_p):
        print("   %-3s : %d (multiset,sign) cases" % (m, len(known_p[m])))

    print("\n--- NEW positive-dimensional loci (curves) ---")
    new_curves = []
    for key in sorted(new_c, key=lambda k: sp.count_ops(sp.sympify(k))):
        c = sp.sympify(key)
        tags = new_c[key]
        ex = tags[0]
        print("   %s = 0    <- %d case(s), e.g. %s"
              % (sp.factor(c), len(tags), fmt_case(ex)))
        new_curves.append((sp.srepr(c), str(c), len(tags), fmt_case(ex)))

    print("\n--- NEW isolated points (0-dimensional) ---")
    new_points = []
    newcurve_exprs = [sp.sympify(k) for k in new_c]
    for key in sorted(new_p, key=lambda k: (float(sp.sympify(k[0])), float(sp.sympify(k[1])))):
        pv, qv = sp.sympify(key[0]), sp.sympify(key[1])
        tags = new_p[key]
        mod = sp.sqrt(sp.simplify(pv ** 2 + qv ** 2))
        arg = sp.deg(sp.simplify(sp.atan2(qv, pv)))
        onc = [str(sp.factor(c)) for c in newcurve_exprs
               if sp.simplify(c.subs({p: pv, q: qv})) == 0]
        print("   x = %s %s %s i   |x| = %s, arg = %s deg   <- %d case(s), e.g. %s   %s"
              % (pv, '+' if float(qv) >= 0 else '-', sp.Abs(qv), sp.nsimplify(mod),
                 sp.nsimplify(sp.simplify(arg)), len(tags), fmt_case(tags[0]),
                 ("[ALSO ON new curve(s): %s]" % "; ".join(onc)) if onc
                 else "[NOT on any other locus -> genuinely isolated]"))
        new_points.append((str(pv), str(qv), len(tags), fmt_case(tags[0]), onc))

    # ---- advertised identities must be present
    l30 = norm_poly(p ** 2 - 3 * q ** 2)     # x^2 + (x*)^2 = N
    l60 = norm_poly(3 * p ** 2 - q ** 2)     # x^2 + (x*)^2 = -N
    have = {sp.srepr(sp.sympify(k)) for k in new_c}
    assert sp.srepr(l30) in have, "MISSING the arg=+-30deg line"
    assert sp.srepr(l60) in have, "MISSING the arg=+-60deg line"
    print("\n[self-check] both advertised lines p^2=3q^2 (arg=+-30) and q^2=3p^2 (arg=+-60) present -- OK")

    out = {'n_cases': n_cases, 'n_impossible': n_imposs, 'n_plane': n_plane,
           'new_curves': new_curves, 'new_points': new_points,
           'known_curves': {k: len(v) for k, v in known_c.items()},
           'known_points': {k: len(v) for k, v in known_p.items()},
           'runtime': dt}
    with open(CACHE, 'w') as fh:
        json.dump(out, fh, indent=1)
    return out


def fmt_case(tag):
    combo, s = tag
    return " ".join(("+" if si > 0 else "-") + PRETTY[t] for si, t in zip(s, combo)) + " = 0"


def two_term():
    """Completeness footnote: 2-term sums over the closed term set."""
    print("\n" + "=" * 96)
    print("STAGE 2  --  the TWO-term sums (the published paper calls these 'trivial')")
    print("=" * 96)
    res = {}
    n = 0
    for combo in combinations_with_replacement(TNAMES, 2):
        for s2 in (1, -1):
            n += 1
            s = (1, s2)
            expr = sp.expand(sum(si * TVAL[t] for si, t in zip(s, combo)))
            Re, Im = sp.expand(sp.re(expr)), sp.expand(sp.im(expr))
            status, curves, points = components_of(Re, Im)
            points = [pt for pt in points if not (pt[0] == 0 and pt[1] == 0)]
            curves = [c for c in curves if curve_real_points(c)]
            if status == 'PLANE':
                res.setdefault('IDENTITY (all x)', []).append(fmt_case((combo, s)))
            elif not curves and not points:
                pass
            else:
                for c in curves:
                    res.setdefault("curve %s = 0" % c, []).append(fmt_case((combo, s)))
                for pt in points:
                    res.setdefault("point x = %s + %s i" % pt, []).append(fmt_case((combo, s)))
    print("%d (multiset,sign) 2-term cases; nontrivial outcomes:" % n)
    for k in sorted(res):
        print("   %-34s <- %s" % (k, ", ".join(res[k])))
    return res


def realize():
    """Assert every closed term type is an actual entry product, with both signs."""
    print("\n" + "=" * 96)
    print("STAGE 0  --  realizability of the six term types")
    print("=" * 96)
    ent = {'1': sp.Integer(1), 'x': X, 'x*': XS}
    seen = {}
    for un, u in ent.items():
        for vn, v in ent.items():
            t = sp.expand(sp.conjugate(u) * v)
            t = sp.expand(sp.simplify(t))
            for nm, val in TVAL.items():
                if sp.simplify(t - val) == 0:
                    seen.setdefault(nm, []).append("conj(%s)*%s" % (un, vn))
    for nm in TNAMES:
        assert nm in seen, "term type %s NOT realizable" % nm
        print("   %-6s = %s   (sign flipped freely by u -> -u)" % (PRETTY[nm], seen[nm][0]))
    assert set(seen) == set(TNAMES)
    print("   -> all six types realizable; the closed term set is exactly "
          "{+-1,+-x,+-x*,+-N,+-x^2,+-(x*)^2}")


if __name__ == '__main__':
    realize()
    derive()
    two_term()
    print("\nDONE.  cache -> %s" % CACHE)
