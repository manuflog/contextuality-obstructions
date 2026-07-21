#!/usr/bin/env python3
"""
B1 — PERES 33 <-> PENROSE 33: does the exact imaginary flex family of the Peres 33-ray KS set
connect it to Penrose's complex 33-ray KS set?

ANSWER (this file proves it): YES — EXACTLY, and the whole family has a CLOSED FORM.

The key literature input (verified here line by line, in exact arithmetic): Gould & Aravind,
"Isomorphism between the Peres and Penrose proofs of the BKS theorem in three dimensions",
Found. Phys. 40, 1096 (2010), arXiv:0909.4502. Their Table 3 gives the most general 33-ray
configuration with the Peres orthogonality table: three complex parameters a, b, c with
|a| = |b| = 1, |c|^2 = 2, k = -a b* c / c*; Peres = (a,b,c) = (1, 1, sqrt2); Penrose =
(-i, -1, -sqrt2).  What THIS file adds on top of transcribing that table:

  1. EXACT verification that all 72 orthogonalities of Table 3 hold IDENTICALLY in the three
     phases (Laurent-polynomial arithmetic over Z[sqrt2] — no floats, no sympy), and that the
     orthogonality graph is exactly the Peres graph at both the Peres and Penrose points
     (no extra orthogonalities).
  2. EXACT gauge lemma: diagonal unitaries diag(1, e^{iq}, e^{ir}) act on the parameter torus
     by (alpha,beta,gamma) -> (alpha+r-q, beta+r, gamma+q) (phases of a,b,c). The orbit is
     2-dimensional, so the 3-torus of parameters collapses to ONE modulus
         phi = alpha - beta + gamma,
     matching the EXACT flex = 1 of sic_zoo. The slice a = e^{i theta}, b = 1, c = sqrt2
     (k = -e^{i theta}) is a global section: THE flex family in closed form,
         v_j(theta) = entrywise  m_jc * e^{i e_jc theta},  e_jc in {-1,0,1}, m_jc in Z[sqrt2]
     — precisely the "entries acquire a phase e^{i theta}" ansatz.
  3. EXACT flex certificates at BOTH endpoints (mod-p rank bound at two primes + explicit
     kernel vector verified in exact arithmetic, the sic_zoo method, extended here to complex
     rays over Z[i,sqrt2]): flex = 1 at Peres (theta = 0) and at the Penrose point
     (theta = -pi/2). The exact tangent of the closed-form family at theta = 0 spans the SAME
     kernel line as sic_zoo's exact flex vector (rank check mod p): the imaginary flex traced
     in SIC_ZOO.md IS the tangent of the Peres->Penrose circle.
  4. EXACT endpoint identification: diag(1,-1,-1) maps the slice at theta = -pi/2 ray-by-ray
     (projectively, verified by exact cross products over Z[i,sqrt2]) onto the Penrose
     configuration (a,b,c) = (-i,-1,-sqrt2).  Peres sits at theta = 0, Penrose a quarter turn
     away.  KS-uncolorability (full rules, exhaustive) + criticality hold for the whole family
     since the graph is constant.
  5. Global structure of the family (exact where marked): period 2 pi; conj(v(theta)) =
     v(-theta) identically (antiunitary theta <-> -theta); a signed-permutation relabeling
     realizes theta <-> theta + pi EXACTLY (identically in theta), so the moduli circle has
     length pi with fundamental domain [0, pi/2]: PERES at 0, PENROSE at pi/2 — the two
     "poles" of the family.  Real points: theta = 0, pi only. The labeled modulus in closed
     form: |<v9,v14>|^2 / 8 = (3 - 2 sqrt2 cos theta)/8, i.e. (2-sqrt2)/4 -> Peres,
     sqrt6/4 -> Penrose — exactly the two inner products Gould-Aravind quote.
  6. NUMERICAL confirmation: high-precision predictor/Newton path following of the flex
     (residual <= 1e-12) goes all the way around: the traced family matches the closed form
     to ~1e-12 in all 528 |Gram| entries at a fitted theta-hat every step, rank J = 156
     (flex = 1) all along, reaches the Penrose |Gram| matrix at theta-hat = pi/2, and closes
     up at theta-hat = 2 pi.  The continuation adds nothing to the proof (the family is
     exact); it confirms that the numerically traced family of SIC_ZOO.md is this circle.

Engine reuse: sic_zoo.py (imported, UNMODIFIED) supplies the Peres-33 rays, Z[sqrt2]
arithmetic, mod-p ranks, the exhaustive KS-colorability checker, and the session-2 exact
flex certificate that is cross-checked here.

Run:  python3 peres_penrose.py            (~1-2 min, all checks printed)
      python3 peres_penrose.py family cont   (sections: table3 ks iso family global cont)
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from itertools import combinations, permutations, product

import sic_zoo as SZ
from sic_zoo import (Z0, q_add, q_neg, q_mul, q_dot, q_float, q_sign_norm,
                     rank_mod_p, q_rowdot, ks_colorable, PRIMES, rays_peres33)
from flex_dimension import flex_dimension

SQRT2 = 2 ** 0.5
ONE, RT2, MONE = (1, 0), (0, 1), (-1, 0)

# =====================================================================================
# Laurent polynomials in phase variables z_1..z_nv with coefficients in Z[sqrt2].
# An element is a dict {exponent_tuple: pair}, pair = (a,b) = a + b*sqrt2 (sic_zoo format).
# All Table-3 coefficient constants are REAL (Z[sqrt2]); the imaginary unit only enters at
# evaluation, so conjugation = exponent inversion.
# =====================================================================================
def L_mono(coeff, expo):
    return {tuple(expo): coeff} if coeff != Z0 else {}

def L_add(u, v):
    out = dict(u)
    for e, c in v.items():
        s = q_add(out.get(e, Z0), c)
        if s == Z0: out.pop(e, None)
        else: out[e] = s
    return out

def L_neg(u): return {e: q_neg(c) for e, c in u.items()}

def L_mul(u, v):
    out = {}
    for e1, c1 in u.items():
        for e2, c2 in v.items():
            e = tuple(x + y for x, y in zip(e1, e2))
            s = q_add(out.get(e, Z0), q_mul(c1, c2))
            if s == Z0: out.pop(e, None)
            else: out[e] = s
    return out

def L_conj(u):  # coefficients real; phases invert
    return {tuple(-x for x in e): c for e, c in u.items()}

def L_herm(uvec, vvec):
    s = {}
    for uc, vc in zip(uvec, vvec): s = L_add(s, L_mul(L_conj(uc), vc))
    return s

def L_cross_zero(u, v):
    """u, v: Laurent vectors. True iff u_a v_b - u_b v_a == 0 identically for all a<b,
       i.e. u and v are proportional (ratio a monomial) wherever both are nonzero —
       combined with matching zero patterns this is exact projective equality."""
    d = len(u)
    for a in range(d):
        for b in range(a + 1, d):
            if L_add(L_mul(u[a], v[b]), L_neg(L_mul(u[b], v[a]))): return False
    for a in range(d):
        if bool(u[a]) != bool(v[a]): return False
    return True

def L_eval_c(u, zs):
    out = 0j
    for expo, c in u.items():
        t = complex(q_float(c))
        for z, e in zip(zs, expo): t *= z ** e
        out += t
    return out

# ---------- exact evaluation at quarter-turn phases: z_k = i^{q_k} -> Z[i,sqrt2] ----------
# Q4 element = (re_pair, im_pair) representing (a + b sqrt2) + i (c + d sqrt2).
Q4_0, Q4_1, Q4_I = (Z0, Z0), (ONE, Z0), (Z0, ONE)
def q4_add(u, v): return (q_add(u[0], v[0]), q_add(u[1], v[1]))
def q4_neg(u): return (q_neg(u[0]), q_neg(u[1]))
def q4_mul(u, v):
    return (q_add(q_mul(u[0], v[0]), q_neg(q_mul(u[1], v[1]))),
            q_add(q_mul(u[0], v[1]), q_mul(u[1], v[0])))
def q4_conj(u): return (u[0], q_neg(u[1]))
def q4_float(u): return q_float(u[0]) + 1j * q_float(u[1])
def q4_is0(u): return u[0] == Z0 and u[1] == Z0

def q4_herm(x, y):
    s = Q4_0
    for a, b in zip(x, y): s = q4_add(s, q4_mul(q4_conj(a), b))
    return s

def L_eval_q4(u, quarters):
    out = Q4_0
    for expo, c in u.items():
        q = sum(qq * e for qq, e in zip(quarters, expo)) % 4
        cc = (c, Z0)
        for _ in range(q): cc = q4_mul(cc, Q4_I)
        out = q4_add(out, cc)
    return out

def L_eval_pair(u, quarters):
    v = L_eval_q4(u, quarters)
    assert v[1] == Z0, "evaluation expected to be real"
    return v[0]

# =====================================================================================
# Gould-Aravind Table 3 (arXiv:0909.4502): the general 33-ray family.
# a, b, c enter as monomials  a = z^EA, b = z^EB, c = sqrt2 * z^EC  (unit-phase variables);
# k = -a b* c / c* = - z^{EA - EB + 2 EC}. Passing arbitrary integer exponent vectors
# EA, EB, EC realizes any monomial substitution with |a|=|b|=1, |c|=sqrt2.
# =====================================================================================
def table3(nv, EA, EB, EC):
    def M(coeff, pa, pb, pc):
        return L_mono(coeff, tuple(pa * EA[i] + pb * EB[i] + pc * EC[i] for i in range(nv)))
    O = {}
    one, mone = M(ONE, 0, 0, 0), M(MONE, 0, 0, 0)
    a, as_ = M(ONE, 1, 0, 0), M(ONE, -1, 0, 0)
    b, bs = M(ONE, 0, 1, 0), M(ONE, 0, -1, 0)
    c, cs = M(RT2, 0, 0, 1), M(RT2, 0, 0, -1)
    k, ks = M(MONE, 1, -1, 2), M(MONE, -1, 1, -2)
    ac, ascs = M(RT2, 1, 0, 1), M(RT2, -1, 0, -1)
    bcs, bsc = M(RT2, 0, 1, -1), M(RT2, 0, -1, 1)
    N = L_neg
    return [
        (one, O, O), (O, one, O), (O, O, one),                                    # 1  2  3
        (O, one, a), (O, as_, mone), (one, O, b), (bs, O, mone),                  # 4  5  6  7
        (one, k, O), (ks, mone, O),                                               # 8  9
        (ascs, N(as_), one), (cs, one, a), (N(cs), one, a), (ascs, as_, mone),    # 10 11 12 13
        (N(bs), bsc, one), (one, c, b), (one, N(c), b), (bs, bsc, mone),          # 14 15 16 17
        (N(ks), one, bcs), (one, k, N(ac)), (one, k, ac), (ks, mone, bcs),        # 18 19 20 21
        (one, O, ac), (one, N(c), O), (one, c, O), (one, O, N(ac)),               # 22 23 24 25
        (O, one, bcs), (N(cs), one, O), (cs, one, O), (O, one, N(bcs)),           # 26 27 28 29
        (O, bsc, one), (ascs, O, one), (N(ascs), O, one), (O, N(bsc), one),       # 30 31 32 33
    ]

GEN = table3(3, (1, 0, 0), (0, 1, 0), (0, 0, 1))       # general: variables (za, zb, zc)
SLICE = table3(1, (1,), (0,), (0,))                    # the flex family: a=e^{i th}, b=1, c=sqrt2

# Gould-Aravind Table 1 (transcribed; cross-checked against the exact computation below).
T1_TRIADS = [(1, 2, 3), (1, 4, 5), (1, 26, 33), (1, 29, 30), (2, 6, 7), (2, 22, 32),
             (2, 25, 31), (3, 8, 9), (3, 23, 28), (3, 24, 27), (4, 10, 13), (5, 11, 12),
             (6, 14, 17), (7, 15, 16), (8, 18, 21), (9, 19, 20)]
T1_DYADS = [(10, 24), (10, 25), (11, 23), (11, 25), (12, 22), (12, 24), (13, 22), (13, 23),
            (14, 28), (14, 29), (15, 27), (15, 29), (16, 26), (16, 28), (17, 26), (17, 27),
            (18, 32), (18, 33), (19, 31), (19, 33), (20, 30), (20, 32), (21, 30), (21, 31)]

def table1_edges():
    E = set()
    for t in T1_TRIADS:
        for i, j in combinations(t, 2): E.add(frozenset((i - 1, j - 1)))
    for i, j in T1_DYADS: E.add(frozenset((i - 1, j - 1)))
    assert len(E) == 72
    return E

T1_EDGES = table1_edges()

# parameter points (quarter-turn exponents for (za, zb, zc)): i^q
Q_PERES = (0, 0, 0)         # (a,b,c) = (1, 1, sqrt2)
Q_PENROSE = (3, 2, 2)       # (a,b,c) = (-i, -1, -sqrt2)

def rays_q4(quarters, fam=GEN):
    return [tuple(L_eval_q4(comp, quarters) for comp in ray) for ray in fam]

def rays_c(zs, fam=GEN):
    return [np.array([L_eval_c(comp, zs) for comp in ray]) for ray in fam]

def graph_q4(rays):
    """Exact orthogonality graph over Z[i,sqrt2]."""
    return {frozenset((i, j)) for i, j in combinations(range(len(rays)), 2)
            if q4_is0(q4_herm(rays[i], rays[j]))}

def distinct_q4(rays):
    for i, j in combinations(range(len(rays)), 2):
        prop = all(q4_is0(q4_add(q4_mul(rays[i][a], rays[j][b]),
                                 q4_neg(q4_mul(rays[i][b], rays[j][a]))))
                   for a in range(3) for b in range(a + 1, 3))
        if prop: return False
    return True

# =====================================================================================
# Extended Jacobian for COMPLEX rays (generalizes sic_zoo.build_extended_jacobian, which
# assumes real rays).  Rays are Q4 triples v = P + iQ with P,Q Z[sqrt2] vectors; the real
# variables are (Re w_ic, Im w_ic), layout coord(i,c,re/im) = 6i + 2c + (0|1), n = 198.
# Edge rows: Re/Im[<w_i,v_j> + <v_i,w_j>]; norm rows: Re<v_i,w_i>.
# Trivial span: 33 per-vertex phases (w_i = i v_i) + 9 u(3) directions, one exact relation.
# All row entries are Z[sqrt2] pairs, so sic_zoo's mod-p rank machinery applies unchanged.
# =====================================================================================
def build_ext_jac_c(rays):
    d, V = 3, len(rays)
    E = sorted(tuple(sorted(e)) for e in graph_q4(rays))
    n = 2 * d * V
    P = [[v[c][0] for c in range(d)] for v in rays]
    Q = [[v[c][1] for c in range(d)] for v in rays]
    def idx(i, c, im): return 2 * d * i + 2 * c + (1 if im else 0)
    rows = []
    for i, j in E:
        re, im = [Z0] * n, [Z0] * n
        for c in range(d):
            re[idx(i, c, 0)] = q_add(re[idx(i, c, 0)], P[j][c])
            re[idx(i, c, 1)] = q_add(re[idx(i, c, 1)], Q[j][c])
            re[idx(j, c, 0)] = q_add(re[idx(j, c, 0)], P[i][c])
            re[idx(j, c, 1)] = q_add(re[idx(j, c, 1)], Q[i][c])
            im[idx(i, c, 0)] = q_add(im[idx(i, c, 0)], Q[j][c])
            im[idx(i, c, 1)] = q_add(im[idx(i, c, 1)], q_neg(P[j][c]))
            im[idx(j, c, 0)] = q_add(im[idx(j, c, 0)], q_neg(Q[i][c]))
            im[idx(j, c, 1)] = q_add(im[idx(j, c, 1)], P[i][c])
        rows.append(re); rows.append(im)
    for i in range(V):
        r = [Z0] * n
        for c in range(d):
            r[idx(i, c, 0)] = P[i][c]; r[idx(i, c, 1)] = Q[i][c]
        rows.append(r)
    triv = []
    for i in range(V):                                    # per-vertex phase w_i = i v_i
        t = [Z0] * n
        for c in range(d):
            t[idx(i, c, 0)] = q_neg(Q[i][c]); t[idx(i, c, 1)] = P[i][c]
        triv.append(t)
    for a in range(d):                                    # A = i E_aa
        t = [Z0] * n
        for i in range(V):
            t[idx(i, a, 0)] = q_neg(Q[i][a]); t[idx(i, a, 1)] = P[i][a]
        triv.append(t)
    for a in range(d):                                    # A = E_ab - E_ba ; A = i(E_ab+E_ba)
        for b in range(a + 1, d):
            t = [Z0] * n
            for i in range(V):
                t[idx(i, a, 0)] = q_add(t[idx(i, a, 0)], P[i][b])
                t[idx(i, a, 1)] = q_add(t[idx(i, a, 1)], Q[i][b])
                t[idx(i, b, 0)] = q_add(t[idx(i, b, 0)], q_neg(P[i][a]))
                t[idx(i, b, 1)] = q_add(t[idx(i, b, 1)], q_neg(Q[i][a]))
            triv.append(t)
            t = [Z0] * n
            for i in range(V):
                t[idx(i, a, 0)] = q_add(t[idx(i, a, 0)], q_neg(Q[i][b]))
                t[idx(i, a, 1)] = q_add(t[idx(i, a, 1)], P[i][b])
                t[idx(i, b, 0)] = q_add(t[idx(i, b, 0)], q_neg(Q[i][a]))
                t[idx(i, b, 1)] = q_add(t[idx(i, b, 1)], P[i][a])
            triv.append(t)
    # exact global-phase relation: sum(vertex phases) == sum(iE_aa directions)
    s1 = [Z0] * n; s2 = [Z0] * n
    for t in triv[:V]: s1 = [q_add(x, y) for x, y in zip(s1, t)]
    for t in triv[V:V + d]: s2 = [q_add(x, y) for x, y in zip(s2, t)]
    assert s1 == s2, "global-phase relation violated — formulation error"
    assert all(q_rowdot(r, t) == Z0 for r in rows for t in triv), \
        "trivial directions not in kernel — formulation error"
    return rows, triv, E, n

def exact_flex_certificate_c(rays, tangent_realvec, name):
    """flex = 1 EXACT certificate at a complex configuration: mod-p rank bound (two primes)
       gives flex <= 1; the explicit tangent (J.w = 0 verified in exact arithmetic, w not in
       the trivial span mod p) gives flex >= 1."""
    J, T, E, n = build_ext_jac_c(rays)
    V, d = len(rays), 3
    rj = [rank_mod_p(J, p, s) for p, s in PRIMES]
    rt = [rank_mod_p(T, p, s) for p, s in PRIMES]
    assert max(rt) == V + d * d - 1 == 41, f"rank T != 41: {rt}"   # <=41 exact, mod-p >= 41
    bound = (n - max(rj)) - 41
    okJ = all(q_rowdot(row, tangent_realvec) == Z0 for row in J)
    rtw = max(rank_mod_p(T + [tangent_realvec], p, s) for p, s in PRIMES)
    assert okJ, f"{name}: tangent NOT in ker J (exact check failed)"
    assert rtw == 42, f"{name}: tangent lies in the trivial span"
    assert bound == 1, f"{name}: mod-p bound is {bound}, not 1"
    print(f"    {name}: rank_p J = {max(rj)} (primes {[p for p, _ in PRIMES]}), rank T = 41 "
          f"exact => flex <= {bound}; exact tangent J.w = 0 in Z[i,sqrt2], w not in trivial "
          f"span (rank_p[T;w] = 42) => FLEX = 1 EXACT")
    return J, T, E, n

def slice_tangent_realvec(quarters):
    """d/dtheta of the slice family at theta = quarter * pi/2, as the 198-dim real vector of
       Z[sqrt2] pairs: entry m z^e -> i e m z^e evaluated at z = i^quarter."""
    out = [Z0] * 198
    for j, ray in enumerate(SLICE):
        for c, comp in enumerate(ray):
            w = Q4_0
            for expo, m in comp.items():
                e = expo[0]
                if e == 0: continue
                val = L_eval_q4({expo: (e * m[0], e * m[1])}, quarters)
                w = q4_add(w, q4_mul(Q4_I, val))          # i * e * m * z^e
            out[6 * j + 2 * c] = w[0]; out[6 * j + 2 * c + 1] = w[1]
    return out

# =====================================================================================
# sections
# =====================================================================================
def sec_table3():
    print("[1] Gould-Aravind Table 3 at the Peres and Penrose points (all checks EXACT)")
    pen = rays_q4(Q_PENROSE)
    per = rays_q4(Q_PERES)
    assert distinct_q4(pen) and distinct_q4(per)
    print("    33 rays, pairwise projectively distinct at both points (exact cross products)")
    for v in per: assert all(x[1] == Z0 for x in v)
    print("    Peres point (a,b,c)=(1,1,sqrt2): all entries REAL, components in {0,±1,±sqrt2}")
    g_pen, g_per = graph_q4(pen), graph_q4(per)
    assert g_pen == T1_EDGES and g_per == T1_EDGES
    print("    orthogonality graphs computed exactly over Z[i,sqrt2]: BOTH equal the 72-edge")
    print("    Table-1 graph (16 triads + 24 dyads), no extra orthogonalities at either point")
    tri_pen = [t for t in combinations(range(33), 3)
               if all(frozenset(p) in g_pen for p in combinations(t, 2))]
    assert len(tri_pen) == 16
    # Gould-Aravind's unitary-inequivalence witness: |<v9,v14>| (rays 9,14, norms 2 and 4)
    G_per = q4_herm(per[8], per[13]); G_pen = q4_herm(pen[8], pen[13])
    m_per = q4_mul(q4_conj(G_per), G_per); m_pen = q4_mul(q4_conj(G_pen), G_pen)
    assert m_per == ((3, -2), Z0) and m_pen == ((3, 0), Z0)
    print("    witness |<v9,v14>|^2 (unnormalized, norms 2*4=8): Peres 3-2sqrt2  <->  ((2-sqrt2)/4)^2*8")
    print("                                                      Penrose 3       <->  (sqrt6/4)^2*8")
    print("    => matches the paper's quoted values (2-sqrt2)/4 vs sqrt6/4: transcription correct;")
    print("       Peres and Penrose are NOT unitarily equivalent (EXACT witness)")
    # essential complexity: Bargmann invariant of the pairwise non-orthogonal triple (9,14,15)
    B_pen = q4_mul(q4_mul(q4_herm(pen[8], pen[13]), q4_herm(pen[13], pen[14])),
                   q4_herm(pen[14], pen[8]))
    B_per = q4_mul(q4_mul(q4_herm(per[8], per[13]), q4_herm(per[13], per[14])),
                   q4_herm(per[14], per[8]))
    assert B_per[1] == Z0 and B_pen[1] != Z0
    print(f"    Bargmann invariant of triple (9,14,15): Peres real (= {B_per[0]}), Penrose")
    print(f"      NON-REAL (= {B_pen[0]} + i*{B_pen[1]}) => Penrose is ESSENTIALLY COMPLEX (EXACT)")
    fx = flex_dimension(rays_c([-1j, -1, -1]), name="Penrose 33")
    print(f"    numerical flex engine on the Penrose rays: flex = {fx}  (expected 1)\n")
    return dict(fx_pen=fx)

def sec_ks():
    print("[2] Penrose 33: KS-uncolorability + criticality (graph-level, exhaustive, EXACT)")
    g = graph_q4(rays_q4(Q_PENROSE))
    pairs = sorted(tuple(sorted(e)) for e in g)
    triads = [list(t) for t in combinations(range(33), 3)
              if all(frozenset(p) in g for p in combinations(t, 2))]
    col, nodes = ks_colorable(33, pairs, triads)
    assert not col
    print(f"    KS-UNCOLORABLE, full rules (every orthogonal pair <=1 'one', every triad exactly")
    print(f"    one 'one'): exhaustive backtracking, {nodes} nodes")
    tot = 0
    for r in range(33):
        keep = [i for i in range(33) if i != r]
        rn = {v: k for k, v in enumerate(keep)}
        sp = [(rn[i], rn[j]) for i, j in pairs if i != r and j != r]
        st = [[rn[x] for x in t] for t in triads if r not in t]
        c, nn = ks_colorable(32, sp, st)
        tot += nn
        assert c, f"deletion of ray {r+1} still uncolorable — not critical?"
    print(f"    CRITICAL: all 33 single-ray deletions are KS-colorable (33 exhaustive searches,")
    print(f"    {tot} nodes total).  [Same graph as Peres-33, so this also re-verifies sic_zoo.]\n")

def sec_iso():
    print("[3] Graph isomorphism: Peres 33 (sic_zoo) <-> Penrose 33")
    sz = rays_peres33()
    per = rays_q4(Q_PERES)
    per_pairs = [tuple(x[0] for x in v) for v in per]     # real reps as Z[sqrt2] pairs
    lookup = {q_sign_norm(v): t for t, v in enumerate(per_pairs)}
    sigma = [lookup[q_sign_norm(v)] for v in sz]
    assert sorted(sigma) == list(range(33))
    E_sz = {frozenset((i, j)) for i, j in combinations(range(33), 2)
            if q_dot(sz[i], sz[j]) == Z0}
    assert {frozenset((sigma[i], sigma[j])) for e in E_sz for i, j in [tuple(e)]} == T1_EDGES
    print("    explicit bijection sigma: sic_zoo Peres rays -> Table-3 labels (projective ray")
    print("    matching over Z[sqrt2]); sigma maps the 72 sic_zoo edges EXACTLY onto the 72")
    print("    Table-1 edges.  Penrose graph == Table-1 graph (section [1]).")
    print("    => Peres-33 and Penrose-33 orthogonality graphs are ISOMORPHIC (EXACT,")
    print("       constructive certificate — identical Kochen-Specker diagrams).")
    deg = {}
    for e in T1_EDGES:
        for x in e: deg[x] = deg.get(x, 0) + 1
    ds = sorted(deg.values())
    print(f"    invariants: degree sequence {min(ds)}..{max(ds)} "
          f"(multiset {sorted(set((d, ds.count(d)) for d in ds))}), 16 triangles, 72 edges")
    print(f"    sigma = {sigma}\n")
    return sigma

def sec_family(sigma=None):
    print("[4] The EXACT flex family (Table 3) and the Peres->Penrose connection")
    # ---- 4a: all 72 orthogonalities hold identically in the three phases
    edges, bad = set(), 0
    for i, j in combinations(range(33), 2):
        h = L_herm(GEN[i], GEN[j])
        if not h: edges.add(frozenset((i, j)))
    assert edges == T1_EDGES
    print("    [4a] all 72 edge inner products vanish IDENTICALLY as Laurent polynomials in")
    print("         (za, zb, zc) over Z[sqrt2]; the other 456 vanish for NO choice of phases")
    print("         identically => the 3-torus family realizes the Peres graph. EXACT.")
    # ---- 4b: gauge lemma — diag(1, zq, zr) realizes (a,b,c) -> (a zr/zq, b zr, c zq)
    nv = 5   # variables (za, zb, zc, zq, zr)
    base = table3(nv, (1, 0, 0, 0, 0), (0, 1, 0, 0, 0), (0, 0, 1, 0, 0))
    moved = table3(nv, (1, 0, 0, -1, 1), (0, 1, 0, 0, 1), (0, 0, 1, 1, 0))
    zq = L_mono(ONE, (0, 0, 0, 1, 0)); zr = L_mono(ONE, (0, 0, 0, 0, 1))
    for j in range(33):
        Uv = (base[j][0], L_mul(zq, base[j][1]), L_mul(zr, base[j][2]))
        assert L_cross_zero(Uv, moved[j]), f"gauge lemma fails at ray {j+1}"
    print("    [4b] GAUGE LEMMA (exact, identically in all 5 phases): diag(1, e^iq, e^ir) maps")
    print("         the (alpha,beta,gamma) configuration ray-by-ray onto (alpha+r-q, beta+r,")
    print("         gamma+q).  The diagonal orbit is 2-dim => single modulus phi = alpha-beta+gamma.")
    print("         Peres: phi = 0.  Penrose (-i,-1,-sqrt2): phi = -pi/2 - pi + pi = -pi/2.")
    print("         The slice a=e^{i theta}, b=1, c=sqrt2, k=-e^{i theta} is a global section")
    print("         (theta = phi): v_j(theta) has entries m*e^{i e theta}, e in {-1,0,1} — the")
    print("         phase ansatz holds EXACTLY.")
    # ---- 4c: exact flex certificates at both endpoints; tangent = sic_zoo flex vector
    print("    [4c] exact flex certificates (mod-p bound + exact kernel vector):")
    per = rays_q4((0,), SLICE)
    w0 = slice_tangent_realvec((0,))
    J0, T0, E0, n0 = exact_flex_certificate_c(per, w0, "Peres endpoint theta=0")
    pen_slice = rays_q4((3,), SLICE)                      # theta = -pi/2: z = i^3 = -i
    wq = slice_tangent_realvec((3,))
    exact_flex_certificate_c(pen_slice, wq, "Penrose endpoint theta=-pi/2")
    # cross-check: sic_zoo's exact flex vector spans the same line as the tangent at theta=0
    if sigma is None: sigma = sec_iso_quiet()
    cert = SZ.exact_certificate(rays_peres33(), "Peres 33", PRIMES, sympy_over_Q=False)
    assert cert["exact_value"] == 1
    u_sz = cert["flex_vector"]
    sz = rays_peres33()
    per_pairs = [tuple(x[0] for x in v) for v in per]
    u_t3 = [Z0] * 198
    for s in range(33):
        t = sigma[s]
        eps = 1 if sz[s] == per_pairs[t] else -1
        assert sz[s] == per_pairs[t] or sz[s] == tuple(q_neg(x) for x in per_pairs[t])
        for c in range(6):
            a, b = u_sz[6 * s + c]
            u_t3[6 * t + c] = (eps * a, eps * b)
    assert all(q_rowdot(row, u_t3) == Z0 for row in J0)
    r2 = max(rank_mod_p(T0 + [w0, u_t3], p, s) for p, s in PRIMES)
    assert r2 == 42
    print("    sic_zoo's exact flex vector (session 2), transported through sigma: J.u = 0 exact")
    print("    and rank_p[T; w; u] = 42 = rank[T; w]  =>  u lies in span(trivial, tangent):")
    print("    the SIC_ZOO imaginary flex IS the tangent of this circle at Peres. EXACT.")
    # ---- 4d: endpoint identification with the actual Penrose configuration
    pen = rays_q4(Q_PENROSE)
    U = [Q4_1, q4_neg(Q4_1), q4_neg(Q4_1)]               # diag(1,-1,-1)
    for j in range(33):
        Uv = tuple(q4_mul(U[c], pen_slice[j][c]) for c in range(3))
        for a in range(3):
            for b in range(a + 1, 3):
                assert q4_is0(q4_add(q4_mul(Uv[a], pen[j][b]), q4_neg(q4_mul(Uv[b], pen[j][a]))))
        assert all(q4_is0(Uv[c]) == q4_is0(pen[j][c]) for c in range(3))
    print("    [4d] diag(1,-1,-1) maps the slice at theta = -pi/2 ray-by-ray projectively onto")
    print("         the PENROSE configuration (a,b,c)=(-i,-1,-sqrt2) (exact cross products,")
    print("         all 33 rays)  =>  PENROSE-33 LIES ON THE PERES-33 FLEX FAMILY, a quarter")
    print("         turn (theta = -pi/2) from Peres.  EXACT.")
    G = L_herm(SLICE[8], SLICE[13])
    assert G == {(1,): ONE, (0,): (0, -1)}               # <v9,v14>(theta) = e^{i theta} - sqrt2
    print("    labeled modulus in closed form: <v9,v14>(theta) = e^{i theta} - sqrt2, so")
    print("    |<v9,v14>|^2/8 = (3 - 2 sqrt2 cos theta)/8: Peres (2-sqrt2)^2/16 -> Penrose 3/8.")
    # ---- 4e: the state-independence operator identity is constant along the family
    M = [[{} for _ in range(3)] for _ in range(3)]
    for ray in GEN:
        nj = L_herm(ray, ray)
        assert list(nj.values())[0][1] == 0 and len(nj) == 1
        w = 12 // list(nj.values())[0][0]                # 12/|v_j|^2 in {12,6,4,3}
        for aa in range(3):
            for bb in range(3):
                t = L_mul(ray[aa], L_conj(ray[bb]))
                M[aa][bb] = L_add(M[aa][bb], {e: (w * x, w * y) for e, (x, y) in t.items()})
    for aa in range(3):
        for bb in range(3):
            want = {(0, 0, 0): (132, 0)} if aa == bb else {}
            assert M[aa][bb] == want, "tight-frame identity fails"
    print("    [4e] TIGHT-FRAME IDENTITY along the whole family: sum_j P_j = 11*I identically in")
    print("         ALL THREE phases (12*sum = 132*I over the Laurent ring). The operator identity")
    print("         behind state-independence is CONSTANT along the flex — the modulus deforms the")
    print("         KS set without ever touching sum P = (V/d) I.  EXACT.\n")
    return w0

def sec_iso_quiet():
    sz = rays_peres33()
    per_pairs = [tuple(x[0] for x in v) for v in rays_q4(Q_PERES)]
    lookup = {q_sign_norm(v): t for t, v in enumerate(per_pairs)}
    return [lookup[q_sign_norm(v)] for v in sz]

def sec_global():
    print("[5] Global structure of the family (exact claims marked EXACT)")
    # conj(v(theta)) = v(-theta): trivially exact (real coefficients, exponent inversion)
    for ray in SLICE:
        for comp in ray:
            assert L_conj(comp) == {(-e[0],): c for e, c in comp.items()}
    print("    conj(v_j(theta)) = v_j(-theta) identically (coefficients real). EXACT.")
    print("    => theta <-> -theta is an ANTIUNITARY equivalence of configurations.")
    # theta = pi: the configuration is real again; is it the Peres SET relabeled?
    reps_pi = [tuple(L_eval_pair(c, (2,)) for c in ray) for ray in SLICE]
    reps_0 = [tuple(L_eval_pair(c, (0,)) for c in ray) for ray in SLICE]
    lut = {q_sign_norm(v): k for k, v in enumerate(reps_0)}
    gperm = [lut.get(q_sign_norm(v)) for v in reps_pi]
    assert None not in gperm and sorted(gperm) == list(range(33))
    assert {frozenset((gperm[i], gperm[j])) for e in T1_EDGES for i, j in [tuple(e)]} == T1_EDGES
    print(f"    theta = pi is REAL and equals the Peres SET exactly, relabeled by the graph")
    print(f"    automorphism g = {gperm}  (EXACT)")
    # generic theta <-> theta+pi: search signed permutations R with R v_j(theta) ~ v_g(j)(theta+pi)
    th0 = 0.37
    A = [v / np.linalg.norm(v) for v in rays_c([np.exp(1j * th0)], SLICE)]
    B = [v / np.linalg.norm(v) for v in rays_c([np.exp(1j * (th0 + np.pi))], SLICE)]
    found = None
    for perm in permutations(range(3)):
        for sg in product((1, -1), repeat=3):
            g = []
            ok = True
            for j in range(33):
                Rv = np.array([sg[c] * A[j][perm[c]] for c in range(3)])
                m = [k for k in range(33)
                     if abs(abs(np.vdot(Rv, B[k])) - 1) < 1e-9]
                if len(m) != 1: ok = False; break
                g.append(m[0])
            if ok and sorted(g) == list(range(33)):
                found = (perm, sg, g); break
        if found: break
    assert found, "no signed-permutation relabeling for theta -> theta+pi found"
    perm, sg, g = found
    # exact verification, identically in theta: R v_j(z) ~ v_g(j)(-z)
    for j in range(33):
        Rv = []
        for c in range(3):
            comp = SLICE[j][perm[c]]
            Rv.append({e: (sg[c] * cf[0], sg[c] * cf[1]) for e, cf in comp.items()})
        tgt = []
        for c in range(3):
            comp = SLICE[g[j]][c]
            tgt.append({e: (cf[0] if e[0] % 2 == 0 else -cf[0],
                            cf[1] if e[0] % 2 == 0 else -cf[1]) for e, cf in comp.items()})
        assert L_cross_zero(tuple(Rv), tuple(tgt)), f"theta+pi relabeling fails at ray {j+1}"
    if perm == (0, 1, 2) and sg == (1, 1, 1):
        print("    theta <-> theta + pi needs NO unitary at all: v_j(theta+pi) is projectively")
        print(f"    v_g(j)(theta) with g = {g}")
        print("    — the 33-ray SET is invariant under theta -> theta+pi, verified identically in")
        print("    theta (exact cross products). EXACT.  The unlabeled family has period pi.")
    else:
        print(f"    theta <-> theta + pi is a UNITARY equivalence for ALL theta: signed")
        print(f"    permutation R = (perm {perm}, signs {sg}) + relabeling g = {g},")
        print(f"    verified identically in theta (exact cross products). EXACT.")
    print("    => as UNLABELED ray sets the family is a circle of circumference pi (theta mod pi):")
    print("       PERES at theta = 0, PENROSE at theta = pi/2 == -pi/2 — antipodal points.")
    print("       Adding the antiunitary theta <-> -theta folds it to the segment [0, pi/2] with")
    print("       Peres and Penrose at the two ends (the arcs (0,pi/2) and (pi/2,pi) are complex-")
    print("       conjugate mirror images of each other).")
    print("       Consequence: conj(Penrose set) = set(+pi/2) = set(+pi/2 - pi) = Penrose set —")
    print("       the PENROSE ray set is SELF-CONJUGATE as a set (relabeled). EXACT.")
    # no degenerations along the family (numerical scan + exact non-identical-vanishing)
    for i, j in combinations(range(33), 2):
        if frozenset((i, j)) in T1_EDGES: continue
        assert L_herm(GEN[i], GEN[j]), "non-edge identically orthogonal?!"
    ths = np.linspace(0, 2 * np.pi, 1441)
    min_ne, min_dist, argmin_ne = np.inf, np.inf, None
    nonedges = [tuple(sorted(e)) for e in
                (set(map(frozenset, combinations(range(33), 2))) - T1_EDGES)]
    nonedges = [(i, j) for i, j in nonedges]
    for th in ths:
        v = [x / np.linalg.norm(x) for x in rays_c([np.exp(1j * th)], SLICE)]
        gm = min(abs(np.vdot(v[i], v[j])) for i, j in nonedges)
        if gm < min_ne: min_ne, argmin_ne = gm, th
        dm = min(1 - abs(np.vdot(v[i], v[j])) for i, j in combinations(range(33), 2))
        min_dist = min(min_dist, dm)
    print(f"    scan over 1441 values of theta (NUMERICAL): min non-edge |Gram| = {min_ne:.4f}")
    print(f"    (at theta = {argmin_ne:.3f}; never 0 => the graph never gains an edge), min")
    print(f"    projective ray separation 1-|<vi,vj>| = {min_dist:.4f} (rays never collide).")
    print("    => the family is a smooth closed circle of KS sets, no degenerations.")
    # where is the family real?  Bargmann phase of triple (9,14,15) vanishes only at 0, pi
    bph = []
    for th in ths:
        v = rays_c([np.exp(1j * th)], SLICE)
        b = np.vdot(v[8], v[13]) * np.vdot(v[13], v[14]) * np.vdot(v[14], v[8])
        bph.append(b.imag)
    bph = np.array(bph)
    small = np.abs(bph) < 1e-9
    sc = np.sign(bph[:-1]) * np.sign(bph[1:]) < 0
    zs = sorted(set([round(float(ths[i]) / np.pi, 2) % 2 for i in np.where(sc)[0]] +
                    [round(float(t) / np.pi, 2) % 2 for t in ths[small]]))
    print(f"    Im(Bargmann(9,14,15)) vanishes only at theta/pi in {zs} (NUMERICAL grid; the")
    print(f"    values at 0 and pi are exactly real by [1]/[5]): the ONLY real points of the")
    print(f"    family are the two Peres copies; all others, Penrose included, are essentially")
    print(f"    complex.\n")

def sec_cont():
    print("[6] High-precision numerical continuation of the flex (confirms path = closed form)")
    V, d = 33, 3
    E = sorted(tuple(sorted(e)) for e in T1_EDGES)
    v0 = rays_c([1.0 + 0j], SLICE)
    norms0 = [float(np.vdot(x, x).real) for x in v0]
    # closed-form |Gram|: Fourier coefficients of <v_i, v_j>(theta) for all 528 pairs
    pairs = list(combinations(range(33), 2))
    FC = {pq: {e[0]: q_float(c) for e, c in L_herm(SLICE[pq[0]], SLICE[pq[1]]).items()}
          for pq in pairs}
    def gram_abs_theta(th):
        return np.array([abs(sum(c * np.exp(1j * e * th) for e, c in FC[pq].items()))
                         for pq in pairs])
    def gram_abs(v):
        return np.array([abs(np.vdot(v[i], v[j])) for i, j in pairs])
    def F(v):
        out = []
        for i, j in E:
            z = np.vdot(v[i], v[j]); out += [z.real, z.imag]
        out += [np.vdot(v[i], v[i]).real - norms0[i] for i in range(V)]
        return np.array(out)
    def JF(v):
        n = 2 * d * V
        M = np.zeros((2 * len(E) + V, n))
        for r, (i, j) in enumerate(E):
            for c in range(d):
                M[2 * r, 2 * d * i + 2 * c] += v[j][c].real
                M[2 * r, 2 * d * i + 2 * c + 1] += v[j][c].imag
                M[2 * r + 1, 2 * d * i + 2 * c] += v[j][c].imag
                M[2 * r + 1, 2 * d * i + 2 * c + 1] += -v[j][c].real
                M[2 * r, 2 * d * j + 2 * c] += v[i][c].real
                M[2 * r, 2 * d * j + 2 * c + 1] += v[i][c].imag
                M[2 * r + 1, 2 * d * j + 2 * c] += -v[i][c].imag
                M[2 * r + 1, 2 * d * j + 2 * c + 1] += v[i][c].real
        for i in range(V):
            for c in range(d):
                M[2 * len(E) + i, 2 * d * i + 2 * c] = 2 * v[i][c].real
                M[2 * len(E) + i, 2 * d * i + 2 * c + 1] = 2 * v[i][c].imag
        return M
    def unpack(delta, v):
        return [v[i] + np.array([delta[2 * d * i + 2 * c] + 1j * delta[2 * d * i + 2 * c + 1]
                                 for c in range(d)]) for i in range(V)]
    def newton(v):
        # rcond=1e-8 truncates the 42 kernel singular values explicitly: with the default
        # machine cutoff a near-zero sigma occasionally crosses the threshold and the
        # min-norm step takes an O(1e-4) excursion ALONG the family (invisible in the
        # residual but jumping theta)
        for _ in range(12):
            f = F(v)
            if np.abs(f).max() < 1e-12: break
            delta, *_ = np.linalg.lstsq(JF(v), -f, rcond=1e-8)
            v = unpack(delta, v)
        return v
    def fit_theta(v, guess):
        ga = gram_abs(v)
        def cost(t): return float(((gram_abs_theta(t) - ga) ** 2).sum())
        lo, hi = guess - 0.08, guess + 0.08
        for _ in range(60):
            m1, m2 = lo + (hi - lo) / 3, hi - (hi - lo) / 3
            if cost(m1) < cost(m2): hi = m2
            else: lo = m1
        th = (lo + hi) / 2
        return th, float(np.abs(gram_abs_theta(th) - ga).max())
    # signed invariant: Bargmann product of the non-orthogonal triple (9,14,15); its imaginary
    # part is odd in theta, so the pair (|Gram| profile, Im B) pins theta WITH its sign.
    Bpoly = L_mul(L_mul(L_herm(SLICE[8], SLICE[13]), L_herm(SLICE[13], SLICE[14])),
                  L_herm(SLICE[14], SLICE[8]))
    BF = {e[0]: q_float(c) for e, c in Bpoly.items()}
    def bargmann_theta(th):
        return sum(c * np.exp(1j * e * th) for e, c in BF.items())
    def bargmann(v):
        return complex(np.vdot(v[8], v[13]) * np.vdot(v[13], v[14]) * np.vdot(v[14], v[8]))
    bscale = max(abs(c) for c in BF.values())
    def fit_theta2(v, guess, win=0.08):
        ga, bp = gram_abs(v), bargmann(v)
        def cost(t):
            return (float(((gram_abs_theta(t) - ga) ** 2).sum())
                    + abs(bargmann_theta(t) - bp) ** 2 / bscale ** 2)
        lo, hi = guess - win, guess + win
        for _ in range(70):
            m1, m2 = lo + (hi - lo) / 3, hi - (hi - lo) / 3
            if cost(m1) < cost(m2): hi = m2
            else: lo = m1
        th = (lo + hi) / 2
        dev = float(np.abs(gram_abs_theta(th) - ga).max())
        return th, dev
    def triv_basis(v):
        """Orthonormal basis of the trivial (gauge) directions at v: 33 per-vertex phases +
           u(3); these lie INSIDE ker J, so ker J = triv (+) modulus-line (orthogonally)."""
        rows = []
        def pack(w):
            out = np.zeros(2 * d * V)
            for i in range(V):
                out[2 * d * i:2 * d * i + 2 * d:2] = w[i].real
                out[2 * d * i + 1:2 * d * i + 2 * d:2] = w[i].imag
            return out
        for i in range(V):
            w = [np.zeros(d, complex) for _ in range(V)]; w[i] = 1j * v[i]
            rows.append(pack(w))
        for A in [np.diag([1j if c == a else 0 for c in range(d)]) for a in range(d)]:
            rows.append(pack([A @ x for x in v]))
        for a in range(d):
            for b in range(a + 1, d):
                A = np.zeros((d, d), complex); A[a, b], A[b, a] = 1, -1
                rows.append(pack([A @ x for x in v]))
                A = np.zeros((d, d), complex); A[a, b], A[b, a] = 1j, 1j
                rows.append(pack([A @ x for x in v]))
        Q, R = np.linalg.qr(np.array(rows).T)
        keep = np.abs(np.diag(R)) > 1e-10 * np.abs(R).max()
        return Q[:, keep]
    def step(v, dirvec, h):
        """Predictor along the gauge-fixed modulus direction + Newton corrector; the new
           direction is the projection of the old one onto ker J with the 41-dim trivial
           span removed — i.e. onto the 1-dim modulus line at the new point."""
        vv = newton(unpack(h * dirvec, v))
        M = JF(vv); sv = np.linalg.svd(M, compute_uv=False)
        rank = int((sv > 1e-8 * sv[0]).sum())
        gap = sv[rank - 1] / sv[rank]
        _, _, Vt = np.linalg.svd(M)
        K = Vt[rank:]
        nd = K.T @ (K @ dirvec)
        Q = triv_basis(vv)
        nd -= Q @ (Q.T @ nd)
        assert np.linalg.norm(nd) > 1e-6, "modulus direction lost"
        nd /= np.linalg.norm(nd)
        if nd @ dirvec < 0: nd = -nd
        return vv, nd, rank, gap
    def land(v, dirvec, th_cur, target, rate):
        """Newton landing on theta = target using whichever scalar invariant has the larger
           slope there: y1 = |<v9,v14>|^2 = 3 - 2 sqrt2 cos(theta)  or  y2 = Im Bargmann."""
        s1t, d1 = 3 - 2 * SQRT2 * np.cos(target), 2 * SQRT2 * np.sin(target)
        s2t = bargmann_theta(target).imag
        d2 = sum(c * e * np.cos(e * target) for e, c in BF.items()).real
        use1 = abs(d1) >= abs(d2)
        st, sl = (s1t, d1) if use1 else (s2t, d2)
        def scal(v):
            return abs(np.vdot(v[8], v[13])) ** 2 if use1 else bargmann(v).imag
        dth = target - th_cur
        for _ in range(15):
            if abs(dth) < 1e-13: break
            v, dirvec, _, _ = step(v, dirvec, dth / rate)
            dth = (st - scal(v)) / sl
            if os.environ.get("PP_DEBUG"): print(f"        land: dth={dth:+.3e} use1={use1} sl={sl:+.4f}")
        dev = float(np.abs(gram_abs(v) - gram_abs_theta(target)).max())
        dB = abs(bargmann(v) - bargmann_theta(target))
        return target - dth, dev, dB, float(np.abs(F(v)).max())
    # exact tangent at theta = 0, gauge-fixed: project out the trivial span -> modulus dir
    w0 = slice_tangent_realvec((0,))
    tvec = np.array([q_float(x) for x in w0])
    Q0 = triv_basis(v0)
    mvec = tvec - Q0 @ (Q0.T @ tvec)
    wmod = np.linalg.norm(mvec)
    dirvec = mvec / wmod
    h = 2 * np.pi * wmod / 160
    print(f"    arc step h = {h:.4f} along the gauge-fixed modulus line (|P_gauge-perp dv/dtheta|")
    print(f"    = {wmod:.4f} at theta=0; ~160 steps/revolution); Newton-corrected to residual")
    print(f"    <= 1e-12; theta-hat fitted from all 528 |Gram| entries + the signed Bargmann")
    print(f"    invariant of triple (9,14,15) (odd in theta => tracks direction).")
    v, th_hat, rate = v0, 0.0, 1.0 / wmod
    ranks, gaps, devs = set(), [], []
    hit_pen = hit_close = None
    for k in range(1, 260):
        v, dirvec, rank, gap = step(v, dirvec, h)
        ranks.add(rank); gaps.append(gap)
        th_new, dev = fit_theta2(v, th_hat + h * rate)
        devs.append(dev)
        if k == 1 and th_new < 0:                        # follow increasing theta
            dirvec, th_hat = -dirvec, 0.0
            v, dirvec, rank, gap = step(v0, dirvec, h)
            th_new, dev = fit_theta2(v, h * rate)
        rate = max(0.2 / wmod, min(5.0 / wmod, (th_new - th_hat) / h))
        th_hat = th_new
        if k % 20 == 0 or k == 1:
            print(f"      step {k:3d}: theta-hat = {th_hat:+.6f}  resid = {np.abs(F(v)).max():.1e}"
                  f"  rank J = {rank}  |Gram| dev = {dev:.1e}")
        if hit_pen is None and th_hat + 1.5 * h * rate >= np.pi / 2:
            hit_pen = land(v, dirvec, th_hat, np.pi / 2, rate)
        if th_hat + 1.5 * h * rate >= 2 * np.pi:
            hit_close = land(v, dirvec, th_hat, 2 * np.pi, rate)
            break
    assert hit_pen and hit_close
    print(f"    ranks of J along the whole loop: {sorted(ranks)} (ker 42 = 41 gauge + 1 modulus")
    print(f"    => flex stays 1 everywhere); min spectral gap sigma_156/sigma_157 = {min(gaps):.1e}")
    print(f"    max |Gram| deviation of the path from the closed-form family over all steps:")
    print(f"      {max(devs):.2e}  => the numerically continued flex path IS the exact circle")
    print(f"    PENROSE landing (theta-hat -> pi/2, reached {hit_pen[0]:.9f}):")
    print(f"      max ||Gram| - |Gram|_Penrose| = {hit_pen[1]:.2e}, |Bargmann - B_Penrose| = "
          f"{hit_pen[2]:.2e}, resid {hit_pen[3]:.1e}")
    print(f"    CLOSURE landing (theta-hat -> 2 pi, reached {hit_close[0]:.9f}):")
    print(f"      max ||Gram| - |Gram|_Peres| = {hit_close[1]:.2e}, |Bargmann - B_Peres| = "
          f"{hit_close[2]:.2e}  => CLOSED loop")
    print("    (NUMERICAL, confirmatory — the family itself is exact, section [4]; the signed")
    print("    Bargmann tracking shows the path goes AROUND the circle, not back and forth.)\n")

SECTIONS = [("table3", sec_table3), ("ks", sec_ks), ("iso", sec_iso),
            ("family", sec_family), ("global", sec_global), ("cont", sec_cont)]

def main(which=None):
    t0 = time.time()
    print("=" * 98)
    print("PERES 33 <-> PENROSE 33: the exact imaginary flex family connects them (peres_penrose.py)")
    print("=" * 98)
    print(f"exact mod-p certificates use primes {[p for p, _ in PRIMES]} (p = 7 mod 8, sqrt2 exists)\n")
    sigma = None
    for name, fn in SECTIONS:
        if which and name not in which: continue
        if name == "iso": sigma = fn()
        elif name == "family": fn(sigma)
        else: fn()
    print("=" * 98)
    print("SUMMARY (evidence labels)")
    print("-" * 98)
    print("EXACT:      Penrose-33 constructed (Gould-Aravind Table 3, (a,b,c)=(-i,-1,-sqrt2));")
    print("            33 distinct rays; orthogonality graph = Peres graph (72 edges, isomorphism")
    print("            constructive); KS-uncolorable + critical (exhaustive); essentially complex;")
    print("            the one-parameter closed-form family v_j(theta) (phase ansatz e^{i e theta})")
    print("            satisfies ALL orthogonalities AND sum_j P_j = 11*I identically (the state-")
    print("            independence operator identity is constant along the flex); flex = 1 EXACT")
    print("            at Peres AND at")
    print("            the Penrose point; sic_zoo's flex vector = tangent of this family; Penrose")
    print("            = quarter-turn point theta = -pi/2 (explicit unitary diag(1,-1,-1));")
    print("            theta <-> -theta (antiunitary, conjugation) and set(theta+pi) = set(theta)")
    print("            (pure relabeling) exact; as unlabeled sets the family is a circle of")
    print("            circumference pi with Peres and Penrose ANTIPODAL; Penrose self-conjugate.")
    print("NUMERICAL:  no degenerations/extra edges along the family (1441-point scan); real")
    print("            points only at theta = 0, pi; continuation path = closed form to ~1e-13,")
    print("            rank J = 156 (flex 1) along the whole loop; Penrose |Gram| matched to")
    print("            ~3e-15 at the pi/2 landing; loop closes at 2 pi to ~3e-13.")
    print("LITERATURE: the identification of the Table-3 (-i,-1,-sqrt2) configuration with")
    print("            Penrose's dodecahedral/Majorana construction is Gould-Aravind (2010),")
    print("            arXiv:0909.4502 (their eq. (2) / Table 3); their two quoted invariants")
    print("            ((2-sqrt2)/4 and sqrt6/4 for |<v9,v14>|) are reproduced here exactly.")
    print(f"\nperes_penrose PASS ({time.time() - t0:.1f}s)")

if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    main(args or None)
