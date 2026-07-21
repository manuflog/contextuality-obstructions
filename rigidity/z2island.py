#!/usr/bin/env python3
"""
B1 — THE Z[sqrt(-2)] FLEXIBLE ISLAND (second test of Dictionary v3).

Kernaghan, "The Algebraic Landscape of Kochen-Specker Sets in Dimension Three",
arXiv:2603.16988 (v2, March 2026), reports six minimal-KS "algebraic islands" in d = 3 and a
Jacobian rigidity analysis (his Observation 28): four islands rigid in C^3, while "the Peres and
Z[sqrt(-2)] islands each have exactly one deformation dimension", and (verbatim, v2 §7.6):

    "The Peres/Z[sqrt(-2)] flex is infinitesimal only: second-order obstruction analysis shows a
     nonzero component in the cokernel of the Jacobian, blocking extension to a finite deformation.
     The set is infinitesimally flexible but finitely rigid — the moduli space has isolated points
     (Peres and Penrose realizations) rather than a continuous family."

His island: ring Z[sqrt(-2)], alphabet {0, +-1, +-sqrt(-2)} (sqrt(-2) = i sqrt2), 49 rays,
120 orthogonal pairs, 16 triads, OCUS-certified pool minimum 33; min-33 set: 72 edges, 16 bases,
degree distribution {4:30, 8:3}; same orthogonality graph as Peres-33 (his Proposition 19).

WHAT THIS FILE PROVES (machine-verified; labels EXACT unless marked):

  [1] RECONSTRUCTION. The island is reconstructed from its defining alphabet (the paper prints
      the generating description, not the ray list): all 49 projective rays over the alphabet,
      exactly 120 Hermitian-orthogonal pairs and 16 triads, pool KS-UNCOLORABLE (full rules,
      exhaustive); the 33 basis-participating rays form the UNIQUE minimal KS subset (min = 33
      re-proved here, not just quoted: auxiliary rays lie in no triad, so every uncolorable
      subset restricts to an uncolorable subset of the 33, and criticality kills every proper
      subset). 72 edges, 16 bases, degrees {4:30, 8:3} — every printed invariant of Kernaghan's
      Z[sqrt(-2)]-33 is matched.  All arithmetic in Z[i, sqrt2], no floats.

  [2] IDENTIFICATION. The 33-ray set is EXACTLY the Gould-Aravind Table-3 configuration at
      (a,b,c) = (1, 1, sqrt(-2)) (ray-by-ray bijection, projective equality on the nose), hence
      carries the SAME 72-edge graph as Peres-33 (constructive isomorphism — Kernaghan's Prop. 19
      re-proved exactly) and sits ON the Gould-Aravind phase circle at modulus phi = +pi/2.

  [3] FLEX = 1 EXACT. sic_zoo's mod-p + exact-kernel certificate at the Z[sqrt(-2)] point:
      rank_p J = 156 at two primes => flex <= 1; the explicit family tangent satisfies J.w = 0 in
      exact arithmetic and is not in the trivial span => FLEX = 1 EXACT (dim ker = 42 = 41 gauge
      + 1 modulus, agreeing with Kernaghan's numerical 42).

  [4] THE FLEX IS FINITE (Kernaghan's "infinitesimal only" is REFUTED at this island too).
      The one-parameter family v_j(theta) with entries m * e^{i e theta} (m in Z[i,sqrt2],
      e in {-1,0,1}; the GA phase ansatz) passes through the Z[sqrt(-2)] set at theta = 0 and
      satisfies ALL 72 orthogonalities and all 33 norm constraints IDENTICALLY in theta (Laurent
      polynomials over Z[i,sqrt2]).  In particular the second-order obstruction Kernaghan claims
      is nonzero is verified here to VANISH exactly: J.v'' = -C''(v',v') has the explicit exact
      solution v'' = d^2v/dtheta^2 (all 72+33 second-order identities checked in Z[i,sqrt2]).
      Endpoints: a quarter turn (theta = -pi/2) lands ray-by-ray on the REAL Peres-33 set (exact
      diagonal unitary diag(1,-i,1)); the base point itself is monomial-unitarily equivalent to
      the PENROSE-33 configuration.  So Kernaghan's TWO flexible islands are the two antipodes
      of ONE exact closed circle of KS sets — his "isolated points (Peres and Penrose)" are
      connected by the very flex he computed.

  [5] WITNESS PRESERVATION (Dictionary v3, second confirmation). Along the whole family,
      sum_j P_j = 11 * I IDENTICALLY in theta (12*sum = 132*I over the Laurent ring): the flex
      moves inside the state-independence variety, exactly as for Peres-33.

Engine reuse: sic_zoo.py and peres_penrose.py are imported UNMODIFIED (Z[sqrt2]/Z[i,sqrt2]
arithmetic, Laurent polynomials, GA Table 3, mod-p ranks, exhaustive KS checker, complex
extended Jacobian + flex certificate).

Run:  python3 z2island.py                (~1 min, all checks printed)
      python3 z2island.py pool flex      (sections: pool min33 ga flex family)
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from itertools import combinations, permutations, product

import sic_zoo as SZ
from sic_zoo import (Z0, q_neg, ks_colorable, PRIMES, rank_mod_p, q_rowdot, rays_peres33)
import peres_penrose as PP
from peres_penrose import (Q4_0, Q4_1, Q4_I, q4_add, q4_neg, q4_mul, q4_conj, q4_float,
                           q4_is0, q4_herm, L_eval_q4, table3, GEN, T1_EDGES,
                           rays_q4, graph_q4, distinct_q4, exact_flex_certificate_c,
                           Q_PERES, Q_PENROSE)
from flex_dimension import flex_dimension

# ----------------------------------------------------------------- ring & alphabet
# Q4 element ((a,b),(c,d)) = (a + b sqrt2) + i (c + d sqrt2)   [peres_penrose format]
ONE4, MONE4 = (( 1, 0), Z0), ((-1, 0), Z0)
RTm2, MRTm2 = (Z0, (0, 1)), (Z0, (0, -1))          # +- sqrt(-2) = +- i sqrt2
ALPHA = [(Z0, Z0), ONE4, MONE4, RTm2, MRTm2]        # Kernaghan's alphabet {0, +-1, +-sqrt(-2)}

def canon(v):
    """Canonical projective representative for vectors with entries in ALPHA:
       divide by sqrt(-2) if every nonzero entry is +-sqrt(-2), then fix the sign so the first
       nonzero entry is 1 or sqrt(-2). Unique per ray (proof: a scalar mapping ALPHA\\{0} into
       ALPHA\\{0} is +-1 or +-sqrt(-2)^{+-1}, and the reduction+sign kills all of them)."""
    assert all(x in ALPHA for x in v) and any(not q4_is0(x) for x in v)
    if all(q4_is0(x) or x[0] == Z0 for x in v):     # all nonzero entries pure +-i sqrt2
        v = tuple(Q4_0 if q4_is0(x) else ((x[1][1], 0), Z0) for x in v)
    neg = next((x in (MONE4, MRTm2)) for x in v if not q4_is0(x))
    return tuple(q4_neg(x) for x in v) if neg else tuple(v)

def proj_eq(u, v):
    """Exact projective equality of Q4 triples (2x2 minors + zero patterns)."""
    for a in range(3):
        for b in range(a + 1, 3):
            if not q4_is0(q4_add(q4_mul(u[a], v[b]), q4_neg(q4_mul(u[b], v[a])))): return False
    return all(q4_is0(u[c]) == q4_is0(v[c]) for c in range(3))

def herm_complement(v, w):
    """Hermitian analogue of the cross product (Kernaghan Def. 14, complex islands):
       cofactors of the conjugated pair — the unique ray orthogonal to both."""
    vb = [q4_conj(x) for x in v]; wb = [q4_conj(x) for x in w]
    u = []
    for k in range(3):
        a, b = (k + 1) % 3, (k + 2) % 3
        u.append(q4_add(q4_mul(vb[a], wb[b]), q4_neg(q4_mul(vb[b], wb[a]))))
    return tuple(u)

# ----------------------------------------------------------------- Q4-Laurent (one variable)
# QL = dict {int exponent e : Q4 coefficient}; represents sum_e  c_e * z^e,  z = e^{i theta}.
def ql_add(u, v):
    out = dict(u)
    for e, c in v.items():
        s = q4_add(out.get(e, Q4_0), c)
        if q4_is0(s): out.pop(e, None)
        else: out[e] = s
    return out

def ql_mul(u, v):
    out = {}
    for e1, c1 in u.items():
        for e2, c2 in v.items():
            s = q4_add(out.get(e1 + e2, Q4_0), q4_mul(c1, c2))
            if q4_is0(s): out.pop(e1 + e2, None)
            else: out[e1 + e2] = s
    return out

def ql_conj(u): return {-e: q4_conj(c) for e, c in u.items()}

def ql_herm(x, y):
    s = {}
    for a, b in zip(x, y): s = ql_add(s, ql_mul(ql_conj(a), b))
    return s

def ql_scale(u, k):  # integer scalar
    return {e: ((k * c[0][0], k * c[0][1]), (k * c[1][0], k * c[1][1])) for e, c in u.items()}

def ql_eval_q(u, q):
    """Evaluate at z = i^q, exactly in Q4."""
    out = Q4_0
    for e, c in u.items():
        for _ in range((q * e) % 4): c = q4_mul(c, Q4_I)
        out = q4_add(out, c)
    return out

def ql_deriv_at1(u):
    """d/dtheta at theta = 0 of sum c_e e^{i e theta}  =  sum (i e) c_e."""
    out = Q4_0
    for e, c in u.items():
        out = q4_add(out, q4_mul((Z0, (e, 0)), c))
    return out

def ql_deriv2_at1(u):
    """d^2/dtheta^2 at theta = 0  =  sum (-e^2) c_e."""
    out = Q4_0
    for e, c in u.items():
        out = q4_add(out, ((-e * e * c[0][0], -e * e * c[0][1]),
                           (-e * e * c[1][0], -e * e * c[1][1])))
    return out

# =====================================================================================
# [1] the pool and the minimal set, reconstructed from the alphabet
# =====================================================================================
def build_pool():
    vecs = [v for v in product(ALPHA, repeat=3) if any(not q4_is0(x) for x in v)]
    pool = sorted({canon(v) for v in vecs})
    E = sorted(tuple(sorted(e)) for e in graph_q4(pool))
    triads = [t for t in combinations(range(len(pool)), 3)
              if all(tuple(sorted(p)) in set(E) for p in combinations(t, 2))]
    return pool, E, triads

def sec_pool():
    print("[1] The Z[sqrt(-2)] island reconstructed from the alphabet {0, +-1, +-sqrt(-2)}")
    pool, E, triads = build_pool()
    assert distinct_q4(pool), "canonicalization produced a projective duplicate"
    print(f"    projective rays over the alphabet: {len(pool)} (Kernaghan: 49)")
    print(f"    Hermitian-orthogonal pairs: {len(E)} (Kernaghan: 120); triads: {len(triads)} "
          f"(Kernaghan: 16)")
    assert (len(pool), len(E), len(triads)) == (49, 120, 16)
    col, nodes = ks_colorable(49, E, [list(t) for t in triads])
    assert not col
    print(f"    the 49-ray pool is KS-UNCOLORABLE (full rules, exhaustive backtracking, "
          f"{nodes} nodes)")
    # island closure (Kernaghan Def. 14, complex case): Hermitian complements of all
    # orthogonal pairs, iterated to stabilization. Like the REAL Peres island (49 -> 145,
    # his Table 5), the 49-ray alphabet pool is NOT complement-closed; Kernaghan's quoted
    # island numbers (49/120/16/min-33) are the raw alphabet pool's, and all minimality
    # claims below are scoped to that pool, exactly as his OCUS certificate is.
    closed = list(pool)
    rounds = 0
    while rounds < 6:
        cur_edges = [(i, j) for i, j in combinations(range(len(closed)), 2)
                     if q4_is0(q4_herm(closed[i], closed[j]))]
        fresh = []
        for i, j in cur_edges:
            u = herm_complement(closed[i], closed[j])
            if not any(proj_eq(u, w) for w in closed + fresh): fresh.append(u)
        if not fresh: break
        closed += fresh; rounds += 1
    assert rounds < 6, "completion did not stabilize"
    cE = [(i, j) for i, j in combinations(range(len(closed)), 2)
          if q4_is0(q4_herm(closed[i], closed[j]))]
    cEs = set(cE)
    cT = [t for t in combinations(range(len(closed)), 3)
          if all(tuple(sorted(p)) in cEs for p in combinations(t, 2))]
    ccol, cn = ks_colorable(len(closed), cE, [list(t) for t in cT])
    assert not ccol
    print(f"    Hermitian-complement completion (Def. 14): stabilizes in {rounds} rounds at "
          f"{len(closed)} rays,")
    print(f"    {len(cE)} pairs, {len(cT)} triads, still KS-UNCOLORABLE ({cn} nodes) — the")
    print(f"    alphabet pool is not complement-closed (cf. real Peres 49 -> 145, his Table 5);")
    print(f"    Kernaghan's quoted 49/120/16 are the raw alphabet pool, used below as he does.")
    return pool, E, triads

def sec_min33(pool, E, triads):
    print("[2] The minimal KS set = the 33 basis-participating rays (UNIQUE minimum, EXACT)")
    inb = sorted({r for t in triads for r in t})
    aux = [r for r in range(49) if r not in inb]
    assert len(inb) == 33 and len(aux) == 16
    print(f"    basis-participating rays: 33; auxiliary rays (in no triad): 16 (Kernaghan: 33%)")
    rn = {v: k for k, v in enumerate(inb)}
    ours = [pool[r] for r in inb]
    E33 = sorted((rn[i], rn[j]) for i, j in E if i in rn and j in rn)
    T33 = [[rn[x] for x in t] for t in triads]
    deg = [0] * 33
    for i, j in E33: deg[i] += 1; deg[j] += 1
    dd = {k: deg.count(k) for k in sorted(set(deg))}
    print(f"    edges: {len(E33)} (Kernaghan: 72); bases: {len(T33)} (Kernaghan: 16); "
          f"degree distribution {dd} (Kernaghan: {{4:30, 8:3}})")
    assert len(E33) == 72 and len(T33) == 16 and dd == {4: 30, 8: 3}
    col, nodes = ks_colorable(33, E33, T33)
    assert not col
    print(f"    KS-UNCOLORABLE (exhaustive, {nodes} nodes)")
    tot = 0
    for r in range(33):
        keep = [i for i in range(33) if i != r]
        rn2 = {v: k for k, v in enumerate(keep)}
        sp = [(rn2[i], rn2[j]) for i, j in E33 if i != r and j != r]
        st = [[rn2[x] for x in t] for t in T33 if r not in t]
        c, nn = ks_colorable(32, sp, st)
        tot += nn
        assert c, f"deletion of ray {r} still uncolorable"
    print(f"    CRITICAL: all 33 single-ray deletions KS-colorable (33 exhaustive searches, "
          f"{tot} nodes)")
    # uniqueness + pool-minimum, re-proved (not quoted):
    #   (i) every 3-clique of the pool graph lies inside the 33 (auxiliary rays are in no triad
    #       — true by construction of inb, asserted above);
    #   (ii) hence any coloring of S∩B33 extends by f=0 on auxiliary rays to a coloring of S,
    #        so S uncolorable => S∩B33 uncolorable;
    #   (iii) KS-colorability is inherited by subsets (triads of a subset are triads of the set),
    #        so criticality kills every proper subset of B33.
    #   => every KS-uncolorable subset of the pool CONTAINS the 33; min = 33, UNIQUE.
    print("    => every KS-uncolorable subset of the 49-pool contains these 33 rays:")
    print("       pool minimum = 33, and the minimal set is UNIQUE (re-proves Kernaghan's")
    print("       OCUS-certified min-33 exactly, and sharpens it to uniqueness). EXACT.\n")
    return ours, E33, T33

# =====================================================================================
# [3] identification with the Gould-Aravind family at (a,b,c) = (1, 1, sqrt(-2))
# =====================================================================================
def sec_ga(ours):
    print("[3] Identification: the 33-set IS Gould-Aravind Table 3 at (a,b,c) = (1,1,sqrt(-2))")
    gv = rays_q4((0, 0, 1), GEN)                    # z_a = 1, z_b = 1, z_c = i (c = i sqrt2)
    assert all(x in ALPHA for v in gv for x in v), "GA point not alphabet-valued?!"
    lut = {canon(v): t for t, v in enumerate(gv)}
    assert len(lut) == 33
    tau = [lut[u] for u in ours]                    # our index -> GA Table-3 label
    assert sorted(tau) == list(range(33))
    eps = []
    for j, u in enumerate(ours):
        g = gv[tau[j]]
        if u == g: eps.append(1)
        else:
            assert u == tuple(q4_neg(x) for x in g)
            eps.append(-1)
    print("    ray-by-ray bijection tau (projective equality ON THE NOSE, signs +-1): every")
    print("    alphabet ray equals a Table-3 ray at (1,1,sqrt(-2)) and vice versa. EXACT.")
    E33_lab = {frozenset((tau[i], tau[j])) for i in range(33) for j in range(i + 1, 33)
               if q4_is0(q4_herm(ours[i], ours[j]))}
    assert E33_lab == T1_EDGES
    print("    the 72 edges map exactly onto the 72-edge Gould-Aravind Table-1 graph =>")
    print("    same orthogonality graph as Peres-33 (constructive; via peres_penrose's sigma")
    print("    this is an explicit Peres-33 <-> Z[sqrt(-2)]-33 isomorphism). Kernaghan's")
    print("    Proposition 19 re-proved EXACTLY (his: VF2, numerical-free but uncertified).")
    print("    Gauge position: phases (alpha,beta,gamma) = (0, 0, pi/2), modulus")
    print("    phi = alpha - beta + gamma = +pi/2 — the ANTIPODE of Peres (phi = 0) on the")
    print("    Gould-Aravind circle, i.e. the Penrose point of PERES_PENROSE.md.\n")
    print(f"    tau = {tau}")
    return tau, eps

# =====================================================================================
# [4] exact flex certificate at the Z[sqrt(-2)] point
# =====================================================================================
def z2_family(tau, eps):
    """The GA one-parameter family through the Z[sqrt(-2)] point, as Q4-Laurent vectors in OUR
       ray order: a = e^{i theta}, b = 1, c = sqrt(-2) fixed. Entries are single monomials
       m z^e, m in Z[i,sqrt2], e in {-1,0,1} (the phase ansatz)."""
    fam = []
    for j in range(33):
        ray = []
        for comp in GEN[tau[j]]:
            ql = {}
            for expo, m in comp.items():
                ea, eb, ec = expo
                c4 = ((m[0] * eps[j], m[1] * eps[j]), Z0)
                for _ in range(ec % 4): c4 = q4_mul(c4, Q4_I)   # z_c = i
                # z_b = 1 contributes nothing; z_a stays symbolic
                ql = ql_add(ql, {ea: c4})
            ray.append(ql)
        fam.append(tuple(ray))
    return fam

def tangent_realvec(fam):
    """d/dtheta at theta = 0 as the 198-dim real vector of Z[sqrt2] pairs (layout of
       peres_penrose.build_ext_jac_c: coord(j,c,re/im) = 6j + 2c + (0|1))."""
    out = [Z0] * 198
    for j, ray in enumerate(fam):
        for c, comp in enumerate(ray):
            w = ql_deriv_at1(comp)
            out[6 * j + 2 * c] = w[0]; out[6 * j + 2 * c + 1] = w[1]
    return out

def sec_flex(ours, tau, eps):
    print("[4] FLEX = 1 EXACT at the Z[sqrt(-2)] point (mod-p bound + exact kernel vector)")
    fam = z2_family(tau, eps)
    for j in range(33):                              # family passes through our rays at theta=0
        assert tuple(ql_eval_q(c, 0) for c in fam[j]) == ours[j]
    w = tangent_realvec(fam)
    exact_flex_certificate_c(ours, w, "Z[sqrt(-2)]-33")
    print("    dim ker(extended J) = 198 - 156 = 42 = 33 vertex phases + u(3) - 1 global")
    print("    relation (41 gauge) + 1 modulus — matches Kernaghan's numerical dim ker = 42")
    print("    (his Observation 28 table), now certified EXACTLY.")
    fx = flex_dimension([np.array([q4_float(x) for x in v]) for v in ours],
                        name="Z[sqrt(-2)]-33")
    print(f"    numerical flex engine cross-check: flex = {fx} (expected 1)\n")
    assert fx == 1
    return fam, w

# =====================================================================================
# [5] the two Dictionary-v3 tests: finiteness + witness preservation
# =====================================================================================
def sec_family(ours, tau, eps, fam, w):
    print("[5] TEST (a): the flex is FINITE — closed-form family, all constraints identical in theta")
    # 5a. phase ansatz + identical orthogonality
    for ray in fam:
        for comp in ray:
            assert len(comp) <= 1 and all(e in (-1, 0, 1) for e in comp)
    print("    [5a] every entry of v_j(theta) is a single monomial m e^{i e theta},")
    print("         m in Z[i,sqrt2], e in {-1,0,1} — the Gould-Aravind phase ansatz, EXACT.")
    edges = {frozenset((i, j)) for i, j in combinations(range(33), 2)
             if not ql_herm(fam[i], fam[j])}
    E33 = {frozenset((i, j)) for i, j in combinations(range(33), 2)
           if q4_is0(q4_herm(ours[i], ours[j]))}
    assert edges == E33 and len(edges) == 72
    norms = []
    for j in range(33):
        n = ql_herm(fam[j], fam[j])
        assert list(n) == [0] and n[0][1] == Z0 and n[0][0][1] == 0
        norms.append(n[0][0][0])
    assert sorted(set(norms)) == [1, 2, 3, 4]
    print("    all 72 edge inner products vanish IDENTICALLY as Laurent polynomials over")
    print("    Z[i,sqrt2]; the 456 non-edge products do not; all 33 norms are CONSTANT in theta.")
    print("    => a genuine one-parameter family of KS sets through the Z[sqrt(-2)] island")
    print("       (uncolorability holds for every theta: the 72-edge/16-triad structure is")
    print("       present identically, and extra accidental edges could only add constraints).")
    print("    THE FLEX IS FINITE. Kernaghan's 'infinitesimal (but not finite) flex' /")
    print("    'blocking extension to a finite deformation' (Obs. 28) is REFUTED at the")
    print("    Z[sqrt(-2)] island — same verdict as for Peres-33 (PERES_PENROSE.md). EXACT.")
    # 5b. the claimed second-order obstruction vanishes, explicitly
    v0 = [tuple(ql_eval_q(c, 0) for c in ray) for ray in fam]
    v1 = [tuple(ql_deriv_at1(c) for c in ray) for ray in fam]
    v2 = [tuple(ql_deriv2_at1(c) for c in ray) for ray in fam]
    assert v0 == ours
    for i, j in [tuple(sorted(e)) for e in
                 {frozenset(p) for p in combinations(range(33), 2)} & edges]:
        s = q4_add(q4_add(q4_herm(v2[i], v0[j]), q4_herm(v0[i], v2[j])),
                   ql_scale({0: q4_herm(v1[i], v1[j])}, 2)[0])
        assert q4_is0(s), f"second-order obstruction nonzero on edge ({i},{j})?!"
    for i in range(33):
        s = q4_add(q4_add(q4_herm(v2[i], v0[i]), q4_herm(v0[i], v2[i])),
                   ql_scale({0: q4_herm(v1[i], v1[i])}, 2)[0])
        assert q4_is0(s), f"second-order norm obstruction nonzero at ray {i}?!"
    print("    [5b] SECOND-ORDER OBSTRUCTION = 0, verified directly: with w = v'(0) the exact")
    print("         tangent, v'' = v''(0) solves J.v'' = -C''(w,w) for all 72 edge and 33 norm")
    print("         constraints (checked in Z[i,sqrt2]. Kernaghan's 'nonzero component in the")
    print("         cokernel' cannot be a property of this configuration/flex; the family is")
    print("         unobstructed to ALL orders, since the constraints vanish identically). EXACT.")
    # 5c. endpoints: quarter-turn = the REAL Peres-33; the base point = Penrose-33
    print("    [5c] where the family goes:")
    per_gen = rays_q4(Q_PERES, GEN)
    U = (Q4_1, q4_neg(Q4_I), Q4_1)                  # diag(1, -i, 1)
    vq = [tuple(ql_eval_q(c, 3) for c in ray) for ray in fam]   # theta = -pi/2
    for j in range(33):
        Uv = tuple(q4_mul(U[c], vq[j][c]) for c in range(3))
        assert proj_eq(Uv, per_gen[tau[j]]), f"Peres landing fails at ray {j}"
    print("         theta = -pi/2: diag(1,-i,1) maps the configuration ray-by-ray onto the")
    print("         REAL Peres-33 set (GA Table 3 at (1,1,sqrt2); = sic_zoo's rays_peres33 by")
    print("         peres_penrose section [3]). A quarter turn of the flex connects the")
    print("         Z[sqrt(-2)] island to the Peres island. EXACT.")
    pen = rays_q4(Q_PENROSE)
    A = [np.array([q4_float(x) for x in v]) for v in ours]
    A = [a / np.linalg.norm(a) for a in A]
    B = [np.array([q4_float(x) for x in v]) for v in pen]
    B = [b / np.linalg.norm(b) for b in B]
    UNITS = (Q4_1, Q4_I, q4_neg(Q4_1), q4_neg(Q4_I))
    UF = (1, 1j, -1, -1j)
    found = None
    for perm in permutations(range(3)):
        for us in product(range(4), repeat=3):
            g, ok = [], True
            for j in range(33):
                Rv = np.array([UF[us[c]] * A[j][perm[c]] for c in range(3)])
                m = [k for k in range(33) if abs(abs(np.vdot(Rv, B[k])) - 1) < 1e-9]
                if len(m) != 1: ok = False; break
                g.append(m[0])
            if ok and sorted(g) == list(range(33)):
                found = (perm, us, g); break
        if found: break
    assert found, "no monomial unitary onto the Penrose configuration found"
    perm, us, g = found
    for j in range(33):                              # exact verification of the numerical find
        Rv = tuple(q4_mul(UNITS[us[c]], ours[j][perm[c]]) for c in range(3))
        assert proj_eq(Rv, pen[g[j]]), f"Penrose identification fails at ray {j}"
    print(f"         theta = 0 (the island itself): the monomial unitary R (permutation {perm},")
    print(f"         diagonal i-powers {us}) maps the 33 alphabet rays ray-by-ray onto the")
    print("         PENROSE-33 configuration (GA Table 3 at (-i,-1,-sqrt2)); relabeling")
    print(f"         g = {g}")
    print("         => Kernaghan's Z[sqrt(-2)] island IS Penrose-33 up to a unitary — his two")
    print("         'isolated points (Peres and Penrose realizations)' are his two flexible")
    print("         islands, and they sit ANTIPODALLY on one exact circle of KS sets. EXACT.")
    # 5d. witness preservation
    M = [[{} for _ in range(3)] for _ in range(3)]
    for j in range(33):
        n = ql_herm(fam[j], fam[j])[0][0][0]
        for a in range(3):
            for b in range(3):
                M[a][b] = ql_add(M[a][b], ql_scale(ql_mul(fam[j][a], ql_conj(fam[j][b])),
                                                   12 // n))
    for a in range(3):
        for b in range(3):
            want = {0: (((132, 0)), Z0)} if a == b else {}
            assert M[a][b] == want, "tight-frame identity fails along the family"
    print("    [5d] TEST (b): WITNESS PRESERVATION. sum_j P_j = 11*I holds IDENTICALLY in theta")
    print("         (12 * sum = 132 * I over the Laurent ring Z[i,sqrt2][z,z^-1]): the flex of")
    print("         the Z[sqrt(-2)] island moves INSIDE the state-independence variety")
    print("         sum P = (V/d) I — the second confirmation of Dictionary v3. EXACT.\n")

SECTIONS = ["pool", "min33", "ga", "flex", "family"]

def main(which=None):
    t0 = time.time()
    print("=" * 98)
    print("THE Z[sqrt(-2)] FLEXIBLE ISLAND (Kernaghan arXiv:2603.16988): reconstruction, exact")
    print("flex, finiteness, witness preservation (z2island.py)")
    print("=" * 98)
    print(f"exact mod-p certificates use primes {[p for p, _ in PRIMES]} (p = 7 mod 8)\n")
    pool, E, triads = sec_pool()
    ours, E33, T33 = sec_min33(pool, E, triads)
    if which and set(which) <= {"pool", "min33"}: return
    tau, eps = sec_ga(ours)
    fam, w = sec_flex(ours, tau, eps)
    sec_family(ours, tau, eps, fam, w)
    print("=" * 98)
    print("SUMMARY (evidence labels)")
    print("-" * 98)
    print("EXACT:      49-ray alphabet pool (completion computed: not complement-closed);")
    print("            120 pairs, 16 triads; pool uncolorable; minimal KS subset = the 33")
    print("            basis-participating rays, UNIQUE, critical (min-33 re-proved, not quoted);")
    print("            72 edges / 16 bases / degrees {4:30, 8:3} — all of Kernaghan's invariants;")
    print("            ray-by-ray = GA Table 3 at (1,1,sqrt(-2)) => Peres graph (Prop. 19 exact);")
    print("            FLEX = 1 (mod-p bound + exact tangent, dim ker = 42 = 41+1);")
    print("            the flex is FINITE: closed-form family m e^{i e theta}, all 72+33")
    print("            constraints identical in theta; second-order obstruction = 0 explicitly;")
    print("            quarter-turn endpoint = REAL Peres-33 (diag(1,-i,1)); base point =")
    print("            Penrose-33 (explicit monomial unitary); sum_j P_j = 11*I identically.")
    print("NUMERICAL:  flex_dimension cross-check (flex = 1); float prefilter of the Penrose")
    print("            unitary search (the found unitary is then verified exactly).")
    print("LITERATURE: Kernaghan's island data used as cross-checks only (49/120/16/33/72/")
    print("            {4:30,8:3}/42) — every number is independently recomputed here; the")
    print("            'infinitesimal only' claim of his Obs. 28 is contradicted by the exact")
    print("            family; GA Table 3 transcription is peres_penrose.py's (verified there).")
    print(f"\nz2island PASS ({time.time() - t0:.1f}s)")

if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    main(args or None)
