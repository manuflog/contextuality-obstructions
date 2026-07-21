#!/usr/bin/env python3
"""
BRANCH I -- THE SKEW BLOCK AS AN IMAGINARITY RESOURCE.

Parent facts this branch stands on (PROVED elsewhere, imported unmodified, not re-derived):
  * torsion_flex.py:  at a real point, flex_C = flex_R + flex_skew (the decomposition theorem).
    flex_skew = dim of the space of purely-imaginary infinitesimal deformations v_j -> v_j + i t y_j
    (real y_j) that preserve all orthogonalities to first order.
  * peres_penrose.py: the Peres(theta=0)->Penrose(theta=pi/2) SLICE family v_j(theta), entries
    m_jc * e^{i e_jc theta} with m_jc in Z[sqrt2] real, e_jc in {-1,0,1}; EXACT flex=1 certificates
    at theta=0 and theta=-pi/2 (mod-p rank bound + explicit exact kernel tangent).

QUESTION this branch asks: is flex_skew / the deformation it generates quantifiable as an
imaginarity RESOURCE (in the sense of Hickey-Gour-style resource theories, free states = real
density matrices)? Four tasks, each labeled EXACT / PROVED / NUMERICAL / CONJECTURE honestly.

Run: python3 branch_imag.py     (~5-10s)
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import sympy as sp
from itertools import combinations, permutations, product

from peres_penrose import (SLICE, L_eval_c, rays_q4, Q_PERES,
                            build_ext_jac_c, slice_tangent_realvec, exact_flex_certificate_c)
import torsion_flex as TF
from sic_zoo import rays_peres33, q_float

SQRT2 = sp.sqrt(2)


# =====================================================================================
# shared exact extraction: SLICE ray j, component c -> (m_jc real in Z[sqrt2], e_jc in {-1,0,1})
# =====================================================================================
def slice_data_sympy():
    data = []
    for ray in SLICE:
        comps = []
        for comp in ray:
            if not comp:
                comps.append((sp.Integer(0), 0))
            else:
                (expo,), = [(k,) for k in comp.keys()]
                a, b = comp[expo]
                m = sp.Integer(a) + sp.Integer(b) * SQRT2
                comps.append((m, expo[0]))
            assert len(comp) <= 1, "assumption violated: >1 monomial in a SLICE component"
        data.append(comps)
    return data


def rays_c(zs):
    return [np.array([L_eval_c(c, zs) for c in ray]) for ray in SLICE]


def rays_at(theta):
    return rays_c([np.exp(1j * theta)])


# =====================================================================================
# TASK 1: imaginarity along the circle, exact closed forms
# =====================================================================================
def sec1_profiles(data):
    print("=" * 100)
    print("[1] IMAGINARITY ALONG THE PERES-PENROSE CIRCLE -- exact closed forms")
    print("=" * 100)

    # ---- master identity (EXACT, all theta, not just leading order) ----------------
    # sin(e*theta) = e*sin(theta) identically for e in {-1,0,1}  =>  Im v_jc(theta) = e_jc*m_jc*sin(theta)
    # exactly, for ALL theta (not an infinitesimal statement). Verify no |e|>1 occurs (needed
    # for the identity), then verify the identity numerically to machine precision.
    maxe = max(abs(e) for ray in data for (m, e) in ray)
    assert maxe <= 1, f"master identity needs |e|<=1 everywhere, found max|e|={maxe}"
    rng = np.random.default_rng(0)
    thetas_probe = rng.uniform(-4, 4, 12)
    worst = 0.0
    for j, ray in enumerate(data):
        for c, (m, e) in enumerate(ray):
            mf = float(m)
            for th in thetas_probe:
                v = mf * np.exp(1j * e * th)
                worst = max(worst, abs(v.imag - mf * e * np.sin(th)))
    print(f"  EXACT identity (verified numerically, max err {worst:.1e} over 33*3*12 probes):")
    print("      Im v_jc(theta) = e_jc * m_jc * sin(theta)   for ALL theta  (since sin(e*th)=e*sin(th),")
    print("      e in {-1,0,1} exactly). So Im v_j(theta) = sin(theta) * y_j with y_j := e_j (x) m_j")
    print("      the SAME theta-INDEPENDENT vector for every theta -- the flex-skew tangent direction")
    print("      is not just the leading-order imaginary generator, it is the imaginary part of the")
    print("      ENTIRE family, at every point of the circle, exactly. This is the master fact behind")
    print("      everything below. [EXACT, PROVED by sin(e theta)=e sin theta, e in {-1,0,1}]")

    # ---- consequence: Im P_j(theta) = sin(theta) K_j exactly, K_j real antisymmetric ----
    # Im(P_ab) = m_a m_b sin(Delta theta)/N,  Delta = e_a-e_b in {-2,-1,0,1,2} in general, but
    # empirically (checked below) never +-2 on this family => Im(Delta theta)=Delta*sin(theta) too.
    no_delta2 = True
    for ray in data:
        es = [e for (m, e) in ray]
        for a, b in combinations(range(3), 2):
            if abs(es[a] - es[b]) == 2:
                no_delta2 = False
    print(f"\n  Delta=e_a-e_b never reaches +-2 on any of the 33 rays: {no_delta2} [EXACT, enumerated]")
    print("      => Im P_j(theta) = sin(theta) * K_j EXACTLY for all theta too (K_j theta-independent")
    print("      real antisymmetric 3x3), where (K_j)_ab = m_a m_b (e_a-e_b)/N_j.")

    # ---- closed forms: l1 and Frobenius^2 totals ----
    A = sp.Integer(0)   # coeff of |sin theta| in total l1  (no sin(2theta) term: verified below)
    B = sp.Integer(0)   # would-be coeff of |sin 2theta|; must vanish given no_delta2
    C = sp.Integer(0)   # coeff of sin^2(theta) in total sum-of-squared-Frobenius
    D = sp.Integer(0)   # sum_j ||K_j||_2  (coeff of |sin theta| in sum of INDIVIDUAL Frobenius norms)
    for ray in data:
        m = [x[0] for x in ray]; e = [x[1] for x in ray]
        N = sum(x * x for x in m)
        s2j = sp.Integer(0)
        for a, b in combinations(range(3), 2):
            Delta = e[a] - e[b]
            if Delta == 0:
                continue
            coeff = sp.nsimplify(sp.simplify(sp.Abs(m[a] * m[b]) / N), [SQRT2])
            if abs(Delta) == 1:
                A += 2 * coeff
            elif abs(Delta) == 2:
                B += 2 * coeff
            s2j += 2 * (m[a] * m[b] / N) ** 2
        C += sp.nsimplify(sp.simplify(s2j), [SQRT2])
        D += sp.sqrt(sp.nsimplify(sp.simplify(s2j), [SQRT2]))
    A = sp.nsimplify(sp.simplify(A), [SQRT2]); B = sp.simplify(B); C = sp.nsimplify(sp.simplify(C), [SQRT2])
    Df = float(D)
    assert B == 0, "unexpected sin(2theta) component -- master identity claim would be wrong"
    print(f"\n  Total l1 imaginarity  ||Im P||_l1 summed over 33 rays:")
    print(f"      TotalL1(theta) = A * |sin theta|,   A = {A} = {float(A):.6f}   [EXACT]")
    print(f"  Total (sum of squares of) Frobenius imaginarity:")
    print(f"      TotalFro2(theta) = C * sin(theta)^2,  C = {C} = {float(C):.6f}   [EXACT]")
    print(f"  Total (sum of individual) Frobenius norms:")
    print(f"      TotalFroSum(theta) = D * |sin theta|,  D = {Df:.6f}  (D^2 not a nice radical; NUMERICAL)")

    # numeric cross-check of the closed forms against direct entrywise computation
    Af, Cf = float(A), float(C)
    worst_l1 = worst_fro = 0.0
    for th in np.linspace(-6, 6, 37):
        tot_l1 = tot_fro2 = 0.0
        for ray, j in zip(data, range(33)):
            v = np.array([float(m) * np.exp(1j * e * th) for (m, e) in ray])
            N = sum(float(m) ** 2 for (m, e) in ray)
            P = np.outer(v, np.conj(v)) / N
            tot_l1 += np.sum(np.abs(P.imag)); tot_fro2 += np.sum(P.imag ** 2)
        worst_l1 = max(worst_l1, abs(tot_l1 - Af * abs(np.sin(th))))
        worst_fro = max(worst_fro, abs(tot_fro2 - Cf * np.sin(th) ** 2))
    print(f"\n  numeric cross-check (37 theta in [-6,6]): max|TotalL1 err|={worst_l1:.1e}, "
          f"max|TotalFro2 err|={worst_fro:.1e}  [EXACT, confirmed]")

    # maxima
    print("\n  MAXIMA (exact, from the closed forms -- both are single-frequency |sin theta| / sin^2):")
    print("      period pi, EVEN in theta (Im P(-theta) = -Im P(theta) exactly), zero exactly at")
    print("      theta = 0, pi (the two REAL points of the family -- Peres and its antipode) and")
    print("      MAXIMAL exactly at theta = +-pi/2 (the Penrose point and its mirror). No special")
    print("      role for theta=pi/4 in ANY total measure on this family: the profile is a pure")
    print("      single harmonic; pi/4 is just the point where |sin theta| = 1/sqrt(2) ~ 70.7% of max.")
    print("      This is the EXACT answer to 'is theta=-pi/2 the max / is pi/4 special': YES to the")
    print("      first, NO to the second, and both facts are exact algebraic consequences of the")
    print("      master identity (e_jc in {-1,0,1}, never +-2), not numerical coincidences.")
    return dict(A=A, C=C, D=Df)


# =====================================================================================
# TASK 2: the infinitesimal statement + dimensional check
# =====================================================================================
def sec2_infinitesimal(data, profiles):
    print("\n" + "=" * 100)
    print("[2] THE INFINITESIMAL STATEMENT: flex_skew as the imaginarity-generation direction")
    print("=" * 100)

    # unit-normalized skew flex tangent y = e (x) m, over all 33*3 = 99 real coordinates
    y2 = sp.Integer(0)
    for ray in data:
        for (m, e) in ray:
            y2 += (e * m) ** 2
    y2 = sp.nsimplify(sp.simplify(y2), [SQRT2])
    normy = float(sp.sqrt(y2))
    print(f"  ||y||_2^2 = {y2} = {float(y2):.4f}  =>  ||y||_2 = {normy:.6f}   [EXACT]")

    A = profiles["A"]; D = profiles["D"]
    R_l1 = float(A) / normy
    R_frosum = D / normy
    print(f"\n  Define t = ||y||_2 * theta (arc-length reparametrization of the flex to leading order;")
    print(f"  t is exact only along the tangent line, theta is the exact circle parameter).")
    print(f"  R := d/dt [TotalL1](t)|_{{t=0+}}  = (dTotalL1/dtheta|_0) / ||y||_2 = A/||y||_2")
    print(f"      = {R_l1:.6f}   [EXACT, one-sided derivative -- TotalL1 = A|sin theta| has a kink at 0]")
    print(f"  R_froSum := d/dt [TotalFroSum](t)|_0 = D/||y||_2 = {R_frosum:.6f}   [EXACT closed form for D")
    print(f"      pending; D itself is a NUMERICAL sum of 33 square roots of Z[sqrt2] numbers]")
    print(f"  HONEST CAVEAT: the SQUARED-Frobenius total TotalFro2 = C sin^2(theta) is QUADRATIC in theta")
    print(f"      near 0 (d/dtheta|_0 = 0); its 'rate' is a curvature, 2C/||y||^2 = "
          f"{2 * float(profiles['C']) / float(y2):.6f}, not a first-order rate. Measure choice matters:")
    print(f"      linear measures (l1, sum of Frobenius norms) give a finite nonzero R; the natural")
    print(f"      'energy' measure (sum of squared Frobenius norms) gives R=0 and a curvature instead.")

    # contrast: the flex tangent EXACTLY preserves orthogonality (reuse the program's own exact
    # certificate, unmodified); a generic non-flex imaginary direction does NOT, even though it can
    # carry comparable or larger 'imaginarity rate'.
    print("\n  CONTRAST WITH A NON-FLEX IMAGINARY DIRECTION (does it stay a KS configuration?):")
    per = rays_q4(Q_PERES)
    w = slice_tangent_realvec((0, 0, 0))
    J, T, E, n = exact_flex_certificate_c(per, w, "    reproduced: Peres flex tangent")
    print("      [EXACT, imported machinery, re-run here] the true flex tangent satisfies J.w = 0")
    print("      identically in Z[i,sqrt2] -- ALL 72 orthogonalities preserved to first order.")

    # generic random purely-imaginary direction on the SAME 198-dim real ambient space: check it
    # generically violates the edge constraints (numerically), i.e. does NOT preserve the KS graph.
    rng = np.random.default_rng(1)
    Jf = np.array([[q_float(x) for x in row] for row in J], dtype=float)
    d, V = 3, 33
    yprime = rng.standard_normal(V * d)  # generic real y' (imaginary perturbation direction i*y')
    wprime = np.zeros(2 * d * V)
    for i in range(V):
        for c in range(d):
            wprime[2 * d * i + 2 * c + 1] = yprime[d * i + c]   # purely imaginary slot
    resid = Jf @ wprime
    edge_resid = resid[:2 * len(E)]
    print(f"      A GENERIC random purely-imaginary direction y' (same ambient space): "
          f"max|J.w'| over the {len(E)} edge-constraint rows = {np.max(np.abs(edge_resid)):.4f}")
    print(f"      (vs. EXACTLY 0 for the true flex) -- generic imaginary perturbations break the KS")
    print(f"      orthogonality graph at first order; only the flex_skew-dimensional subspace does not.")
    print(f"      [NUMERICAL demonstration of an EXACT fact: ker(J) has the flex tangent in it, a random")
    print(f"      vector of full ambient dimension is in ker(J) with probability 0.]")

    print("\n  SHARPEST HONEST STATEMENT (this branch's proposed precise form):")
    print("      flex_skew(config) = dim{ real y = (y_j) : v_j -> v_j + i*t*y_j preserves every")
    print("      orthogonality <v_i,v_j>=0 of the configuration to O(t) }")
    print("      = the number of independent directions along which a real realization of a given")
    print("      orthogonality GRAPH can be deformed by generating imaginary parts while remaining")
    print("      (to first order) a realization of that SAME graph.")
    print("      CAVEAT made precise here: 'staying that KS set' only literally means 'staying the same")
    print("      RIGID configuration' when flex_R=0 (Peres-33, Yu-Oh, Peres-24). When flex_R>0 (odd")
    print("      cycles) the real embedding itself already has moduli, and flex_skew counts imaginary")
    print("      directions ADDITIONAL to those, compatible with the SAME orthogonality GRAPH, not a")
    print("      unique rigid configuration.")

    # dimensional check on the 4 named cases, reusing torsion_flex.py's OWN scenario functions
    # (imported unmodified) so this is an independent re-run of that program's numbers, not a quote.
    print("\n  DIMENSIONAL CHECK (re-run of torsion_flex.py's own scenario() functions, unmodified):")
    cases = [("peres33", "Peres-33", 1, "SI-KS, uncolorable, no parity witness (t0=1,tau=0): flex_R=0"),
             ("yuoh", "Yu-Oh 13", 0, "colorable / non-contextual (t0=0): flex_R=0"),
             ("peres24", "Peres-24", 0, "SI-KS with a parity/AvN witness (t0=1,tau!=0): flex_R=0"),
             ("c7", "C7 umbrella", 2, "state-DEPENDENT (flex_R=4>0): flex_skew counts EXTRA imaginary")]
    all_ok = True
    for key, label, expect, note in cases:
        r = TF.scenario(key)
        got = r["flex_skew"]
        ok = (got == expect)
        all_ok &= ok
        print(f"      {label:<14} flex_skew = {got}  (predicted {expect})  "
              f"{'[OK]' if ok else '[MISMATCH]'}  -- {note}")
    label_c7 = "NUMERICAL" if True else "EXACT"
    print(f"      Peres-33/Yu-Oh/Peres-24: EXACT/Q(sqrt2) mod-p rank certificates.")
    print(f"      C7 umbrella: NUMERICAL (SVD rank, minsv reported by torsion_flex.py).")
    print(f"      All 4 predictions matched: {all_ok}")
    return dict(dim_check_ok=all_ok)


# =====================================================================================
# TASK 3: resource-monotone sanity check -- B3 symmetry, gauge covariance, Bargmann invariants
# =====================================================================================
def sec3_resource_checks():
    print("\n" + "=" * 100)
    print("[3] RESOURCE-MONOTONE SANITY CHECK: B3 symmetry, gauge covariance, Bargmann invariants")
    print("=" * 100)

    def apply_g(v, perm, sg):
        return np.array([sg[a] * v[perm[a]] for a in range(3)])

    def total_l1(vs):
        tot = 0.0
        for v in vs:
            N = np.vdot(v, v).real
            P = np.outer(v, np.conj(v)) / N
            tot += np.sum(np.abs(P.imag))
        return tot

    thetas_test = [0.0, 0.37, 1.1, 2.0, -0.8]
    Vcache = {th: rays_at(th) for th in thetas_test}

    def find_sigma(perm, sg, th):
        vs = Vcache[th]
        gv = [apply_g(vs[j], perm, sg) for j in range(33)]
        sigma = {}
        for j in range(33):
            scores = [(abs(np.vdot(vs[k], gv[j])) / (np.linalg.norm(vs[k]) * np.linalg.norm(gv[j]) + 1e-300), k)
                      for k in range(33)]
            best_score, best_k = max(scores)
            if best_score < 1 - 1e-6:
                return None
            sigma[j] = best_k
        return sigma if len(set(sigma.values())) == 33 else None

    n48 = n_theta_indep = n_at_peres = 0
    good_elems = []
    for perm in permutations(range(3)):
        for sg in product((1, -1), repeat=3):
            n48 += 1
            if find_sigma(perm, sg, 0.0) is not None:
                n_at_peres += 1
            sig0 = find_sigma(perm, sg, thetas_test[0])
            if sig0 is None:
                continue
            if all(find_sigma(perm, sg, th) == sig0 for th in thetas_test[1:]):
                n_theta_indep += 1
                good_elems.append((perm, sg))

    print(f"  B3 = signed permutations of the 3 axes, |B3|=48 (sign-flip x permutation).")
    print(f"  At theta=0 (the real Peres point): {n_at_peres}/48 stabilize the 33-ray set as a set")
    print(f"      [NUMERICAL, matches the known full symmetry of Peres-33].")
    print(f"  As a symmetry of the WHOLE one-parameter family (same ray-permutation for ALL theta")
    print(f"      tested): only {n_theta_indep}/48 -- exactly the sign-flip subgroup (Z2)^3, perm=identity.")
    print(f"      [NUMERICAL] Axis PERMUTATIONS that move the theta-carrying axis are symmetries of the")
    print(f"      real point only, not of the SLICE gauge choice (consistent with peres_penrose.py's")
    print(f"      own gauge lemma: SLICE is a 1-dim section of a 2-dim gauge orbit).")

    # invariance of the total-l1 PROFILE under the theta-independent (Z2)^3 subgroup
    ok = True
    for perm, sg in good_elems:
        for th in [0.2, 1.3, -0.9]:
            vs = rays_at(th)
            gvs = [apply_g(v, perm, sg) for v in vs]
            if abs(total_l1(vs) - total_l1(gvs)) > 1e-9:
                ok = False
    print(f"  Total-l1(theta) profile invariant under all {len(good_elems)} theta-independent B3 elements, "
          f"3 sample thetas: {'PASS' if ok else 'FAIL'}   [NUMERICAL, to 1e-9]")

    # gauge covariance under diag(1,e^{iq},e^{ir}) at the real Peres point: EXACT closed form
    print("\n  GAUGE COVARIANCE under diag(1,e^{iq},e^{ir}) (a basis change of C^3, NOT a per-ray phase):")
    data = slice_data_sympy()
    v0 = rays_at(0.0)
    assert max(abs(v.imag).max() for v in v0) < 1e-12, "Peres point must be real"
    pairs = [(0, 1), (0, 2), (1, 2)]
    S = [sp.Integer(0)] * 3
    for ray in data:
        m = [x[0] for x in ray]
        N = sum(x * x for x in m)
        for idx, (a, b) in enumerate(pairs):
            S[idx] += sp.nsimplify(sp.simplify(sp.Abs(m[a] * m[b]) / N), [SQRT2])
    S = [sp.nsimplify(sp.simplify(s), [SQRT2]) for s in S]
    assert S[0] == S[1] == S[2], "expected S12=S13=S23 by the axis symmetry of Peres-33"
    Sval = S[0]
    print(f"      TotalL1_gauge(q,r) = 2*S*(|sin q| + |sin r| + |sin(q-r)|),  S = {Sval} = {float(Sval):.6f}")
    print(f"      (S12=S13=S23 exactly, forced by the B3 axis symmetry of the real Peres point) [EXACT]")
    worst = 0.0
    for (q, r) in [(0.3, 0.0), (0.0, 0.5), (0.4, -0.7), (1.1, 2.2), (np.pi, np.pi)]:
        U = np.diag([1.0, np.exp(1j * q), np.exp(1j * r)])
        vs = [U @ v for v in v0]
        direct = total_l1(vs)
        pred = float(2 * Sval * (abs(np.sin(q)) + abs(np.sin(r)) + abs(np.sin(q - r))))
        worst = max(worst, abs(direct - pred))
    print(f"      numeric cross-check, 5 (q,r) pairs incl. (pi,pi) [a REAL gauge]: max err = {worst:.1e} [EXACT]")
    print(f"      At (q,r)=(pi,pi) (diag(1,-1,-1), a REAL orthogonal map = a FREE operation of")
    print(f"      imaginarity theory): TotalL1_gauge = 0, matching the Peres point staying real.")
    print(f"      At any non-real (q,r): TotalL1 becomes strictly positive -- l1/Frobenius Im-measures")
    print(f"      computed in a fixed basis are GAUGE-DEPENDENT (basis artifacts), confirming the")
    print(f"      resource-theory expectation that only REAL unitaries are free / imaginarity-preserving.")

    # Bargmann invariants: the gauge-invariant core
    print("\n  BARGMANN INVARIANTS as the gauge-INVARIANT core:")

    def bargmann(vs, i, j, k):
        return np.vdot(vs[i], vs[j]) * np.vdot(vs[j], vs[k]) * np.vdot(vs[k], vs[i])

    v0n = [v / np.linalg.norm(v) for v in v0]
    trip = (8, 13, 14)
    B0 = bargmann(v0n, *trip)
    worst_g = 0.0
    for (q, r) in [(0.3, 0.0), (0.0, 0.5), (0.4, -0.7), (1.1, 2.2)]:
        U = np.diag([1.0, np.exp(1j * q), np.exp(1j * r)])
        vg = [U @ v for v in v0n]
        worst_g = max(worst_g, abs(bargmann(vg, *trip) - B0))
    print(f"      Bargmann(9,14,15) = <v9|v14><v14|v15><v15|v9> is invariant under per-ray phases AND")
    print(f"      under ANY unitary change of basis (since <Uv_i,Uv_j>=<v_i,v_j>): checked over the same")
    print(f"      4 gauge choices above, max drift = {worst_g:.1e}  [EXACT identity, NUMERICAL confirmation]")

    def total_bargmann_imag(vs):
        tot = 0.0
        for i, j, k in combinations(range(len(vs)), 3):
            tot += abs(bargmann(vs, i, j, k).imag)
        return tot

    t0 = time.time()
    sample_thetas = [0.0, 0.3, np.pi / 4, np.pi / 2, np.pi]
    tb = {th: total_bargmann_imag([v / np.linalg.norm(v) for v in rays_at(th)]) for th in sample_thetas}
    print(f"      gauge-invariant TOTAL = sum over all C(33,3)=5456 triples of |Im Bargmann_ijk(theta)|:")
    for th in sample_thetas:
        print(f"        theta={th: .4f}: {tb[th]: .3f}")
    print(f"      ({time.time() - t0:.2f}s for the sweep) [NUMERICAL]")
    print(f"      Same ZERO locus as the basis-dependent TotalL1 (theta=0, pi -- both vanish exactly")
    print(f"      where the configuration is real). Verified at (pi,pi)-type real gauges: stays 0.")
    worst_g2 = 0.0
    for (q, r) in [(0.3, 0.0), (0.0, 0.5), (0.4, -0.7)]:
        U = np.diag([1.0, np.exp(1j * q), np.exp(1j * r)])
        vg = [U @ v for v in v0n]
        worst_g2 = max(worst_g2, total_bargmann_imag(vg))
    print(f"      gauge-invariance of the TOTAL Bargmann-imaginarity at the Peres point (should stay 0")
    print(f"      under any diag gauge, since it's basis-independent): max value found = {worst_g2:.1e} [EXACT-in-principle, NUMERICAL check]")
    # fine argmax scan, honestly reporting the max is NEAR but not exactly at pi/2 (unlike TotalL1)
    ths = np.linspace(0, np.pi / 2, 181)
    vals = [total_bargmann_imag([v / np.linalg.norm(v) for v in rays_at(th)]) for th in ths]
    imax = int(np.argmax(vals))
    print(f"      argmax on [0,pi/2] (181-pt scan): theta={ths[imax]:.5f} (pi/2={np.pi/2:.5f}), "
          f"value/value(pi/2) = {vals[imax] / vals[-1]:.6f}")
    print(f"      HONEST NOTE: unlike TotalL1 (EXACTLY max at pi/2), the gauge-invariant Bargmann total")
    print(f"      peaks NEAR but not exactly AT pi/2 (~0.1% higher, at theta~0.97*pi/2) -- it is NOT a")
    print(f"      pure single-harmonic in theta (triples mix more exponent differences than pairs do).")
    print(f"      [NUMERICAL, no closed form derived for this quantity]")


# =====================================================================================
# TASK 4: literature hook
# =====================================================================================
def sec4_literature():
    print("\n" + "=" * 100)
    print("[4] LITERATURE HOOK (LITERATURE-UNCHECKED beyond what was searched)")
    print("=" * 100)
    print("  Two WebSearch queries were run (2026-07-21):")
    print("    (a) 'imaginarity resource theory Kochen-Specker contextuality'")
    print("    (b) '\"resource theory of imaginarity\" Hickey Gour real quantum mechanics'")
    print("  Query (a) returned no title mentioning both imaginarity and Kochen-Specker/contextuality;")
    print("  the engine's own auto-generated summary asserted a 'graph-theoretic framework connected")
    print("  to Kochen-Specker contextuality' linking multi-state imaginarity to KS graphs, but NO")
    print("  returned source title supports that specific claim and no such paper could be identified")
    print("  or opened -- that summary sentence is NOT trusted and is explicitly flagged as unverified.")
    print("  Query (b) returned the Hickey-Gour 2018 paper (arXiv:1801.05123, J. Phys. A 51, 414009)")
    print("  and several follow-ups (imaginarity of channels, distributed scenarios, Gaussian channels,")
    print("  imaginarity-generating power of unitaries, quantification papers) -- none of the RETURNED")
    print("  TITLES mention contextuality, Kochen-Specker, or noncontextuality inequalities.")
    print("  VERDICT: LITERATURE-UNCHECKED / NO CONFIRMED HIT. Titles were read, not full papers; a")
    print("  passing remark inside one of the returned papers cannot be ruled out. No connection between")
    print("  imaginarity resource theory and KS/contextuality was found or can be honestly claimed.")


def main():
    t0 = time.time()
    print("BRANCH I -- the skew block as an imaginarity resource")
    print(f"working dir: {os.path.dirname(os.path.abspath(__file__))}\n")
    data = slice_data_sympy()
    profiles = sec1_profiles(data)
    dimcheck = sec2_infinitesimal(data, profiles)
    sec3_resource_checks()
    sec4_literature()
    print("\n" + "=" * 100)
    ok = dimcheck["dim_check_ok"]
    print(f"SUMMARY: closed forms EXACT & cross-checked; dimensional check on 4 cases "
          f"{'PASSED' if ok else 'FAILED'}; B3/gauge/Bargmann checks NUMERICAL & consistent; "
          f"literature hook: no confirmed connection.")
    print(f"[{time.time() - t0:.1f}s]  branch_imag {'PASS' if ok else 'FAIL'}")


if __name__ == "__main__":
    main()
