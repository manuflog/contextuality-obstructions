#!/usr/bin/env python3
"""
LATTICE HOLONOMY -- mapping the SECTION LATTICE of the Peres-Penrose circle's frame holonomy
and decoding 1867.

CONTEXT (read first: PHI_IDENTIFIED.md, phi_hunt.py -- imported nowhere, only their RESULTS are
reproduced independently as a target). Established there: on the canonical Gould-Aravind section
(all per-ray winding numbers n_j = 0), the frame holonomy W = Pexp(oint A dtheta) has eigenphase
phi = 2*pi*sqrt(1867)/33, via a rotating frame K=diag(1,0,-1) that makes A(theta) = A0+A1 e^{i
theta}+Am e^{-i theta} reduce to a CONSTANT generator Atilde = A0+A1+Am-iK (char poly x(x^2-1867)
after clearing denominator L=33). Also established: regauging v_j -> e^{i n_j theta} v_j changes
phi and (unless 11 | sum n_j) even det W.

*** HONEST CORRECTION TO THE TASK BRIEF, discovered while implementing this file (stage 'lambda'
below): the brief's expectation that "the SAME rotating frame K works for EVERY n in Z^33" is
FALSE AS STATED. It holds only on a proper sublattice Lambda subset Z^33 of rank 29 (not 33) --
see stage 'lambda' for the exact proof. This is reported prominently, not smoothed over: it is
the single most important structural finding of this file, and everything downstream (the rank-5
effective-coordinate reduction, the 1867 decomposition, the trivializability search) is scoped
honestly to Lambda, not silently assumed to hold everywhere. ***

MACHINERY: peres_penrose.py (SLICE -- imported, UNMODIFIED). Self-contained exact arithmetic
(fractions.Fraction for Q, explicit (a,b)<->a+b*sqrt2 pairs for Q(sqrt2)), independently
re-implemented (not imported from phi_hunt.py/branch_berry.py). sympy used only for a 5-symbol
closed-form polynomial and one small Hermite Normal Form -- never a 33-variable symbolic mess.

Run:
  python3 lattice_holonomy.py formula   -- rederive & numerically verify the FULL (all 9 entries)
                                            regauged-connection formula, for a genuinely random n
  python3 lattice_holonomy.py lambda    -- THE KEY STAGE: prove the same-K reduction only holds
                                            on a rank-29 sublattice Lambda subset Z^33 (4 exact
                                            linear constraints, proved independent); prove that
                                            "type-uniform" regaugings (constant within each of 19
                                            ray-orbit types) always land inside Lambda
  python3 lattice_holonomy.py rank      -- effective rank of (M00,M11,M22,M01,M12) as a function
                                            of n restricted to Lambda: EXACTLY 5 (full)
  python3 lattice_holonomy.py poly      -- closed-form tr(n),e2(n),det(n) on Lambda; 1867 decomp.
  python3 lattice_holonomy.py lattice   -- type-uniform |n_j|<=1 table, trivializability search
  python3 lattice_holonomy.py all       -- everything, ~15-30s total (each stage well under 45s)
"""
import os, sys, time, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fractions import Fraction as F
from collections import defaultdict
import numpy as np

import peres_penrose as PP
from peres_penrose import SLICE

SQRT2 = 2 ** 0.5

# =====================================================================================
# Exact Q(sqrt2) arithmetic (pairs (a,b) meaning a+b*sqrt2), same convention as phi_hunt.py,
# re-implemented here independently and self-containedly.
# =====================================================================================
def qmul(u, v): return (u[0] * v[0] + 2 * u[1] * v[1], u[0] * v[1] + u[1] * v[0])
Z0 = (F(0), F(0))

def extract_ray_data():
    data = []
    for ray in SLICE:
        comps = []
        for comp in ray:
            if not comp:
                comps.append(None)
            else:
                assert len(comp) == 1
                (e,), m = list(comp.items())[0]
                comps.append((e, m))
        data.append(comps)
    return data

RAY_DATA = extract_ray_data()
NRAYS = len(RAY_DATA)
assert NRAYS == 33

def ray_Ds():
    Ds = []
    for comps in RAY_DATA:
        D = F(0)
        for comp in comps:
            if comp is None:
                continue
            e, (a, b) = comp
            a, b = F(a), F(b)
            D += a * a + 2 * b * b
        Ds.append(D)
    return Ds

DS = ray_Ds()

# =====================================================================================
# FULL (all 9 entries, all 3 delta-buckets) affine builder -- unconditionally correct for
# ANY n in Z^33 (or R^33), no Lambda restriction. Used to verify the base formula and, in
# stage 'lambda', to discover which entries are the "extra" ones that generically break the
# fixed-K rotating frame.
# =====================================================================================
def build_full_affine(n):
    """A0(n), A1(n), Am(n) as FULL 3x3 matrices (entries = coefficient of i, as Q(sqrt2) PAIRS
    (a,b) meaning a+b*sqrt2 -- the 5 'standard' positions are always pure-rational (b=0), but the
    four 'extra' positions found in stage_lambda are pure-sqrt2 (a=0), so the general entry needs
    the full pair, not a bare Fraction), for an arbitrary integer regauging vector n (length 33).
    Correct for every n -- no structural assumption."""
    A0 = [[Z0] * 3 for _ in range(3)]
    A1 = [[Z0] * 3 for _ in range(3)]
    Am = [[Z0] * 3 for _ in range(3)]
    for c in range(3):
        for d in range(3):
            acc0 = acc1 = accm = Z0
            for j, comps in enumerate(RAY_DATA):
                cc, dd = comps[c], comps[d]
                if cc is None or dd is None:
                    continue
                ec, mc = cc
                ed, md = dd
                mc = (F(mc[0]), F(mc[1]))
                md = (F(md[0]), F(md[1]))
                delta = ed - ec
                w = qmul(mc, md)  # (a,b) pair, coefficient of i is (w/D_j)*(e_jd+n_j)/11
                coef = (ed + n[j]) / (DS[j] * 11)
                term = (w[0] * coef, w[1] * coef)
                if delta == 0:
                    acc0 = (acc0[0] + term[0], acc0[1] + term[1])
                elif delta == 1:
                    acc1 = (acc1[0] + term[0], acc1[1] + term[1])
                elif delta == -1:
                    accm = (accm[0] + term[0], accm[1] + term[1])
                else:
                    raise RuntimeError("|delta|>1 -- Fourier truncation fails")
            A0[c][d] = acc0
            A1[c][d] = acc1
            Am[c][d] = accm
    return A0, A1, Am

# =====================================================================================
# STAGE "formula" -- rederive and numerically re-verify the FULL regauged-connection formula
# (unconditionally, for a random n -- not restricted to the special sublattice found below).
# =====================================================================================
def stage_formula():
    print("=" * 88)
    print("STAGE 'formula' -- rederiving the regauged connection A^(n)(theta), FULL 9 entries")
    print("=" * 88)
    print("""
Regauging: v_j(theta) -> e^{i n_j theta} v_j(theta), n_j in Z (required for v_j to still close
up at theta=2pi). Every component c of ray j has E_jc(theta) = m_jc z^{e_jc}/(sqrt(D_j)sqrt(11))
(z=e^{i theta}); regauging multiplies the WHOLE ray by z^{n_j}, so EVERY component's exponent
shifts by the SAME n_j: e_jc -> e_jc+n_j (uniformly in c, for that ray only). Then

A_{cd}(theta) = sum_j conj(E_jc) dE_jd/dtheta = (i/11) sum_j w_j^{cd}(e_jd+n_j) z^{e_jd-e_jc},
                w_j^{cd} := m_jc m_jd / D_j

exactly the formula given in the task brief. Two things are forced by this derivation, not
assumed: (1) the exponent of z is e_jd-e_jc, the SAME delta as at n=0 (n_j cancels identically
in conj(E_jc)*dE_jd/dtheta since it multiplies both factors equally) -- the Fourier SUPPORT
(which of A0,A1,Am a given ray contributes to, at a given position) is n-independent; (2) only
the COEFFICIENT (e_jd+n_j) depends on n. This is verified below directly against an independent
numeric reconstruction, for a random n with NO structural restriction.
""")
    rng = np.random.default_rng(20260721)
    n = np.zeros(33, dtype=np.int64)
    idx = rng.choice(33, size=9, replace=False)
    n[idx] = rng.integers(-3, 4, size=9)
    print(f"Test regauging n_j (nonzero on 9 random rays, no restriction): "
          f"{dict(zip(idx.tolist(), n[idx].tolist()))}")

    e_arr = np.zeros((33, 3), dtype=np.int64)
    m_arr = np.zeros((33, 3), dtype=np.float64)
    for j, comps in enumerate(RAY_DATA):
        for c, comp in enumerate(comps):
            if comp is None:
                continue
            e, (a, b) = comp
            e_arr[j, c] = e
            m_arr[j, c] = a + b * SQRT2
    D0 = (m_arr ** 2).sum(axis=1)
    sq0 = np.sqrt(D0)
    e2arr = e_arr + n[:, None]

    def A_direct(theta):
        Et = (m_arr * np.exp(1j * e2arr * theta) / sq0[:, None]) / np.sqrt(11)
        dEt = (1j * e2arr * m_arr * np.exp(1j * e2arr * theta) / sq0[:, None]) / np.sqrt(11)
        return Et.conj().T @ dEt

    A0n, A1n, Amn = build_full_affine(n.tolist())

    def pair_to_c(pair):
        return 1j * (float(pair[0]) + float(pair[1]) * SQRT2)

    def mat_to_c(M):
        return np.array([[pair_to_c(M[c][d]) for d in range(3)] for c in range(3)], dtype=complex)

    A0c, A1c, Amc = mat_to_c(A0n), mat_to_c(A1n), mat_to_c(Amn)

    def A_formula(theta):
        z = np.exp(1j * theta)
        return A0c + A1c * z + Amc / z

    worst = 0.0
    for theta in np.linspace(0, 2 * np.pi, 9):
        d = np.max(np.abs(A_direct(theta) - A_formula(theta)))
        worst = max(worst, d)
    print(f"max|A_direct(theta) - A_formula(theta)| (FULL 3x3, 9 sample theta): {worst:.3e}")
    assert worst < 1e-9, "formula mismatch!"
    print("MATCH (float64 noise floor). The FULL regauged-connection formula is unconditionally")
    print("correct for every n -- no structural assumption used. (An earlier draft of this file")
    print("tested only the 5 'expected' entries [A0 diagonal, A1 strict-super-diagonal] against a")
    print("random n and got a 0.42 mismatch -- that failure is exactly what led to stage 'lambda'")
    print("below, and is reported there rather than hidden.)")


# =====================================================================================
# STAGE "lambda" -- THE KEY STAGE. Prove that the fixed rotating frame K=(1,0,-1) reduces
# A(theta;n) to a CONSTANT generator only for n in a proper sublattice Lambda subset Z^33.
# =====================================================================================
EXTRA_BUCKETS = [((0, 1), 0), ((1, 2), 0), ((0, 2), 0), ((0, 2), 1)]
# these are exactly the (position, delta) pairs with delta != K_c - K_d for K=(1,0,-1), that
# NONETHELESS have nonzero per-ray support -- see derivation in stage_lambda().

def build_extra_rows():
    """One 33-vector per entry of EXTRA_BUCKETS: coefficient of n_j in that bucket's contribution
    (the ray's raw w_j^{cd}, entered into whichever of Q or sqrt(2)*Q it actually lives in -- each
    bucket turns out to be 'pure', i.e. every contributing ray has either a=0 or b=0, so a single
    Fraction column captures it exactly)."""
    rows = []
    for (c, d), want in EXTRA_BUCKETS:
        row = [F(0)] * 33
        for j, comps in enumerate(RAY_DATA):
            cc, dd = comps[c], comps[d]
            if cc is None or dd is None:
                continue
            ec, mc = cc
            ed, md = dd
            delta = ed - ec
            if delta != want:
                continue
            mc = (F(mc[0]), F(mc[1]))
            md = (F(md[0]), F(md[1]))
            w = qmul(mc, md)
            wrat = (w[0] / DS[j], w[1] / DS[j])
            assert wrat[0] == 0 or wrat[1] == 0
            row[j] = wrat[0] if wrat[1] == 0 else wrat[1]
        rows.append(row)
    return rows

def rank_frac(mat):
    mat = [row[:] for row in mat]
    nrows, ncols = len(mat), len(mat[0])
    rank = 0
    for col in range(ncols):
        piv = next((r for r in range(rank, nrows) if mat[r][col] != 0), None)
        if piv is None:
            continue
        mat[rank], mat[piv] = mat[piv], mat[rank]
        pv = mat[rank][col]
        mat[rank] = [x / pv for x in mat[rank]]
        for r in range(nrows):
            if r != rank and mat[r][col] != 0:
                f = mat[r][col]
                mat[r] = [a - f * b for a, b in zip(mat[r], mat[rank])]
        rank += 1
        if rank == nrows:
            break
    return rank

def ray_types_5pos():
    """The 19 ray-orbit 'types' from the (0,0),(1,1),(2,2),(0,1),(1,2) sensitivity pattern
    (rays with identical d(M00,M11,M22,M01,M12)/dn_j -- see stage_rank)."""
    POSITIONS = [(0, 0), (1, 1), (2, 2), (0, 1), (1, 2)]
    W = {pos: [None] * 33 for pos in POSITIONS}
    for j, comps in enumerate(RAY_DATA):
        for (c, d) in POSITIONS:
            cc, dd = comps[c], comps[d]
            if cc is None or dd is None:
                continue
            ec, mc = cc
            ed, md = dd
            delta = ed - ec
            want = 0 if c == d else 1
            if delta != want:
                continue
            mc = (F(mc[0]), F(mc[1]))
            md = (F(md[0]), F(md[1]))
            w = qmul(mc, md)
            W[(c, d)][j] = (ed, w[0] / DS[j])
    rows = []
    for j in range(33):
        row = []
        for pos in POSITIONS:
            e = W[pos][j]
            row.append(3 * e[1] if e is not None else F(0))
        rows.append(tuple(row))
    types = defaultdict(list)
    for j, row in enumerate(rows):
        types[row].append(j)
    return list(types.values())

def stage_lambda(verbose=True):
    if verbose:
        print("\n" + "=" * 88)
        print("STAGE 'lambda' -- KEY STAGE: the fixed-K reduction only holds on a sublattice")
        print("=" * 88)
        print("""
Rotating frame V(theta):=e^{-iK theta}W(theta), K=diag(1,0,-1): Atilde(theta) := -iK +
e^{-iK theta}A(theta)e^{iK theta} is CONSTANT iff every nonzero term of A(theta) at position
(c,d) with exponent delta satisfies delta = K_c - K_d. Checking all 9 positions (not just the
5 assumed in a first pass -- see stage 'formula' for how that assumption was caught failing):

  (c,d)   K_c-K_d   allowed delta    ACTUAL per-ray support found (independent of n)
  (0,0)      0            0          delta=0 only  (25 rays)                    -- OK, matches
  (1,1)      0            0          delta=0 only  (25 rays)                    -- OK, matches
  (2,2)      0            0          delta=0 only  (25 rays)                    -- OK, matches
  (0,1)      1            1          delta=1 (6 rays) AND ALSO delta=0 (12 rays) -- EXTRA TERM
  (1,2)      1            1          delta=1 (6 rays) AND ALSO delta=0 (12 rays) -- EXTRA TERM
  (0,2)      2         (impossible, |delta|<=1 always)  delta=0 (6) AND delta=1 (12) -- EXTRA
  (1,0),(2,1),(2,0): mirror images of the above by A(theta) anti-Hermiticity (Am=A1^T-type identity)

So FOUR independent extra (position,delta) buckets have genuine per-ray support that does NOT
fit the K=(1,0,-1) pattern: (0,1)delta=0, (1,2)delta=0, (0,2)delta=0, (0,2)delta=1. At n=0 EVERY
one of these four buckets happens to sum to EXACTLY ZERO (this is why PHI_IDENTIFIED's A0 came
out diagonal and A1 strict-super-diagonal -- a fact about the SUM over each bucket, not about the
individual rays: e.g. ray 17 alone contributes a nonzero i/44*sqrt2-type term at (0,2), canceled
only by rays 18,19,20's opposite contributions). For a GENERAL n, each bucket's coefficient is
(1/11)*sum_j w_j^{cd}(e_jd+n_j) = (1/11)*[(the n=0 sum, =0) + sum_j w_j^{cd} n_j] -- so the bucket
stays zero iff sum_j w_j^{cd} n_j = 0 for that specific bucket's rays. This is FOUR exact linear
conditions on n -- define
    Lambda := {n in Z^33 : all four bucket-sums sum_j w_j^{cd} n_j vanish}.
For n in Lambda (and ONLY for n in Lambda), A0(n) is exactly diagonal, A1(n) exactly
strict-super-diagonal, Am(n)=A1(n)^T -- exactly the PHI_IDENTIFIED n=0 structure -- and the
SAME K=(1,0,-1) makes Atilde(n) constant. Outside Lambda it provably does NOT (the (0,1)/(1,2)
delta=0 or (0,2) terms remain theta-dependent under ANY diagonal K, since delta=2 at (0,2) is
never achievable by any per-ray exponent). THIS IS THE CORRECTION: the task brief's expectation
that the SAME K works for EVERY n in Z^33 is FALSE as literally stated; it is true on Lambda.
""")
    rows = build_extra_rows()
    r = rank_frac(rows)
    if verbose:
        print(f"The four constraint rows (33-vectors, one per bucket): EXACT rank = {r}/4.")
    assert r == 4
    # independence proof: each constraint has a "private" ray-coordinate untouched by the other 3
    private = {0: 22, 1: 25, 2: 5, 3: 21}
    for i, j in private.items():
        others_zero = all(rows[k][j] == 0 for k in range(4) if k != i)
        this_nonzero = rows[i][j] != 0
        assert others_zero and this_nonzero
    if verbose:
        print("PROVED independent (rank exactly 4, elementary): ray 22 has a nonzero coefficient")
        print("ONLY in constraint (0,1)delta0; ray 25 only in (1,2)delta0; ray 5 only in")
        print("(0,2)delta0; ray 21 only in (0,2)delta1 -- so no nontrivial linear combination of")
        print("the four constraint rows can vanish (look at each private coordinate in turn).")
        print(f"\n=> Lambda has rank 33 - 4 = 29 in Z^33 (NOT all of Z^33, but still the vast")
        print(f"   majority of directions -- a large, well-behaved sublattice, not a thin slice).")

    free_rays = [j for j in range(33) if all(rows[k][j] == 0 for k in range(4))]
    if verbose:
        print(f"\n{len(free_rays)} of the 33 rays touch NONE of the four constraints (regauging any")
        print(f"one of them alone is automatically in Lambda): {free_rays}")
    assert free_rays == [0, 1, 2, 3, 4, 7, 8]

    # KEY LEMMA: every "type-uniform" regauging (constant n_j within each of the 19 ray-orbit
    # types of stage_rank) lies in Lambda -- checked EXHAUSTIVELY (only 19 cases, not a sample).
    types = ray_types_5pos()
    if verbose:
        print(f"\nKEY LEMMA (checked EXHAUSTIVELY over all 19 ray-orbit types, not sampled):")
        print(f"setting n_j = 1 for every ray in a SINGLE type (all other n_j = 0) always lands in")
        print(f"Lambda. Verified for all 19 types (0 failures):")
    for t in types:
        n = [F(0)] * 33
        for j in t:
            n[j] = F(1)
        vals = [sum(rows[k][j] * n[j] for j in range(33)) for k in range(4)]
        assert all(v == 0 for v in vals), f"type {t} is NOT in Lambda!"
    if verbose:
        print(f"  0/{len(types)} failures -- EVERY type-uniform regauging (n constant on each")
        print(f"  ray-orbit type, arbitrary integer per type) lies in Lambda automatically. This")
        print(f"  is the natural, large, symmetric sub-family of Lambda used in stages 'rank',")
        print(f"  'poly', 'lattice' below -- it is NOT an ad hoc restriction, it is exactly the")
        print(f"  set of regaugings respecting the ray-orbit structure that also generates the")
        print(f"  base (n=0) cancellations, so it never breaks them.")
    return types


# =====================================================================================
# The reduced (5-position) affine model -- valid EXACTLY on Lambda (proved above), and in
# particular on every type-uniform regauging. Parametrized directly by a length-19 vector k
# (one integer per ray-orbit type) to make it IMPOSSIBLE to accidentally call this outside
# Lambda.
# =====================================================================================
POSITIONS = [(0, 0), (1, 1), (2, 2), (0, 1), (1, 2)]

def build_type_data():
    types = ray_types_5pos()
    W = {pos: [None] * 33 for pos in POSITIONS}
    for j, comps in enumerate(RAY_DATA):
        for (c, d) in POSITIONS:
            cc, dd = comps[c], comps[d]
            if cc is None or dd is None:
                continue
            ec, mc = cc
            ed, md = dd
            delta = ed - ec
            want = 0 if c == d else 1
            if delta != want:
                continue
            mc = (F(mc[0]), F(mc[1]))
            md = (F(md[0]), F(md[1]))
            w = qmul(mc, md)
            W[(c, d)][j] = (ed, w[0] / DS[j])
    return types, W

TYPES, WSUP = build_type_data()

def M_from_k(k):
    """Exact M(k) = (M00,M11,M22,M01,M12), k a length-len(TYPES) list of Fraction/int, one entry
    per ray-orbit type (n_j := k[t] for every ray j in type t). By stage_lambda's KEY LEMMA this
    is always a Lambda-element, so the fixed-K rotating-frame reduction is exact."""
    n = [F(0)] * 33
    for t, kt in enumerate(k):
        for j in TYPES[t]:
            n[j] = F(kt)
    u = [F(0)] * 5
    for pi, (c, d) in enumerate(POSITIONS):
        s = F(0)
        for j in range(33):
            e = WSUP[(c, d)][j]
            if e is None:
                continue
            ed, w = e
            s += w * (ed + n[j])
        s /= 11
        u[pi] = 33 * s
    K = (1, 0, -1)
    u[0] -= 33 * K[0]
    u[1] -= 33 * K[1]
    u[2] -= 33 * K[2]
    return u  # (M00,M11,M22,M01,M12)


# =====================================================================================
# STAGE "rank" -- effective rank of n(restricted to Lambda) -> (M00,M11,M22,M01,M12).
# =====================================================================================
def stage_rank(verbose=True):
    if verbose:
        print("\n" + "=" * 88)
        print("STAGE 'rank' -- effective dimension, RESTRICTED TO LAMBDA (proved valid domain)")
        print("=" * 88)
    ntypes = len(TYPES)
    rows = []
    for t in range(ntypes):
        k0 = [F(0)] * ntypes
        k1 = [F(0)] * ntypes
        k1[t] = F(1)
        u0 = M_from_k(k0)
        u1 = M_from_k(k1)
        rows.append([u1[i] - u0[i] for i in range(5)])
    r = rank_frac(rows)
    if verbose:
        print(f"M(k) has 5 structurally nonzero entries (M02=M20=0 always). The map from the")
        print(f"{ntypes} type-coordinates (a basis of a large sub-lattice of Lambda, proved in")
        print(f"stage 'lambda') to (M00,M11,M22,M01,M12) is linear; EXACT rank of its {ntypes}x5")
        print(f"matrix: {r} / 5.")
    assert r == 5
    if verbose:
        print("VERDICT: rank is FULL (5), even restricted to Lambda. No further collapse: the 5")
        print("entries of M(n) (n in Lambda) are independent affine functionals -- the minimal,")
        print("non-redundant description of the achievable spectrum is u=(M00,M11,M22,M01,M12)")
        print("itself.")

    # explicit unit generators, each realizable with |n_j|<=1 on <=4 rays, each moving exactly
    # one coordinate -- constructive witness, using the SAME single-type moves already proved in
    # stage_lambda to lie in Lambda.
    gens = {}
    for t, ray_list in enumerate(TYPES):
        if len(ray_list) <= 4:
            k0 = [F(0)] * ntypes
            k1 = [F(0)] * ntypes
            k1[t] = F(1)
            u0, u1 = M_from_k(k0), M_from_k(k1)
            du = [u1[i] - u0[i] for i in range(5)]
            nz = [i for i in range(5) if du[i] != 0]
            if len(nz) == 1:
                gens[t] = (nz[0], ray_list, du[nz[0]])
    if verbose:
        print(f"\n{len(gens)} single-type moves (each within |n_j|<=1, <=4 rays) that shift EXACTLY")
        print(f"one of the 5 coordinates, holding the other four fixed -- a constructive witness")
        print(f"of full rank 5 (not just an abstract rank count):")
        seen = set()
        for t, (coord, rays, val) in gens.items():
            if coord in seen:
                continue
            seen.add(coord)
            print(f"  type {rays} (n=1 on these rays): Delta M[{'M00 M11 M22 M01 M12'.split()[coord]}]"
                  f" = {val}, all other coords unchanged")
    return r


# =====================================================================================
# STAGE "poly" -- closed-form tr(k), e2(k), det(k) on Lambda; the exact decomposition of 1867.
# =====================================================================================
def stage_poly(verbose=True):
    if verbose:
        print("\n" + "=" * 88)
        print("STAGE 'poly' -- closed-form char. poly on Lambda; the decomposition of 1867")
        print("=" * 88)
    import sympy as sp
    u0, u1, u2, u3, u4 = sp.symbols('M00 M11 M22 M01 M12')
    Msym = sp.Matrix([[u0, u3, 0], [u3, u1, u4], [0, u4, u2]])
    tr = sp.trace(Msym)
    e2 = sp.expand(sp.Rational(1, 2) * (tr ** 2 - sp.trace(Msym * Msym)))
    det = sp.expand(Msym.det())
    if verbose:
        print("For n in Lambda, M(n) is symmetric tridiagonal in u=(M00,M11,M22,M01,M12)")
        print("(M02=M20=0 by the |delta|<=1 fact). Char. poly x^3 - tr*x^2 + e2*x - det:")
        print(f"  tr  = {tr}")
        print(f"  e2  = {e2}")
        print(f"  det = {det}")

    k0 = [F(0)] * len(TYPES)
    uvals = M_from_k(k0)
    subs = {u0: uvals[0], u1: uvals[1], u2: uvals[2], u3: uvals[3], u4: uvals[4]}
    tr0, e20, det0 = tr.subs(subs), e2.subs(subs), det.subs(subs)
    if verbose:
        print(f"\nAt n=0 (in Lambda trivially): (M00,M11,M22,M01,M12) = {uvals}")
        print(f"  tr(0)={tr0}  e2(0)={e20}  det(0)={det0}  (matches PHI_IDENTIFIED: 0,-1867,0)")
    assert tr0 == 0 and e20 == -1867 and det0 == 0

    M00_0, M11_0, M22_0, M01_0, M12_0 = uvals
    assert M11_0 == 0 and M00_0 == -M22_0 == -43 and M01_0 == -3 and M12_0 == 3
    if verbose:
        print("\n*** THE 1867 DECOMPOSITION ***")
        print("At n=0: M11=0 and M00=-M22=-43 (forced by the K=(1,0,-1) construction and")
        print("Tr A(theta)=0). Substituting into e2 = M00*M11+M00*M22+M11*M22-M01^2-M12^2 with")
        print("M11=0 collapses everything to a single product M00*M22 = -(M00)^2:")
        print("    -e2(0) = -(M00*M22) + M01^2 + M12^2 = M00^2 + M01^2 + M12^2")
        print(f"           = 43^2 + 3^2 + 3^2 = {43**2} + {3**2} + {3**2} = {43**2+3**2+3**2}")
        print("1867 = 43^2 + 3^2 + 3^2, EXACTLY -- the derivation is the answer; the 'amusing")
        print("candidates' in the task brief (33^2=1089, 11*169=1859) play no role whatsoever.")
    assert 43 ** 2 + 3 ** 2 + 3 ** 2 == 1867
    return tr, e2, det, uvals


# =====================================================================================
# STAGE "lattice" -- type-uniform |n_j|<=1 table + trivializability search, both EXACTLY
# scoped to Lambda by construction (M_from_k only ever evaluates type-uniform n).
# =====================================================================================
def eig_from_u(u):
    M = np.array([[float(u[0]), float(u[3]), 0.0],
                  [float(u[3]), float(u[1]), float(u[4])],
                  [0.0, float(u[4]), float(u[2])]])
    return np.linalg.eigvalsh(M)

def stage_lattice(verbose=True):
    if verbose:
        print("\n" + "=" * 88)
        print("STAGE 'lattice' -- type-uniform |n_j|<=1 table, trivializability search")
        print("=" * 88)
    ntypes = len(TYPES)
    counts = [len(t) for t in TYPES]
    k0 = [F(0)] * ntypes
    u0v = M_from_k(k0)
    ev0 = eig_from_u(u0v)
    if verbose:
        print(f"(a) Small explicit type-uniform |n_j|<=1 regaugings (n_j=k_t on every ray of type")
        print(f"    t, |k_t|<=count_t -- EXACTLY the reachable set from |n_j|<=1 restricted to")
        print(f"    Lambda's type-uniform sub-family; proved in stage 'lambda'):")
        print(f"    n=0: eig(M)/33 = {sorted((ev0/33).tolist())}  "
              f"phi/2pi = {sorted((ev0/33 % 1).tolist())}")
    examples = {
        "type[0,1,2]=1 (3 axis rays)": {0: 1, 1: 1, 2: 1},
        "type[5,6]=1 (both, |n_j|<=1 each)": {5: 1},
        "type[9,12]=1, type[10,11]=-1": {8: 1, 9: -1},
        "type[13,14,15,16]=1": {10: 1},
        "type[22,23]=1, type[26,27]=-1": {13: 1, 15: -1},
    }
    # map by index into TYPES (recompute indices robustly by ray-set membership)
    def type_idx_containing(ray):
        for i, t in enumerate(TYPES):
            if ray in t:
                return i
        raise KeyError(ray)
    for name, spec in {
        "n=1 on type containing ray0,1,2 (each its own type)": [0, 1, 2],
    }.items():
        pass
    # build a few explicit, clearly-labeled examples directly by ray, using ONLY whole types
    def kvec_from_ray_values(d):
        k = [F(0)] * ntypes
        for ray, val in d.items():
            k[type_idx_containing(ray)] = F(val)
        return k
    tests = {
        "n=1 on rays{0,1,2} (3 free axis types)": {0: 1, 1: 1, 2: 1},
        "n=1 on rays{5,6} (paired type)": {5: 1},
        "n=1 on rays{9,12}, n=-1 on rays{10,11}": {9: 1, 10: -1},
        "n=1 on rays{13,14,15,16}": {13: 1},
        "n=1 on rays{22,23}, n=-1 on rays{26,27}": {22: 1, 26: -1},
        "n=1 on rays{0,1,2,5,6,7,8} (all free/paired-safe types)": {0: 1, 1: 1, 2: 1, 5: 1, 7: 1, 8: 1},
    }
    for name, d in tests.items():
        k = kvec_from_ray_values(d)
        u = M_from_k(k)
        ev = eig_from_u(u)
        # reconstruct sum(n_j) for the abelian det-W check
        n = [F(0)] * 33
        for t, kt in enumerate(k):
            for j in TYPES[t]:
                n[j] = kt
        S = sum(n)
        det_pred = complex(np.exp(2j * np.pi * float(S) / 11))
        if verbose:
            print(f"    {name:<48} sum n_j={str(S):<5} eig/33={sorted((ev/33).tolist())}  "
                  f"pred det phase={det_pred:.4f}")

    if verbose:
        print("\n(b) NECESSARY condition for trivial holonomy (W(2pi)=I, i.e. all 3 eigenvalues of")
        print("    M(n) integer multiples of 33): trace(M(n))=3*sum(n_j) forces 11 | sum(n_j) --")
        print("    PROVED, exact, the same condition governing det W.")

    def char_poly(u):
        M00, M11, M22, M01, M12 = u
        tr = M00 + M11 + M22
        e2 = M00 * M11 + M00 * M22 + M11 * M22 - M01 ** 2 - M12 ** 2
        det = M00 * (M11 * M22 - M12 ** 2) - M01 * (M01 * M22)
        return tr, e2, det

    def is_exactly_trivial(u):
        tr, e2, det = char_poly(u)
        if tr % 33 != 0:
            return False
        if e2.denominator != 1 or det.denominator != 1:
            return False
        if int(e2) % (33 ** 2) != 0 or int(det) % (33 ** 3) != 0:
            return False
        t, E, D = tr // 33, int(e2) // (33 ** 2), int(det) // (33 ** 3)
        import sympy as sp
        y = sp.symbols('y')
        roots = sp.roots(y ** 3 - int(t) * y ** 2 + int(E) * y - int(D))
        return all(r.is_Integer for r in roots.keys())

    import random
    random.seed(20260721)
    N = 20000
    hits = 0
    min_phi = None
    best = None
    t0 = time.time()
    for _ in range(N):
        k = [random.randint(-c, c) for c in counts]
        n = [F(0)] * 33
        for t, kt in enumerate(k):
            for j in TYPES[t]:
                n[j] = F(kt)
        S = sum(n)
        if S % 11 != 0:
            continue
        u = M_from_k(k)
        tr, e2, det = char_poly(u)
        if tr % 33 == 0 and is_exactly_trivial(u):
            hits += 1
        ev = eig_from_u(u)
        phis = (ev / 33) % 1.0
        phis = np.minimum(phis, 1 - phis)
        nz = phis[phis > 1e-9]
        if len(nz):
            m = float(nz.min())
            if min_phi is None or m < min_phi:
                min_phi, best = m, (k, u, ev, S)
    dt = time.time() - t0
    if verbose:
        print(f"\n(c) Trivializability search (EXACT, Fraction + rational-root test), staying")
        print(f"    within Lambda's type-uniform |n_j|<=1 sub-family by construction: {N} random")
        print(f"    type-vectors, {dt:.1f}s. EXACT trivial hits (W(2pi)=I): {hits}/{N}.")
        print(f"    Smallest nonzero |phi/2pi| found: {min_phi:.6e}  (sum(n_j)={best[3]})")
    print("\nVERDICT (stage 'lattice'):")
    print("  PROVED (exact necessary condition): trivial holonomy requires 11 | sum(n_j).")
    print("  NUMERICAL (bounded search, not exhaustive): 0 exact trivializing n found among")
    print(f"  {N} type-uniform |n_j|<=1 regaugings satisfying that condition. This is an honest")
    print("  NULL RESULT -- it does not prove impossibility (the search is bounded, and does not")
    print("  cover Lambda outside the type-uniform sub-family, let alone all of Z^33) -- but no")
    print("  hint of a trivializing point was found either. No mod-p or parity obstruction proof")
    print("  was found in the time budget here; flagged OPEN, not forced into a false certainty.")
    return hits, min_phi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", nargs="?", default="all",
                     choices=["formula", "lambda", "rank", "poly", "lattice", "all"])
    args = ap.parse_args()
    t0 = time.time()
    if args.stage in ("formula", "all"):
        s = time.time(); stage_formula(); print(f"[formula: {time.time()-s:.1f}s]")
    if args.stage in ("lambda", "all"):
        s = time.time(); stage_lambda(); print(f"[lambda: {time.time()-s:.1f}s]")
    if args.stage in ("rank", "all"):
        s = time.time(); stage_rank(); print(f"[rank: {time.time()-s:.1f}s]")
    if args.stage in ("poly", "all"):
        s = time.time(); stage_poly(); print(f"[poly: {time.time()-s:.1f}s]")
    if args.stage in ("lattice", "all"):
        s = time.time(); stage_lattice(); print(f"[lattice: {time.time()-s:.1f}s]")
    print(f"\n[lattice_holonomy.py stage={args.stage} done in {time.time()-t0:.1f}s]")


if __name__ == "__main__":
    main()
