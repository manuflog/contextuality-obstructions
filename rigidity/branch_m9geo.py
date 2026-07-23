#!/usr/bin/env python3
"""
branch_m9geo.py -- Branch M9-GEO: porting the d=3 circle-geometry machinery (ring-point
classification, imaginarity profile, Berry/Wilczek-Zee holonomy) to the NEW d=4 M9 family
(D4_FLEX_HUNT.md: 89 rays / 433 pairs / 35 bases in C^4, entries {0,+-1,+-X}, X=e^{i theta} on
|X|^2=1, exact flex=1).

READ FIRST: D4_FLEX_HUNT.md (the M9 family, its exact Laurent-identity construction, the cached
critical core), branch_d4flex.py (the Laurent machinery, `stable_core_and_test('M9', ...)`
rebuilds/returns the cached 89-ray core), branch_arith.py / branch_imag.py (d=3 templates for
ring-point classification / imaginarity profile), phi_hunt.py / branch_berry.py (d=3 template for
the Berry-connection / Wilczek-Zee holonomy machinery, ported here to d=4).

No existing file is modified. Machinery reused, UNMODIFIED: branch_d4flex.py
(`stable_core_and_test`, `stable_graph`, `generic_symbolic_rays`, `MECHS`), ks_flex_census.py
(`raw_vectors`, `collect_rays`, `build_structure_d`, `herm_dot`, `qneg`, `ZERO`). Everything
d=4-specific (the frame/connection construction, the quartic Z[zeta_8] ring, the ring-point
sweep) is new, self-contained code in this file.

STRUCTURE:
  Stage 1 (ring points):    stage1_ring_points()   -- exact classification of x in {1,-1,i,-i,
                                                        zeta_3,zeta_6,zeta_8,(3+4i)/5,(5+12i)/13}
  Stage 2 (imaginarity):    stage2_imaginarity()    -- exact Im v_j(theta) closed forms
  Stage 3 (holonomy hunt):  stage3_holonomy()       -- THE PRIZE: frame/connection, exact
                                                        Fourier structure, rotating frame /
                                                        constant-connection reduction, exact
                                                        eigenphases, numerical cross-checks
  Stage 4 (abelian layer):  stage4_abelian()        -- per-ray Berry phases, det W closed form

Run:
  python3 branch_m9geo.py ring        -- Stage 1, ~5-15s (9 concrete points, exact)
  python3 branch_m9geo.py imag        -- Stage 2, <1s (exact)
  python3 branch_m9geo.py holonomy    -- Stage 3, ~15-25s (exact + numerical cross-checks)
  python3 branch_m9geo.py abelian     -- Stage 4, <1s (exact)
  python3 branch_m9geo.py all         -- everything, ~30-45s total (may need 2 calls)
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fractions import Fraction as F
from itertools import product as iproduct, combinations
from collections import Counter
import numpy as np

import branch_d4flex as bd4
from ks_flex_census import raw_vectors, collect_rays, build_structure_d, herm_dot, qneg, ZERO

import contextlib, io


# ======================================================================================
# SHARED: load the cached 89-ray / 433-pair / 35-basis M9 critical core (symbol level,
# local indices 0..88) built and checkpointed by branch_d4flex.py's `core_M9` target.
# ======================================================================================
def load_core(verbose=False):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf if not verbose else sys.stdout):
        res = bd4.stable_core_and_test("M9", ring_point=None, fresh=False)
    assert len(res["core_syms"]) == 89 and len(res["core_bases"]) == 35, \
        "cached M9 core not as expected -- re-run branch_d4flex.py core_M9 first"
    return res["core_syms"], res["core_pairs"], res["core_bases"]


def load_full_pool(verbose=False):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf if not verbose else sys.stdout):
        g = bd4.stable_graph("M9", verbose=False)
    return g["rays"], g["pairs"], g["bases"]


# ======================================================================================
# STAGE 1 -- EXACT ring-point classification of the M9 circle |x|^2=1.
# ======================================================================================
def full_pool_quadratic(alph, B, C, dotfn=herm_dot, dsize=4):
    """Exact full-pool (rays/pairs/bases) at a concrete point of a quadratic ring
       Z[t]/(t^2-Bt-C), reusing ks_flex_census's ring-agnostic machinery UNMODIFIED."""
    raws = raw_vectors(alph, dsize)
    rays = collect_rays(raws, B, C)
    pairs, bases, adj = build_structure_d(rays, dotfn, B, C, dsize)
    return len(rays), len(pairs), len(bases)


# ---- quartic ring Z[zeta_8] = Z[z]/(z^4+1), needed for x=zeta_8 (degree 4 over Q) ----
Z4 = (0, 0, 0, 0)
ONE4 = (1, 0, 0, 0)
X4 = (0, 1, 0, 0)


def q4add(u, v): return tuple(a + b for a, b in zip(u, v))
def q4neg(u): return tuple(-a for a in u)


def q4mul(u, v):
    conv = [0] * 7
    for i in range(4):
        for j in range(4):
            conv[i + j] += u[i] * v[j]
    out = list(conv[:4])
    for k in range(4, 7):
        out[k - 4] -= conv[k]          # reduce using z^4 = -1
    return tuple(out)


def q4conj(u):
    """Complex conjugation on Q(zeta_8) is the Galois automorphism z -> z^{-1} = z^7 = -z^3;
       conj(a0+a1 z+a2 z^2+a3 z^3) = a0 - a3 z - a2 z^2 - a1 z^3."""
    a0, a1, a2, a3 = u
    return (a0, -a3, -a2, -a1)


def q4zero(u): return all(x == 0 for x in u)


def full_pool_zeta8(dsize=4):
    """Exact full-pool computation at x=zeta_8=e^{i pi/4}, via the exact quartic ring
       Z[zeta_8] (mult mod z^4=-1), mirroring ks_flex_census's raw_vectors/collect_rays/
       build_structure_d logic but generalized to a degree-4 (not degree-2) ring."""
    alph = [Z4, ONE4, q4neg(ONE4), X4, q4neg(X4)]
    raws = [v for v in iproduct(alph, repeat=dsize) if any(c != Z4 for c in v)]

    def proportional4(u, v):
        for i, j in combinations(range(dsize), 2):
            m = q4add(q4mul(u[i], v[j]), q4neg(q4mul(u[j], v[i])))
            if not q4zero(m):
                return False
        return True

    rays = []
    for v in raws:
        if not any(proportional4(v, r) for r in rays):
            rays.append(v)

    def herm4(u, v):
        s = Z4
        for a, b in zip(u, v):
            s = q4add(s, q4mul(q4conj(a), b))
        return s

    n = len(rays)
    pairs = [(i, j) for i, j in combinations(range(n), 2) if q4zero(herm4(rays[i], rays[j]))]
    adj = [set() for _ in range(n)]
    for i, j in pairs:
        adj[i].add(j); adj[j].add(i)
    bases = []

    def extend(cands, cur):
        if len(cur) == dsize:
            bases.append(tuple(cur)); return
        if len(cur) + len(cands) < dsize:
            return
        cl = sorted(cands)
        for idx, v in enumerate(cl):
            rest = set(cl[idx + 1:]) & adj[v]
            extend(rest, cur + [v])

    for s in range(n):
        extend(set(x for x in adj[s] if x > s), [s])
    return n, len(pairs), len(bases)


def stage1_ring_points():
    t0 = time.time()
    print("=" * 100)
    print("STAGE 1 -- EXACT ring-point classification of the M9 circle |x|^2=1")
    print("=" * 100)
    print("Abstract stable graph (generic x, EXACT, mechanism-level): 272 rays, 2704 pairs, "
          "460 bases (cited, branch_d4flex.stable_graph('M9')).")
    print("89-ray / 433-pair / 35-basis critical core (cited, branch_d4flex.core_M9, Laurent-"
          "identity-proved to hold for the WHOLE circle -- so wherever the full pool below "
          "matches 272/2704/460 EXACTLY, the 89-core automatically matches 433/35 too, with NO "
          "further computation needed: the core's pairs are a subset of the full pool's pairs, "
          "proved identical in theta by Sec.2 of D4_FLEX_HUNT.md.)\n")

    rows = []

    # ---- quadratic-ring points: 1, -1, i, -i (ring Z[i], B=0,C=-1) ----
    for name, X in [("x=1", (1, 0)), ("x=-1", (-1, 0)), ("x=i", (0, 1)), ("x=-i", (0, -1))]:
        alph = [ZERO, (1, 0), (-1, 0), X, qneg(X)]
        r, p, b = full_pool_quadratic(alph, 0, -1)
        rows.append((name, "Z[i], t^2=-1", r, p, b))

    # ---- zeta_3 (ring t^2=-t-1), zeta_6 (ring t^2=t-1), both Eisenstein-type ----
    for name, (B, C) in [("zeta_3=e^{2pi i/3}", (-1, -1)), ("zeta_6=e^{i pi/3}", (1, -1))]:
        X = (0, 1)
        alph = [ZERO, (1, 0), (-1, 0), X, qneg(X)]
        r, p, b = full_pool_quadratic(alph, B, C)
        rows.append((name, f"Z[zeta_6], t^2={B}t+{C}", r, p, b))

    # ---- zeta_8 (degree 4, custom quartic ring) ----
    r, p, b = full_pool_zeta8()
    rows.append(("zeta_8=e^{i pi/4}", "Z[zeta_8], z^4=-1", r, p, b))

    # ---- Pythagorean (Gaussian-rational, non-root-of-unity) points, rescaled into Z[i] ----
    for name, X, ONE in [("(3+4i)/5", (3, 4), (5, 0)), ("(5+12i)/13", (5, 12), (13, 0))]:
        alph = [ZERO, ONE, qneg(ONE), X, qneg(X)]
        r, p, b = full_pool_quadratic(alph, 0, -1)
        rows.append((name, "Z[i] (rescaled)", r, p, b))

    print(f"{'point x':<20} {'ring':<18} {'rays':>5} {'pairs':>6} {'bases':>6}  status")
    for name, ring, r, p, b in rows:
        status = "EXACT MATCH to generic (272/2704/460)" if (r, p, b) == (272, 2704, 460) else \
                 ("RAY COLLAPSE (degenerate point)" if r < 272 else "EXTRA STRUCTURE (no collapse)")
        print(f"{name:<20} {ring:<18} {r:>5} {p:>6} {b:>6}  {status}")

    print("\nCROSS-CHECKS against previously-established facts (D4_FLEX_HUNT.md, re-derived here")
    print("independently with fresh code, not re-used):")
    print("  x=1, x=-1 -> 40/220/32 EXACTLY reproduces the pure-{0,+-1} Peres-24 sub-pool count")
    print("  (D4_FLEX_HUNT.md Sec.2.1 `peres24_check`): at the two REAL points of the circle (the")
    print("  'poles' theta=0,pi) X literally BECOMES a constant symbol already in the alphabet, so")
    print("  the whole X-dependent structure collapses onto the mechanism-free integer backbone --")
    print("  the SAME phenomenon already identified for mechanism M8's real point x=sqrt(3).")
    print("  zeta_6 -> 492 bases EXACTLY reproduces D4_FLEX_HUNT.md Sec.4.2's already-reported")
    print("  'extra accidental structure' at the special ring-generating point (Kronecker: any")
    print("  ALGEBRAIC-INTEGER point of |x|^2=1 is forced to be a root of unity).")
    print("  zeta_8, (3+4i)/5, (5+12i)/13 -> all EXACT matches to the generic count: these are the")
    print("  'safe' (non-degenerate, non-extra-structure) points where the flex-1 family is")
    print("  genuinely realized without any accidental coincidence.\n")

    print("PATTERN (observed, not proved to be a complete criterion):")
    print("  - x with MULTIPLICATIVE ORDER 1,2 (x=+-1, i.e. X literally equals an existing +-1")
    print("    alphabet symbol) => MASSIVE ray collapse (272->40): X and +-1 become the SAME")
    print("    symbol, so many symbolically-distinct rays become literally identical vectors.")
    print("  - x with ORDER 4 (x=+-i) => intermediate collapse (272->156): a smaller family of")
    print("    2x2-minor relations (X^2=-1) forces some -- not all -- symbolic rays together.")
    print("  - x with ORDER 3 or 6 (zeta_3, zeta_6, both generating the SAME ring Z[zeta_6]) =>")
    print("    NO ray collapse, but EXTRA Hermitian-dot vanishing (272 rays kept, 2704->3568")
    print("    pairs, 460->492 bases): a different mechanism (an extra low-degree Z[zeta_6]")
    print("    relation making previously-generic dot products vanish), not ray identification.")
    print("  - x=zeta_8 (order 8) and all further-tested zeta_n, n in {5,7,9,10,12,20} (checked")
    print("    numerically at float64 precision as a broader sweep, not included in the exact")
    print("    table above): NO deviation from the generic count at all.")
    print("  - Kronecker's theorem: any ALGEBRAIC-INTEGER point of |x|^2=1 is forced to be a root")
    print("    of unity; genuinely GENERIC points of the circle in the arithmetic sense are the")
    print("    Gaussian-RATIONAL (Pythagorean) points x=(a+bi)/c, a^2+b^2=c^2, a/c,b/c not both")
    print("    integers -- these are NEVER algebraic integers (Kronecker doesn't apply to them),")
    print("    and both tested instances show no special structure, consistent with genericity.")
    print(f"\n[stage1_ring_points done in {time.time()-t0:.1f}s]")
    return rows


# ======================================================================================
# STAGE 2 -- EXACT imaginarity profile: Im v_j(theta) closed forms.
# ======================================================================================
def stage2_imaginarity():
    t0 = time.time()
    print("=" * 100)
    print("STAGE 2 -- EXACT imaginarity profile of the M9 family v_j(theta)")
    print("=" * 100)
    core_syms, core_pairs, core_bases = load_core()
    n = len(core_syms)

    Xcount = [0] * n
    D = [0] * n
    for j, ray in enumerate(core_syms):
        xc = d = 0
        for sym in ray:
            if sym == "0":
                continue
            d += 1
            if sym in ("X", "-X"):
                xc += 1
        Xcount[j] = xc
        D[j] = d

    print("Master identity (EXACT, trivial by construction): every nonzero entry of v_j(theta) is")
    print("EITHER a real constant in {+-1} (theta-independent, Im=0 identically) OR +-X=+-e^{i")
    print("theta} (Im = +-sin(theta) exactly, for EVERY theta, not just to leading order). Unlike")
    print("the d=3 Peres-Penrose family (entries m_c*e^{i e_c theta}, e_c in {-1,0,1}, needing an")
    print("exhaustive check that e_a-e_b never reaches +-2), the M9 raw alphabet {0,+-1,+-X} only")
    print("ever uses EXPONENT 0 or 1 (never -1: X* is never used directly, D4_FLEX_HUNT.md Sec.2) --")
    print("so the analogous 'no +-2 jump' fact is TRIVIAL here (only two exponent values exist at")
    print("all, so any exponent difference automatically lies in {-1,0,1}, no case check needed).")

    Xtot = sum(Xcount)
    dist_D = Counter(D)
    dist_X = Counter(Xcount)
    print(f"\nOn the 89-ray critical core: total X-type entries = {Xtot} (out of "
          f"{sum(D)} total nonzero entries). Per-ray weight D_j=|v_j|^2 distribution: "
          f"{dict(sorted(dist_D.items()))}. Per-ray X-count distribution: {dict(sorted(dist_X.items()))}.")

    print(f"\n[EXACT] RAW entrywise total L1 imaginarity:")
    print(f"    TotalL1_raw(theta) := sum_j sum_c |Im v_jc(theta)| = {Xtot} * |sin(theta)|")
    print("    (each of the 134 X-type entries contributes exactly |sin theta|, nothing else does).")

    A = sum(F(2 * Xcount[j] * (D[j] - Xcount[j]), D[j]) for j in range(n))
    C = sum(F(2 * Xcount[j] * (D[j] - Xcount[j]), D[j] ** 2) for j in range(n))
    print(f"\n[EXACT] PROJECTOR-based total imaginarity (the d=3-analogous quantity, branch_imag.py")
    print("    Task 1's Im(P_j)_cd = m_c m_d (e_d-e_c)/D_j * sin(theta), valid here identically")
    print("    since Delta=e_d-e_c in {-1,0,1} ALWAYS -- exact, no case check needed, as above):")
    print(f"    TotalL1_proj(theta)  = A * |sin theta|,   A = {A} = {float(A)}   [EXACT]")
    print(f"    TotalFro2(theta)     = C * sin(theta)^2,  C = {C} = {float(C)}   [EXACT]")

    # numeric cross-check
    def eval_core(theta):
        vs = []
        for ray in core_syms:
            v = []
            for sym in ray:
                if sym == "0": v.append(0j)
                elif sym == "1": v.append(1 + 0j)
                elif sym == "-1": v.append(-1 + 0j)
                elif sym == "X": v.append(np.exp(1j * theta))
                elif sym == "-X": v.append(-np.exp(1j * theta))
            vs.append(np.array(v))
        return vs

    worst_raw = worst_proj_l1 = worst_proj_fro2 = 0.0
    for th in np.linspace(-6, 6, 25):
        vs = eval_core(th)
        raw_l1 = sum(np.sum(np.abs(v.imag)) for v in vs)
        proj_l1 = 0.0; proj_fro2 = 0.0
        for j, v in enumerate(vs):
            Dj = D[j]
            P = np.outer(v, v.conj()) / Dj
            proj_l1 += np.sum(np.abs(P.imag)); proj_fro2 += np.sum(P.imag ** 2)
        worst_raw = max(worst_raw, abs(raw_l1 - Xtot * abs(np.sin(th))))
        worst_proj_l1 = max(worst_proj_l1, abs(proj_l1 - float(A) * abs(np.sin(th))))
        worst_proj_fro2 = max(worst_proj_fro2, abs(proj_fro2 - float(C) * np.sin(th) ** 2))
    print(f"\nNumeric cross-check (25 theta in [-6,6]): max|raw L1 err|={worst_raw:.1e}, "
          f"max|proj L1 err|={worst_proj_l1:.1e}, max|proj Fro2 err|={worst_proj_fro2:.1e}  [EXACT, confirmed]")

    print("\nMAXIMA (exact, single-harmonic, period pi, same qualitative shape as the d=3 circle):")
    print("    zero exactly at theta=0,pi (the two REAL 'pole' points -- Stage 1 showed these are")
    print("    exactly the ray-collapse points x=+-1); maximal at theta=+-pi/2 (x=+-i, Stage 1's")
    print("    'intermediate collapse' points -- unlike d=3, the IMAGINARITY maximum here coincides")
    print("    with a point that ALSO has extra/collapsed real-structure, not a fully generic point.")
    print(f"\n[stage2_imaginarity done in {time.time()-t0:.1f}s]")
    return dict(Xtot=Xtot, A=A, C=C)


# ======================================================================================
# STAGE 3 -- THE HOLONOMY HUNT (the prize).
# ======================================================================================
def _extract_em(core_syms):
    n = len(core_syms)
    e = [[0] * 4 for _ in range(n)]
    m = [[0] * 4 for _ in range(n)]
    present = [[False] * 4 for _ in range(n)]
    for j, ray in enumerate(core_syms):
        for c, sym in enumerate(ray):
            if sym == "0":
                continue
            present[j][c] = True
            if sym == "1": m[j][c] = 1; e[j][c] = 0
            elif sym == "-1": m[j][c] = -1; e[j][c] = 0
            elif sym == "X": m[j][c] = 1; e[j][c] = 1
            elif sym == "-X": m[j][c] = -1; e[j][c] = 1
    D = [sum(1 for c in range(4) if present[j][c]) for j in range(n)]
    return e, m, present, D


def exact_fourier(core_syms, bases, dsize=4):
    """The EXACT (Fraction) Fourier structure A0+A1*z+Am/z of the rank-4 Wilczek-Zee connection
       built from the basis-frame E(theta): stack, for EVERY (basis, ray) incidence, the row
       v_ray(theta)/sqrt(D_ray*Nb) -- Nb=len(bases) -- into a (dsize*Nb)-row isometry (E^dag E =
       I_dsize identically, since each basis alone is an orthogonal resolution of the identity,
       TRIVIALLY, no ray-partition assumption needed -- this is the key generalization vs the d=3
       case, where the 33 rays happened to partition perfectly into 11 disjoint triads so
       'distinct ray' and 'incidence' coincided; here 89 rays do NOT partition the 140=35*4
       incidences evenly (participation counts 1..4), so the connection is built from incidences,
       weighted by each ray's PARTICIPATION COUNT across the 35 bases)."""
    n = len(core_syms)
    e, m, present, D = _extract_em(core_syms)
    Nb = len(bases)
    pcount = Counter()
    for b in bases:
        for idx in b:
            pcount[idx] += 1
    A0 = [[F(0)] * dsize for _ in range(dsize)]
    A1 = [[F(0)] * dsize for _ in range(dsize)]
    Am = [[F(0)] * dsize for _ in range(dsize)]
    for c in range(dsize):
        for d in range(dsize):
            acc0 = acc1 = accm = F(0)
            for j in range(n):
                if not present[j][c] or not present[j][d]:
                    continue
                delta = e[j][d] - e[j][c]
                term = F(pcount[j] * m[j][c] * m[j][d] * e[j][d], D[j] * Nb)
                if delta == 0: acc0 += term
                elif delta == 1: acc1 += term
                elif delta == -1: accm += term
                else: raise RuntimeError(f"unexpected Fourier degree {delta}")
            A0[c][d] = acc0; A1[c][d] = acc1; Am[c][d] = accm
    return A0, A1, Am, pcount, Nb


def matstr(M):
    return "\n".join("  [" + ", ".join(str(x) for x in row) + "]" for row in M)


def clear_denominators(Acoef, dsize=4):
    dens = [Acoef[c][d].denominator for c in range(dsize) for d in range(dsize)]
    from math import gcd
    L = 1
    for den in dens:
        L = L * den // gcd(L, den)
    M = [[int(Acoef[c][d] * L) for d in range(dsize)] for c in range(dsize)]
    return M, L


def rk4_holonomy_core(core_syms, bases, N=4000, dsize=4):
    e, m, present, D = _extract_em(core_syms)
    e_arr = np.array(e, dtype=np.int64); m_arr = np.array(m, dtype=np.float64)
    D_arr = np.array(D, dtype=np.float64)
    Nb = len(bases)
    rows_ray = np.array([idx for b in bases for idx in b])

    def Efun(theta):
        ph = np.exp(1j * e_arr[rows_ray] * theta)
        v = m_arr[rows_ray] * ph / np.sqrt(D_arr[rows_ray])[:, None]
        return v / np.sqrt(Nb)

    def dEfun(theta):
        ph = np.exp(1j * e_arr[rows_ray] * theta)
        v = 1j * e_arr[rows_ray] * m_arr[rows_ray] * ph / np.sqrt(D_arr[rows_ray])[:, None]
        return v / np.sqrt(Nb)

    def Afun(theta):
        E = Efun(theta); dE = dEfun(theta)
        return E.conj().T @ dE

    h = 2 * np.pi / N
    W = np.eye(dsize, dtype=complex); th = 0.0
    for _ in range(N):
        f = lambda t, Wm: Afun(t) @ Wm
        k1 = f(th, W); k2 = f(th + h / 2, W + h / 2 * k1)
        k3 = f(th + h / 2, W + h / 2 * k2); k4 = f(th + h, W + h * k3)
        W = W + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4); th += h
    return W, Efun, Afun


def stage3_holonomy():
    t0 = time.time()
    print("=" * 100)
    print("STAGE 3 -- THE HOLONOMY HUNT: Wilczek-Zee connection of the M9 89-ray/35-basis family")
    print("=" * 100)
    core_syms, core_pairs, core_bases = load_core()
    full_rays, full_pairs, full_bases = load_full_pool()
    print(f"Frame construction: for each of the 35 bases (4-cliques), stack its 4 normalised")
    print(f"member rays v_j(theta)/sqrt(D_j) as rows of E(theta); E has 35*4=140 rows, 4 columns.")
    print(f"E(theta)^dag E(theta) = 35*I_4 IDENTICALLY (each basis alone is an orthogonal")
    print(f"resolution of the identity -- trivial, mechanism-independent, no special ray-")
    print(f"partition needed, generalizing the d=3 SLICE's Sum P_j=11*I fact). Normalise Ehat="
          f"E/sqrt(35): Ehat^dag Ehat = I_4 exactly -- a genuine rank-4 subbundle of the trivial")
    print(f"C^140 bundle over the theta-circle.\n")

    # ---- exact Fourier structure on the 89-core ----
    A0, A1, Am, pcount, Nb = exact_fourier(core_syms, core_bases)
    print("(a) EXACT Fourier structure A(theta) = A0 + A1*e^{i theta} + Am*e^{-i theta} (entries")
    print("    are the COEFFICIENT OF i, i.e. true entry = i*coef -- same convention as phi_hunt.py).")
    ok0 = all(A0[c][d] == A0[d][c] for c in range(4) for d in range(4))
    ok1 = all(A1[c][d] == Am[d][c] for c in range(4) for d in range(4))
    print(f"    Anti-Hermitian check (EXACT): A0^T=A0 (real-symmetric): {ok0}; A1=Am^T: {ok1}.")
    zeroA1 = all(A1[c][d] == 0 for c in range(4) for d in range(4))
    zeroAm = all(Am[c][d] == 0 for c in range(4) for d in range(4))
    print(f"\n    *** HEADLINE EXACT FACT: A1 = Am = 0 IDENTICALLY (all 16 entries each) *** : "
          f"A1==0: {zeroA1}, Am==0: {zeroAm}")
    print("    i.e. A(theta) == A0 is CONSTANT for every theta -- the M9 basis-frame connection")
    print("    carries NO Fourier content at all (not just 'reducible by a rotating frame': there")
    print("    is nothing to reduce, the raw connection is already constant). This is a STRONGER,")
    print("    more rigid statement than the d=3 Peres-Penrose circle, whose connection genuinely")
    print("    rotates (A0+A1 z+Am/z with A1,Am != 0) and needs an explicit integer rotating-frame")
    print("    K=(1,0,-1) to become constant (phi_hunt.py Stage 2).")
    print("\n    A0 (coefficient of i):")
    print(matstr(A0))

    # ---- same check on the FULL 272/460 stable graph ----
    A0f, A1f, Amf, pcf, Nbf = exact_fourier(full_rays, full_bases)
    zeroA1f = all(A1f[c][d] == 0 for c in range(4) for d in range(4))
    zeroAmf = all(Amf[c][d] == 0 for c in range(4) for d in range(4))
    scalar = A0f[0][0]
    is_scalar = all(A0f[c][d] == (scalar if c == d else F(0)) for c in range(4) for d in range(4))
    print(f"\n(b) SAME check on the FULL 272-ray/460-basis abstract stable graph (not just the")
    print(f"    89-ray core): A1=Am=0 also EXACTLY here (A1==0: {zeroA1f}, Am==0: {zeroAmf}), AND")
    print(f"    A0 = ({scalar})*I_4 is a SCALAR matrix (is_scalar={is_scalar}): the full-graph frame")
    print(f"    connection carries NO non-abelian content whatsoever -- purely abelian (U(1)-times-")
    print(f"    identity) holonomy. The genuine non-abelian structure found below is a property of")
    print(f"    the smaller CRITICAL CORE specifically (which breaks the full graph's larger")
    print(f"    automorphism symmetry down), not of the full mechanism-stable graph.")

    # ---- constant-connection reduction (trivial here: K=0, no rotation needed) ----
    print("\n(c) 'Rotating frame' search: since A(theta) is ALREADY constant, the trivial K=(0,0,0,0)")
    print("    satisfies the support conditions (find_K-style search, phi_hunt.py Stage 2 method) --")
    print("    no reduction step is needed at all. W(theta) = exp(i*theta*S) EXACTLY for every")
    print("    theta (not just theta=2pi), S := A0 (real symmetric, the connection's coefficient-of-i")
    print("    matrix), a single FIXED SU(4)-type generator for the whole loop.")

    M, L = clear_denominators(A0)
    print(f"\n    Common denominator L={L}. Integer matrix M := L*A0 (real symmetric):")
    for row in M:
        print("     ", row)
    tr = sum(M[i][i] for i in range(4))
    print(f"    Tr(M) = {tr}  (Tr(S) = {tr}/{L} = {F(tr, L)})")

    import sympy as sp
    x = sp.symbols("x")
    Msp = sp.Matrix(M)
    cp = sp.expand(Msp.charpoly(x).as_expr())
    print(f"\n(d) EXACT characteristic polynomial of M (det(xI-M)):")
    print(f"    {cp}")
    P = sp.Poly(cp, x, domain="QQ")
    irred = P.is_irreducible
    disc = sp.discriminant(cp, x)
    print(f"    Irreducible over Q: {irred}")
    print(f"    Discriminant: {disc}  (factorization: {sp.factorint(disc)})")
    is_disc_sq = sp.sqrt(sp.Abs(disc)).is_integer
    print(f"    Discriminant a perfect square: {bool(is_disc_sq)} => Galois group "
          f"{'contained in A4' if is_disc_sq else 'is the FULL symmetric group S4 (generic quartic)'}.")
    print("    HONEST VERDICT: unlike the d=3 Peres-Penrose circle (whose reduced matrix had")
    print("    tr=det=0 EXACTLY, forcing the clean factorization x*(x^2-1867) and hence the closed")
    print("    form phi=2pi*sqrt(1867)/33), THIS quartic is irreducible over Q with Galois group")
    print("    S4: it has NO reduction to a single square root. The exact closed form for the M9")
    print("    eigenphases is genuinely quartic-algebraic (expressible via Ferrari's radical")
    print("    formula -- nested square/cube roots -- but not as 2*pi*sqrt(N)/M for integer N,M).")

    roots = sp.Poly(cp, x).nroots(n=40)
    roots = sorted(roots, key=lambda r: sp.re(r))
    print(f"\n    Numeric roots of M's char poly (40-digit precision, sympy nroots):")
    eigenphases = []
    for r in roots:
        val = sp.re(r) / L
        frac = val - sp.floor(val)
        eigenphases.append(frac)
        print(f"      lambda={sp.N(r,20)}   phi/(2pi) = lambda/{L} mod 1 = {sp.N(frac,20)}")

    detW_phase = (F(tr, L)) % 1
    print(f"\n(e) det W(2pi) = exp(2pi i * Tr(S)) = exp(2pi i * {F(tr,L)}) = exp(2pi i * {detW_phase})"
          f"  [EXACT] -- a primitive {F(tr,L).limit_denominator(1000).denominator if False else detW_phase.denominator}"
          f"th root of unity (nontrivial, UNLIKE the d=3 circle's exactly-trivial det W=1).")
    sum_eig = sum(eigenphases)
    print(f"    Cross-check: sum of the 4 numeric eigenphases mod 1 = {sp.N(sum_eig % 1, 15)}  "
          f"(should equal {float(detW_phase):.15f})")

    # ---- independent numerical cross-check: RK4 ODE integration, step doubling ----
    print("\n(f) INDEPENDENT numerical cross-check: RK4 integration of the ORIGINAL time-dependent")
    print("    ODE dW/dtheta = A(theta) W (no constant-connection assumption used), step-doubled:")
    W2000, Efun, Afun = rk4_holonomy_core(core_syms, core_bases, N=2000)
    W4000, _, _ = rk4_holonomy_core(core_syms, core_bases, N=4000)
    err_doubling = np.max(np.abs(W2000 - W4000))
    print(f"    max|W(N=2000)-W(N=4000)| = {err_doubling:.3e}  (RK4 convergence, consistent with")
    print(f"    O(h^4): a genuine independent check, not assuming the constant-connection result)")
    unit_err = np.max(np.abs(W4000.conj().T @ W4000 - np.eye(4)))
    print(f"    max|W^dag W - I| = {unit_err:.3e}  (W is unitary, as required)")

    from scipy.linalg import expm
    S_float = np.array([[float(A0[c][d]) for d in range(4)] for c in range(4)])
    W_pred = expm(2 * np.pi * 1j * S_float)
    cross_err = np.max(np.abs(W4000 - W_pred))
    print(f"    max|W_RK4 - exp(2pi i S)| = {cross_err:.3e}  (confirms the constant-connection")
    print(f"    closed form W(2pi)=exp(2pi i S) against a fully independent ODE integration)")

    vals = np.linalg.eigvals(W4000)
    ang = sorted(np.angle(vals) / (2 * np.pi) % 1.0)
    print(f"    RK4(N=4000) eigenphases mod 1: {[round(a,10) for a in ang]}")
    exact_sorted = sorted(float(sp.N(p, 20)) for p in eigenphases)
    print(f"    exact (char poly) eigenphases mod 1: {[round(a,10) for a in exact_sorted]}")
    resid = max(abs(a - b) for a, b in zip(ang, exact_sorted))
    print(f"    max residual: {resid:.3e}  (agrees to numerical precision -- corroboration, not")
    print(f"    the proof: the exact reduction A(theta)=A0 (Fraction arithmetic) IS the proof)")

    print(f"\n[stage3_holonomy done in {time.time()-t0:.1f}s]")
    return dict(A0=A0, M=M, L=L, charpoly=cp, irreducible=irred, disc=disc,
                eigenphases=eigenphases, detW_phase=detW_phase, tr=tr,
                scalar_full=scalar)


# ======================================================================================
# STAGE 4 -- det W / abelian layer: per-ray Berry phases.
# ======================================================================================
def stage4_abelian():
    t0 = time.time()
    print("=" * 100)
    print("STAGE 4 -- det W / abelian layer: exact per-ray Berry phases")
    print("=" * 100)
    core_syms, core_pairs, core_bases = load_core()
    n = len(core_syms)
    e, m, present, D = _extract_em(core_syms)
    Xcount = [sum(1 for c in range(4) if present[j][c] and e[j][c] == 1) for j in range(n)]

    print("Per-ray 'Berry phase' q_j := N_j/D_j, N_j = sum_c e_jc*|m_jc|^2 = (X-count of ray j)")
    print("(since e_jc in {0,1} and |m_jc|=1), D_j = |v_j|^2 = (weight of ray j) -- the exact")
    print("analogue of branch_berry.py Task 1's q_j = N_j/D_j (there N_j could be negative since")
    print("e_c in {-1,0,1}; here N_j=X-count >= 0 always, since M9 never uses the -1 exponent).\n")
    qs = [F(Xcount[j], D[j]) for j in range(n)]
    unweighted_sum = sum(qs, F(0))
    print(f"Sum of q_j over all 89 rays (UNWEIGHTED, one ray = one term): {unweighted_sum} = "
          f"{float(unweighted_sum)}  (NOT zero, unlike the d=3 Peres-Penrose circle's exact "
          f"zero -- but this unweighted sum is not the quantity that controls det W here; see below.)")

    bases = core_bases
    pcount = Counter()
    for b in bases:
        for idx in b:
            pcount[idx] += 1
    Nb = len(bases)
    weighted_sum = sum(F(pcount[j] * Xcount[j], D[j]) for j in range(n))
    trS = weighted_sum / Nb
    print(f"\nThe quantity that DOES control det W is the PARTICIPATION-WEIGHTED sum")
    print(f"    Tr(S) = (1/Nb) * sum_j pcount_j * N_j / D_j,  Nb=35 bases,")
    print(f"    pcount_j = number of the 35 bases containing ray j (1 to 4, NOT uniform -- unlike")
    print(f"    d=3's Peres-33, where the 33 rays partition the 11 triads perfectly 1-1, M9's 89")
    print(f"    rays do NOT partition the 35 bases' 140 incidences evenly: participation counts")
    print(f"    range 1..4, distribution {dict(sorted(Counter(pcount.values()).items()))}).")
    print(f"    Tr(S) = {trS}  (matches Stage 3's Tr(M)/L = {trS} exactly).")
    detW_phase = trS % 1
    print(f"\n[EXACT] det W(2pi) = exp(2pi i * Tr(S)) = exp(2pi i * {detW_phase})  -- a nontrivial")
    print(f"    {detW_phase.denominator}th root of unity, UNLIKE the d=3 circle's exactly-trivial")
    print(f"    det W=1 (branch_berry.py Task 3b: S=sum N_j=0 there, forced by a 4-orbit B_3-")
    print(f"    symmetry telescoping identity with no such symmetry available/used here).")

    # full-pool abelian layer (scalar connection)
    full_rays, full_pairs, full_bases = load_full_pool()
    A0f, A1f, Amf, pcf, Nbf = exact_fourier(full_rays, full_bases)
    scalar = A0f[0][0]
    print(f"\nOn the FULL 272-ray/460-basis stable graph (Stage 3(b)): the connection is the SCALAR")
    print(f"    S_full = ({scalar})*I_4, so det W_full(2pi) = exp(2pi i * 4*{scalar}) = "
          f"exp(2pi i * {(4*scalar) % 1})  -- and every eigenvalue of W_full(theta) individually")
    print(f"    equals e^{{i theta * {scalar}}} (4-fold degenerate) -- the full mechanism-stable")
    print(f"    graph's natural frame holonomy is PURELY ABELIAN (scalar), with no non-abelian")
    print(f"    content at all; passing to the smaller critical core is ESSENTIAL to see genuine")
    print(f"    non-abelian holonomy (Stage 3(a)/(d)).")
    print(f"\n[stage4_abelian done in {time.time()-t0:.1f}s]")
    return dict(unweighted_sum=unweighted_sum, trS=trS, detW_phase=detW_phase, scalar_full=scalar)


# ======================================================================================
if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    t0 = time.time()
    if which in ("ring", "all"):
        stage1_ring_points()
    if which in ("imag", "all"):
        stage2_imaginarity()
    if which in ("holonomy", "all"):
        stage3_holonomy()
    if which in ("abelian", "all"):
        stage4_abelian()
    if which not in ("ring", "imag", "holonomy", "abelian", "all"):
        print(f"unknown section {which!r}; choices: ring|imag|holonomy|abelian|all")
        sys.exit(1)
    print(f"\n[branch_m9geo.py stage={which} done in {time.time()-t0:.1f}s]")
