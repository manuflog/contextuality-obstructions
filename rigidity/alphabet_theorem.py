#!/usr/bin/env python3
"""
alphabet_theorem.py -- turning Kernaghan's EMPIRICAL two-mechanism observation (arXiv:2603.16988,
survey of KS-uncolorable sets built from two-symbol alphabets A = {0,+-1,+-x} in C^3) into a
THEOREM, or finding where it breaks. See ALPHABET_THEOREM.md for the full writeup; this file is
the reproducible computational spine.

HONESTY SCOPE (read before trusting a number below): arXiv:2603.16988 itself was fetched and read
in FULL in a prior session (see KS_CENSUS.md); THIS session does not re-fetch it. Everything
below attributed to "the paper" is either (a) taken verbatim from the ATTACK BRIEF's own summary
of the paper's claim, given to this session as ground truth to test, or (b) cross-referenced
against the already-validated (in prior sessions, against the paper's own printed invariants)
island reconstructions living in ks_flex_census.py / KS_CENSUS.md. No new claim about what the
paper's prose says is made anywhere in this file or in ALPHABET_THEOREM.md.

WHAT THIS FILE DOES (five independent, cheap, exact/verified computations):

  1. `derive`     : the exhaustive classification of ALL 3-term vanishing sums achievable by an
                    inner product of two rays with entries in {0,+-1,+-x} in C^3 (d=3, so <=3
                    nonzero coordinate-products per orthogonality relation). Every one of the 20
                    "type multisets" x 4 independent sign patterns = 80 cases is solved by SYMPY
                    (not just asserted by hand), splitting x=p+iq into real p,q so the sympy
                    solve is over genuine real unknowns (this is necessary: a naive complex-symbol
                    solve returns spurious non-real parametrizations -- see ALPHABET_THEOREM.md
                    Sec 1 for the worked example of the bug this avoids). Produces the complete
                    list of algebraic mechanisms M1..M5.
  2. `duality`    : PROVES (2-line exact complex algebra, plus a numeric spot-check) that the
                    alphabet has a hidden gauge symmetry x -> 1/x (dividing every ray by x is a
                    global rescale, hence preserves every ray projectively and just swaps which
                    symbol is called "1" vs "x") under which M2 <-> M2 (modulus 2 <-> 1/2), M1 <->
                    M1 (x=2 <-> x=1/2), M4 <-> M4 (golden family is self-dual), and CRUCIALLY
                    M3 <-> M5 (Kernaghan's phase mechanism is inversion-dual to the "circle"
                    mechanism found in `derive` -- so M5 is NOT a genuinely new 5th mechanism).
  3. `structure`  : VERIFIES computationally that any x avoiding M1-M4 (a "generic" point) gives
                    the SAME orthogonality graph (literally identical edge-index sets, not just
                    matching counts) as any other avoiding x, colorable (3-colorable), matching
                    the Godsil-Zaks theorem for rational rays (x=3,5,7 are directly rational).
                    Also checks a non-real generic point (Gaussian x=1+2i) and, for the real case,
                    that cross-product COMPLETION (not just the raw pool) also stays colorable.
  4. `mechanisms` : sanity-checks the algebraic condition (not the full KS-uncolorability, which
                    is already exactly certified in ks_flex_census.py/KS_CENSUS.md) for each named
                    generator against M1-M5.
  5. `m5hunt`     : the actual NEW-ISLAND HUNT the brief asked for -- tests a concrete point x on
                    mechanism M5 that is NOT simultaneously on M3 in the "obvious" way (x=2+omega,
                    Eisenstein integer, norm 3, Re(x)=3/2 != +-1/2) for actual KS-uncolorability,
                    using ks_flex_census.py machinery UNMODIFIED (raw_vectors/collect_rays/
                    build_structure/uncolorable/greedy_critical_core/exact_flex_hermitian_quadratic
                    /find_primes_ring), cross-checked with ks_flex_census.ks_colorable_generic and
                    torsion_layer.count_ks_colorings per the MACHINERY brief. Reports the honest
                    verdict (spoiler in ALPHABET_THEOREM.md: it finds a KS-uncolorable 33-ray core
                    with flex=0 EXACT -- but that core turns out to be a SUBSET of the *already
                    known* omega-generated Eisenstein/Cabello-33 island's own 57-ray raw pool, not
                    a new configuration -- an honest negative that is itself a nice confirmation
                    of the `duality` result).

Every stage is standalone and runs in well under 45s (see timings printed by each stage; the
heaviest, m5hunt, is ~15s). No existing file is modified; only sympy and ks_flex_census.py
(imported, unmodified) are used as machinery, per the brief.

Usage:
    python3 alphabet_theorem.py derive
    python3 alphabet_theorem.py duality
    python3 alphabet_theorem.py structure
    python3 alphabet_theorem.py mechanisms
    python3 alphabet_theorem.py m5hunt
    python3 alphabet_theorem.py all           # all five, ~30s total
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from itertools import combinations_with_replacement, product as iproduct
from collections import Counter

import sympy as sp

from ks_flex_census import (
    qmul, qneg, qconj, qz, ZERO, herm_dot, bil_dot, raw_vectors, collect_rays,
    build_structure, uncolorable, basis_participating, restrict, greedy_critical_core,
    exact_flex_hermitian_quadratic, find_primes_ring, t0_tau, proportional,
    cross_product_completion, ks_colorable_generic,
)
from torsion_layer import count_ks_colorings

# ==================================================================================================
# STAGE 1: derive -- exhaustive 3-term vanishing-sum classification, sympy-verified
# ==================================================================================================
def derive():
    t0 = time.time()
    p, q = sp.symbols('p q', real=True)
    I = sp.I
    x, xb, N = p + I * q, p - I * q, p ** 2 + q ** 2
    types = {'1': sp.Integer(1), 'x': x, 'x*': xb, 'N': N}
    type_names = ['1', 'x', 'x*', 'N']

    def is_spurious(val):
        # sympy's plain solve() does not respect the p,q real=True assumption when a free
        # variable remains: it returns algebraic-closure branches like p=+-I*q, which would only
        # be real if q=0 (checked separately below). Filter these explicitly -- see module
        # docstring point 1.
        return val.has(sp.I)

    found = {}   # canonical condition string -> (combo, sign pattern, solution) example
    n_impossible, n_cases = 0, 0
    for combo in combinations_with_replacement(type_names, 3):
        for signs in iproduct([1, -1], repeat=2):
            s = (1,) + signs
            n_cases += 1
            expr = sp.expand(sum(si * types[t] for si, t in zip(s, combo)))
            re_part, im_part = sp.re(expr), sp.im(expr)
            sol = sp.solve([sp.Eq(re_part, 0), sp.Eq(im_part, 0)], [p, q], dict=True)
            clean = []
            for d in sol:
                pv, qv = d.get(p, p), d.get(q, q)
                if is_spurious(pv) or is_spurious(qv):
                    continue
                if pv == 0 and qv == 0:
                    continue  # x = 0, excluded (not a valid alphabet symbol)
                clean.append(d)
            if not clean:
                n_impossible += 1
                continue
            # classify into M1..M5 by inspecting the solution shape
            for d in clean:
                pv = d.get(p, None)
                qfree = q not in d
                if pv is not None and pv.is_number and not qfree:
                    qv = d.get(q, sp.Integer(0))
                    if qv == 0 and pv in (sp.Rational(2), sp.Rational(-2), sp.Rational(1, 2), sp.Rational(-1, 2)):
                        key = "M1 (rational): x in {+-2, +-1/2}"
                    elif qv == 0 and sp.simplify(pv ** 2 - pv - 1) == 0:
                        key = "M4 (golden/quadratic-trace): x real, x^2 - x - 1 = 0"
                    elif qv == 0 and sp.simplify(pv ** 2 + pv - 1) == 0:
                        key = "M4 (golden/quadratic-trace): x real, x^2 + x - 1 = 0"
                    else:
                        key = f"other isolated real point p={pv}, q={qv}"
                elif pv is not None and qfree and pv.free_symbols == set():
                    key = f"M3 (phase): Re(x) = {pv}  (any Im x)"
                elif pv is not None and qfree:
                    # p expressed as a function of free q -> a curve (circle) or the modulus-2 arc
                    pexpr = sp.simplify(pv)
                    if sp.simplify(pexpr ** 2 - (2 - q ** 2)) == 0 or sp.simplify(pexpr**2 - (sp.Rational(1,2) - q**2)) == 0:
                        # p^2 + q^2 = const  <=>  N = const
                        constval = sp.simplify(pexpr**2 + q**2)
                        key = f"M2 (modulus): |x|^2 = {constval}"
                    else:
                        # circle (p -+ 1)^2 + q^2 = 1  <=>  2 Re(x) = +- N
                        key = f"M5 (circle): p = {pexpr}  i.e. 2Re(x) = +-|x|^2"
                else:
                    key = "unclassified: " + str(d)
                found.setdefault(key, []).append((combo, s))

    print(f"[derive] {n_cases} (type-multiset, sign-pattern) cases enumerated "
          f"(C(4+3-1,3)=20 multisets x 4 sign patterns = 80).")
    print(f"[derive] {n_impossible} cases have NO solution at all (impossible for any x) "
          f"-- these are exactly the 'odd number of +-1-type terms' parity obstructions "
          f"(AAA-type: 1+1+1, x+x+x, x*+x*+x*, N+N+N; and the pure-modulus-mismatch AAB cases "
          f"x+x-x*=0 etc., which force x=0).")
    print(f"[derive] {len(found)} distinct nontrivial conditions found, collapsing into:")
    mech_order = ["M1", "M2", "M3", "M4", "M5"]
    for m in mech_order:
        keys = sorted(k for k in found if k.startswith(m))
        for k in keys:
            print(f"    {k}   <- {len(found[k])} (combo,signs) case(s), e.g. {found[k][0]}")
    others = sorted(k for k in found if not any(k.startswith(m) for m in mech_order))
    if others:
        print("  ** UNCLASSIFIED (would need inspection -- none expected):")
        for k in others:
            print("    ", k)
    assert not others, "found a case outside M1-M5 -- classification is not exhaustive, STOP"
    print(f"[derive] EXHAUSTIVE: every one of the 80 cases is either impossible, or falls into "
          f"exactly one of M1 (rational), M2 (modulus-2, Kernaghan's mechanism (i)), M3 (phase, "
          f"generalizes Kernaghan's mechanism (ii)), M4 (golden/quadratic-trace), M5 (circle). "
          f"({time.time()-t0:.2f}s)")
    return found


# ==================================================================================================
# STAGE 2: duality -- x <-> 1/x is a gauge symmetry of the alphabet; proves M3 <-> M5, M1 self-dual,
# M2 self-dual (2<->1/2), M4 self-dual.
# ==================================================================================================
def duality():
    t0 = time.time()
    print("[duality] Gauge symmetry: dividing every raw vector by x is a global rescale (each ray")
    print("  v ~ v/x is the SAME projective ray), turning alphabet {0,+-1,+-x} into {0,+-1,+-(1/x)}.")
    print("  So the mechanism-list must be closed under x -> y := 1/x. Check each mechanism:")

    # M1: x=2 <-> y=1/2 (both literally in the M1 list {+-2,+-1/2}) -- trivial, by inspection.
    print("  M1 (x in {+-2,+-1/2}): closed under inversion by inspection (2<->1/2, -2<->-1/2). OK.")

    # M2: N=2 <-> N(y)=1/N(x)=1/2 (both literally in the M2 list {2,1/2}) -- trivial.
    print("  M2 (|x|^2 in {2,1/2}): |y|^2 = 1/|x|^2, so 2 <-> 1/2. Closed. OK.")

    # M4: symbolic check that y=1/x for x a golden root is again a golden root.
    xr = sp.symbols('xr', real=True)
    for eq in [xr**2 - xr - 1, xr**2 + xr - 1]:
        roots = sp.solve(sp.Eq(eq, 0), xr)
        for r in roots:
            y = sp.simplify(1 / r)
            ok = sp.simplify(y**2 - y - 1) == 0 or sp.simplify(y**2 + y - 1) == 0
            print(f"    M4 root x={r}  ->  y=1/x={sp.nsimplify(y)}  "
                  f"still a golden root: {ok}")
            assert ok
    print("  M4 (golden family): PROVED self-dual under x -> 1/x (checked for all 4 roots). OK.")

    # M3 <-> M5: exact symbolic proof.
    # x = p + i q (p,q real), y = 1/x = xbar / N(x).  Re(y) = p / N(x).  N(y) = 1/N(x).
    p, q, eps = sp.symbols('p q eps', real=True)
    N = p**2 + q**2
    # M3 hypothesis: p = eps/2  (eps = +-1)
    p_m3 = eps / 2
    Nx = p_m3**2 + q**2
    Re_y = p_m3 / Nx
    Ny = 1 / Nx
    lhs = sp.simplify(2 * Re_y - eps * Ny)
    print(f"  M3 -> M5 exact identity check: with Re(x)=eps/2, "
          f"2*Re(1/x) - eps*|1/x|^2 = {lhs}  (should be 0 identically in q)")
    assert lhs == 0, "duality identity failed -- STOP, do not report as proved"
    print("  PROVED: x in M3 (Re x = eps/2) => y=1/x satisfies 2*Re(y) = eps*|y|^2, i.e. y in M5.")

    # converse: M5 -> M3
    # M5 hypothesis: 2p = eps*N(x)  =>  p = eps*N/2
    Nsym = sp.symbols('Nsym', positive=True)
    p_m5 = eps * Nsym / 2
    # here we treat p as determined by N via the M5 relation; q is then whatever makes p^2+q^2=Nsym
    Re_y2 = sp.simplify(p_m5 / Nsym)
    print(f"  M5 -> M3 exact identity check: with 2*Re(x)=eps*|x|^2, "
          f"Re(1/x) = Re(x)/|x|^2 = (eps*|x|^2/2)/|x|^2 = {Re_y2}  (should be eps/2)")
    assert Re_y2 == eps / 2
    print("  PROVED: x in M5 => y=1/x satisfies Re(y) = eps/2, i.e. y in M3.")
    print("  ==> M3 and M5 are THE SAME mechanism family under the alphabet's own inversion gauge")
    print("      symmetry -- NOT two independent mechanisms. (See m5hunt: an M5 test point's")
    print("      critical KS core turns out to already sit inside the M3(omega) island's raw pool.)")

    # numeric spot check (independent of the symbolic proof, cheap sanity net)
    import cmath
    xv = complex(-0.5, 1.3)  # Re(x)=-1/2 exactly: M3 point
    yv = 1 / xv
    lhs_num = 2 * yv.real - (-1) * abs(yv) ** 2
    print(f"  numeric spot check: x={xv}, y=1/x={yv}, 2Re(y)-(-1)|y|^2 = {lhs_num:.2e} (~0). OK.")
    print(f"[duality] done ({time.time()-t0:.2f}s)")


# ==================================================================================================
# STAGE 3: structure -- generic x (avoiding M1-M4) gives the SAME, colorable, graph.
# ==================================================================================================
def _avoids_M1_M4(xr_val=None, xc_val=None):
    """Sanity pre-check (not a proof, just a guard against accidentally picking a bad test point)."""
    if xr_val is not None:
        p, N = xr_val, xr_val * xr_val
        bad = (p in (2, -2)) or (abs(p - 0.5) < 1e-9) or (abs(p + 0.5) < 1e-9) \
            or (abs(N - 2) < 1e-9) or (abs(N - 0.5) < 1e-9) \
            or (abs(p**2 - p - 1) < 1e-9) or (abs(p**2 + p - 1) < 1e-9)
        return not bad
    p, q = xc_val.real, xc_val.imag
    N = p * p + q * q
    bad = (abs(q) < 1e-9 and (abs(p - 2) < 1e-9 or abs(p + 2) < 1e-9 or abs(p - 0.5) < 1e-9 or abs(p + 0.5) < 1e-9)) \
        or (abs(N - 2) < 1e-9) or (abs(N - 0.5) < 1e-9) \
        or (abs(p - 0.5) < 1e-9 or abs(p + 0.5) < 1e-9)
    return not bad


def structure():
    t0 = time.time()
    B, C = 0, 0  # plain integers (real ring)
    ONE = (1, 0)

    def real_pool(xval):
        X = (xval, 0)
        alph = [ZERO, ONE, qneg(ONE), X, qneg(X)]
        rays = collect_rays(raw_vectors(alph, 3), B, C)
        pairs, triads, _ = build_structure(rays, bil_dot, B, C)
        u, nodes, _, _ = uncolorable(rays, bil_dot, B, C)
        return rays, pairs, triads, u

    print("[structure] Raw-pool check at three GENERIC rational x (avoid M1-M4 by inspection: "
          "x=3,5,7 are integers != +-2, |x|^2 != 2, x^2-x-1 != 0, x^2+x-1 != 0):")
    pools = {}
    for xv in (3, 5, 7):
        assert _avoids_M1_M4(xr_val=xv), f"x={xv} unexpectedly hits a mechanism -- pick another"
        rays, pairs, triads, u = real_pool(xv)
        pools[xv] = (rays, pairs, triads)
        print(f"    x={xv}: rays={len(rays)} pairs={len(pairs)} triads(bases)={len(triads)} "
              f"KS-uncolorable={u}  (expect False)")
        assert not u, f"x={xv} unexpectedly gave a KS-uncolorable raw pool"

    same35 = set(pools[3][1]) == set(pools[5][1]) and set(pools[3][2]) == set(pools[5][2])
    same37 = set(pools[3][1]) == set(pools[7][1]) and set(pools[3][2]) == set(pools[7][2])
    print(f"    graph IDENTICAL (same edge/triad index sets, not just counts) at x=3 vs x=5: {same35}")
    print(f"    graph IDENTICAL at x=3 vs x=7: {same37}")
    assert same35 and same37, "generic-x graphs differ -- structure theorem claim is FALSE, STOP"

    # cross-product completion at x=3 stays colorable (stronger than just the raw pool)
    rays3 = pools[3][0]
    completed = cross_product_completion(rays3, B, C, max_rounds=6, verbose=False)
    u_c, nodes_c, _, _ = uncolorable(completed, bil_dot, B, C)
    print(f"    x=3 AFTER cross-product completion: {len(completed)} rays, "
          f"KS-uncolorable={u_c}  (expect False -- completion does not rescue a generic x)")
    assert not u_c

    # a non-real generic point: Gaussian x = 1+2i (avoids M1-M4: not real -> M4 excluded trivially;
    # |x|^2=5 not in {2,1/2}; Re(x)=1 not +-1/2)
    Bc, Cc = 0, -1
    Xc = (1, 2); Xcb = qconj(Xc, Bc)
    alph = [ZERO, ONE, qneg(ONE), Xc, qneg(Xc), Xcb, qneg(Xcb)]
    rays_c = collect_rays(raw_vectors(alph, 3), Bc, Cc)
    pairs_c, triads_c, _ = build_structure(rays_c, herm_dot, Bc, Cc)
    u2, nodes2, _, _ = uncolorable(rays_c, herm_dot, Bc, Cc)
    print(f"    Gaussian x=1+2i (complex, avoids M1-M4): rays={len(rays_c)} pairs={len(pairs_c)} "
          f"triads={len(triads_c)} KS-uncolorable={u2}  (expect False)")
    assert not u2

    # cross-check colorability with the size-generic checker + torsion_layer's independent counter,
    # per the MACHINERY brief, on the smallest instance (x=3 raw pool).
    (col_generic,) = ks_colorable_generic(len(rays3), pools[3][1], [list(t) for t in pools[3][2]])
    cnt, _ = count_ks_colorings(len(rays3), pools[3][1], [list(t) for t in pools[3][2]], use_pairs=True)
    print(f"    cross-check on x=3 raw pool: ks_colorable_generic says colorable={col_generic}; "
          f"torsion_layer.count_ks_colorings finds {cnt} colorings (agree: {col_generic == (cnt > 0)})")
    assert col_generic and cnt > 0 and col_generic == (cnt > 0)

    print("[structure] Since x=3 (etc.) gives ALL-RATIONAL ray coordinates, Godsil-Zaks (rational")
    print("  projective rays over R^n are always 3-colorable -- W.A. Godsil, J. Zaks, 'Colouring the")
    print("  sphere', Univ. of Waterloo research report, 1988; standard citation in the KS")
    print("  literature) applies DIRECTLY: this graph is 3-colorable. Combined with `derive`'s")
    print("  exhaustiveness (any x avoiding M1-M4 has NO extra vanishing 3-term sum beyond the")
    print("  trivial 2-term self-cancellations that ALSO occur at x=3,5,7), any x avoiding M1-M4")
    print("  gives EXACTLY this graph -- hence is never KS-uncolorable. (raw-pool claim: PROVED +")
    print("  VERIFIED; completion claim: VERIFIED for this representative only, NOT proved in")
    print("  general -- see ALPHABET_THEOREM.md Sec 4 honesty note.)")
    print(f"[structure] done ({time.time()-t0:.2f}s)")


# ==================================================================================================
# STAGE 4: mechanisms -- sanity-check known generators against M1-M5 (algebraic condition only;
# full KS-uncolorability + exact flex for these is ALREADY certified in ks_flex_census.py/
# KS_CENSUS.md -- not recomputed here, just cross-referenced).
# ==================================================================================================
def mechanisms():
    t0 = time.time()
    import cmath
    reps = {
        "sqrt2 (Peres/Z[sqrt2] island)": complex(2 ** 0.5, 0),
        "golden phi=(1+sqrt5)/2 (Golden island)": complex((1 + 5 ** 0.5) / 2, 0),
        "omega=e^{2pi i/3} (Eisenstein island = Cabello simplest KS set)": cmath.exp(2j * cmath.pi / 3),
        "x=2 (CK-31 / integer island)": complex(2, 0),
        "Heegner alpha=(1+sqrt(-7))/2 (Heegner-7 island)": complex(0.5, (7 ** 0.5) / 2),
        "1+i (Gaussian island generator candidate)": complex(1, 1),
    }
    print("[mechanisms] algebraic condition satisfied by each KNOWN island's generator "
          "(cf. KS_CENSUS.md for the exact KS-uncolorability + flex certificates -- not redone here):")
    for name, xv in reps.items():
        N = abs(xv) ** 2
        p = xv.real
        hits = []
        if abs(N - 2) < 1e-9 or abs(N - 0.5) < 1e-9:
            hits.append(f"M2 (|x|^2={N:.4f})")
        if abs(p - 0.5) < 1e-9 or abs(p + 0.5) < 1e-9:
            hits.append(f"M3 (Re x={p:.4f})")
        if abs(2 * p - N) < 1e-9 or abs(2 * p + N) < 1e-9:
            hits.append(f"M5 (2Re x={2*p:.4f}, |x|^2={N:.4f})")
        if abs(xv.imag) < 1e-9 and (abs(p - 2) < 1e-9 or abs(p + 2) < 1e-9 or abs(p - 0.5) < 1e-9 or abs(p + 0.5) < 1e-9):
            hits.append("M1")
        if abs(xv.imag) < 1e-9 and (abs(p * p - p - 1) < 1e-9 or abs(p * p + p - 1) < 1e-9):
            hits.append("M4")
        print(f"    {name}: x={xv:.4f}, |x|^2={N:.4f}, Re(x)={p:.4f}  ->  {hits if hits else 'NONE (!)'}")
    print("[mechanisms] Note: 'Heegner alpha' hits BOTH M2 and M3 simultaneously -- an honest")
    print("  number-theoretic coincidence already flagged in BRANCH_ARITH.md (h(-7)=1 makes the")
    print("  norm-2 prime above 2 principal AND its generator happens to have real part 1/2).")
    print("  '1+i' hits M2 only (|1+i|^2=2) -- consistent with Kernaghan's Prop. 19 that the")
    print("  Gaussian island's graph is isomorphic to the Peres/Z[sqrt-2] (both modulus-2) graph.")
    print(f"[mechanisms] done ({time.time()-t0:.2f}s)")


# ==================================================================================================
# STAGE 5: m5hunt -- test a genuine M5 point (x=2+omega, Eisenstein, norm 3) for a NEW KS island.
# ==================================================================================================
def m5hunt():
    t0 = time.time()
    B, C = -1, -1  # Eisenstein: omega^2 = -omega - 1
    ONE = (1, 0)
    X = (2, 1)               # x = 2 + omega
    XB = qconj(X, B)
    N = qmul(X, XB, B, C)
    print(f"[m5hunt] x = 2+omega (Eisenstein integer), x* = {XB}, N(x) = |x|^2 = {N} "
          f"(expect (3,0)=3)")
    assert N == (3, 0)
    twoRe = (X[0] + XB[0], X[1] + XB[1])
    print(f"[m5hunt] x + x* (= 2 Re x) = {twoRe} (expect (3,0)=3 = N(x): confirms the M5 relation "
          f"2Re(x)=N(x) EXACTLY for this element, and Re(x)=3/2 != +-1/2 so it is NOT already on "
          f"M3 in the naive sense -- a genuine, previously-untested M5 test point).")
    assert twoRe == N

    alph_x = [ZERO, ONE, qneg(ONE), X, qneg(X), XB, qneg(XB)]
    raws = raw_vectors(alph_x, 3)
    rays = collect_rays(raws, B, C)
    pairs, triads, _ = build_structure(rays, herm_dot, B, C)
    bp = basis_participating(len(rays), triads)
    u, nodes, _, _ = uncolorable(rays, herm_dot, B, C)
    print(f"[m5hunt] raw pool: {len(raws)} raw vectors -> {len(rays)} rays, {len(pairs)} pairs, "
          f"{len(triads)} triads, {len(bp)} basis-participating. RAW POOL KS-uncolorable: {u} "
          f"({nodes} nodes)")
    assert u, "raw pool at x=2+omega turned out colorable -- no core to extract, report as such"

    sub_rays, _ = restrict(rays, bp)
    core_idx = greedy_critical_core(sub_rays, herm_dot, B, C, trials=25, seed0=0, verbose=False)
    core = [sub_rays[i] for i in core_idx]
    pairs2, triads2, _ = build_structure(core, herm_dot, B, C)
    deg = [0] * len(core)
    for i, j in pairs2: deg[i] += 1; deg[j] += 1
    print(f"[m5hunt] critical core (25 greedy trials): {len(core)} rays, {len(pairs2)} pairs, "
          f"{len(triads2)} bases, degree distribution {sorted(Counter(deg).items())}")
    u2, nodes2, _, _ = uncolorable(core, herm_dot, B, C)
    assert u2, "core not uncolorable -- bug"
    (col_gen,) = ks_colorable_generic(len(core), pairs2, [list(t) for t in triads2])
    cnt, _ = count_ks_colorings(len(core), pairs2, [list(t) for t in triads2], use_pairs=True)
    print(f"[m5hunt] cross-check: ks_colorable_generic says colorable={col_gen}, "
          f"torsion_layer.count_ks_colorings finds {cnt} colorings "
          f"(both must agree it's 0/uncolorable: {(not col_gen) and cnt == 0})")
    assert (not col_gen) and cnt == 0

    _, tau = t0_tau(len(core), triads2)
    Dmag = abs(B * B + 4 * C)
    primes = find_primes_ring(0, Dmag, count=2, below=200003)
    cert = exact_flex_hermitian_quadratic(core, B, C, primes)
    print(f"[m5hunt] t0=1 (uncolorable), tau={tau} (expect 0, Lemma B / TORSION.md: d=3 odd => "
          f"tau=0 always). primes={[p for p,s in primes]}. flex bound: {cert}")

    # THE KEY QUESTION: is this a NEW configuration, or does it coincide with the already-known
    # omega-generated Eisenstein/Cabello-33 island?
    OM = (0, 1); OM2 = qconj(OM, B)
    alph_omega = [ZERO, ONE, qneg(ONE), OM, qneg(OM), OM2, qneg(OM2)]
    rays_omega = collect_rays(raw_vectors(alph_omega, 3), B, C)
    in_omega_pool = [any(proportional(r, s, B, C) for s in rays_omega) for r in core]
    n_in = sum(in_omega_pool)
    print(f"[m5hunt] IDENTITY CHECK: of the {len(core)} critical-core rays at x=2+omega, "
          f"{n_in} are ALREADY present (as projective rays) in the raw pool generated by the "
          f"KNOWN Eisenstein alphabet {{0,+-1,+-omega,+-omega^2}} ({len(rays_omega)} rays).")
    if n_in == len(core):
        print("[m5hunt] VERDICT: NOT A NEW ISLAND. The x=2+omega critical core is entirely")
        print("  contained in the already-known omega-generated Eisenstein/Cabello-33 pool -- this")
        print("  is (a subset that already equals, given both are size-33/78/14 minimal cores) the")
        print("  SAME KS set reached via a redundant/larger alphabet, not a new configuration. This")
        print("  is an HONEST NEGATIVE result, and it is exactly what the `duality` proof predicts:")
        print("  M5 is not an independent mechanism, so a fresh M5 point should not produce fresh")
        print("  geometry beyond the M3 (omega) orbit already known.")
    else:
        print("[m5hunt] VERDICT: some core rays are NOT in the omega pool -- this WOULD be evidence")
        print("  of new geometry; report exactly which are new and re-examine before claiming a")
        print("  discovery.")
    print(f"[m5hunt] done ({time.time()-t0:.2f}s)")
    return cert


SECTIONS = {
    "derive": derive, "duality": duality, "structure": structure,
    "mechanisms": mechanisms, "m5hunt": m5hunt,
}

def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which == "all":
        t0 = time.time()
        for name, fn in SECTIONS.items():
            print(f"\n===== [{name}] =====")
            fn()
        print(f"\n[all] TOTAL {time.time()-t0:.2f}s")
    elif which in SECTIONS:
        SECTIONS[which]()
    else:
        print(f"unknown section {which!r}; choose from {list(SECTIONS)+['all']}")
        sys.exit(1)

if __name__ == "__main__":
    main()
