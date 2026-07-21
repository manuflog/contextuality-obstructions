#!/usr/bin/env python3
"""
alphabet_d4.py -- extending ALPHABET_THEOREM.md (the d=3 exhaustive classification of
vanishing-sum "mechanisms" for two-symbol alphabets A = {0,+-1,+-x} in C^3, giving M1..M4/M3-M5)
to d=4. See ALPHABET_D4.md for the full writeup; this file is the reproducible spine.

CONTEXT (read ALPHABET_THEOREM.md first): in d=3, an orthogonality relation <u,v>=0 has at most
3 nonzero coordinate-products, each in the TERM ALPHABET {0,+-1,+-x,+-x*,+-N} (N:=|x|^2). The
exhaustive 3-term classification gives exactly 4 independent mechanism families (M1 rational, M2
modulus-2, M3/M5 phase, M4 golden), and "avoid all of them" => the raw pool is always the SAME
3-colorable graph => never KS-uncolorable.

WHAT'S DIFFERENT IN d=4: an orthogonality relation now has UP TO 4 nonzero coordinate-products
(d=4 ambient slots), all still drawn from the SAME term alphabet {0,+-1,+-x,+-x*,+-N} -- Stage 0
below (`termlemma`) re-verifies this membership claim exhaustively and confirms x^2 NEVER appears
(each term is conj(u_c)*v_c with u_c,v_c in {0,+-1,+-x}, so conj(u_c) in {0,+-1,+-x*}; the product
of one un-conjugated and one conjugated alphabet symbol never produces x*x=... wait it DOES
produce x*x=N, but never x*x (un-conjugated product) since only ONE of the two factors is ever
conjugated). The 1-, 2-, 3-nonzero-term relations are already the complete d=3 story (M1-M5); what
is NEW for d=4 is the 4-nonzero-term relations, enumerated exhaustively in Stage 1 (`derive4`).

THE KEY STRUCTURAL DIFFERENCE FROM d=3 (this is the headline finding, not a computational
curiosity): Peres' own d=4 KS set (Peres-24, and its 18-ray subset CEG-18) uses ONLY entries in
{0,+-1} -- no irrational symbol at all. Since {0,+-1} subset {0,+-1,+-x} for EVERY x, the raw pool
of EVERY two-symbol d=4 alphabet {0,+-1,+-x} contains, as a sub-multiset of rays, the entire
Peres-24 ray set -- hence is KS-uncolorable for literally every x, mechanism or no mechanism. So
the d=3 theorem's SHAPE ("avoid M1-M4 => provably colorable") cannot even be stated in d=4: there
is no colorable regime to fall into. Stage 2 (`anchor`) verifies this containment computationally
and states the corrected theorem shape. The interesting question becomes: does a given x add
GENUINELY NEW structure -- a critical KS-uncolorable core that actually uses an x-coordinate ray,
as opposed to just re-deriving a sub-copy of Peres-24/CEG-18 -- and does the mechanism it satisfies
predict this? Stages `pool_*`/`core_*` (sqrt2, sqrt3, golden, omega, generic x=5) test this per
representative, per the MACHINERY brief, reusing ks_flex_census.py's d=4 machinery
(build_structure_d / uncolorable_d / greedy_critical_core_d / ks_colorable_generic) UNMODIFIED.

HONESTY SCOPE: every "critical core" reported by `core_*` below is CRITICAL (independently
re-verified: every single-ray deletion is checked to restore colorability) but found by a
GREEDY randomized peel with a bounded trial count (dictated by the ~45s/stage wall-clock budget
after live timing calibration; each greedy_critical_core_d trial on a ~270-400 ray d=4
basis-participating pool costs 5-15s depending on the ring) -- so a smaller true minimum core is
NOT ruled out, exactly the same caveat this repo's other greedy-core sections (e.g.
ks_flex_census.py's own d4sqrt2 target, m5hunt in alphabet_theorem.py) already carry and disclose.

Usage (each stage independently <45s; heavy pool/core stages are per-mechanism, mirroring
ks_flex_census.py's own pool_X/core_X staging pattern for the identical wall-clock reason):
    python3 alphabet_d4.py termlemma      # term-alphabet lemma, exhaustive, ~1s
    python3 alphabet_d4.py derive4        # exhaustive 4-term vanishing-sum classification, ~1s
    python3 alphabet_d4.py duality4       # x<->1/x duality LEMMA for the general 1-term family, ~1s
    python3 alphabet_d4.py anchor         # Peres-24 containment => corrected theorem shape, ~1s
    python3 alphabet_d4.py pool_sqrt2 / core_sqrt2      # ~1s / ~30s
    python3 alphabet_d4.py pool_sqrt3 / core_sqrt3      # ~1s / ~30s  (THE flagged new mechanism)
    python3 alphabet_d4.py pool_golden / core_golden    # ~1s / ~30s
    python3 alphabet_d4.py pool_omega / core_omega      # ~1s / ~30s (complex ring, fewer trials)
    python3 alphabet_d4.py pool_x5 / core_x5            # ~1s / ~30s (generic rational control)
    python3 alphabet_d4.py sqrt3_flex     # exact flex certificate on the cached sqrt3 core, ~5s
    python3 alphabet_d4.py all            # the four cheap analytical stages only (~3s total)
No existing file is modified; only sympy and ks_flex_census.py / exact_rigidity.py (imported,
unmodified) are used as machinery.
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from itertools import combinations_with_replacement, product as iproduct
from fractions import Fraction as Fr

import sympy as sp

from ks_flex_census import (
    qmul, qneg, qconj, qz, ZERO, herm_dot, bil_dot, raw_vectors, collect_rays,
    build_structure_d, uncolorable_d, basis_participating, restrict, greedy_critical_core_d,
    exact_flex_real_quadratic, find_primes_ring, t0_tau, proportional, ks_colorable_generic,
    cache_save, cache_load,
)
from exact_rigidity import integer_rays_peres24

# ==================================================================================================
# STAGE 0: termlemma -- exhaustive check that conj(u_c)*v_c, u_c,v_c in {0,+-1,+-x}, always lands
# in {0,+-1,+-x,+-x*,+-N} and NEVER produces x^2 (or x*^2).
# ==================================================================================================
def termlemma():
    t0 = time.time()
    p, q = sp.symbols('p q', real=True)
    I = sp.I
    x, xb, N = p + I * q, p - I * q, p ** 2 + q ** 2
    symbols = {'0': sp.Integer(0), '1': sp.Integer(1), '-1': sp.Integer(-1), 'x': x, '-x': -x}
    target_set = {'0': sp.Integer(0), '1': sp.Integer(1), '-1': sp.Integer(-1),
                  'x': x, '-x': -x, 'x*': xb, '-x*': -xb, 'N': N, '-N': -N}
    print("[termlemma] Checking every conj(u_c)*v_c for u_c,v_c in {0,+-1,+-x} (5x5=25 ordered")
    print("  pairs) lands in the term alphabet {0,+-1,+-x,+-x*,+-N} and that x^2 / x*^2 NEVER")
    print("  appear (the reason: the product always has AT MOST ONE conjugated factor -- conj(u_c)")
    print("  in {0,+-1,+-x*}, v_c UNCONJUGATED in {0,+-1,+-x} -- so an x*x combination gives N, but")
    print("  an x*x* or x*x (both un-conjugated) combination NEVER occurs structurally):")
    seen = set()
    bad = []
    for name_u, u_c in symbols.items():
        conj_u = sp.conjugate(u_c) if name_u not in ('x', '-x') else (xb if name_u == 'x' else -xb)
        for name_v, v_c in symbols.items():
            term = sp.expand(conj_u * v_c)
            matched = None
            for tname, tval in target_set.items():
                if sp.simplify(term - tval) == 0:
                    matched = tname
                    break
            seen.add(matched if matched else f"UNMATCHED[{sp.simplify(term)}]")
            if matched is None:
                bad.append((name_u, name_v, term))
            # explicit x^2 check
            has_x2 = term.has(x ** 2) or (sp.expand(term - p ** 2 + q ** 2 - 2 * I * p * q) == 0 and name_u in ('x', '-x'))
    print(f"[termlemma] 25 ordered pairs checked; distinct term-values produced: {sorted(seen)}")
    print(f"[termlemma] UNMATCHED (would indicate a term outside {{0,+-1,+-x,+-x*,+-N}}): {bad}")
    assert not bad, "found a term outside the claimed alphabet -- STOP, lemma is false"
    # separately: explicitly confirm x*x (BOTH un-conjugated) is never among the 25 computed terms,
    # i.e. x^2 is not reachable, by checking none of the 25 raw products symbolically equals x**2
    # (as an expression in p,q) unless it also equals a term already in target_set (it does not,
    # generically, since x^2 = p^2-q^2+2ipq != N = p^2+q^2 for q!=0).
    x2 = sp.expand(x * x)
    is_x2_new = sp.simplify(x2 - N) != 0 and sp.simplify(x2 - x) != 0 and sp.simplify(x2) != 0
    print(f"[termlemma] sanity: x^2 = {sp.expand(x2)} is symbolically DISTINCT from every term in "
          f"the alphabet (checked != N, != x, != 0): {is_x2_new}. It never arose above because no "
          f"un-conjugated pair (x,x) is ever multiplied by this construction (only conj(u)*v).")
    print(f"[termlemma] PROVED (exhaustive, sympy, 25/25 pairs): the term alphabet for ANY <u,v> "
          f"with u,v in {{0,+-1,+-x}}^d (any d) is EXACTLY {{0,+-1,+-x,+-x*,+-N}} -- independent of "
          f"d. This is what makes the d=3 M1-M5 vanishing-sum classification reusable verbatim as "
          f"the complete <=3-term story in d=4, with only the NEW 4-term case (`derive4`) to add.")
    print(f"[termlemma] done ({time.time()-t0:.2f}s)")


# ==================================================================================================
# STAGE 1: derive4 -- exhaustive classification of 4-nonzero-term vanishing sums over
# {1,x,x*,N} (with signs). Every such sum, after separating real/imaginary parts, reduces EXACTLY
# to:   a1 + aRx*Re(x) + aN*|x|^2 = 0     (real part; a1,aRx,aN integers, |.|<=4)
#       aIx * Im(x) = 0                    (imaginary part; aIx integer)
# -- this is proved by direct computation below (no sympy solve needed; Re(1)=1,Re(x)=Re(x*)=p,
# Re(N)=N is exact by construction, Im(1)=Im(N)=0, Im(x)=q, Im(x*)=-q), and is EXACT (integer
# coefficients, sympy Rational/Fraction throughout -- no floats anywhere in this stage).
# ==================================================================================================
def _classify_point(p):
    """A single isolated real point x=p (q=0). Returns a canonical description string,
       filtering out the degenerate x in {0,+-1} (not a genuinely new alphabet symbol)."""
    p = sp.Rational(p)
    if p in (0, 1, -1):
        return "DEGENERATE(x in {0,+-1}, excluded -- alphabet needs a genuinely new symbol)"
    if p in (2, -2, sp.Rational(1, 2), sp.Rational(-1, 2)):
        return f"M1: x = {p}"
    return f"NEW(rational): x = {p}"


def _solve_real_quadratic(aN, aRx, a1):
    """aN*p^2 + aRx*p + a1 = 0 (q=0 forced). Returns a description string, or None if
       identically-true (0=0) or a list of (kind, value) for IMPOSSIBLE / roots."""
    aN, aRx, a1 = sp.Integer(aN), sp.Integer(aRx), sp.Integer(a1)
    if aN == 0 and aRx == 0:
        return "TRIVIAL(always true, any real x)" if a1 == 0 else "IMPOSSIBLE(no real x)"
    if aN == 0:
        p = -a1 / aRx
        return [p]
    disc = aRx ** 2 - 4 * aN * a1
    if disc < 0:
        return "IMPOSSIBLE(no real x)"
    sq = sp.sqrt(disc)
    if disc == 0:
        return [-aRx / (2 * aN)]
    if sq.is_rational:
        return [(-aRx + sq) / (2 * aN), (-aRx - sq) / (2 * aN)]
    return ("QUADRATIC_IRRATIONAL", aN, aRx, a1, disc)


def _describe_family_A(aN, aRx, a1):
    """q free (Im x unconstrained by this relation): a1 + aRx*p + aN*(p^2+q^2) = 0.
       Returns a canonical, human-readable description string. The input triple is first reduced
       by its gcd and sign-normalized (first nonzero entry of (a1,aRx,aN) made positive) so that
       two witnesses describing the SAME geometric locus via differently-scaled coefficients
       (e.g. (2,2,0) and (1,1,0), both "N+Re(x)=0") always produce the IDENTICAL description
       string -- this is what makes string-equality dedup (in derive4's `found` dict and in
       duality4's dual-lookup) correct instead of accidentally fragmenting one mechanism into
       several differently-scaled entries."""
    from math import gcd as _gcd
    aNi, aRxi, a1i = int(aN), int(aRx), int(a1)
    g = _gcd(_gcd(abs(aNi), abs(aRxi)), abs(a1i)) or 1
    aNi, aRxi, a1i = aNi // g, aRxi // g, a1i // g
    first_nz = next((v for v in (a1i, aRxi, aNi) if v != 0), 0)
    if first_nz < 0:
        aNi, aRxi, a1i = -aNi, -aRxi, -a1i
    aN, aRx, a1 = sp.Integer(aNi), sp.Integer(aRxi), sp.Integer(a1i)
    if aN == 0 and aRx == 0:
        return "TRIVIAL(always true, any x)" if a1 == 0 else "IMPOSSIBLE(no x, any Im)"
    if aN == 0:
        p = -a1 / aRx
        if p in (sp.Rational(1, 2), sp.Rational(-1, 2)):
            return f"M3: Re(x) = {p}"
        return f"NEW(constant-Re): Re(x) = {p}"
    # circle form: (p + aRx/(2aN))^2 + q^2 = R2
    c = -aRx / (2 * aN)
    R2 = c ** 2 - a1 / aN
    if R2 < 0:
        return "IMPOSSIBLE(no x, negative radius^2)"
    if R2 == 0:
        # degenerates to the single point x=c (q=0) -- reuse the isolated-point classifier
        # (this is how N=0 (x=0, DEGENERATE) and x=+-1-on-a-zero-radius-circle get filtered).
        return _classify_point(c)
    if c == 0:
        # pure modulus
        if R2 in (2, sp.Rational(1, 2)):
            return f"M2: |x|^2 = {R2}"
        return f"NEW(modulus): |x|^2 = {R2}"
    # off-center circle: check against M5 (2Re(x) = +- N, i.e. aN=+-1, aRx=-+2, a1=0)
    if a1 == 0 and abs(aN) == 1 and abs(aRx) == 2 and aRx == -2 * aN:
        eps = -aN
        return f"M5: 2Re(x) = {eps}*|x|^2  (center {c}, R^2={R2})"
    return f"NEW(circle): {aN}*|x|^2 + {aRx}*Re(x) + {a1} = 0  (center Re(x)={c}, R^2={R2})"


def derive4():
    t0 = time.time()
    type_names = ['1', 'x', 'x*', 'N']
    n_cases = 0
    found = {}   # description -> list of (combo, signs) example witnesses
    for combo in combinations_with_replacement(type_names, 4):
        for signs in iproduct([1, -1], repeat=3):
            s = (1,) + signs
            n_cases += 1
            a1 = sum(si for si, t in zip(s, combo) if t == '1')
            ax = sum(si for si, t in zip(s, combo) if t == 'x')
            axs = sum(si for si, t in zip(s, combo) if t == 'x*')
            aN = sum(si for si, t in zip(s, combo) if t == 'N')
            aRx, aIx = ax + axs, ax - axs

            if aIx != 0:
                res = _solve_real_quadratic(aN, aRx, a1)
                if res == "TRIVIAL(always true, any real x)":
                    key = "TRIVIAL: holds automatically whenever x already real (Im x forced 0, no other condition)"
                elif res == "IMPOSSIBLE(no real x)":
                    key = "IMPOSSIBLE"
                elif isinstance(res, tuple) and res[0] == "QUADRATIC_IRRATIONAL":
                    _, aNv, aRxv, a1v, disc = res
                    # normalize to monic x^2 + (aRx/aN) x + (a1/aN) = 0, check golden family
                    A, B_ = sp.Rational(aRxv, aNv), sp.Rational(a1v, aNv)
                    if (A, B_) in ((-1, -1), (1, -1)):
                        key = f"M4: x real, x^2 {'+' if A>0 else '-'} x - 1 = 0  (golden family)"
                    else:
                        key = f"NEW(real quadratic irrational): x real, x^2 + ({A})x + ({B_}) = 0  (disc={disc})"
                else:
                    # list of rational roots -- classify each via the shared point classifier
                    vals = sorted(set(res))
                    labels = sorted(set(_classify_point(v) for v in vals))
                    non_degenerate = [l for l in labels if not l.startswith("DEGENERATE")]
                    key = " & ".join(non_degenerate) if non_degenerate else labels[0]
            else:
                key = _describe_family_A(aN, aRx, a1)

            found.setdefault(key, []).append((combo, s, (a1, aRx, aIx, aN)))

    print(f"[derive4] {n_cases} (type-multiset, sign-pattern) cases enumerated "
          f"(C(4+4-1,4)=35 multisets x 2^3=8 sign patterns = 280).")
    buckets = {"TRIVIAL": [], "IMPOSSIBLE": [], "DEGENERATE": [], "M": [], "NEW": []}
    for k in found:
        if k.startswith("TRIVIAL"):
            buckets["TRIVIAL"].append(k)
        elif k.startswith("IMPOSSIBLE"):
            buckets["IMPOSSIBLE"].append(k)
        elif k.startswith("DEGENERATE"):
            buckets["DEGENERATE"].append(k)
        elif k.startswith("M") or " & " in k:
            buckets["M"].append(k)
        else:
            buckets["NEW"].append(k)
    print(f"[derive4] {sum(len(found[k]) for k in buckets['TRIVIAL'])} case(s) TRIVIAL (already-known "
          f"d=3-style 2+2 self-cancellation, no condition on x)")
    print(f"[derive4] {sum(len(found[k]) for k in buckets['IMPOSSIBLE'])} case(s) IMPOSSIBLE for any x")
    print(f"[derive4] {sum(len(found[k]) for k in buckets['DEGENERATE'])} case(s) DEGENERATE "
          f"(force x in {{0,+-1}}, not a genuinely new symbol)")
    print(f"[derive4] {len(buckets['M'])} distinct conditions RECOGNIZED as already-known d=3 "
          f"mechanisms M1-M5:")
    for k in sorted(buckets["M"]):
        print(f"    {k}   <- {len(found[k])} case(s), e.g. {found[k][0][:2]}")
    print(f"[derive4] {len(buckets['NEW'])} distinct GENUINELY NEW conditions (not expressible via "
          f"a <=3-term sum, i.e. not in M1-M5) found among the 4-term sums:")
    for k in sorted(buckets["NEW"]):
        print(f"    {k}   <- {len(found[k])} case(s), e.g. {found[k][0][:2]}")
    print(f"[derive4] EXHAUSTIVE: every one of the 280 cases is TRIVIAL, IMPOSSIBLE, DEGENERATE, a "
          f"known M1-M5 mechanism, or one of the {len(buckets['NEW'])} new conditions listed above. "
          f"({time.time()-t0:.2f}s)")
    return found


# ==================================================================================================
# STAGE 2: duality4 -- the x<->1/x gauge symmetry (same argument as ALPHABET_THEOREM.md Sec.2,
# d-independent) acting on the GENERAL 1-parameter-family form a1 + aRx*Re(x) + aN*|x|^2 = 0:
# PROVES the coefficient triple (a1,aRx,aN) maps to (aN,aRx,a1) under x -> y=1/x -- i.e. "swap the
# constant and modulus coefficients, keep the Re(x) coefficient" -- which recovers M2<->M2 (2<->1/2)
# and M3<->M5 EXACTLY as special cases, and now classifies every NEW d=4 mechanism found in
# `derive4` by the same rule (self-dual, or paired with another entry in the list).
# ==================================================================================================
def duality4():
    t0 = time.time()
    a1, aRx, aN = sp.symbols('a1 aRx aN', integer=True)
    p, q = sp.symbols('p q', real=True, positive=False)
    N = p ** 2 + q ** 2
    lhs = a1 + aRx * p + aN * N
    # y = 1/x = xbar/N(x); Re(y) = p/N; N(y) = 1/N.  Substitute the DUAL relation
    # aN + aRx*Re(y) + a1*N(y) = 0  and multiply through by N(x) (nonzero for x!=0):
    dual_lhs_times_N = sp.simplify((aN + aRx * (p / N) + a1 * (1 / N)) * N)
    print("[duality4] LEMMA: for x != 0, x satisfies  a1 + aRx*Re(x) + aN*|x|^2 = 0  IFF  y=1/x "
          "satisfies  aN + aRx*Re(y) + a1*|y|^2 = 0  (i.e. the coefficient triple maps "
          "(a1,aRx,aN) -> (aN,aRx,a1) under x -> 1/x).")
    print(f"  Proof: multiply the DUAL relation by N(x) (nonzero): aN*N(x) + aRx*Re(y)*N(x) + "
          f"a1*N(y)*N(x). Using Re(y)=Re(x)/N(x) and N(y)=1/N(x): = aN*N(x) + aRx*Re(x) + a1.")
    print(f"  sympy check: aN*N(x) + aRx*Re(y)*N(x) + a1*N(y)*N(x) simplifies to "
          f"{dual_lhs_times_N} == aN*(p^2+q^2) + aRx*p + a1 (the ORIGINAL relation with a1,aN "
          f"swapped) -- by inspection these are literally the same polynomial in a1,aRx,aN,p,q.")
    check = sp.simplify(dual_lhs_times_N - (aN * N + aRx * p + a1))
    print(f"  exact difference (should be 0 identically): {check}")
    assert check == 0, "duality4 coefficient-swap lemma failed -- STOP"
    print("[duality4] PROVED (exact, no floats): x<->1/x acts on the coefficient triple by "
          "(a1,aRx,aN) -> (aN,aRx,a1). Recovers, as special cases already proved in "
          "ALPHABET_THEOREM.md Sec.2: M2 (2,0,-1)<->(-1,0,2) i.e. N=2<->N=1/2; M3 (mp1,+-2,0) <-> "
          "M5 (0,+-2,mp1) i.e. Re(x)=+-1/2 <-> 2Re(x)=-+N. Applying the SAME swap to every NEW "
          "condition found in `derive4`:")
    # ---- exhaustive application: pull every distinct NEW/M condition found by derive4() (which
    # already stores the exact integer coefficient triple (a1,aRx,aIx,aN) per witness), canonicalize
    # each triple up to an overall nonzero scalar (divide by gcd, fix sign of first nonzero entry),
    # apply the swap, and re-run it through the SAME classifier (_describe_family_A/_classify_point)
    # used by derive4 to name the dual condition -- this is a genuine independent cross-check, not
    # just a coefficient bookkeeping exercise: if the swap rule or the classifier had a bug, the
    # dual's re-derived name would not match a condition already present in the derive4 census.
    def canon(a1v, aRxv, aNv):
        from math import gcd
        g = gcd(gcd(abs(a1v), abs(aRxv)), abs(aNv)) or 1
        t = (a1v // g, aRxv // g, aNv // g)
        first_nonzero = next((v for v in t if v != 0), 0)
        return tuple(-v for v in t) if first_nonzero < 0 else t

    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        found = derive4()   # re-run derive4's fast (<0.01s) exhaustive scan to get its census dict
    print("\n[duality4] Applying the swap EXHAUSTIVELY to every distinct Family-A (q-free: circle / "
          "modulus / constant-Re) condition `derive4` found -- re-classifying each dual through the "
          "IDENTICAL classifier code path (`_describe_family_A`) used by derive4 itself, which is an "
          "independent cross-check: if the swap rule or the classifier disagreed, the dual's "
          "re-derived name would not land back in derive4's own census.")
    seen_pairs = set()
    for key, witnesses in sorted(found.items()):
        if not (key.startswith("NEW") or key.startswith("M2") or key.startswith("M3") or key.startswith("M5")):
            continue
        a1v, aRxv, aIxv, aNv = witnesses[0][2]
        if aIxv != 0:
            continue  # this stage covers the Family-A (q-free) mechanisms, where the swap rule
                       # gives the richest new pairing structure; M1/M4 real-branch self-duality is
                       # already proved by direct root substitution in ALPHABET_THEOREM.md Sec.2
                       # (unchanged, d-independent) and re-checked qualitatively there, not repeated.
        t0v = canon(a1v, aRxv, aNv)
        dual_t = canon(aNv, aRxv, a1v)
        dual_key = _describe_family_A(dual_t[2], dual_t[1], dual_t[0])
        pair = tuple(sorted([key, dual_key]))
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)
        found_dual_in_census = dual_key in found
        selfdual = (key == dual_key)
        tag = "SELF-DUAL" if selfdual else ("PAIRED (both in census)" if found_dual_in_census else
                                             "PAIRED (dual not among the 280 witnesses -- outside the direct 4-term list, but still a valid consequence of the swap lemma)")
        print(f"    {key}\n      (a1,aRx,aN)={t0v} -> dual (a1,aRx,aN)={dual_t} -> {dual_key}   [{tag}]")
    print(f"[duality4] done ({time.time()-t0:.2f}s)")


# ==================================================================================================
# STAGE 3: anchor -- the KEY structural fact: Peres-24's own alphabet is {0,+-1} (no x at all), so
# {0,+-1} subset {0,+-1,+-x} for EVERY x means the raw two-symbol d=4 pool ALWAYS contains Peres-24
# (as a ray sub-multiset), hence is trivially KS-uncolorable for literally every x -- there is no
# "avoid the mechanisms => colorable" regime in d=4 at all, unlike d=3. This is verified
# computationally (not just asserted) for one representative generic x.
# ==================================================================================================
def anchor():
    t0 = time.time()
    p24 = integer_rays_peres24()
    print(f"[anchor] Peres-24 (exact_rigidity.integer_rays_peres24): {len(p24)} rays, alphabet "
          f"{{0,+-1}} ONLY (entries are 0, 1, -1 -- verified by inspection of the generator: "
          f"{p24[0]}, {p24[4]}, {p24[-1]}).")
    entries = set(c for v in p24 for c in v)
    assert entries <= {0, 1, -1}, "Peres-24 uses an entry outside {0,+-1} -- anchor claim is FALSE"
    print(f"    entries used across all 24 rays: {sorted(entries)} (subset check: PASSED)")

    B, C = 0, 3   # a representative real generic x = sqrt(3); the CONTAINMENT claim is x-independent
    ONE = (1, 0)
    X = (0, 1)
    alph = [ZERO, ONE, qneg(ONE), X, qneg(X)]
    print("[anchor] Building the raw d=4 pool for a representative x=sqrt(3) (any x would do -- "
          "the containment argument below never uses a property of sqrt(3) itself, only that "
          "{0,+-1} subset {0,+-1,+-x}):")
    raws = raw_vectors(alph, 4)
    rays = collect_rays(raws, B, C)
    print(f"    raw A^4\\{{0}}: {len(raws)} vectors -> {len(rays)} distinct rays")
    p24_pairs = [tuple((c, 0) for c in v) for v in p24]
    contained = [any(proportional(pr, r, B, C) for r in rays) for pr in p24_pairs]
    n_in = sum(contained)
    print(f"    CONTAINMENT CHECK: {n_in} of {len(p24)} Peres-24 rays found (exact proportionality "
          f"in the ring Z[sqrt3]) inside the x=sqrt(3) raw pool.")
    assert n_in == len(p24), "Peres-24 is NOT fully contained -- containment claim is FALSE, STOP"
    print("    ALL 24/24 contained. Since Peres-24 alone is already KS-uncolorable (this repo's "
          "own certified fact, sic_zoo.py sec_peres24 / exact_rigidity.py; not re-derived here), "
          "and it sits inside the x=sqrt(3) raw pool as a sub-configuration, the WHOLE pool is "
          "automatically KS-uncolorable too -- no need to even run the (expensive) uncolorable_d "
          "search on the full pool to know this.")
    print("[anchor] CORRECTED THEOREM SHAPE (replacing the d=3 'avoid M1-M4 => colorable' shape, "
          "which has NO d=4 analogue): for EVERY x != 0,+-1, the raw pool of {0,+-1,+-x} in C^4 is "
          "KS-uncolorable, unconditionally -- because it always contains the mechanism-free "
          "integer island Peres-24/CEG-18. The mechanism list from `derive4` does NOT classify "
          "colorable-vs-uncolorable in d=4 (everything is uncolorable); it classifies whether x "
          "contributes GENUINELY NEW critical structure (a critical KS core using an x-coordinate "
          "ray, not reducible to a subset of the {0,+-1}-only pool) -- tested per-mechanism in the "
          "pool_*/core_* stages below.")
    print(f"[anchor] done ({time.time()-t0:.2f}s)")


# ==================================================================================================
# STAGE 4: per-mechanism pool_X / core_X -- build the d=4 raw pool for a representative x of each
# mechanism family, and greedily peel a CRITICAL KS-uncolorable core (verified critical), reporting
# its size and whether it uses any x-coordinate ray (the "genuinely new" test from `anchor`).
# Reuses ks_flex_census.py's build_structure_d / uncolorable_d / greedy_critical_core_d / restrict /
# basis_participating UNMODIFIED (the same machinery already used for the d4sqrt2 target there).
# ==================================================================================================
def _uses_x(ray):
    return any(c[1] != 0 for c in ray)

def _real_rep(name, B, C, dotfn=bil_dot):
    ONE = (1, 0); X = (0, 1)
    return [ZERO, ONE, qneg(ONE), X, qneg(X)], B, C, dotfn

REPS_REAL = {
    "sqrt2":  (0, 2),
    "sqrt3":  (0, 3),   # THE flagged new mechanism: |x|^2=3
    "golden": (1, 1),
}

def _pool_common(name, B, C, dotfn, trials, seed0):
    ONE = (1, 0); X = (0, 1)
    alph = [ZERO, ONE, qneg(ONE), X, qneg(X)]
    t0 = time.time()
    raws = raw_vectors(alph, 4)
    rays = collect_rays(raws, B, C)
    pairs, bases, adj = build_structure_d(rays, dotfn, B, C, 4)
    bp = basis_participating(len(rays), bases)
    (col,) = ks_colorable_generic(len(rays), pairs, [list(b) for b in bases])
    print(f"[{name}] raw={len(raws)} rays={len(rays)} pairs={len(pairs)} bases={len(bases)} "
          f"basis-participating={len(bp)}  KS-uncolorable={not col}  ({time.time()-t0:.2f}s)")
    assert not col, f"{name} pool unexpectedly colorable -- contradicts the anchor containment claim"
    cache_save(f"d4_{name}_pool", rays)
    cache_save(f"d4_{name}_bp", bp)
    return rays, bp

def pool_sqrt2(): return _pool_common("sqrt2", 0, 2, bil_dot, 0, 0)
def pool_sqrt3(): return _pool_common("sqrt3", 0, 3, bil_dot, 0, 0)
def pool_golden(): return _pool_common("golden", 1, 1, bil_dot, 0, 0)
def pool_x5(): return _pool_common("x5", 0, 25, bil_dot, 0, 0)   # x=5: t^2=25 (B=0,C=25) -> t=5 exactly (rational control point)

def pool_omega():
    B, C = -1, -1
    ONE = (1, 0); OM = (0, 1); OM2 = qconj(OM, B)
    alph = [ZERO, ONE, qneg(ONE), OM, qneg(OM), OM2, qneg(OM2)]
    t0 = time.time()
    raws = raw_vectors(alph, 4)
    rays = collect_rays(raws, B, C)
    pairs, bases, adj = build_structure_d(rays, herm_dot, B, C, 4)
    bp = basis_participating(len(rays), bases)
    (col,) = ks_colorable_generic(len(rays), pairs, [list(b) for b in bases])
    print(f"[omega] raw={len(raws)} rays={len(rays)} pairs={len(pairs)} bases={len(bases)} "
          f"basis-participating={len(bp)}  KS-uncolorable={not col}  ({time.time()-t0:.2f}s)")
    assert not col
    cache_save("d4_omega_pool", rays)
    cache_save("d4_omega_bp", bp)
    return rays, bp

def _core_common(name, B, C, dotfn, trials, seed0):
    rays = cache_load(f"d4_{name}_pool"); bp = cache_load(f"d4_{name}_bp")
    if rays is None or bp is None:
        rays, bp = _pool_common(name, B, C, dotfn, 0, 0)
    rays = [tuple(tuple(c) for c in v) for v in rays]
    sub, _ = restrict(rays, bp)
    t0 = time.time()
    core_idx = greedy_critical_core_d(sub, dotfn, B, C, 4, trials=trials, seed0=seed0, verbose=True)
    core = [sub[i] for i in core_idx]
    n_x = sum(1 for r in core if _uses_x(r))
    pairs, bases, _ = build_structure_d(core, dotfn, B, C, 4)
    (col,) = ks_colorable_generic(len(core), pairs, [list(b) for b in bases])
    print(f"[{name}] critical core: {len(core)} rays ({n_x} use x-coordinate, {len(core)-n_x} pure "
          f"integer), {len(pairs)} pairs, {len(bases)} bases, uncolorable={not col} "
          f"({time.time()-t0:.2f}s, {trials} greedy trials)")
    assert not col
    verdict = "GENUINELY NEW (uses x, irreducible to a {0,+-1}-only core)" if n_x > 0 else \
              "NOT new (greedy peel found a pure-integer core -- consistent with/subset of Peres-24/CEG-18)"
    size_vs_ceg18 = "SMALLER than CEG-18 (18)" if len(core) < 18 else \
                    ("EQUAL to CEG-18" if len(core) == 18 else "LARGER than CEG-18 (18)")
    print(f"[{name}] VERDICT: {verdict}; size {len(core)} is {size_vs_ceg18}.")
    cache_save(f"d4_{name}_core", core)
    return core

def core_sqrt2(): return _core_common("sqrt2", 0, 2, bil_dot, trials=6, seed0=0)
def core_sqrt3(): return _core_common("sqrt3", 0, 3, bil_dot, trials=3, seed0=0)
def core_golden(): return _core_common("golden", 1, 1, bil_dot, trials=3, seed0=0)
def core_x5(): return _core_common("x5", 0, 25, bil_dot, trials=2, seed0=0)
def core_omega(): return _core_common("omega", -1, -1, herm_dot, trials=1, seed0=0)


# ==================================================================================================
# STAGE 5: sqrt3_flex -- exact rigidity certificate on the cached sqrt3 critical core (the flagged
# |x|^2=3 mechanism), reusing exact_flex_real_quadratic UNMODIFIED (same function used for the
# Golden island in ks_flex_census.py, generic over d and over the real quadratic ring).
# ==================================================================================================
def sqrt3_flex():
    t0 = time.time()
    core = cache_load("d4_sqrt3_core")
    if core is None:
        core = core_sqrt3()
    core = [tuple(tuple(c) for c in v) for v in core]
    B, C = 0, 3
    pairs, bases, _ = build_structure_d(core, bil_dot, B, C, 4)
    _, tau = t0_tau(len(core), bases)
    print(f"[sqrt3_flex] core: {len(core)} rays, {len(pairs)} pairs, {len(bases)} bases. "
          f"t0=1 (uncolorable, verified in core_sqrt3), tau={tau} (d=4 EVEN -- Lemma B/TORSION.md "
          f"does NOT force tau=0 here, unlike the odd-d=3 case; this is a live, computed invariant).")
    Dmag = abs(B * B + 4 * C)
    primes = find_primes_ring(B, C, count=2, below=200003)
    print(f"[sqrt3_flex] mod-p primes for Q(sqrt{Dmag}): {[p for p, s in primes]}")
    cert = exact_flex_real_quadratic(core, B, C, primes)
    print(f"[sqrt3_flex] n={cert['n']}, E={cert['E']}, trivial-expected={cert['triv_expected']}, "
          f"mod-p bound on flex: 0 <= flex <= {cert['bound']}  ({time.time()-t0:.2f}s)")
    if cert["bound"] == 0:
        print("=> FLEX = 0 EXACT (over Q(sqrt3)) -- the sqrt3 d=4 critical core is RIGID.")
    else:
        print(f"=> mod-p bound is {cert['bound']} > 0 -- NOT pinned to 0 by this bound; report as "
              f"NUMERICAL/UNRESOLVED, not a rigidity certificate.")
    return cert


SECTIONS = {
    "termlemma": termlemma, "derive4": derive4, "duality4": duality4, "anchor": anchor,
    "pool_sqrt2": pool_sqrt2, "core_sqrt2": core_sqrt2,
    "pool_sqrt3": pool_sqrt3, "core_sqrt3": core_sqrt3,
    "pool_golden": pool_golden, "core_golden": core_golden,
    "pool_omega": pool_omega, "core_omega": core_omega,
    "pool_x5": pool_x5, "core_x5": core_x5,
    "sqrt3_flex": sqrt3_flex,
}
CHEAP_ALL = ["termlemma", "derive4", "duality4", "anchor"]

def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which == "all":
        t0 = time.time()
        for name in CHEAP_ALL:
            print(f"\n===== [{name}] =====")
            SECTIONS[name]()
        print(f"\n[all] (cheap analytical stages only -- run pool_X/core_X/sqrt3_flex separately, "
              f"each is its own <45s stage) TOTAL {time.time()-t0:.2f}s")
    elif which in SECTIONS:
        SECTIONS[which]()
    else:
        print(f"unknown section {which!r}; choose from {list(SECTIONS)+['all']}")
        sys.exit(1)

if __name__ == "__main__":
    main()
