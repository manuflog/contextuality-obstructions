#!/usr/bin/env python3
"""
branch_theta2.py -- BRANCH T2: restating the Theta-coherence congruence (parent item A1,
verification/INDEX.md V52) in fully-defined variables, and recovering as much of a closed
form for gamma_{t'} - gamma_t as can actually be derived and machine-verified.

============================================================================================
CONTEXT (read arxiv/rigidity/BRANCH_THETA.md first -- prior audit, not modified here)
============================================================================================
The prior branch (BRANCH_THETA.md / branch_theta.py, this program) established that the
quoted congruence

    gamma_{t'} - gamma_t == 2[ Sum_c y_c s_c + Sum_v h_v (d/g_v) t_v ]   (mod 2d)

is UNSTARTABLE AS WRITTEN: y_c and h_v have zero operational definition anywhere in the
repository (h_v exists elsewhere with a provably different meaning), and INDEX.md now
carries a correction. That conclusion is NOT revisited or re-litigated here -- it stands.

What IS fully implemented and precisely defined is verification/paired_classification_
theorem.py's star_check ([P6]): it groups per-context characters t=(C,a,b) into psi-classes
and asserts equal (label, Theta) pairs within a class. This branch:

  (1) restates star_check's own machinery -- psi-class, label z_t, phase Theta_t/gamma_t,
      s_c, g_v, t_v -- in fully precise, code-grounded notation (no invented symbols);
  (2) states precisely what [P6] verifies in that notation;
  (3) DERIVES, from the exact matrix multiplication law of the Weyl operators built in
      verification/weyl.py, a closed-form formula for gamma_t itself (hence trivially for
      any difference gamma_{t'} - gamma_t), and VERIFIES it exactly against the code's own
      r-extraction across many fresh random families at d=4,6,8 (0 mismatches target);
  (4) derives and verifies a genuine gauge-invariance law for the DIFFERENCE within a
      psi-class (does not depend on which base point lambda0 in L(F) is used);
  (5) reports, with equal honesty, a *failed* attempted further reduction (eliminating
      lambda0 in favor of only t_v, g_v data) -- caught and falsified by direct numerical
      testing, not asserted -- to show exactly where the "local, cross-context sum" shape
      of the lost y_c/h_v formula remains genuinely open.

NUMERICAL vs DERIVED vs CONJECTURE is labeled at every step. Runtime target: well under 45s.

Usage:
  python3 branch_theta2.py audit      # print the definitions/restatement only (instant)
  python3 branch_theta2.py verify     # matrix identities + family sweep (~15-30s)
  python3 branch_theta2.py all        # same as verify (default)
"""
import sys, os, time, argparse, importlib.util
import numpy as np

# ---------------------------------------------------------------------------------
# Locate verification/paired_classification_theorem.py (read-only import, no repo file
# touched or modified anywhere in this branch).
# ---------------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_CANDIDATES = [
    os.path.abspath(os.path.join(_HERE, "..", "..", "verification")),
    "/sessions/friendly-exciting-ptolemy/mnt/contextuality-obstructions/verification",
    "/Users/manuflog/Developer/contextuality-obstructions/verification",
]
VERDIR = next((p for p in _CANDIDATES if os.path.isdir(p)), None)
if VERDIR is None:
    raise SystemExit("FATAL: could not locate verification/ directory in any known "
                      f"location; tried {_CANDIDATES}")
if VERDIR not in sys.path:
    sys.path.insert(0, VERDIR)

_pct_path = os.path.join(VERDIR, "paired_classification_theorem.py")
if not os.path.isfile(_pct_path):
    raise SystemExit(f"FATAL: {_pct_path} not found -- V52 machinery not present.")
_spec = importlib.util.spec_from_file_location("pct_v52_t2", _pct_path)
pct = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pct)

from weyl import build  # same read-only import star_check itself relies on


# ============================================================================================
# PART 1 -- DEFINITIONS, restated precisely from the code (task 1). No symbol here is
# invented; every one is traced to a specific line/function.
# ============================================================================================
DEFINITIONS = """
================================================================================================
PART 1: PRECISE DEFINITIONS (all traced to code, none invented)
================================================================================================
  d              even integer; N = d^2 (two "sites", m=2 in weyl.build(d,2)).
  tau            exp(i*pi/d);  w := tau^2 = exp(2*pi*i/d)  (called "omega" in the docstring).
  X, Z           d x d clock/shift matrices: X_{ij}=1 iff i==j+1 (mod d); Z=diag(w^0..w^{d-1}).
  T(p,q)         := X^p Z^q  (depends on p,q only mod d, since X^d=Z^d=I).
  q(v)           for v=(v1,v2,v3,v4) in Z^4 (ANY integer representative):
                 q(v) := v1*v2 + v3*v4          [weyl.py: "q=sum(v[2i]*v[2i+1])"]
  W(v)           := tau^{-q(v)} * T(v1,v2) (x) T(v3,v4)     [weyl.build's W(v), always called
                 in the rest of the codebase on v in [0,d)^4]
  context C      an ordered triple (u,v,z) of observables with sform(u,v)=0 (commuting pair);
                 z is determined by the closure condition u+v+z=0 (mod d).
  s_c            Fam.S[ci]: the integer in Z_d with  prod_{w in C} W(w) = w^{s_c} * I
                 (the matrix product over a context is proved central and an omega-power;
                 [P2a] in paired_classification_theorem.py).
  g_v            Fam.gv[v] = gcd(v1,v2,v3,v4,d): [P1]'s Lemma 0 proves spec(W(v)) (the set of
                 eigen-phases k with W(v) having eigenvalue w^k) is EXACTLY one coset of
                 g_v*Z_d inside Z_d.
  t_v            a representative of that coset: spec(W(v)) = t_v + g_v*Z_d. t_v is fixed only
                 MOD g_v (any two representatives differ by a multiple of g_v); nothing in the
                 code ever names a canonical choice, matching the prior audit's "HALF-FOUND".
  L(F)           the finite set of "deterministic assignments" lambda: obs -> Z_d with
                 (i) lambda_v in spec(W(v)) for every observable v, and
                 (ii) sum_{w in C} lambda_w == s_c (mod d) for every context C.
                 (Fam._enum_L; used throughout [P4]-[P6].)
  lambda0        := L(F)[0], the FIRST solution found by the code's deterministic backtracking
                 enumeration -- a fixed but NOT canonical base point (see Part 3/4: choosing any
                 other element of L in its place is a proven gauge symmetry).
  K              := { lambda - lambda0 (mod d) : lambda in L(F) }  (the "section-translation
                 group"; star_check's docstring asserts, and [P4]/[P5] machinery uses, that this
                 is a subgroup of Z_d^obs, i.e. L(F) is an affine coset of K).
  character t    a triple (C,a,b): a context C=(u,v,*) together with (a,b) in Z_d^2\\{(0,0)}.
                 (The double loop inside star_check.)
  label z_t      := (a*u + b*v) mod d,  componentwise, in Z_d^4.
  psi_t          the FUNCTION L(F) -> Z_d,  l |-> (a*(l_u - lambda0_u) + b*(l_v - lambda0_v))
                 mod d.  Two characters t, t' are in the SAME psi-CLASS iff psi_t = psi_t' as
                 functions on L(F) (equal at every l, not just at lambda0).
  Q_t            := W(u)^a @ W(v)^b  (an ordinary matrix power/product, NOT W(a*u+b*v) computed
                 directly -- this distinction matters, see Part 3).
  Theta_t/gamma_t  [P1] + the code's own runtime assert prove Q_t = c_t * W(z_t) for a SCALAR
                 c_t with |c_t| = 1 EXACTLY (extracted as c_t = tr(W(z_t)^dagger @ Q_t) / N).
                 Define  Theta_t := c_t * w^{-(a*lambda0_u + b*lambda0_v)}
                                   = c_t * tau^{-2*(a*lambda0_u + b*lambda0_v)}.
                 [P6]'s own assert PROVES Theta_t is exactly a 2d-th root of unity:
                 Theta_t = tau^{gamma_t}, gamma_t := round(arg(Theta_t)/(pi/d)) mod 2d, verified
                 to < 1e-9 by direct comparison, not merely asserted.

================================================================================================
PART 2: WHAT [P6] (star_check) PRECISELY VERIFIES, in the notation above
================================================================================================
For every character t = (C,a,b) with (a,b) != (0,0):
  (A) LABEL EQUALITY (a THEOREM, proved elsewhere -- closure + spec-coset lemma +
      ann(K) = rowspan(constraint matrix) + Sum_v (d/g_v)*Z_d): for all t, t' with
      psi_t = psi_t',  z_t = z_t'.
  (B) PHASE EQUALITY (the OPEN lemma this file certifies per family, NUMERICALLY, not
      proved): for all t, t' with psi_t = psi_t',  gamma_t == gamma_t'  (mod 2d).
  (C) TRIVIALITY: the class psi=0 (i.e. a*k_u+b*k_v==0 for every k in K) must have
      (z,gamma) = (0,0).
[P6] PASS means (A),(B),(C) all hold for every psi-class in the family, checked by direct
construction-and-comparison (not via any closed form). (B) is exactly
"gamma_{t'} - gamma_t == 0 (mod 2d) whenever psi_t = psi_t'" -- the SAME left-hand side as
the target congruence, restricted to the (unproven-in-general) case where the difference is
claimed to vanish. The lost formula's actual content -- what gamma_{t'} - gamma_t equals for
GENERAL t, t' (not just psi-equal pairs, where it is asserted 0) -- is the open target.
"""


# ============================================================================================
# PART 3 -- THE DERIVATION. Fully worked in BRANCH_THETA2.md; reproduced here as executable
# code + numerical certification.
# ============================================================================================
def qf(v):
    v = [int(x) for x in v]
    return v[0] * v[1] + v[2] * v[3]


def E_exponent(d, u, v, a, b):
    """DERIVED (see BRANCH_THETA2.md Part 3 for the full proof): the tau-exponent E(t) with

        W(u)^a @ W(v)^b = tau^{E(t)} * W(z_t)      (z_t = (a*u+b*v) mod d)

    obtained from two exact identities proved from weyl.py's own X,Z definitions:
      (I)   W(u)^a  = tau^{q(u)*a*(a-2) + q(a*u mod d)} * W(a*u mod d)      [single-generator]
      (III) W(x)@W(y) = tau^{q(z)-q(x)-q(y)+2*(x2*y1+x4*y3)} * W(z), z=(x+y) mod d [two-gen law]
    Substituting x = (a*u mod d), y = (b*v mod d) into (III) and using (I) for W(u)^a, W(v)^b
    makes the intermediate q((a*u) mod d), q((b*v) mod d) terms CANCEL exactly, leaving:
    """
    au = (a * np.array(u)) % d
    bv = (b * np.array(v)) % d
    zt = (a * np.array(u) + b * np.array(v)) % d
    return (qf(u) * a * (a - 2) + qf(v) * b * (b - 2) + qf(zt)
            + 2 * (int(au[1]) * int(bv[0]) + int(au[3]) * int(bv[2])))


def gamma_formula(d, u, v, a, b, lam0_u, lam0_v):
    """DERIVED closed form for gamma_t (see BRANCH_THETA2.md):
         gamma_t == E(t) - 2*(a*lambda0_u + b*lambda0_v)   (mod 2d)
    Fully determined by: the standard quadratic form q (from weyl.py), the character's own
    (a,b,u,v) [hence z_t], and the TWO coordinates lambda0_u, lambda0_v of the family's base
    point at the context's own two generators. No y_c, h_v, or any other undefined symbol
    is used anywhere in this formula."""
    E = E_exponent(d, u, v, a, b)
    return (E - 2 * (a * lam0_u + b * lam0_v)) % (2 * d)


def matrix_identity_checks(rng, d_list=(4, 6, 8, 10, 12), ntrials=50):
    """Verify identities (I) and (III) directly against raw matrices (independent of the
    Fam/context/family machinery) -- this is the DERIVATION's own base case, not a
    consequence of anything the rest of the sweep does."""
    worst1 = worst3 = 0.0
    for d in d_list:
        X, Z, w, tau, W, N = build(d, 2)
        for _ in range(ntrials):
            u = tuple(int(x) for x in rng.integers(0, d, 4))
            v = tuple(int(x) for x in rng.integers(0, d, 4))
            a = int(rng.integers(0, d))
            zu = tuple((a * np.array(u)) % d)
            lhs = np.linalg.matrix_power(W(np.array(u)), a)
            E1 = qf(u) * a * (a - 2) + qf(zu)
            rhs = (tau ** (E1 % (2 * d))) * W(np.array(zu))
            worst1 = max(worst1, float(np.abs(lhs - rhs).max()))

            z = tuple((np.array(u) + np.array(v)) % d)
            lhs3 = W(np.array(u)) @ W(np.array(v))
            E3 = qf(z) - qf(u) - qf(v) + 2 * (u[1] * v[0] + u[3] * v[2])
            rhs3 = (tau ** (E3 % (2 * d))) * W(np.array(z))
            worst3 = max(worst3, float(np.abs(lhs3 - rhs3).max()))
    return worst1, worst3


def star_check_r(fam, ci, a, b, lam0):
    """Reproduce EXACTLY the r-extraction inside star_check for one character, given an
    explicit base point lam0 (so we can test other valid base points too, not just fam.L[0])."""
    d, N = fam.d, fam.N
    C = fam.CTX[ci]
    u, v = C[0], C[1]
    Q = np.linalg.matrix_power(fam.W(np.array(u)), a) @ np.linalg.matrix_power(fam.W(np.array(v)), b)
    z = tuple(int(x) for x in (a * np.array(u) + b * np.array(v)) % d)
    ph = np.trace(fam.W(np.array(z)).conj().T @ Q) / N
    assert abs(abs(ph) - 1) < 1e-9
    ang = ph * np.exp(-2j * np.pi / d * (a * lam0[u] + b * lam0[v]))
    r = int(np.round(np.angle(ang) / (np.pi / d))) % (2 * d)
    assert abs(ang - np.exp(1j * np.pi / d * r)) < 1e-9
    return r


def sweep_families(seed0=101, per_d=10, d_list=(4, 6, 8), time_budget=32.0):
    """Fresh random families (pct.rand_family) at d=4,6,8. For EVERY character in EVERY
    family: compare the code's own r (star_check_r) against gamma_formula. Also runs the
    gauge-shift and gauge-invariance-of-difference checks using TWO different valid base
    points (fam.L[0] and fam.L[1]) when |L|>=2."""
    t0 = time.time()
    total_chars = 0
    mismatches = 0
    gauge_shift_checked = 0
    gauge_shift_bad = 0
    diff_invariance_checked = 0
    diff_invariance_bad = 0
    cross_context_psi_pairs = 0
    families_done = 0
    by_d = {}
    for d in d_list:
        rng = np.random.default_rng(seed0 + d)
        made, tried = 0, 0
        d_chars = 0
        d_mismatch = 0
        while made < per_d and tried < 5 * per_d and time.time() - t0 < time_budget:
            tried += 1
            ctxs = pct.rand_family(rng, d, int(rng.integers(4, 7)), max_obs=9, scale=1)
            if ctxs is None:
                continue
            try:
                fam = pct.Fam(d, ctxs, f"t2_d{d}_{made}")
            except Exception:
                continue
            if not fam.L or len(fam.L) > 1200:
                continue
            made += 1
            families_done += 1
            lam0 = fam.L[0]
            oi = {vv: k for k, vv in enumerate(fam.obs)}
            for ci, C in enumerate(fam.CTX):
                u, v = C[0], C[1]
                for a in range(d):
                    for b in range(d):
                        if (a, b) == (0, 0):
                            continue
                        total_chars += 1
                        d_chars += 1
                        r_code = star_check_r(fam, ci, a, b, lam0)
                        r_formula = gamma_formula(d, u, v, a, b, lam0[u], lam0[v])
                        if r_code != r_formula:
                            mismatches += 1
                            d_mismatch += 1

            # gauge-shift + gauge-invariance-of-difference, using a second base point
            if len(fam.L) >= 2:
                lam0b = fam.L[1]
                Kv = [tuple((l[vv] - lam0[vv]) % d for vv in fam.obs) for l in fam.L]
                classes = {}
                for ci, C in enumerate(fam.CTX):
                    u, v = C[0], C[1]
                    iu, iv = oi[u], oi[v]
                    for a in range(d):
                        for b in range(d):
                            if (a, b) == (0, 0):
                                continue
                            psi = tuple((a * k[iu] + b * k[iv]) % d for k in Kv)
                            classes.setdefault(psi, []).append((ci, a, b))
                # gauge-shift law on a sample of characters
                sample = [(ci, a, b) for ci, C in enumerate(fam.CTX) for a in range(d)
                          for b in range(d) if (a, b) != (0, 0)][:60]
                ku_kv_cache = {}
                for ci, a, b in sample:
                    C = fam.CTX[ci]
                    u, v = C[0], C[1]
                    g_a = gamma_formula(d, u, v, a, b, lam0[u], lam0[v])
                    g_b = gamma_formula(d, u, v, a, b, lam0b[u], lam0b[v])
                    ku = (lam0b[u] - lam0[u]) % d
                    kv = (lam0b[v] - lam0[v]) % d
                    pred = (g_a - 2 * (a * ku + b * kv)) % (2 * d)
                    gauge_shift_checked += 1
                    if pred != g_b:
                        gauge_shift_bad += 1
                # gauge-invariance of the DIFFERENCE within a psi-class, cross-context
                for psi, lst in classes.items():
                    cis = set(x[0] for x in lst)
                    if len(cis) < 2 or len(lst) < 2:
                        continue
                    ci0, a0, b0 = lst[0]
                    C0 = fam.CTX[ci0]
                    for ci1, a1, b1 in lst[1:]:
                        cross_context_psi_pairs += 1
                        C1 = fam.CTX[ci1]
                        gA0 = gamma_formula(d, C0[0], C0[1], a0, b0, lam0[C0[0]], lam0[C0[1]])
                        gA1 = gamma_formula(d, C1[0], C1[1], a1, b1, lam0[C1[0]], lam0[C1[1]])
                        gB0 = gamma_formula(d, C0[0], C0[1], a0, b0, lam0b[C0[0]], lam0b[C0[1]])
                        gB1 = gamma_formula(d, C1[0], C1[1], a1, b1, lam0b[C1[0]], lam0b[C1[1]])
                        diffA = (gA1 - gA0) % (2 * d)
                        diffB = (gB1 - gB0) % (2 * d)
                        diff_invariance_checked += 1
                        if diffA != diffB:
                            diff_invariance_bad += 1
        by_d[d] = (d_chars, d_mismatch, made)
    return dict(total_chars=total_chars, mismatches=mismatches,
                gauge_shift_checked=gauge_shift_checked, gauge_shift_bad=gauge_shift_bad,
                diff_invariance_checked=diff_invariance_checked,
                diff_invariance_bad=diff_invariance_bad,
                cross_context_psi_pairs=cross_context_psi_pairs,
                families_done=families_done, by_d=by_d, elapsed=time.time() - t0)


def falsified_local_reduction_demo(rng, d=4, ntrip=6, per_d=4, time_budget=6.0):
    """HONEST NEGATIVE RESULT (task instructions require reporting exactly where a
    derivation gap is, not silently dropping it). An earlier draft of this derivation
    conjectured that, for psi-equal t=(C,a,b), t'=(C',a',b'):
        a'*lambda0_{u'} + b'*lambda0_{v'} == a*lambda0_u + b*lambda0_v   (mod d)
    on the theory that evaluating the psi-equality function at l=lambda0 would give this.
    That evaluation is ALGEBRAICALLY TRIVIAL (both sides collapse to 0=0 by construction,
    since psi_t(lambda0) = a*(lambda0_u-lambda0_u)+b*(lambda0_v-lambda0_v) = 0 identically
    for every t) and therefore carries NO information -- the claimed identity is unfounded.
    Direct numerical testing (this function) confirms it is FALSE in general. This is kept
    in the branch, run, and reported so the failure is verifiable rather than asserted."""
    t0 = time.time()
    checked = bad = 0
    made = 0
    while made < per_d and time.time() - t0 < time_budget:
        ctxs = pct.rand_family(rng, d, ntrip, max_obs=9, scale=1)
        if ctxs is None:
            continue
        try:
            fam = pct.Fam(d, ctxs, "falsify_demo")
        except Exception:
            continue
        if not fam.L:
            continue
        made += 1
        lam0 = fam.L[0]
        oi = {vv: k for k, vv in enumerate(fam.obs)}
        Kv = [tuple((l[vv] - lam0[vv]) % d for vv in fam.obs) for l in fam.L]
        classes = {}
        for ci, C in enumerate(fam.CTX):
            u, v = C[0], C[1]
            iu, iv = oi[u], oi[v]
            for a in range(d):
                for b in range(d):
                    if (a, b) == (0, 0):
                        continue
                    psi = tuple((a * k[iu] + b * k[iv]) % d for k in Kv)
                    classes.setdefault(psi, []).append((ci, a, b))
        for psi, lst in classes.items():
            cis = set(x[0] for x in lst)
            if len(cis) < 2:
                continue
            ci0, a0, b0 = lst[0]
            u0, v0 = fam.CTX[ci0][0], fam.CTX[ci0][1]
            lhs0 = (a0 * lam0[u0] + b0 * lam0[v0]) % d
            for ci1, a1, b1 in lst[1:]:
                u1, v1 = fam.CTX[ci1][0], fam.CTX[ci1][1]
                rhs = (a1 * lam0[u1] + b1 * lam0[v1]) % d
                checked += 1
                if lhs0 != rhs:
                    bad += 1
    return checked, bad


# ============================================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", nargs="?", default="all", choices=["audit", "verify", "all"])
    ap.add_argument("--seed", type=int, default=101)
    ap.add_argument("--per-d", type=int, default=10)
    args = ap.parse_args()
    t0 = time.time()
    print(DEFINITIONS)
    if args.mode == "audit":
        print(f"done in {time.time()-t0:.1f}s")
        return

    print("=" * 96)
    print("PART 3a: base-case matrix identities (I) [single-generator power law] and")
    print("(III) [two-generator product law] -- verified directly against raw Weyl matrices,")
    print("independent of the Fam/context machinery.")
    print("=" * 96)
    rng = np.random.default_rng(args.seed)
    w1, w3 = matrix_identity_checks(rng)
    print(f"  Identity (I)  W(u)^a = tau^{{q(u)a(a-2)+q(a*u mod d)}} W(a*u mod d): "
          f"worst |LHS-RHS| = {w1:.2e} {'PASS' if w1 < 1e-8 else 'FAIL'}")
    print(f"  Identity (III) W(x)W(y) = tau^{{...}} W((x+y) mod d): "
          f"worst |LHS-RHS| = {w3:.2e} {'PASS' if w3 < 1e-8 else 'FAIL'}")

    print()
    print("=" * 96)
    print("PART 3b: end-to-end DERIVED closed form for gamma_t vs the code's own r-extraction,")
    print(f"fresh random families ({args.per_d} per d) at d=4,6,8 (NUMERICAL certification)")
    print("=" * 96)
    res = sweep_families(seed0=args.seed, per_d=args.per_d)
    for d, (nch, nbad, nfam) in sorted(res["by_d"].items()):
        print(f"  d={d}: {nfam} families, {nch} characters checked, {nbad} mismatches")
    print(f"  TOTAL: {res['families_done']} families, {res['total_chars']} characters, "
          f"{res['mismatches']} mismatches "
          f"{'(EXACT MATCH -- DERIVED closed form certified)' if res['mismatches']==0 else '(FORMULA WRONG)'}")

    print()
    print("=" * 96)
    print("PART 3c: gauge-shift law and gauge-invariance of the psi-class DIFFERENCE")
    print("=" * 96)
    print(f"  gauge-shift law gamma_t(lambda0')-gamma_t(lambda0) == -2(a k_u+b k_v): "
          f"{res['gauge_shift_checked']} checks, {res['gauge_shift_bad']} mismatches "
          f"{'PASS' if res['gauge_shift_bad']==0 else 'FAIL'}")
    print(f"  gauge-invariance of gamma_t'-gamma_t within a psi-class (cross-context pairs), "
          f"tested with TWO different base points lambda0=L[0] and L[1]: "
          f"{res['diff_invariance_checked']} checks ({res['cross_context_psi_pairs']} "
          f"cross-context psi-equal pairs found), {res['diff_invariance_bad']} mismatches "
          f"{'PASS' if res['diff_invariance_bad']==0 else 'FAIL'}")

    print()
    print("=" * 96)
    print("PART 3d: HONEST NEGATIVE RESULT -- a further reduction attempt (eliminating")
    print("lambda0 in favor of only the FIRST context's coordinates) that was tried and")
    print("FALSIFIED by direct testing, kept here for verifiability (see docstring).")
    print("=" * 96)
    rng2 = np.random.default_rng(args.seed + 999)
    checked, bad = falsified_local_reduction_demo(rng2)
    print(f"  naive substitution a'*lam0_u'+b'*lam0_v' == a*lam0_u+b*lam0_v (mod d) for "
          f"psi-equal cross-context pairs: {checked} checks, {bad} mismatches "
          f"{'(holds -- unexpected)' if bad==0 else '(FALSIFIED, as expected -- the substitution was an unfounded step, not a theorem)'}")

    print()
    print("=" * 96)
    print("HONEST SUMMARY")
    print("=" * 96)
    ok_formula = res["mismatches"] == 0
    ok_gauge = res["gauge_shift_bad"] == 0 and res["diff_invariance_bad"] == 0
    print(f"  - DERIVED + NUMERICALLY VERIFIED: closed form gamma_t == E(t) - "
          f"2(a*lambda0_u+b*lambda0_v) (mod 2d), 0/{res['total_chars']} mismatches across "
          f"{res['families_done']} fresh families (d=4,6,8). Status: {'CONFIRMED' if ok_formula else 'REFUTED -- see mismatches above'}.")
    print(f"  - DERIVED + VERIFIED: gauge-shift law and gauge-invariance-of-difference within "
          f"a psi-class ({res['diff_invariance_checked']} cross-context pair checks, "
          f"{res['gauge_shift_checked']} gauge-shift checks). Status: "
          f"{'CONFIRMED' if ok_gauge else 'REFUTED'}.")
    print(f"  - NOT recovered: the specific y_c/h_v shorthand (still undefined, per prior "
          f"audit) or any purely-LOCAL (per-context s_c, per-observable t_v/g_v) closed form "
          f"for gamma_{{t'}}-gamma_t that avoids referencing lambda0 -- one natural attempt "
          f"was tried and explicitly falsified above ({bad}/{checked} mismatches).")
    print(f"done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
