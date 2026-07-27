#!/usr/bin/env python3
"""
T2 = 0 ON FIBER CYCLES -- corrected structural proof.

    T2(cycle) := sum_{a<b in the word} symp(x_a, x_b)  mod 2

where the word is the concatenation of the lifted contexts selected by a fiber cycle, and
x_v = (v - v mod d)/d are the lift bits.  T2 = 0 is the "T_mix = 0" ingredient of the doubling law
(Paper B).

WHY THIS FILE WAS REWRITTEN (2026-07-24).
-----------------------------------------
The previous version attempted the proof via two sufficient conditions:

    (P1) WITHIN each context, sum_{a<b} symp(x_a,x_b) = 0 mod 2      <-- FALSE
    (P2) CROSS-context sum = 0 mod 2                                 <-- true

and printed "If P1 and P2 both hold universally, T2 = ... is PROVED structurally".  It then
printed P1 = False and exited 0 with no verdict, so the failure was invisible to a runner that
gates on exit codes, and INDEX.md cited the file as establishing the result "(structural)".

P1 is false in bulk: 424 of 896 lifted contexts at d=4 and 344 of 768 at d=6 have a NONZERO
within-context pairing.  So that proof route does not close.  The RESULT is nevertheless true --
T2 = 0 on all 760 (d=4) and 632 (d=6) fiber cycles, verified directly below -- and this file now
proves it by the correct route.  The error was in the proof, not in the theorem.

THE CORRECTED PROOF.
--------------------
Let q(x) = sum_i x_{2i} x_{2i+1} be the standard GF(2) quadratic refinement, so that

    q(x + y) = q(x) + q(y) + symp(x,y)   (mod 2).                                   (Q)

Step 1 (iterate Q over a context C, with S_C := sum_{v in C} x_v):

    w(C) := sum_{a<b in C} symp(x_a,x_b) = q(S_C) + sum_{v in C} q(x_v).            (1)

Step 2.  Summing (1) over the contexts selected by a fiber cycle, the per-observable term is

    sum_C sum_{v in C} q(x_v) = sum_v q(x_v) * mult(v) = 0,                          (2)

because on a cycle every lifted observable occurs an EVEN number of times.  Hence the total
within-context contribution collapses to sum_C q(S_C).

Step 3.  S_C is determined by the BASE context alone (verified below), so writing n(Cbar) for the
number of selected lifts of base context Cbar,

    sum_C q(S_C) = sum_Cbar n(Cbar) * q(S_Cbar) = < n , qS_base >.                   (3)

Step 4.  Every lifted observable having even multiplicity forces every BASE observable to have
even multiplicity, i.e. (n mod 2) lies in the left-kernel of the BASE incidence matrix.  So (3)
vanishes for every fiber cycle iff qS_base is orthogonal to the REALIZED sublattice

    N := { (n mod 2) : n from a fiber cycle }   <=   ker_L(base incidence).

The distinction between N and the full base kernel matters and is not cosmetic: random commuting
families were found whose q(S) pairs ODDLY with a base kernel vector, yet which still have T2 = 0
on every fiber cycle -- because the offending base cycle is never realized (their N = {0}).
Orthogonality to the full base kernel is therefore SUFFICIENT but not NECESSARY.

Step 5.  N is computed below.  For the two seeds of this script (the cert4 family, and the
Mermin-square base scaled to level 6) N turns out to be TRIVIAL, so (3) vanishes with nothing
further to check and T2 = 0 is automatic.

That is not the whole story, and the sweep at the end of this file shows why.  For the SCALED
MERMIN-SQUARE base  pm_base(d) := (d/2) * (qubit labels)  mod d, N obeys a clean dichotomy:

      d = 0 mod 4 :  N = {0, (1,1,1,1,1,1)}   -- the all-ones pattern IS realized
      d = 2 mod 4 :  N = {0}                  -- only the trivial pattern is realized

This is PROVED for all even d in the companion file mod4_dichotomy.py, not merely observed.  The
proof: N is contained in span{all-ones} because the base left-kernel is 1-dimensional; all-ones is
realized iff a GLOBALLY CONSISTENT LIFT of the Mermin square exists; that is the consistency of an
affine GF(2) system whose pair-equation right-hand side is symp(V_i,V_j)/d = (d/2)*s with
symp(l_i,l_j) = 2s, hence vanishes mod 2 exactly when d = 0 mod 4; the whole system therefore
depends on d only through d mod 4, and two base cases (d=4 consistent, rank 20; d=6 inconsistent,
rank 25 with 3 inconsistent rows) settle every even d.  The branches also differ in size -- 1536
lifted contexts / 1400 cycles versus 768 / 632.  This is the same d mod 4 dichotomy that governs
the attained valuation group elsewhere in the program.  Note pm_base(4) is a DIFFERENT family from
the cert4 seed (1536 lifted contexts against 896), which is why the seeds above show N trivial
at d = 4 while pm_base(4) does not.

Consequently, for the Mermin-square tower:
  * at d = 2 mod 4, T2 = 0 is AUTOMATIC -- N is trivial;
  * at d = 0 mod 4, T2 = 0 rests on <all-ones, q(S)> = weight(q(S)) mod 2, and pm_base(d) has
    q(S_Cbar) = 1 for EXACTLY TWO of its six base contexts, at every d tested.  The weight-2
    parity is load-bearing there: a base family of the same shape with an ODD number of q(S)=1
    contexts, and the all-ones pattern realized, would have T2 = 1 on some fiber cycle.

    ==> sum_C w(C) = 0, and with (P2) cross = 0, T2 = within + cross = 0.  QED

So T2 = 0 is not a per-context identity (P1) at all -- it is a parity fact about the base family,
automatic in half the dimensions and a checkable coincidence in the other half.  This script
computes N and the pairing explicitly rather than assuming either.

CAVEAT recorded honestly: orthogonality to the FULL base kernel is sufficient but not necessary.
Random commuting families were found whose q(S) pairs oddly with a base-kernel vector yet which
still satisfy T2 = 0 on every fiber cycle, because their N is trivial.  No family has yet been
exhibited with T2 = 1; whether one exists is OPEN.

Run: python3 close_T2_proof.py     (prints PASS/FAIL and asserts)
"""
import json
import os
import sys
from collections import defaultdict

import numpy as np

from arf_global import symp, build_fiber_pool

HERE = os.path.dirname(os.path.abspath(__file__))


def q(x):
    """standard GF(2) quadratic refinement: q(x+y) = q(x)+q(y)+symp(x,y) mod 2."""
    m = len(x) // 2
    return sum(x[2 * i] * x[2 * i + 1] for i in range(m)) % 2


def left_kernel(A):
    """GF(2) left kernel of A."""
    rows, ncols = A.shape
    aug = np.concatenate([A, np.eye(rows, dtype=int)], axis=1) % 2
    r = 0
    for c in range(ncols):
        pr = next((i for i in range(r, rows) if aug[i, c]), None)
        if pr is None:
            continue
        aug[[r, pr]] = aug[[pr, r]]
        for i in range(rows):
            if i != r and aug[i, c]:
                aug[i] ^= aug[r]
        r += 1
        if r == rows:
            break
    return [aug[i, ncols:].copy() for i in range(rows) if not aug[i, :ncols].any()]


def seed_base(seed):
    if seed == "cert4":
        cert = json.load(open(os.path.join(HERE, "cert4_min.json")))
        return [[tuple(v) for v in it["ctx"]] for it in cert["items"]]
    lbl = {"XI": (1, 0, 0, 0), "IX": (0, 0, 1, 0), "XX": (1, 0, 1, 0),
           "IY": (0, 0, 1, 1), "YI": (1, 1, 0, 0), "YY": (1, 1, 1, 1),
           "XY": (1, 0, 1, 1), "YX": (1, 1, 1, 0), "ZZ": (0, 1, 0, 1)}
    R = [["XI", "IX", "XX"], ["IY", "YI", "YY"], ["XY", "YX", "ZZ"]]
    Co = [["XI", "IY", "XY"], ["IX", "YI", "YX"], ["XX", "YY", "ZZ"]]
    return [[tuple((3 * np.array(lbl[l])) % 6) for l in C] for C in R + Co]


def analyse(tag, d, seed):
    base = seed_base(seed)
    pool = build_fiber_pool(base, d)
    obs = sorted({tuple(v) for C in pool for v in C})
    oi = {v: k for k, v in enumerate(obs)}

    A = np.zeros((len(pool), len(obs)), int)
    w = np.zeros(len(pool), int)
    qS = np.zeros(len(pool), int)
    bkey = []
    for r, C in enumerate(pool):
        xc = [(np.array(v) - np.array(v) % d) // d for v in C]
        w[r] = sum(symp(xc[a], xc[b])
                   for a in range(len(xc)) for b in range(a + 1, len(xc))) % 2
        S = np.zeros(len(xc[0]), int)
        for x in xc:
            S = (S + x) % 2
        qS[r] = q(S)
        bkey.append(tuple(sorted(tuple(np.array(v) % d) for v in C)))
        for v in C:
            A[r, oi[tuple(v)]] ^= 1
    A %= 2

    print(f"\n=== {tag}  (d={d}, seed={seed}):  "
          f"{len(pool)} lifted contexts, {len(obs)} lifted observables")

    # -- the previous route: P1 is FALSE in bulk (recorded, deliberately not asserted) ----
    print(f"  [historical] contexts violating the old (P1): {int(w.sum())} of {len(pool)}"
          f"   -> the earlier proof route does NOT close")

    # -- Step 1: the quadratic-refinement identity ---------------------------------------
    id_ok = True
    for r, C in enumerate(pool):
        xc = [(np.array(v) - np.array(v) % d) // d for v in C]
        if (qS[r] + sum(q(x) for x in xc)) % 2 != w[r]:
            id_ok = False
    print(f"  step 1  w(C) = q(S_C) + sum_v q(x_v)                 : {id_ok}")
    assert id_ok, "quadratic-refinement identity failed"

    # -- Step 3: q(S_C) depends only on the base context ---------------------------------
    seen = defaultdict(set)
    for r in range(len(pool)):
        seen[bkey[r]].add(int(qS[r]))
    base_det = all(len(v) == 1 for v in seen.values())
    print(f"  step 3  q(S_C) determined by the base context alone  : {base_det}")
    assert base_det, "q(S_C) is not a base-context invariant"

    # -- Steps 4-5: the base-level pairing ------------------------------------------------
    ubase = sorted(seen)
    qvec = np.array([next(iter(seen[k])) for k in ubase])
    bobs = sorted({v for k in ubase for v in k})
    bi = {v: i for i, v in enumerate(bobs)}
    B = np.zeros((len(ubase), len(bobs)), int)
    for r, k in enumerate(ubase):
        for v in k:
            B[r, bi[v]] ^= 1
    B %= 2
    K = left_kernel(B)
    print(f"  step 4  {len(ubase)} base contexts, base left-kernel dim {len(K)};"
          f"  q(S) = {list(map(int, qvec))}  (weight {int(qvec.sum())})")

    # -- the REALIZED sublattice N, plus direct confirmation, in one pass ------------------
    bidx = {k: i for i, k in enumerate(ubase)}
    cyc = left_kernel(A)
    realized = set()
    bad_t2 = bad_cross = bad_pair = n_cyc = 0
    for lam in cyc:
        sel = [i for i in range(len(pool)) if lam[i] % 2]
        if not sel:
            continue
        n_cyc += 1
        nvec = np.zeros(len(ubase), int)
        for i in sel:
            nvec[bidx[bkey[i]]] += 1
        nvec %= 2
        realized.add(tuple(nvec))
        if int(nvec @ qvec) % 2:
            bad_pair += 1
        xs = [[(np.array(v) - np.array(v) % d) // d for v in pool[i]] for i in sel]
        word = [v for xc in xs for v in xc]
        T2 = sum(symp(word[a], word[b])
                 for a in range(len(word)) for b in range(a + 1, len(word))) % 2
        SX = [sum(xc) % 2 for xc in xs]
        cross = sum(symp(SX[i], SX[j])
                    for i in range(len(SX)) for j in range(i + 1, len(SX))) % 2
        bad_t2 += int(bool(T2))
        bad_cross += int(bool(cross))

    nz = sorted(p for p in realized if any(p))
    allones = any(all(p) for p in nz)
    print(f"  step 5  realized sublattice N: {len(realized)} pattern(s), "
          f"{len(nz)} nonzero{'  [all-ones REALIZED]' if allones else '  [only the trivial pattern]'}")
    print(f"          <n, q(S)> odd on {bad_pair} of {n_cyc} cycles"
          f"   -> {'automatic (N trivial)' if not nz else 'saved by weight(q(S)) even'}")
    print(f"          d mod 4 = {d % 4}"
          f"  ({'all-ones realized, parity is load-bearing' if allones else 'N trivial, T2=0 automatic'})")
    assert bad_pair == 0, "q(S) pairs oddly with a REALIZED base pattern -- T2 = 0 fails"

    print(f"  direct  T2 != 0 on {bad_t2} of {n_cyc} fiber cycles;  cross != 0 on {bad_cross}")
    assert bad_t2 == 0, "T2 nonzero on some fiber cycle"
    assert bad_cross == 0, "cross term nonzero on some fiber cycle"
    return True


def pm_base(d):
    """the Mermin-square base scaled to level d: (d/2) * qubit labels, mod d."""
    h = d // 2
    lbl = {"XI": (1, 0, 0, 0), "IX": (0, 0, 1, 0), "XX": (1, 0, 1, 0),
           "IY": (0, 0, 1, 1), "YI": (1, 1, 0, 0), "YY": (1, 1, 1, 1),
           "XY": (1, 0, 1, 1), "YX": (1, 1, 1, 0), "ZZ": (0, 1, 0, 1)}
    R = [["XI", "IX", "XX"], ["IY", "YI", "YY"], ["XY", "YX", "ZZ"]]
    Co = [["XI", "IY", "XY"], ["IX", "YI", "YX"], ["XX", "YY", "ZZ"]]
    return [[tuple((h * np.array(lbl[l])) % d) for l in C] for C in R + Co]


def sweep_mod4(ds=(4, 6)):
    """the d mod 4 dichotomy of the realized sublattice N, for the Mermin-square base.

    Two representatives per residue class suffice here because the dichotomy is PROVED for all
    even d in mod4_dichotomy.py (the global-lift system depends on d only through d mod 4).  Pass
    a longer tuple for a wider spot-check; d=4..24 was the original sweep.
    """
    print("\n" + "=" * 78)
    print("SWEEP: realized sublattice N for the scaled Mermin-square base pm_base(d)")
    print("       (dichotomy PROVED for all even d in mod4_dichotomy.py; this is a spot-check)")
    print(f"{'d':>4} {'d%4':>4} {'pool':>6} {'cycles':>7} {'N nonzero':>10} "
          f"{'wt q(S)':>8} {'<1,qS>':>7}  T2=0")
    print("-" * 62)
    ok = True
    for d in ds:
        base = pm_base(d)
        pool = build_fiber_pool(base, d)
        obs = sorted({tuple(v) for C in pool for v in C})
        oi = {v: k for k, v in enumerate(obs)}
        A = np.zeros((len(pool), len(obs)), int)
        bkey = []
        qsb = {}
        for r, C in enumerate(pool):
            k = tuple(sorted(tuple(np.array(v) % d) for v in C))
            bkey.append(k)
            xc = [(np.array(v) - np.array(v) % d) // d for v in C]
            S = np.zeros(len(xc[0]), int)
            for x in xc:
                S = (S + x) % 2
            qsb.setdefault(k, q(S))
            for v in C:
                A[r, oi[tuple(v)]] ^= 1
        A %= 2
        ub = sorted(qsb)
        bidx = {k: i for i, k in enumerate(ub)}
        qv = np.array([qsb[k] for k in ub])
        pats = set()
        ncyc = bad = 0
        for lam in left_kernel(A):
            sel = [i for i in range(len(pool)) if lam[i] % 2]
            if not sel:
                continue
            ncyc += 1
            n = np.zeros(len(ub), int)
            for i in sel:
                n[bidx[bkey[i]]] += 1
            n %= 2
            pats.add(tuple(n))
            if int(n @ qv) % 2:
                bad += 1
        nz = [p for p in pats if any(p)]
        wt = int(qv.sum())
        tag = "   <- all-ones realized" if any(all(x) for x in nz) else ""
        print(f"{d:>4} {d%4:>4} {len(pool):>6} {ncyc:>7} {len(nz):>10} {wt:>8} "
              f"{wt%2:>7}  {'holds' if bad == 0 else 'FAILS(' + str(bad) + ')'}{tag}")
        if bad:
            ok = False
        assert wt % 2 == 0, f"weight(q(S)) is odd at d={d} -- T2=0 would be at risk"
    return ok


if __name__ == "__main__":
    ok = True
    for tag, (d, seed) in {"d4": (4, "cert4"), "d6": (6, "PM6")}.items():
        ok &= analyse(tag, d, seed)
    ok &= sweep_mod4()
    print("\n" + "=" * 78)
    print("T2 = 0 is PROVED structurally, by the corrected route:")
    print("  the per-observable term dies on cycles (even multiplicity); the remainder is the")
    print("  base-level pairing <n, q(S)> over the REALIZED sublattice N.  N is trivial for both")
    print("  seeds; for the Mermin-square tower N is trivial at d=2 mod 4 and contains all-ones")
    print("  at d=0 mod 4, where vanishing rests on weight(q(S)) = 2 being even.")
    print("NOTE: this is NOT the old per-context identity (P1), which is false in bulk.")
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)
