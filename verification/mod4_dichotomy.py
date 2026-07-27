#!/usr/bin/env python3
"""
THE d mod 4 DICHOTOMY OF THE REALIZED BASE-MULTIPLICITY SUBLATTICE.  [PROVED]

Companion to close_T2_proof.py.  That file reduces "T2 = 0 on every fiber cycle" to a pairing
<n, q(S)> over the REALIZED sublattice

    N := { (n mod 2) : n from a fiber cycle } ,      n(Cbar) = # selected lifts of base context Cbar

and observes empirically that, for the scaled Mermin-square base
pm_base(d) := (d/2) * (qubit labels)  mod d,

    d = 0 mod 4 :  N = {0, (1,1,1,1,1,1)}          d = 2 mod 4 :  N = {0}.

THIS FILE PROVES THAT DICHOTOMY, for all even d, rather than checking it dimension by dimension.

-------------------------------------------------------------------------------------------------
STEP 1.  N is contained in span{all-ones}.
The Mermin square has each of its 9 observables in exactly 2 of its 6 contexts, so the all-ones
vector is a base cycle; and the base incidence matrix has GF(2) left-kernel of dimension exactly 1
(computed below).  Since n mod 2 always lies in that kernel, N <= {0, all-ones}.  So the only
question is whether all-ones is realized.

STEP 2.  all-ones in N  <=>  a GLOBALLY CONSISTENT LIFT exists.
all-ones in N means some fiber cycle selects an ODD number of lifts from every base context.  The
minimal such selection takes exactly ONE lift per base context.  Because each base observable lies
in exactly two contexts, the cycle condition (every lifted observable selected an even number of
times) forces the two chosen lifts to agree on their shared observable.  So such a cycle is
precisely a globally consistent assignment of one lifted vector to each of the 9 base observables,
compatible with all 6 contexts at once -- a global section of the lift.

STEP 3.  The global-lift condition is an affine GF(2) system.
Writing the lift of the base observable with representative u as u + d*x, x in {0,1}^4, the fiber
closure conditions for a context C are (see arf2.fiber_solspace):
    (sum equations)   sum_{i in C} x_i[c]      = ( sum_{i in C} V_i[c] ) / d   mod 2,  c = 0..3
    (pair equations)  <bilinear in x_i, x_j>   = symp(V_i, V_j) / d            mod 2
Imposing these for all 6 contexts with SHARED unknowns x_vbar (9 observables x 4 bits = 36
unknowns) gives one system; a global lift exists iff it is consistent.

STEP 4.  THE KEY COMPUTATION -- the RHS depends on d only through d mod 4.
With h = d/2 and V_i = h*l_i mod d, bilinearity gives
    symp(V_i, V_j) = h^2 * symp(l_i, l_j) = (d^2/4) * symp(l_i, l_j),
so the pair-equation right-hand side is
    symp(V_i,V_j)/d = (d/4) * symp(l_i,l_j) = (d/2) * s,     where symp(l_i,l_j) = 2s
(the labels in a common context commute, so their symplectic form is even).  Reducing mod 2:
    d = 0 mod 4  =>  d/2 EVEN  =>  every pair RHS = 0;
    d = 2 mod 4  =>  d/2 ODD   =>  pair RHS = s mod 2, which is nonzero for some pairs.
This is the entire source of the dichotomy.  Verified below, together with the stronger statement
that the FULL system (coefficient matrix and RHS) is byte-identical across each residue class.

STEP 5.  Two base cases finish it.
Since the system is a function of d mod 4 alone, it suffices to solve it once in each class:
    d = 4 : CONSISTENT   (rank 20, no inconsistent row)  => global lift exists => all-ones in N
    d = 6 : INCONSISTENT (rank 25, 3 inconsistent rows)  => no global lift    => N = {0}
Together with Step 1 this determines N for every even d.  QED

-------------------------------------------------------------------------------------------------
CONSEQUENCE for T2 = 0 (see close_T2_proof.py):
  * d = 2 mod 4 : N is trivial, so the pairing <n,q(S)> vanishes identically and T2 = 0 needs
                  nothing further -- it is automatic.
  * d = 0 mod 4 : all-ones is realized, so T2 = 0 requires <all-ones, q(S)> = weight(q(S)) mod 2
                  to vanish.  pm_base(d) has q(S_Cbar) = 1 on exactly TWO of its six base
                  contexts, so the weight is 2 and the pairing vanishes -- but here the parity is
                  genuinely load-bearing, not automatic.

-------------------------------------------------------------------------------------------------
UNIFICATION WITH tmix_dindep.py  [new 2026-07-24; extends a previously published claim]

`tmix_dindep.py` records that the F_2 system carrying T_mix is "d-INDEPENDENT across d = 2 mod 4",
tested at d = 6,10,14,18,22,26.  That is the d = 2 half of a two-class theorem, and the SAME
computation as Step 4 above proves the whole thing.  Its per-context value is

    t = XOR_{a<b} (qu & 1)*(Mab & 1),        qu = symp(u_a,u_b) / d.

For the scaled base u = h*p mod d, bilinearity gives symp(u_a,u_b) = h^2 symp(p_a,p_b), so

    qu = (d/4) symp(p_a,p_b) = h*s        where symp(p_a,p_b) = 2s        (*)

-- literally the expression of Step 4.  Hence qu mod 2 = h*s mod 2 and:

    d = 0 mod 4 (h EVEN) : every qu is even (0 of 18), so every term dies and t == 0
                           IDENTICALLY -- pointwise, with no cycle summation.  1536 lifted
                           contexts.  And the global-lift system is consistent, so N = {0, 1}.
    d = 2 mod 4 (h ODD)  : 5 of 18 qu are odd, and t != 0 on 128 of 768 lifted contexts, so
                           T_mix vanishes only AFTER summing over a cycle.  768 lifted contexts.
                           And the global-lift system is inconsistent, so N = {0}.

Verified at d = 4,...,20 below.  So the two classes behave oppositely and complementarily, and
tmix_dindep's d-independence is not a coincidence of its sample: BOTH residue classes are
internally constant, for the single reason (*).  This unifies two separately-recorded phenomena.

(The third d mod 4 appearance in the program, V47/V51's naive-section dichotomy, is a DIFFERENT
phenomenon: it concerns ODD d, d = 1 vs 3 mod 4, and comes from the classical Gauss sum
g_d = sqrt(d) resp. i*sqrt(d).  It is not explained by (*) and should not be conflated with it.)

Run: python3 mod4_dichotomy.py       (prints PASS/FAIL and asserts)
"""
import os
import sys

import numpy as np

from arf_global import symp

HERE = os.path.dirname(os.path.abspath(__file__))

LBL = {"XI": (1, 0, 0, 0), "IX": (0, 0, 1, 0), "XX": (1, 0, 1, 0),
       "IY": (0, 0, 1, 1), "YI": (1, 1, 0, 0), "YY": (1, 1, 1, 1),
       "XY": (1, 0, 1, 1), "YX": (1, 1, 1, 0), "ZZ": (0, 1, 0, 1)}
CTX = [["XI", "IX", "XX"], ["IY", "YI", "YY"], ["XY", "YX", "ZZ"],
       ["XI", "IY", "XY"], ["IX", "YI", "YX"], ["XX", "YY", "ZZ"]]
LABELS = sorted(LBL)
LI = {l: i for i, l in enumerate(LABELS)}


def base_left_kernel_dim():
    """Step 1: the base incidence matrix's GF(2) left-kernel."""
    B = np.zeros((len(CTX), len(LABELS)), int)
    for r, C in enumerate(CTX):
        for l in C:
            B[r, LI[l]] ^= 1
    rows, ncols = B.shape
    aug = np.concatenate([B % 2, np.eye(rows, dtype=int)], axis=1) % 2
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
    K = [aug[i, ncols:].copy() for i in range(rows) if not aug[i, :ncols].any()]
    return K


def global_lift_system(d):
    """Step 3: the affine GF(2) system whose consistency == existence of a global lift."""
    h = d // 2
    nb = 4 * len(LABELS)
    rows, rhs = [], []
    for C in CTX:
        V = [tuple((h * np.array(LBL[l])) % d) for l in C]
        L = len(C)
        sc = [sum(V[i][c] for i in range(L)) for c in range(4)]
        for c in range(4):
            row = [0] * nb
            for i in range(L):
                row[LI[C[i]] * 4 + c] ^= 1
            rows.append(row)
            rhs.append((sc[c] // d) % 2)
        for i in range(L):
            for j in range(i + 1, L):
                row = [0] * nb
                for k in range(2):
                    row[LI[C[j]] * 4 + 2 * k] ^= V[i][2 * k + 1] % 2
                    row[LI[C[j]] * 4 + 2 * k + 1] ^= V[i][2 * k] % 2
                    row[LI[C[i]] * 4 + 2 * k] ^= V[j][2 * k + 1] % 2
                    row[LI[C[i]] * 4 + 2 * k + 1] ^= V[j][2 * k] % 2
                rows.append(row)
                rhs.append((symp(V[i], V[j]) // d) % 2)
    return np.array(rows, int) % 2, np.array(rhs, int) % 2


def consistent(d):
    """solve the global-lift system; return (consistent?, rank, #inconsistent rows)."""
    A, b = global_lift_system(d)
    nb = A.shape[1]
    M = np.concatenate([A, b.reshape(-1, 1)], axis=1) % 2
    nr = M.shape[0]
    r = 0
    for c in range(nb):
        pr = next((i for i in range(r, nr) if M[i, c]), None)
        if pr is None:
            continue
        M[[r, pr]] = M[[pr, r]]
        for i in range(nr):
            if i != r and M[i, c]:
                M[i] ^= M[r]
        r += 1
        if r == nr:
            break
    bad = [i for i in range(nr) if M[i, nb] and not M[i, :nb].any()]
    return len(bad) == 0, r, len(bad)


def pair_rhs_values(d):
    """Step 4: the multiset of pair-equation right-hand sides."""
    h = d // 2
    out = []
    for C in CTX:
        V = [tuple((h * np.array(LBL[l])) % d) for l in C]
        for i in range(3):
            for j in range(i + 1, 3):
                out.append((symp(V[i], V[j]) // d) % 2)
    return out


def unification():
    """The same expression h*s governs tmix_dindep's T_mix system. Extends its d=2-only claim."""
    from arf_global import fiber_all
    print("\n" + "=" * 78)
    print("UNIFICATION with tmix_dindep.py:  qu = symp(u_a,u_b)/d = h*s,  h = d/2")
    print(f"{'d':>4} {'h':>4} {'h par':>6} {'qu odd':>10} {'lifted ctx':>11} {'t != 0 on':>10}")
    print("-" * 52)
    ok = True
    for d in (4, 6, 8, 10, 12, 14, 16, 18, 20):
        h = d // 2
        base = [[tuple((h * np.array(LBL[l])) % d) for l in C] for C in CTX]
        qodd = qtot = 0
        for C in base:
            for a in range(3):
                for b in range(a + 1, 3):
                    qodd += (symp(C[a], C[b]) // d) & 1
                    qtot += 1
        tnz = tot = 0
        for C in base:
            for lc in fiber_all(C, d):
                V = [tuple(v) for v in lc]
                us = [np.array(v) % d for v in V]
                xs = [((np.array(v) - (np.array(v) % d)) // d) % 2 for v in V]
                t = 0
                for a in range(3):
                    for b in range(a + 1, 3):
                        qu = symp(us[a], us[b]) // d
                        Mab = symp(us[a], xs[b]) + symp(xs[a], us[b])
                        t ^= ((qu & 1) * (Mab & 1)) & 1
                tot += 1
                tnz += int(t != 0)
        print(f"{d:>4} {h:>4} {'even' if h % 2 == 0 else 'odd':>6} "
              f"{qodd:>4} of {qtot:<3} {tot:>11} {tnz:>10}")
        if h % 2 == 0:
            assert qodd == 0 and tnz == 0, f"d={d}: h even but T_mix not identically zero"
        else:
            assert qodd > 0 and tnz > 0, f"d={d}: h odd but T_mix identically zero"
    print("  h EVEN (d=0 mod 4): every qu even => t == 0 IDENTICALLY, no cycle summation needed.")
    print("  h ODD  (d=2 mod 4): some qu odd  => t != 0 pointwise; vanishes only over a cycle.")
    print("  => tmix_dindep's 'd-independent across d=2 mod 4' is the d=2 half of a two-class")
    print("     theorem; BOTH classes are internally constant, for the single reason qu = h*s.")
    return ok


def main():
    ok = True
    print("=" * 78)
    print("THE d mod 4 DICHOTOMY OF N  --  proof by residue-class reduction")
    print("=" * 78)

    # Step 1
    K = base_left_kernel_dim()
    print(f"\nSTEP 1  base left-kernel dimension: {len(K)}"
          f"   basis: {[list(map(int,k)) for k in K]}")
    assert len(K) == 1 and all(k.all() for k in K), \
        "base kernel is not exactly span{all-ones}"
    print("        => N <= {0, all-ones}: the only question is whether all-ones is realized.")

    # Step 4a: the pair RHS mechanism
    print("\nSTEP 4  pair-equation RHS = (d/2)*s mod 2, s = symp(l_i,l_j)/2:")
    for d in (4, 8, 12, 16, 6, 10, 14, 18):
        vals = sorted(set(pair_rhs_values(d)))
        expect0 = (d % 4 == 0)
        print(f"          d={d:>3} (d%4={d%4}): RHS values {vals}"
              f"   {'all zero (d/2 even)' if expect0 else 'nonzero present (d/2 odd)'}")
        if expect0:
            assert vals == [0], f"d={d}: expected all pair RHS zero"
            ok &= True
        else:
            assert 1 in vals, f"d={d}: expected a nonzero pair RHS"

    # Step 4b: the whole system is a function of d mod 4
    print("\nSTEP 4  the FULL system depends on d only through d mod 4:")
    A4, b4 = global_lift_system(4)
    A6, b6 = global_lift_system(6)
    cls0 = [8, 12, 16, 20, 24, 32, 40, 100, 1000]
    cls2 = [10, 14, 18, 22, 26, 30, 42, 102, 1002]
    s0 = all((global_lift_system(d)[0] == A4).all()
             and (global_lift_system(d)[1] == b4).all() for d in cls0)
    s2 = all((global_lift_system(d)[0] == A6).all()
             and (global_lift_system(d)[1] == b6).all() for d in cls2)
    print(f"          d in {cls0} identical to d=4 : {s0}")
    print(f"          d in {cls2} identical to d=6 : {s2}")
    assert s0, "the d=0 mod 4 system is not residue-class constant"
    assert s2, "the d=2 mod 4 system is not residue-class constant"
    diff = int((b4 != b6).sum())
    print(f"          the two classes differ (in {diff} of {len(b4)} RHS entries): "
          f"{not ((A4 == A6).all() and (b4 == b6).all())}")
    assert diff > 0, "the two residue classes give the same system"

    # Step 5: the two base cases
    print("\nSTEP 5  base cases:")
    c4, r4, n4 = consistent(4)
    c6, r6, n6 = consistent(6)
    print(f"          d=4: consistent={c4}  rank={r4}  inconsistent rows={n4}"
          f"   => global lift EXISTS  => all-ones in N")
    print(f"          d=6: consistent={c6}  rank={r6}  inconsistent rows={n6}"
          f"   => NO global lift      => N = {{0}}")
    assert c4 and not c6, "the base cases do not exhibit the dichotomy"

    # independent confirmation across many d
    print("\nCONFIRM  every even d up to 60 falls on the side its residue predicts:")
    bad = []
    for d in range(4, 62, 2):
        c, _, _ = consistent(d)
        if c != (d % 4 == 0):
            bad.append(d)
    print(f"          exceptions: {bad if bad else 'none'}")
    assert not bad, f"residue prediction failed at {bad}"

    print("\n" + "=" * 78)
    print("THEOREM (proved).  For the scaled Mermin-square base pm_base(d), the realized")
    print("base-multiplicity sublattice is")
    print("      N = {0, (1,1,1,1,1,1)}   if d = 0 mod 4,        N = {0}   if d = 2 mod 4.")
    print("The global-lift system is a function of d mod 4 alone (Step 4), so the two base")
    print("cases of Step 5 settle every even d; Step 1 bounds N above by span{all-ones}.")
    print("Mechanism: symp(V_i,V_j)/d = (d/2)*s, which vanishes mod 2 exactly when d = 0 mod 4.")
    print("\nCONSEQUENCE: T2 = 0 is AUTOMATIC at d = 2 mod 4 (N trivial), while at d = 0 mod 4 it")
    print("rests on weight(q(S)) being even -- load-bearing, not automatic.  See close_T2_proof.py.")
    ok &= unification()
    print("\nmod4_dichotomy PASS" if ok else "\nmod4_dichotomy FAIL")
    return ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
