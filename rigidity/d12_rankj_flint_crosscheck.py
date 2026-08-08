#!/usr/bin/env python3
"""
d12_rankj_flint_crosscheck.py -- close the engine/prime asymmetry in the d=12 flex certificate.

WHY THIS FILE EXISTS
--------------------
branch_d12flexcert.py establishes flex EXACTLY 1 at d=12 by the same two-sided squeeze used at
d=8 and d=10:

    lower bound on rank_Q(J):  rank_p(C) <= rank_Q(J) for C = R.J with ANY integer R
                               (proof-grade for any R and any p -- no genericity needed)
    upper bound on rank_Q(J):  the exact kernel certificate, which exhibits nullity >= 556

and the two meet at 9332, forcing nullity exactly 556 and flex exactly 1.

That argument is sound.  What was NOT uniform across the three rungs is the *redundancy* behind
the lower bound.  At d=8 and d=10 the compressed rank is computed by two independent engines
(python-flint nmod_mat, and a float64-carrier BLAS eliminator with a proven exactness bound)
over four primes.  At d=12 branch_d8flexcert.stage_rankj gates the flint units on
cfg["full_rankJ"], which d=12 sets False because the dense 40182 x 9888 mod-p elimination does
not fit -- so d=12 shipped with the BLAS engine only, at the two small primes.

The gate is too coarse: the COMPRESSED rank does not need full J to fit.  C is only
(target+150) x 9888.  This script runs exactly the units the gate skipped -- flint nmod_mat on
the compressed matrix at the two big primes -- at both circle points.

A single engine agreeing with itself at two primes is weaker evidence against an implementation
bug than two engines agreeing at four primes.  The mathematics was never in question; the
cross-check is.

WHAT IT PROVES
--------------
Nothing new about the mathematics.  It removes an asymmetry in how the three top rungs are
certified, so that the sentence "flex exactly 1 at d = 8, 10 and 12, by two-sided squeezes" is
backed by the same weight of computation at all three.

Expected output (both points):

    [x5]  compressed rank_p(R.J), flint, p=998244353 = 9332   (target 9332)
    [x5]  compressed rank_p(R.J), flint, p=999999937 = 9332   (target 9332)
    [x13] compressed rank_p(R.J), flint, p=998244353 = 9332   (target 9332)
    [x13] compressed rank_p(R.J), flint, p=999999937 = 9332   (target 9332)
    VERDICT: d=12 lower bound reproduced by a second engine at two further primes.

Usage:  python3 d12_rankj_flint_crosscheck.py
No existing file is modified.  Results cache to d12flexcert_rankJ_flint.cache.json.
"""
import os
import sys
import time

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import branch_d8flexcert as fc
from branch_d12flexcert import CFG_D12
from ks_flex_census import cache_save, cache_load

CACHE = "d12flexcert_rankJ_flint"


def main():
    base = cache_load("d12flexcert_rankJ")
    if not base:
        print("ERROR: run `python3 branch_d12flexcert.py all` first -- no d12flexcert_rankJ cache.")
        return 1

    out = cache_load(CACHE) or {"primes_flint": list(CFG_D12["primes"])}
    ok = True

    for ptn in ("x5", "x13"):
        prev = base.get(ptn) or {}
        target = prev.get("target")
        if target is None:
            print(f"ERROR: no target recorded for {ptn}")
            return 1

        res = out.setdefault(ptn, {})
        res["target"] = target
        res.setdefault("comp_flint", {})

        for p in CFG_D12["primes"]:
            if str(p) in res["comp_flint"]:
                r = res["comp_flint"][str(p)]
                print(f"  [{ptn}] compressed rank_p(R.J), flint, p={p} = {r}   (cached)")
            else:
                C, m = fc.build_C(CFG_D12, ptn, target)
                t1 = time.time()
                import scipy.sparse as sps
                r = fc.flint_rank_coo(sps.coo_matrix(C), p)
                res["comp_flint"][str(p)] = r
                cache_save(CACHE, out)
                print(f"  [{ptn}] compressed rank_p(R.J), flint, p={p} = {r}   "
                      f"(m={m}, {time.time() - t1:.1f}s)")
            if r != target:
                ok = False
                print(f"    MISMATCH: expected {target}")

        # the BLAS values this is cross-checking
        blas = prev.get("comp_blas", {})
        agree = all(v == target for v in blas.values())
        res["blas_agreement"] = agree
        print(f"  [{ptn}] BLAS engine values {dict(blas)} -- agreement with target: {agree}")
        ok = ok and agree

    cache_save(CACHE, out)
    print()
    if ok:
        print("VERDICT: the d=12 compressed-rank lower bound 9332 is reproduced at both circle")
        print("points by a SECOND independent engine (python-flint nmod_mat) at two further")
        print("primes (998244353, 999999937).  d=12 now matches d=8 and d=10: two engines,")
        print("four primes.  The exact kernel certificate supplies the opposite inequality, so")
        print("nullity is exactly 556 and mechanism flex is exactly 1.")
    else:
        print("VERDICT: MISMATCH -- do not cite the d=12 certificate until this is resolved.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
