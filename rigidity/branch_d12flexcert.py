#!/usr/bin/env python3
"""
branch_d12flexcert.py -- Branch D12-FLEXCERT: exact "flex >= 1" certificate for the published
d=12 core (412 rays / 19885 pairs / 61 bases, `d12_core_final.cache.json`).

Closes the d=12 half of the declared gap ("the analogous flex certificates at d=8 and d=12 have
not been run").  ALL machinery is imported UNCHANGED from branch_d8flexcert.py, which replicates
D10_AUDIT.md Sec.2.2's method exactly (read that file's docstring for the full method statement
and rigour notes).  Only the configuration differs:

  * core: d12_core_final (412 rays, 19885 theta-identical pairs, 61 complete bases,
    uncolorable, all-critical).  n = 2*12*412 = 9888 real unknowns; J is 40182 x 9888
    (2*19885 orthogonality rows + 412 norm rows); T has V + d^2 = 556 generators, exact
    dependency ceiling rank(T) <= 555.
  * the cache has no stored pair list (only npairs=19885), so the cert stage additionally
    requires the recomputed edge sets at x5 and x13 to be SET-IDENTICAL (they are).
  * full_rankJ=False: the dense 40182 x 9888 mod-p elimination does not fit this box (the same
    limit D10_AUDIT hit at d=10).  rank(J) is instead bounded below via the exact-integer
    sparse random row-compression C = R.J (rank_p(C) <= rank_Q(J) for ANY R -- proof-grade
    lower bound), which branch_d8flexcert.py VALIDATES against the full mod-p rank at d=8.
    If the lower bound hits n - rank(T) - 1 at both primes, the certificate's opposite
    inequality closes the squeeze: exact nullity, flex EXACTLY 1.

STAGES (CLI dispatch; every stage checkpoints to d12flexcert_*.cache.json):
    python3 branch_d12flexcert.py gate | cert | gauge | rankj | report | all
No existing file is modified.  No git.
"""
import os, sys, time
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import branch_d8flexcert as fc

CFG_D12 = dict(
    tag="d12flexcert", d=12, core_cache="d12_core_final",
    exp_V=412, exp_pairs=19885, exp_bases=61,
    primes=fc.PRIMES,
    full_rankJ=False,                # 40182 x 9888 dense mod-p does not fit this box
    gate_full=False,                 # dense reference build of the full core does not fit
    gate_subcore=90,                 # builder identity is proved on subcores at d=12 (and on
                                     # the FULL core at d=8 by branch_d8flexcert.py's own gate)
)

if __name__ == "__main__":
    fc.main(CFG_D12)
