#!/usr/bin/env python3
"""
branch_d10flexcert.py -- Branch D10-FLEXCERT: the EXACT flex dimension of the published d=10
core (360 rays / 10872 theta-identical pairs / 47 bases, `d10_core_final.cache.json`).

CLOSES THE LAST OPEN CELL OF THE TOWER'S FLEX ROW.  D10_AUDIT.md Sec.2.2 supplied an exact
`flex >= 1` certificate at d=10 (an exact kernel vector of the constraint Jacobian lying outside
the gauge tangent) but explicitly left the EXACT flex dimension uncomputed ("the 22,104 x 7,200
mod-p elimination does not fit this box").  branch_d8flexcert.py / branch_d12flexcert.py have
since pinned flex = 1 EXACTLY at d=8 and d=12.  This file does the same at d=10, so the even
rungs d=8, 10, 12 all carry an exact number rather than a bound.

ALL machinery is imported UNCHANGED from branch_d8flexcert.py (read that file's docstring for the
full method statement and rigour notes).  Nothing in any existing file is modified.  Only the
configuration and one stage-local extension differ:

  * core: d10_core_final (360 rays, 10872 pairs, 47 complete bases, uncolorable, all-critical).
    n = 2*10*360 = 7200 real unknowns; J is 22104 x 7200 (2*10872 orthogonality rows + 360 norm
    rows); T has V + d^2 = 460 generators with the single exact dependency
    sum(phase rows) == sum(diagonal u(d) rows), so the exact ceiling is rank(T) <= 459.
  * gate_full=True: unlike d=12, the dense reference build of the FULL d=10 core does fit
    (branch_d6geo.build_flex_rows: 22104 x 7200, ~1.4 GB, ~1 s), so the sparse row builder is
    proved entry-by-entry identical to the published convention on the WHOLE core at BOTH
    points -- no subcore surrogate is needed.
  * the cache stores only `npairs` (no explicit pair list), exactly as at d=12, so the cert
    stage additionally requires the recomputed edge sets at x5 and x13 to be SET-IDENTICAL.
  * full_rankJ=False: the 22104 x 7200 dense mod-p elimination is the same wall D10_AUDIT hit.
    rank(J) is instead pinned from below by the exact-integer sparse random row-compression
    C = R.J (rank_p(C) <= rank_Q(J) for ANY integer R -- a proof-grade lower bound, VALIDATED
    against the full mod-p rank of J at d=8 by branch_d8flexcert.py's own rankj stage).  The
    certificate supplies the opposite inequality rank_Q(J) <= n - rank(T) - 1 = 6740, so a
    compressed modular rank that HITS 6740 closes the squeeze and gives the EXACT nullity.
  * ONE stage-local extension (`stage_rankj` below, a thin re-assembly of branch_d8flexcert's
    own unit primitives, nothing recomputed differently): the d=8 file gates its flint
    compressed-rank family on `full_rankJ` because there it exists only to validate the
    compression against the full rank.  At d=10 the compressed matrix (6890 x 7200) is small
    enough for python-flint directly, so that family is run on its own merits at the two big
    primes 998244353 / 999999937 IN ADDITION to the exact-BLAS engine at 1000003 / 1048583.
    d=10 therefore reports FOUR primes, all > 1e6, all required to agree.

METHODOLOGICAL NOTE (the d=8/d=12 finding, re-confirmed here and now understood exactly).
The non-membership witness subset S in certificate step (3) must NOT be a complete basis.  For a
d-ray subset that IS one of the core's bases, u(d) acts transitively on frames, so the whole
constraint tangent of the restricted configuration is gauge: at d=10 every one of the 47 bases
gives rank_Q(T|_S) = 100 with NO jump when v_theta is adjoined.  A generic (non-basis) d-ray
subset gives the full generic rank_Q(T|_S) = d + d^2 - 1 = 109, and adjoining v_theta jumps it to
110.  branch_d8flexcert.cert_at_point already searches bases first and then random subsets, so it
finds a valid witness automatically; the bases are simply never the certificate.

STAGES (CLI dispatch; every stage checkpoints to d10flexcert_*.cache.json):
    python3 branch_d10flexcert.py gate | cert | gauge | rankj [budget] | report | all
No existing file is modified.  No git.
"""
import os, sys, time
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ks_flex_census import cache_save, cache_load
import branch_d8flexcert as fc

# Scratch location for the compression / elimination checkpoints ONLY (no proof content lives
# here; both files are regenerable from the core cache).  /tmp on this box is nearly full, so a
# roomier scratch dir is preferred when it exists.  This is a runtime setting, not an edit.
for _cand in ("/sessions/friendly-exciting-ptolemy/tmp", "/tmp"):
    if os.path.isdir(_cand) and os.access(_cand, os.W_OK):
        fc.TMP = os.path.join(_cand, "d10flexcert_scratch")
        os.makedirs(fc.TMP, exist_ok=True)
        break

CFG_D10 = dict(
    tag="d10flexcert", d=10, core_cache="d10_core_final",
    exp_V=360, exp_pairs=10872, exp_bases=47,
    primes=fc.PRIMES,                # 998244353, 999999937 (flint, on the compression)
    full_rankJ=False,                # 22104 x 7200 dense mod-p does not fit this box
    gate_full=True,                  # the dense reference build of the FULL core DOES fit at d=10
    gate_subcore=90,                 # plus the same subcore check the d=8/d=12 gates run
)


# ======================================================================================
# STAGE RANKJ (d=10 assembly) -- identical primitives to branch_d8flexcert.stage_rankj, but the
# flint compressed-rank family is run independently of `full_rankJ` (see docstring).  Every unit
# is cached the moment it finishes, so the stage is resumable inside this box's ~45 s windows.
# ======================================================================================
def stage_rankj(cfg, points=("x5", "x13"), budget=32.0):
    print("=" * 100)
    print(f"[{cfg['tag']}] RANKJ -- rank(J) over GF(p) at 4 primes, squeezed for the exact nullity")
    print("=" * 100)
    t0 = time.time()
    gauge = cache_load(f"{cfg['tag']}_gauge")
    assert gauge is not None, "run stage gauge first"
    out = cache_load(f"{cfg['tag']}_rankJ") or {}
    out["primes_flint"] = list(cfg["primes"])
    out["primes_blas"] = list(fc.SMALL_PRIMES)
    out["scratch"] = fc.TMP
    pending = False
    for ptn in points:
        res = out.get(ptn) or {}
        if res.get("done"):
            print(f"[rankJ:{ptn}] done (cached): lower bound {res['rank_lower_bound']} "
                  f"target {res['target']} flex_exact={res['flex_exact']}")
            continue
        assert gauge[ptn]["exact"], "gauge rank not exact -- cannot squeeze"
        rankT = gauge[ptn]["gauge_dim"]
        if "n" not in res:
            n = 2 * cfg["d"] * cfg["exp_V"]
            res.update(n=n, rankT=rankT, target=n - rankT - 1)
        target = res["target"]
        print(f"[rankJ:{ptn}] n={res['n']} rank(T)={rankT} => certificate ceiling "
              f"rank_Q(J) <= {target}; nullity >= {rankT + 1}")

        # ---- unit family 1: compressed C = R.J, python-flint, the two BIG primes -----------
        res.setdefault("comp_flint", {})
        for p in cfg["primes"]:
            if str(p) in res["comp_flint"]:
                continue
            if time.time() - t0 > budget:
                pending = True
                break
            import scipy.sparse as sps
            C, m = fc.build_C(cfg, ptn, target)
            t1 = time.time()
            res["comp_flint"][str(p)] = fc.flint_rank_coo(sps.coo_matrix(C), p)
            print(f"[rankJ:{ptn}] compressed rank_p(R.J) mod {p} (m={m}, flint) = "
                  f"{res['comp_flint'][str(p)]} ({time.time()-t1:.1f}s)")
            del C
            out[ptn] = res
            cache_save(f"{cfg['tag']}_rankJ", out)

        # ---- unit family 2: compressed C = R.J, exact-BLAS engine, the two SMALL primes ----
        if not pending:
            res.setdefault("comp_blas", {})
            for p in fc.SMALL_PRIMES:
                if str(p) in res["comp_blas"]:
                    continue
                left = budget - (time.time() - t0)
                if left < 6:
                    pending = True
                    break
                C, m = fc.build_C(cfg, ptn, target)
                done, r = fc.blas_rank_resumable(f"{cfg['tag']}_{ptn}", C, p, budget=left)
                del C
                if not done:
                    print(f"[rankJ:{ptn}] BLAS engine mod {p}: checkpointed at partial rank {r} "
                          f"-- RE-RUN this stage to continue")
                    pending = True
                    break
                res["comp_blas"][str(p)] = r
                print(f"[rankJ:{ptn}] compressed rank_p(R.J) mod {p} (m={m}, exact-BLAS) = {r}")
                out[ptn] = res
                cache_save(f"{cfg['tag']}_rankJ", out)

        if pending:
            out[ptn] = res
            cache_save(f"{cfg['tag']}_rankJ", out)
            break

        # ---- close out the point -----------------------------------------------------------
        if (len(res["comp_flint"]) == len(cfg["primes"])
                and len(res["comp_blas"]) == len(fc.SMALL_PRIMES)):
            vals = set(res["comp_flint"].values()) | set(res["comp_blas"].values())
            assert len(vals) == 1, f"the 4 modular ranks DISAGREE: {res}"
            print(f"[rankJ:{ptn}] 4-prime agreement (flint {list(res['comp_flint'])} + "
                  f"exact-BLAS {list(res['comp_blas'])}): rank_p(R.J) = {vals.pop()}")
            out[ptn] = fc._finish_rankj_point(cfg, ptn, res)
            cache_save(f"{cfg['tag']}_rankJ", out)
    if pending:
        print(f"[rankJ] BUDGET REACHED -- progress cached; re-run `rankj` to continue")
    else:
        print(f"[rankJ] ALL UNITS DONE")
        if all(isinstance(out.get(q), dict) and out[q].get("done") for q in points):
            fx = {q: out[q]["flex_exact"] for q in points}
            assert len(set(fx.values())) == 1, f"x5 and x13 DISAGREE on the flex: {fx}"
            print(f"[rankJ] x5 and x13 AGREE: flex_exact = {list(fx.values())[0]}")
    return out


def main(cfg):
    which = sys.argv[1] if len(sys.argv) > 1 else "report"
    if which == "gate":
        fc.stage_gate(cfg)
    elif which == "cert":
        fc.stage_cert(cfg)
    elif which == "gauge":
        fc.stage_gauge(cfg)
    elif which == "rankj":
        stage_rankj(cfg, budget=float(sys.argv[2]) if len(sys.argv) > 2 else 32.0)
    elif which == "report":
        fc.report(cfg)
    elif which == "all":
        fc.stage_gate(cfg)
        fc.stage_cert(cfg)
        fc.stage_gauge(cfg)
        stage_rankj(cfg)
        fc.report(cfg)
    else:
        print(f"unknown stage {which!r}")
        sys.exit(1)
    print(f"\n[{cfg['tag']} stage={which} total {time.time()-fc.T0:.1f}s]")


if __name__ == "__main__":
    main(CFG_D10)
