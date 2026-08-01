#!/usr/bin/env python3
"""
branch_d6flexcert.py -- Branch D6-FLEXCERT: the EXACT mechanism flex dimension of the d=6
Galois core (280 rays / 3284 theta-identical pairs / 95 bases, `d6galois_core_final.cache.json`).

CERTIFIES THE d=6 RUNG'S FLEX NUMBER.  branch_d8flexcert.py / branch_d10flexcert.py /
branch_d12flexcert.py pinned flex = 1 EXACTLY at d=8/10/12.  An uncertified two-prime probe at
d=6 gave rank_p(J) = 3039, nullity 321 = gauge(315) + 6, i.e. flex = 6, NOT 1.  This file
replaces that probe with a certificate in the same METHOD (branch_d8flexcert's machinery,
imported unchanged where possible; same Jacobian/trivial-motion convention `branch_d6geo.
build_flex_rows`, same two rational circle points x5=(3+4i)/5 / x13=(5+12i)/13, same
restriction trick, same two-prime modular standard), EXTENDED from a 1-vector to a k-vector
certificate:

  * LOWER BOUND (exact, float-free -- python ints / Fractions only on the proof path):
      (1) J . v_theta = 0 exactly, every row (the mechanism direction, D10_AUDIT Sec.2.2
          convention verbatim);
      (2) J . t = 0 exactly for every gauge generator t (int64 sparse product, proven
          overflow bound), so span(T) is inside ker J;
      (3) FIVE further EXACT kernel vectors w_1..w_5: candidates are DISCOVERED mod p
          (flint nullspace + quotient-by-gauge -- discovery is NOT part of the proof), then
          SOLVED EXACTLY over Q (flint fmpz_mat.solve on a 3039x3039 integer subsystem;
          the returned rational vector is then verified J . w = 0 on ALL 6848 rows in exact
          integer arithmetic after clearing denominators -- the verification, not the solver,
          is the certificate).  The vectors are startlingly small (integerized: max entry
          148 with lcm denominator <= 90 at x5, max entry 4732 with lcm denominator <= 650
          at x13).
      (4) INDEPENDENCE BEYOND GAUGE: restrict all vectors to one witness subset S of rays
          (restriction is a linear projection, so a rank jump on ANY subset proves global
          non-membership); exact Fraction Gaussian elimination gives
          rank_Q(T|_S + {v_theta, w_1..w_5}|_S) = rank_Q(T|_S) + 6.
          (S has |S| = 10 > d rays, so S is never one of the 95 bases -- the d=8/10/12
          finding that a complete basis cannot witness, u(d) being transitive on frames,
          is moot here by construction.)
    (1)-(4)  =>  dim ker J >= rank(T) + 6  =>  FLEX >= 6, independently at x5 AND x13.
  * UPPER BOUND (modular, the program's accepted standard): rank_p(J) with python-flint at
    998244353 and 999999937 (full 6848x3360 J -- it fits at d=6, as at d=8), PLUS the
    exact-integer sparse random row-compression C = R.J at the two exact-BLAS primes
    1000003 / 1048583 (rank_p(C) <= rank_Q(J) for ANY integer R) and a flint cross-check of
    C at the big primes -- THREE unit families, FOUR primes, all > 1e6, all required to
    agree.  rank_p <= rank_Q always, and the certificate supplies the opposite inequality
    rank_Q(J) <= n - rank(T) - 6 = 3360 - 315 - 6 = 3039; a modular rank that HITS 3039
    closes the squeeze: rank_Q(J) = 3039, nullity = 321, FLEX EXACTLY 6.
  * gauge tangent: rank_Q(T) = V + d^2 - 1 = 315 exactly (exact dependency ceiling + two-prime
    modular saturation, exactly as at d=8/10/12).
  * CHARACTERISATION (stage `support`): is the extra 5-dim block "decoration flex" (motions
    of a few decoration rays), as the tower paper suggests for its d=6 cores?  Answer: NO.
      - single-ray scan (EXACT, Fraction ranks of every ray's local constraint block, both
        points): every one of the 280 rays has local kernel = its own phase, i.e. ZERO
        single-ray decoration flex;
      - the certified vectors each move 101..280 of the 280 rays;
      - (mod-p, descriptive) a greedy joint-avoidance scan finds that representatives of all
        6 flex classes can simultaneously avoid only ~5-7 rays: the deformation space is
        GLOBAL, not decorative.  The d=6 rung genuinely breaks the d=8/10/12 flex=1 pattern.

STAGES (CLI dispatch; every stage checkpoints to d6flexcert_*.cache.json):
    python3 branch_d6flexcert.py gate      # sparse builder == branch_d6geo.build_flex_rows
    python3 branch_d6flexcert.py cert      # exact flex >= 6 certificate at x5 AND x13
    python3 branch_d6flexcert.py gauge     # gauge-tangent rank: exact ceiling + 2-prime ranks
    python3 branch_d6flexcert.py rankj [budget]   # rank(J): 3 unit families, 4 primes
    python3 branch_d6flexcert.py support   # characterisation of the 6 kernel directions
    python3 branch_d6flexcert.py report    # PASS/FAIL summary + the tower-v3 sentence
    python3 branch_d6flexcert.py all
No existing file is modified.  No git.
"""
import os, sys, time, random, hashlib
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fractions import Fraction
from math import lcm
from collections import defaultdict, Counter

import numpy as np

from ks_flex_census import cache_save, cache_load
import branch_d6geo as bg6           # build_flex_rows (read-only, the CONVENTION)
import branch_d8flexcert as fc       # the d=8/10/12 machinery, imported UNCHANGED

# Scratch for compression/elimination checkpoints only (regenerable, no proof content).
for _cand in ("/sessions/friendly-exciting-ptolemy/tmp", "/tmp"):
    if os.path.isdir(_cand) and os.access(_cand, os.W_OK):
        fc.TMP = os.path.join(_cand, "d6flexcert_scratch")
        os.makedirs(fc.TMP, exist_ok=True)
        break

CFG_D6 = dict(
    tag="d6flexcert", d=6, core_cache="d6galois_core_final",
    exp_V=280, exp_pairs=3284, exp_bases=95,
    primes=fc.PRIMES,                # 998244353, 999999937
    full_rankJ=True,                 # J is 6848 x 3360: full mod-p rank fits easily
    gate_full=True,
    gate_subcore=90,
    disc_prime=fc.PRIMES[0],         # discovery prime (NOT part of the proof path)
)


# ======================================================================================
# Core loading.  The d=6 Galois core cache predates the d8/10/12 flag convention: it has no
# `uncolorable`/`all_critical`/`complete` flags (those closes live in the d6galois stage-3
# caches), but it DOES carry the explicit 3284-pair list.  Validation here: exact counts,
# every basis is a 6-clique of the pair graph, pair list well-formed.
# ======================================================================================
def load_core(cfg):
    core = cache_load(cfg["core_cache"])
    assert core is not None, f"{cfg['core_cache']}.cache.json missing"
    core_syms = [tuple(v) for v in core["core_syms"]]
    core_bases = [tuple(sorted(b)) for b in core["core_bases"]]
    core_pairs = sorted(tuple(p) for p in core["core_pairs"])
    assert len(core_syms) == cfg["exp_V"], (len(core_syms), cfg["exp_V"])
    assert len(core_bases) == cfg["exp_bases"], (len(core_bases), cfg["exp_bases"])
    assert len(core_pairs) == cfg["exp_pairs"], (len(core_pairs), cfg["exp_pairs"])
    assert all(len(b) == cfg["d"] for b in core_bases)
    pairset = set(core_pairs)
    assert all(i < j for i, j in core_pairs)
    from itertools import combinations as _comb
    for b in core_bases:
        assert all((x, y) in pairset for x, y in _comb(b, 2)), "basis not a clique of pairs"
    assert all(sum(1 for s in v if s in ("X", "-X")) == 2 for v in core_syms), \
        "core is documented as the X-count==2-only Galois core"
    return core_syms, core_bases, core_pairs


# ======================================================================================
# STAGE GATE -- fc.sparse_flex_rows is entry-by-entry IDENTICAL to branch_d6geo.build_flex_rows
# on THIS core (fc.stage_gate's own check, re-assembled locally only because fc.load_core
# expects the d8/10/12 cache flag convention).
# ======================================================================================
def stage_gate(cfg):
    print("=" * 100)
    print(f"[{cfg['tag']}] GATE -- sparse row builder == branch_d6geo.build_flex_rows, exactly")
    print("=" * 100)
    t0 = time.time()
    core_syms, core_bases, _ = load_core(cfg)
    checks = []
    tasks = []
    if cfg.get("gate_full"):
        tasks.append(("full core", core_syms))
    tasks.append((f"subcore[:{cfg['gate_subcore']}]", core_syms[:cfg["gate_subcore"]]))
    for label, syms in tasks:
        for ptn in ("x5", "x13"):
            rays_ri = fc.rays_at(syms, ptn)
            ref = bg6.build_flex_rows(rays_ri)          # the published dense convention
            got = fc.sparse_flex_rows(rays_ri)          # the d=8/10/12 sparse mirror
            ok = (ref["E"] == got["E"] and ref["n"] == got["n"]
                  and len(ref["rows"]) == len(got["rows"])
                  and len(ref["triv"]) == len(got["triv"]))
            assert ok, f"shape/edge mismatch on {label}@{ptn}"
            for rd, rs in zip(ref["rows"], got["rows"]):
                assert rd == fc.densify(rs, got["n"]), f"J row mismatch on {label}@{ptn}"
            for td, ts in zip(ref["triv"], got["triv"]):
                assert td == fc.densify(ts, got["n"]), f"T row mismatch on {label}@{ptn}"
            print(f"[gate] {label}@{ptn}: |E|={len(got['E'])} rows={len(got['rows'])} "
                  f"triv={len(got['triv'])} n={got['n']} -- IDENTICAL")
            checks.append([label, ptn, len(got["E"])])
            del ref, got
    out = dict(passed=True, checks=checks, secs=round(time.time() - t0, 2))
    cache_save(f"{cfg['tag']}_gate", out)
    print(f"[gate] PASS  ({out['secs']}s)")
    return out


# ======================================================================================
# mod-p helpers (DISCOVERY ONLY -- nothing below feeds the proof path except through the
# exact re-verification that follows it).
# ======================================================================================
def _flint_from_sparse(rows, n, p):
    import flint
    M = flint.nmod_mat(len(rows), n, p)
    for r, row in enumerate(rows):
        for i, v in row:
            vv = v % p
            if vv:
                M[r, i] = vv
    return M


def _pivots_of_rref(Rm, rk, ncols):
    piv = []
    for row in range(rk):
        for col in range(ncols):
            if int(Rm[row, col]) != 0:
                piv.append(col)
                break
    return piv


def _mm_modp(A, B, p):
    """(A @ B) % p, int64-safe via 15-bit split of B."""
    A = A % p
    B = B % p
    return ((A @ (B & 32767)) % p + (((A @ (B >> 15)) % p) << 15)) % p


def _rref_modp_np(A, p):
    A = A.copy() % p
    m, nc = A.shape
    r = 0
    pivs = []
    for col in range(nc):
        nz = np.nonzero(A[r:, col])[0]
        if nz.size == 0:
            continue
        i = r + int(nz[0])
        if i != r:
            A[[r, i]] = A[[i, r]]
        inv = pow(int(A[r, col]), p - 2, p)
        A[r] = (A[r] * inv) % p
        colv = A[:, col].copy()
        colv[r] = 0
        mask = np.nonzero(colv)[0]
        if mask.size:
            A[mask] = (A[mask] - np.outer(colv[mask], A[r]) % p) % p
        pivs.append(col)
        r += 1
        if r == m:
            break
    return A[:r], pivs


def _discover_structure(fr, p):
    """mod-p structure of J at the discovery prime: pivot/free columns, an independent row
    subset, the nullspace, and the free columns whose reduced-echelon kernel vector is
    OUTSIDE the gauge span (the candidates for exact solving)."""
    import flint
    n = fr["n"]
    M = _flint_from_sparse(fr["rows"], n, p)
    R1 = M.rref()
    Rm, rk = R1 if isinstance(R1, tuple) else (R1, R1.rank())
    pivcols = _pivots_of_rref(Rm, rk, n)
    free = sorted(set(range(n)) - set(pivcols))
    del M, Rm
    # independent rows via the transpose
    Mt = flint.nmod_mat(n, len(fr["rows"]), p)
    for r, row in enumerate(fr["rows"]):
        for i, v in row:
            vv = v % p
            if vv:
                Mt[i, r] = vv
    R2 = Mt.rref()
    Rt, rkt = R2 if isinstance(R2, tuple) else (R2, R2.rank())
    assert rkt == rk
    rowsR = _pivots_of_rref(Rt, rkt, len(fr["rows"]))
    del Mt, Rt
    # nullspace
    M = _flint_from_sparse(fr["rows"], n, p)
    X, nullity = M.nullspace()
    assert nullity == n - rk
    N = np.zeros((n, nullity), dtype=np.int64)
    for jj in range(nullity):
        for ii in range(n):
            v = int(X[ii, jj])
            if v:
                N[ii, jj] = v
    del M, X
    for k in range(nullity):                       # flint nullspace column k <-> free col k
        assert int(N[free[k], k]) % p == 1
    # gauge echelon, then quotient: which nullspace columns create NEW pivots beyond gauge?
    Tm = np.zeros((len(fr["triv"]), n), dtype=np.int64)
    for r, row in enumerate(fr["triv"]):
        for i, v in row:
            Tm[r, i] += v
    Tech, pivT = _rref_modp_np(Tm, p)
    reps, reppivs, newcols = [], [], []
    for k in range(nullity):
        v = N[:, k].copy() % p
        a = v[pivT] % p
        v = (v - _mm_modp(a[None, :], Tech, p)[0]) % p
        for rp, rv in zip(reppivs, reps):
            aa = int(v[rp])
            if aa:
                v = (v - aa * rv) % p
        nz = np.nonzero(v)[0]
        if nz.size:
            inv = pow(int(v[int(nz[0])]), p - 2, p)
            reps.append((v * inv) % p)
            reppivs.append(int(nz[0]))
            newcols.append(k)
    nongauge_free = [free[k] for k in newcols]
    return dict(rank_p=rk, pivcols=pivcols, free=free, rowsR=rowsR,
                nullity_p=int(nullity), gauge_rank_p=len(pivT),
                quotient_dim_p=len(newcols), nongauge_free=nongauge_free)


def _solve_kernel_vectors_exact(fr, pivcols, rowsR, freecols):
    """EXACT rational kernel vectors: for each f in freecols the vector v with v_f = 1,
    v_g = 0 on the other free columns, v_pivcols solved by flint fmpz_mat.solve over Q.
    Returns {f: dense list of Fraction}.  (The solver output is certified only by the exact
    re-verification in the caller.)"""
    import flint
    m = len(pivcols)
    colpos = {c: k for k, c in enumerate(pivcols)}
    Am = flint.fmpz_mat(m, m)
    B = flint.fmpz_mat(m, len(freecols))
    fpos = {f: k for k, f in enumerate(freecols)}
    for a, r in enumerate(rowsR):
        for i, v in fr["rows"][r]:
            if i in colpos:
                Am[a, colpos[i]] = int(Am[a, colpos[i]]) + int(v)
            elif i in fpos:
                B[a, fpos[i]] = int(B[a, fpos[i]]) - int(v)
    X = Am.solve(B)                                  # fmpq_mat, A X = B
    out = {}
    for f in freecols:
        v = [Fraction(0)] * fr["n"]
        v[f] = Fraction(1)
        k = fpos[f]
        for i, c in enumerate(pivcols):
            v[c] = Fraction(str(X[i, k]))
        out[f] = v
    return out


def _integerize(v):
    dens = [x.denominator for x in v if x != 0]
    L = lcm(*dens) if dens else 1
    return [int(x * L) for x in v], L


def _support_rays(vec, V, d):
    return [i for i in range(V) if any(vec[12 * i + t] != 0 for t in range(2 * d))]


# ======================================================================================
# STAGE CERT -- the exact flex >= 6 certificate (k-vector extension of D10_AUDIT Sec.2.2).
# ======================================================================================
def cert_at_point(cfg, point_name, core_syms, core_bases, core_pairs):
    d = cfg["d"]
    t0 = time.time()
    rays_ri = fc.rays_at(core_syms, point_name)
    fr = fc.sparse_flex_rows(rays_ri)
    V, n = fr["V"], fr["n"]
    print(f"[cert:{point_name}] V={V} d={d} n={n} |E|={len(fr['E'])} "
          f"(expect {cfg['exp_pairs']}) Jrows={len(fr['rows'])} triv={len(fr['triv'])}")
    assert len(fr["E"]) == cfg["exp_pairs"], "point-degenerate: extra/missing orthogonal pairs"
    assert sorted(fr["E"]) == core_pairs, "edge set differs from the cached theta-identical set"
    assert len(fr["rows"]) == 2 * cfg["exp_pairs"] + V
    assert len(fr["triv"]) == V + d * d

    # ---- (1) J . v_theta == 0, EVERY row, exact python integers -------------------------
    vt = fc.v_theta(core_syms, rays_ri)
    assert any(vt)
    bad = sum(1 for row in fr["rows"] if fc.sparse_dot(row, vt) != 0)
    print(f"[cert:{point_name}] (1) J.v_theta == 0 exactly: violations = {bad} of "
          f"{len(fr['rows'])} rows")
    assert bad == 0

    # ---- (2) J . t == 0 for every gauge generator (exact int64, overflow-bounded) -------
    J = fc.scipy_J(fr)
    maxJrowsum = int(np.abs(J).sum(axis=1).max())
    Tmat = np.zeros((len(fr["triv"]), n), dtype=np.int64)
    for r, trow in enumerate(fr["triv"]):
        for i, v in trow:
            Tmat[r, i] += v
    bound = maxJrowsum * int(np.abs(Tmat).max())
    assert bound < 2 ** 62
    nzero = int(np.count_nonzero(J @ Tmat.T))
    print(f"[cert:{point_name}] (2) J.t == 0 for all {len(fr['triv'])} gauge generators: "
          f"nonzeros = {nzero} (overflow bound {bound} < 2^62)")
    assert nzero == 0
    del J, Tmat

    # ---- (3) exact extra kernel vectors: mod-p discovery, EXACT solve, EXACT verify -----
    p_disc = cfg["disc_prime"]
    disc = _discover_structure(fr, p_disc)
    print(f"[cert:{point_name}] (3) discovery mod {p_disc} (NOT proof): rank_p={disc['rank_p']} "
          f"nullity_p={disc['nullity_p']} gauge_rank_p={disc['gauge_rank_p']} "
          f"quotient_dim_p={disc['quotient_dim_p']} nongauge free cols {disc['nongauge_free']}")
    kv = _solve_kernel_vectors_exact(fr, disc["pivcols"], disc["rowsR"], disc["nongauge_free"])
    wvecs = []                                       # (name, integer vector, lcm denominator)
    for f in disc["nongauge_free"]:
        vi, L = _integerize(kv[f])
        nbad = sum(1 for row in fr["rows"] if fc.sparse_dot(row, vi) != 0)
        hnum = max(abs(x) for x in vi)
        print(f"[cert:{point_name}]     w(f={f}): EXACT J.w == 0 violations {nbad}/"
              f"{len(fr['rows'])}  (integerized: lcm(den)={L}, max|entry|={hnum})")
        assert nbad == 0, f"exact verification FAILED for candidate f={f}"
        wvecs.append((f"w{f}", vi, L))

    # ---- (4) independence beyond gauge: exact Fraction rank jump on a witness subset ----
    # |S| = 10 > d, so S is never one of the 95 complete bases (those have exactly d rays).
    def restrict(vec_or_sparse, S):
        cols = [fc.coord(i, c, rl, d) for i in S for c in range(d) for rl in (True, False)]
        if vec_or_sparse and isinstance(vec_or_sparse[0], tuple):
            dense = defaultdict(int)
            for i, v in vec_or_sparse:
                dense[i] += v
            return [dense.get(cc, 0) for cc in cols]
        return [vec_or_sparse[cc] for cc in cols]

    cands = [("v_theta", vt)] + [(nm, vi) for nm, vi, _ in wvecs]
    rnd = random.Random(20260731)
    subset_used = r0 = rfin = None
    kept = []
    for trial in range(60):
        size = 10 + 2 * (trial // 20)
        S = tuple(sorted(rnd.sample(range(V), size)))
        TS = [restrict(t, S) for t in fr["triv"]]
        ra = fc.rank_fraction(TS)
        cur = list(TS)
        rprev = ra
        got = []
        for name, vv in cands:
            vS = restrict(vv, S)
            if not any(vS):
                continue
            rn = fc.rank_fraction(cur + [vS])
            if rn == rprev + 1:
                got.append(name)
                cur.append(vS)
                rprev = rn
        if rprev - ra >= len(got) and len(got) > len(kept):
            subset_used, r0, rfin, kept = S, ra, rprev, got
        if len(kept) >= 6:
            break
    k_cert = rfin - r0
    print(f"[cert:{point_name}] (4) witness subset S = {subset_used} (|S|={len(subset_used)}, "
          f"not a basis)")
    print(f"[cert:{point_name}]     rank_Q(T|_S) = {r0} -> rank_Q(T|_S + kept) = {rfin}  "
          f"(exact Fractions; kept = {kept})")
    assert k_cert == len(kept) >= 1
    print(f"[cert:{point_name}] *** {k_cert} exact kernel vectors independent beyond gauge  "
          f"=>  dim ker J >= rank(T) + {k_cert}  =>  FLEX >= {k_cert} ***")

    supports = {"v_theta": len(_support_rays(vt, V, d))}
    for nm, vi, _ in wvecs:
        supports[nm] = len(_support_rays(vi, V, d))
    E_hash = hashlib.md5(repr(sorted(fr["E"])).encode()).hexdigest()
    secs = round(time.time() - t0, 2)
    return dict(point=point_name, V=V, n=n, E=len(fr["E"]), kernel_violations=0,
                gauge_product_nonzeros=0, disc_prime=p_disc,
                rank_p_disc=disc["rank_p"], nullity_p_disc=disc["nullity_p"],
                quotient_dim_p_disc=disc["quotient_dim_p"],
                nongauge_free=disc["nongauge_free"],
                vectors={nm: dict(lcm_den=L,
                                  entries=[[int(i), int(v)] for i, v in enumerate(vi) if v])
                         for nm, vi, L in wvecs},
                supports=supports, subset=list(subset_used), rank_TS=r0,
                rank_TS_plus_kept=rfin, kept=kept, k_cert=k_cert,
                flex_ge=k_cert, E_hash=E_hash, secs=secs)


def stage_cert(cfg):
    print("=" * 100)
    print(f"[{cfg['tag']}] CERT -- exact flex >= k certificate (k-vector extension of the "
          f"D10_AUDIT Sec.2.2 method) at x5 and x13")
    print("=" * 100)
    core_syms, core_bases, core_pairs = load_core(cfg)
    results = {}
    for ptn in ("x5", "x13"):                        # resumable: each point cached on finish
        res = cache_load(f"{cfg['tag']}_cert_{ptn}")
        if res and res.get("k_cert") and "E_hash" in res:
            print(f"[cert:{ptn}] cached: rank_Q(T|_S)={res['rank_TS']} -> "
                  f"{res['rank_TS_plus_kept']} with {res['kept']} => FLEX >= {res['k_cert']}")
        else:
            res = cert_at_point(cfg, ptn, core_syms, core_bases, core_pairs)
            cache_save(f"{cfg['tag']}_cert_{ptn}", res)
        results[ptn] = res
    ks = {q: results[q]["k_cert"] for q in results}
    assert len(set(ks.values())) == 1, f"x5 and x13 certify different k: {ks}"
    print(f"[cert] OVERALL: PASS -- flex >= {list(ks.values())[0]} certified at both points")
    return results


# ======================================================================================
# STAGE GAUGE -- gauge tangent dimension (fc.stage_gauge re-assembled with the local loader).
# ======================================================================================
def stage_gauge(cfg):
    print("=" * 100)
    print(f"[{cfg['tag']}] GAUGE -- rank of the trivial-motion span (V + d^2 generators)")
    print("=" * 100)
    t0 = time.time()
    d = cfg["d"]
    core_syms, _, _ = load_core(cfg)
    out = dict(primes=list(cfg["primes"]))
    for ptn in ("x5", "x13"):
        rays_ri = fc.rays_at(core_syms, ptn)
        fr = fc.sparse_flex_rows(rays_ri)
        V, n = fr["V"], fr["n"]
        ceiling = V + d * d - 1
        acc = defaultdict(int)
        for t in fr["triv"][:V]:
            for i, v in t:
                acc[i] += v
        for t in fr["triv"][V:V + d]:
            for i, v in t:
                acc[i] -= v
        dep_ok = all(v == 0 for v in acc.values())
        print(f"[gauge:{ptn}] generators = V + d^2 = {V + d*d}; exact dependency "
              f"sum(phase)==sum(diag): {dep_ok}  =>  rank_Q(T) <= {ceiling}")
        assert dep_ok
        ranks = {str(p): fc.flint_rank_sparse(fr["triv"], n, p) for p in cfg["primes"]}
        agree = len(set(ranks.values())) == 1
        rp = list(ranks.values())[0]
        exact = agree and rp == ceiling
        print(f"[gauge:{ptn}] rank_p(T) = {ranks}  agree={agree}  "
              f"{'SATURATES the ceiling => rank_Q(T) = ' + str(ceiling) + ' EXACTLY' if exact else 'below ceiling'}")
        assert agree
        out[ptn] = dict(generators=V + d * d, ceiling=ceiling, ranks=ranks,
                        exact=bool(exact), gauge_dim=rp)
    out["secs"] = round(time.time() - t0, 2)
    cache_save(f"{cfg['tag']}_gauge", out)
    print(f"[gauge] done ({out['secs']}s)")
    return out


# ======================================================================================
# STAGE RANKJ -- rank(J) modularly, squeezed against n - rank(T) - k_cert.
# Three unit families / four primes: full J with flint at the two big primes; the exact-
# integer compression C = R.J with flint at the two big primes AND with the exact-BLAS
# engine at the two small primes.  All must agree.
# ======================================================================================
def _build_C_d6(cfg, ptn, target):
    """fc.build_C re-assembled with the local loader (identical construction, identical
    overflow bound: C = R.J exactly over Z, rank(C) <= rank_Q(J) for ANY integer R)."""
    import scipy.sparse as sps
    m = target + 150
    path = os.path.join(fc.TMP, f"{cfg['tag']}_C_{ptn}_m{m}_r{fc.COMP_ROUNDS}_s{fc.COMP_SEED}.npy")
    if os.path.exists(path):
        return np.load(path), m
    core_syms, _, _ = load_core(cfg)
    fr = fc.sparse_flex_rows(fc.rays_at(core_syms, ptn))
    J = fc.scipy_J(fr)
    nrJ = J.shape[0]
    rng = random.Random(fc.COMP_SEED)
    data, ri, ci = [], [], []
    for rd in range(fc.COMP_ROUNDS):
        perm = list(range(nrJ))
        rng.shuffle(perm)
        for k, rowidx in enumerate(perm):
            data.append(rng.randrange(1, 1 << 20))
            ri.append(k % m)
            ci.append(rowidx)
    R = sps.coo_matrix((data, (ri, ci)), shape=(m, nrJ), dtype=np.int64).tocsr()
    maxJ = int(np.abs(J).max())
    bound = fc.COMP_ROUNDS * (nrJ // m + 1) * (1 << 20) * maxJ
    assert bound < 2 ** 62, "int64 overflow bound violated in compression"
    C = np.asarray((R @ J).todense(), dtype=np.int64)
    np.save(path, C)
    return C, m


def _flint_rank_dense(Cint, p):
    """rank over GF(p) of an exact-integer dense matrix (flat-list nmod_mat constructor)."""
    import flint
    Cm = (Cint % p).ravel().tolist()
    return int(flint.nmod_mat(Cint.shape[0], Cint.shape[1], Cm, p).rank())


def _finish_rankj_point_d6(cfg, ptn, res):
    n, rankT, k_cert, target = res["n"], res["rankT"], res["k_cert"], res["target"]
    vals = {**res.get("full", {}), **res.get("comp_flint", {}), **res.get("comp_blas", {})}
    rmax = max(vals.values()) if vals else -1
    res["rank_lower_bound"] = rmax
    two_prime_agree = len(set(res.get("full", {}).values())) == 1 and len(vals) >= 2
    if rmax == target and two_prime_agree:
        res["nullity_exact"] = rankT + k_cert
        res["flex_exact"] = k_cert
        print(f"[rankJ:{ptn}] *** SQUEEZE CLOSED: modular rank = {target} = n - rank(T) - "
              f"{k_cert} (>=2 primes, agreement)  =>  rank_Q(J) = {target}, nullity = "
              f"{rankT + k_cert}, FLEX EXACTLY {k_cert} ***")
    else:
        res["nullity_exact"] = None
        res["flex_exact"] = None
        res["flex_interval"] = [k_cert, n - rmax - rankT]
        print(f"[rankJ:{ptn}] squeeze not closed: best modular lower bound {rmax} vs target "
              f"{target}; flex in [{k_cert}, {n - rmax - rankT}]")
    res["done"] = True
    return res


def stage_rankj(cfg, points=("x5", "x13"), budget=32.0):
    print("=" * 100)
    print(f"[{cfg['tag']}] RANKJ -- rank(J) over GF(p) at 4 primes, squeezed for the exact "
          f"nullity")
    print("=" * 100)
    t0 = time.time()
    core_syms, _, _ = load_core(cfg)
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
        cert = cache_load(f"{cfg['tag']}_cert_{ptn}")
        assert cert is not None and cert.get("k_cert"), "run stage cert first"
        assert gauge[ptn]["exact"], "gauge rank not exact -- cannot squeeze"
        rankT = gauge[ptn]["gauge_dim"]
        k_cert = cert["k_cert"]
        if "n" not in res:
            n = 2 * cfg["d"] * cfg["exp_V"]
            res.update(n=n, rankT=rankT, k_cert=k_cert, target=n - rankT - k_cert)
        target = res["target"]
        print(f"[rankJ:{ptn}] n={res['n']} rank(T)={rankT} k_cert={k_cert} => certificate "
              f"ceiling rank_Q(J) <= {target}; nullity >= {rankT + k_cert}")

        # ---- unit family 1: FULL J, flint, big primes --------------------------------
        res.setdefault("full", {})
        for p in cfg["primes"]:
            if str(p) in res["full"]:
                continue
            if time.time() - t0 > budget:
                pending = True
                break
            fr = fc.sparse_flex_rows(fc.rays_at(core_syms, ptn))
            t1 = time.time()
            res["full"][str(p)] = fc.flint_rank_sparse(fr["rows"], fr["n"], p)
            print(f"[rankJ:{ptn}] FULL rank_p(J) mod {p} = {res['full'][str(p)]} "
                  f"({time.time()-t1:.1f}s)")
            out[ptn] = res
            cache_save(f"{cfg['tag']}_rankJ", out)

        # ---- unit family 2: compressed C = R.J, flint, big primes --------------------
        if not pending:
            res.setdefault("comp_flint", {})
            for p in cfg["primes"]:
                if str(p) in res["comp_flint"]:
                    continue
                if time.time() - t0 > budget:
                    pending = True
                    break
                C, m = _build_C_d6(cfg, ptn, target)
                t1 = time.time()
                res["comp_flint"][str(p)] = _flint_rank_dense(C, p)
                print(f"[rankJ:{ptn}] compressed rank_p(R.J) mod {p} (m={m}, flint) = "
                      f"{res['comp_flint'][str(p)]} ({time.time()-t1:.1f}s)")
                del C
                out[ptn] = res
                cache_save(f"{cfg['tag']}_rankJ", out)

        # ---- unit family 3: compressed C = R.J, exact-BLAS engine, small primes ------
        if not pending:
            res.setdefault("comp_blas", {})
            for p in fc.SMALL_PRIMES:
                if str(p) in res["comp_blas"]:
                    continue
                left = budget - (time.time() - t0)
                if left < 6:
                    pending = True
                    break
                C, m = _build_C_d6(cfg, ptn, target)
                done, r = fc.blas_rank_resumable(f"{cfg['tag']}_{ptn}", C, p, budget=left)
                del C
                if not done:
                    print(f"[rankJ:{ptn}] BLAS engine mod {p}: checkpointed at partial rank "
                          f"{r} -- RE-RUN this stage to continue")
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

        # ---- close out the point -------------------------------------------------------
        if (len(res["full"]) == len(cfg["primes"])
                and len(res["comp_flint"]) == len(cfg["primes"])
                and len(res["comp_blas"]) == len(fc.SMALL_PRIMES)):
            fvals = set(res["full"].values())
            cvals = set(res["comp_flint"].values()) | set(res["comp_blas"].values())
            assert len(fvals) == 1 and cvals == fvals, \
                f"engine/prime cross-validation FAILED: {res}"
            print(f"[rankJ:{ptn}] 4-prime / 3-family agreement: rank_p = {list(fvals)[0]}")
            out[ptn] = _finish_rankj_point_d6(cfg, ptn, res)
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


# ======================================================================================
# STAGE SUPPORT -- characterisation of the 6 kernel directions ("decoration or not?").
# ======================================================================================
def stage_support(cfg):
    print("=" * 100)
    print(f"[{cfg['tag']}] SUPPORT -- do the extra kernel directions move only decoration "
          f"rays?")
    print("=" * 100)
    t0 = time.time()
    d = cfg["d"]
    core_syms, _, core_pairs = load_core(cfg)
    V = len(core_syms)
    adj = {i: [] for i in range(V)}
    for i, j in core_pairs:
        adj[i].append(j)
        adj[j].append(i)
    out = cache_load(f"{cfg['tag']}_support") or {}
    for ptn in ("x5", "x13"):
        if isinstance(out.get(ptn), dict) and out[ptn].get("done"):
            print(f"[support:{ptn}] cached: single-ray movers "
                  f"{len(out[ptn]['single_ray_movers'])}, joint avoidable "
                  f"{out[ptn]['joint_avoidable_rays']}")
            continue
        rays_ri = fc.rays_at(core_syms, ptn)
        # ---- (a) EXACT single-ray scan: local kernel of every ray, Fraction ranks -------
        # A kernel vector supported on ray i alone must satisfy, restricted to ray i's 2d
        # coordinates: the Re/Im orthogonality rows of every edge at i, and i's norm row.
        # Local nullity 1 == the ray's own phase only == NO single-ray decoration flex.
        movers = []
        for i in range(V):
            rows = []
            for j in adj[i]:
                re = [0] * (2 * d)
                im = [0] * (2 * d)
                for cc in range(d):
                    Rej, Imj = rays_ri[j][cc]
                    re[2 * cc], re[2 * cc + 1] = Rej, Imj
                    im[2 * cc], im[2 * cc + 1] = Imj, -Rej
                rows.append(re)
                rows.append(im)
            nr = [0] * (2 * d)
            for cc in range(d):
                Re, Im = rays_ri[i][cc]
                nr[2 * cc], nr[2 * cc + 1] = Re, Im
            rows.append(nr)
            local_null = 2 * d - fc.rank_fraction(rows)
            assert local_null >= 1                    # the phase is always there
            if local_null > 1:
                movers.append([i, local_null - 1])
        print(f"[support:{ptn}] (a) EXACT single-ray scan: rays with local kernel beyond "
              f"their own phase = {len(movers)} of {V}"
              + (f" -- {movers}" if movers else "  =>  ZERO single-ray decoration flex"))
        # ---- (b) supports of the certified vectors --------------------------------------
        cert = cache_load(f"{cfg['tag']}_cert_{ptn}")
        assert cert is not None, "run stage cert first"
        supp = dict(cert["supports"])
        print(f"[support:{ptn}] (b) certified-vector ray supports (of {V} rays): " +
              ", ".join(f"{k}={v}" for k, v in sorted(supp.items())))
        # ---- (c) mod-p DESCRIPTIVE: greedy joint avoidance -------------------------------
        # How many rays can representatives of ALL flex classes simultaneously avoid?
        p = cfg["disc_prime"]
        fr = fc.sparse_flex_rows(rays_ri)
        import flint
        M = _flint_from_sparse(fr["rows"], fr["n"], p)
        X, nullity = M.nullspace()
        N = np.zeros((fr["n"], nullity), dtype=np.int64)
        for jj in range(nullity):
            for ii in range(fr["n"]):
                v = int(X[ii, jj])
                if v:
                    N[ii, jj] = v
        del M, X
        Tm = np.zeros((len(fr["triv"]), fr["n"]), dtype=np.int64)
        for r, row in enumerate(fr["triv"]):
            for i, v in row:
                Tm[r, i] += v
        Tech, pivT = _rref_modp_np(Tm, p)
        reps, reppivs, Phi_cols = [], [], []
        for k in range(nullity):
            v = N[:, k].copy() % p
            a = v[pivT] % p
            v = (v - _mm_modp(a[None, :], Tech, p)[0]) % p
            coords = {}
            for jix, (rp, rv) in enumerate(zip(reppivs, reps)):
                aa = int(v[rp])
                if aa:
                    v = (v - aa * rv) % p
                    coords[jix] = aa
            nz = np.nonzero(v)[0]
            if nz.size:
                jix = len(reps)
                lead = int(v[int(nz[0])])
                reps.append((v * pow(lead, p - 2, p)) % p)
                reppivs.append(int(nz[0]))
                coords[jix] = lead
            Phi_cols.append(coords)
        qd = len(reps)
        Phi = np.zeros((qd, nullity), dtype=np.int64)
        for k, coords in enumerate(Phi_cols):
            for jix, aa in coords.items():
                Phi[jix, k] = aa
        avoid_counts = []
        for seed in (0, 1, 2):
            rnd = random.Random(seed)
            order = list(range(V))
            rnd.shuffle(order)
            B = np.eye(nullity, dtype=np.int64)
            avoided = 0
            for i in order:
                Mi = _mm_modp(N[12 * i:12 * i + 2 * d, :], B, p)
                R, pivs = _rref_modp_np(Mi, p)
                freec = [jj for jj in range(Mi.shape[1]) if jj not in set(pivs)]
                Bn = np.zeros((Mi.shape[1], len(freec)), dtype=np.int64)
                for kk, f in enumerate(freec):
                    Bn[f, kk] = 1
                    for r2, pc in enumerate(pivs):
                        Bn[pc, kk] = (-int(R[r2, f])) % p
                B2 = _mm_modp(B, Bn, p)
                if B2.shape[1] >= qd and len(_rref_modp_np(_mm_modp(Phi, B2, p), p)[1]) == qd:
                    B = B2
                    avoided += 1
            avoid_counts.append(avoided)
        print(f"[support:{ptn}] (c) mod-{p} DESCRIPTIVE greedy: representatives of all {qd} "
              f"flex classes can jointly avoid only {avoid_counts} rays (3 seeds) of {V} "
              f"=>  the deformation space is GLOBAL, not decoration")
        out[ptn] = dict(single_ray_movers=movers, supports=supp,
                        quotient_dim_p=qd, joint_avoidable_rays=avoid_counts,
                        decoration_flex=len(movers) > 0, done=True)
        cache_save(f"{cfg['tag']}_support", out)
    out["secs"] = round(time.time() - t0, 2)
    cache_save(f"{cfg['tag']}_support", out)
    print(f"[support] done ({out['secs']}s)")
    return out


# ======================================================================================
# REPORT
# ======================================================================================
def report(cfg):
    print("=" * 100)
    print(f"[{cfg['tag']}] REPORT")
    print("=" * 100)
    rows = []
    g = cache_load(f"{cfg['tag']}_gate")
    rows.append(("gate: sparse builder == branch_d6geo.build_flex_rows (entry-by-entry)",
                 bool(g and g["passed"])))
    kc = None
    for ptn in ("x5", "x13"):
        c = cache_load(f"{cfg['tag']}_cert_{ptn}")
        if c:
            kc = c["k_cert"]
            rows.append((f"cert@{ptn}: J.v=0 exactly for v_theta + {len(c['vectors'])} solved "
                         f"vectors ({2*c['E']+c['V']} rows each), rank_Q(T|_S)={c['rank_TS']} "
                         f"-> +{c['k_cert']} with {c['kept']} => FLEX >= {c['k_cert']}",
                         c["k_cert"] >= 1))
    ga = cache_load(f"{cfg['tag']}_gauge")
    if ga:
        for ptn in ("x5", "x13"):
            e = ga[ptn]
            rows.append((f"gauge@{ptn}: dim = {e['gauge_dim']} (= V+d^2-1 = {e['ceiling']}, "
                         f"exact={e['exact']}, primes {list(e['ranks'].keys())})", e["exact"]))
    rj = cache_load(f"{cfg['tag']}_rankJ")
    fx = None
    if rj:
        for ptn in ("x5", "x13"):
            if ptn in rj and isinstance(rj[ptn], dict) and rj[ptn].get("done"):
                e = rj[ptn]
                if e.get("flex_exact") is not None:
                    fx = e["flex_exact"]
                    rows.append((f"rankJ@{ptn}: rank_Q(J)={e['target']} nullity="
                                 f"{e['nullity_exact']} => FLEX EXACTLY {e['flex_exact']} "
                                 f"(4 primes, 3 unit families)", True))
                else:
                    rows.append((f"rankJ@{ptn}: modular lower bound {e['rank_lower_bound']} "
                                 f"(target {e['target']}), flex in {e.get('flex_interval')}",
                                 False))
    su = cache_load(f"{cfg['tag']}_support")
    if su:
        for ptn in ("x5", "x13"):
            e = su[ptn]
            rows.append((f"support@{ptn}: single-ray decoration flex = "
                         f"{len(e['single_ray_movers'])} rays (EXACT); vector supports "
                         f"{e['supports']}; all classes jointly avoid only "
                         f"{e['joint_avoidable_rays']} rays (mod-p descriptive)",
                         len(e["single_ray_movers"]) == 0))
    allok = True
    for name, ok in rows:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
        allok &= bool(ok)
    print(f"\n  OVERALL: {'PASS' if allok else 'FAIL/INCOMPLETE'}")
    if allok and fx is not None and su is not None:
        print(f"\n  TOWER v3 NOTE: the d=6 Galois core's exact flex is {fx} (gauge 315, "
              f"nullity {315 + fx}, certified at x5 and x13): one mechanism direction "
              f"v_theta plus {fx - 1} further exact kernel directions -- and these are NOT "
              f"decoration flex: no single ray carries local flex, and every representative "
              f"set of the {fx} classes must move all but ~5 of the 280 rays; the d=6 rung "
              f"genuinely breaks the flex=1 uniformity of d=8/10/12 with a 6-dimensional "
              f"GLOBAL deformation space.")
    return allok


def main(cfg):
    which = sys.argv[1] if len(sys.argv) > 1 else "report"
    if which == "gate":
        stage_gate(cfg)
    elif which == "cert":
        stage_cert(cfg)
    elif which == "gauge":
        stage_gauge(cfg)
    elif which == "rankj":
        stage_rankj(cfg, budget=float(sys.argv[2]) if len(sys.argv) > 2 else 32.0)
    elif which == "support":
        stage_support(cfg)
    elif which == "report":
        report(cfg)
    elif which == "all":
        stage_gate(cfg)
        stage_cert(cfg)
        stage_gauge(cfg)
        stage_rankj(cfg)
        stage_support(cfg)
        report(cfg)
    else:
        print(f"unknown stage {which!r}")
        sys.exit(1)
    print(f"\n[{cfg['tag']} stage={which} total {time.time()-fc.T0:.1f}s]")


if __name__ == "__main__":
    main(CFG_D6)
