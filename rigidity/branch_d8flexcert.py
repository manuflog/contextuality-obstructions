#!/usr/bin/env python3
"""
branch_d8flexcert.py -- Branch D8-FLEXCERT: exact "flex >= 1" certificate for the published d=8
core (311 rays / 6128 pairs / 56 bases, `d8galois_core_final.cache.json`).

CLOSES A DECLARED GAP.  The tower paper's "What is not claimed" section states that the d=10
deformation is certified a genuine flex (an exact kernel vector of the constraint Jacobian
outside the gauge tangent, D10_AUDIT.md Sec.2.2) but that "the analogous flex certificates at
d=8 and d=12 have not been run".  This file runs the d=8 one; `branch_d12flexcert.py` (which
imports this file's machinery unchanged) runs the d=12 one.  The METHOD is the d=10 audit's,
replicated exactly -- same Jacobian/trivial-motion convention, same two rational circle points,
same restriction trick, same two-prime modular standard:

  * J and the trivial-motion (gauge) generator set T are `branch_d6geo.build_flex_rows`'s OWN
    convention, unchanged: variables = (Re,Im) of every coordinate of every ray (n = 2*d*V);
    J rows = 2 rows per orthogonal pair (Re/Im of the linearized Hermitian orthogonality)
    + 1 norm row per ray; T = V per-ray-phase directions + d^2 global-unitary u(d) directions
    (d diagonal + d(d-1) off-diagonal anti-Hermitian generators), so |T| = V + d^2 generators
    with the single exact dependency sum(phase rows) == sum(diagonal rows), hence
    rank(T) <= V + d^2 - 1 (the GAUGE TANGENT DIMENSION -- at d=10 this is 360+100-1 = 459;
    here 311+64-1 = 374 at d=8 and 412+144-1 = 555 at d=12).  This file only rebuilds those rows
    SPARSELY (the d=12 dense build does not fit memory); the `gate` stage proves the sparse
    build is entry-by-entry IDENTICAL to branch_d6geo.build_flex_rows.
  * v_theta = d/dtheta(rays), i.e. delta_{j,c} = i * e_{jc} * v_{j,c} (e_{jc} = X-exponent of
    the (j,c) entry), evaluated at the integer representatives of the published rational circle
    points x5 = (3+4i)/5 and x13 = (5+12i)/13 (branch_d6flex.POINTS, scaled by 5 resp. 13).
  * CERTIFICATE (exact, float-free -- the proof path uses ONLY python ints / Fractions and
    bounded int64 integer arithmetic):
      (1) J . v_theta = 0 exactly, EVERY row, pure-python integers;
      (2) J . t     = 0 exactly for every gauge generator t (int64 sparse product with a proven
          overflow bound), so span(T) is inside ker J;
      (3) v_theta NOT in span(T): restrict all vectors to one d-ray subset S (restriction is a
          linear projection, so non-membership of the restriction implies non-membership
          globally); exact Fraction Gaussian elimination gives rank(T|_S) and
          rank(T|_S + v_theta|_S) = rank(T|_S) + 1.
    (1)+(2)+(3)  =>  dim ker J >= rank(T) + 1  =>  FLEX >= 1: theta is a genuine deformation,
    not a hidden gauge orbit.  This is verified independently at BOTH points x5 and x13.
  * RANKS (modular, over the TWO primes 998244353 and 999999937 -- both > 1e6, agreement
    required; this is the program's accepted standard, cf. D10_AUDIT.md Sec.2.3):
      - rank_p(T): since rank_p <= rank_Q <= V + d^2 - 1 (exact ceiling above), a mod-p rank
        that HITS the ceiling proves rank_Q(T) = V + d^2 - 1 EXACTLY.
      - rank_p(J): the certificate gives rank_Q(J) <= n - rank(T) - 1, and rank_p <= rank_Q
        always, so a mod-p rank that hits n - rank(T) - 1 proves rank_Q(J), hence the EXACT
        nullity n - rank_Q(J) = rank(T) + 1, hence FLEX EXACTLY 1.  At d=8 J fits in memory and
        the full mod-p rank is computed directly (python-flint nmod_mat); the same run also
        validates the sparse random row-compression used at d=12 (rank of an exact-integer
        matrix C = R.J, always a LOWER bound for rank_Q(J), whatever R is).
  * Floats appear NOWHERE in the certificate path.  (No float cross-check is needed: every
    quantity is either exact or a two-prime modular rank.)

STAGES (CLI dispatch; every stage checkpoints to d8flexcert_*.cache.json):
    python3 branch_d8flexcert.py gate      # sparse builder == branch_d6geo.build_flex_rows
    python3 branch_d8flexcert.py cert      # the exact flex>=1 certificate at x5 AND x13
    python3 branch_d8flexcert.py gauge     # gauge-tangent rank: exact ceiling + 2-prime ranks
    python3 branch_d8flexcert.py rankj     # rank(J) mod 2 primes (+ compression validation)
    python3 branch_d8flexcert.py report    # PASS/FAIL summary
    python3 branch_d8flexcert.py all
No existing file is modified.  No git.
"""
import os, sys, time, random
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from itertools import combinations
from fractions import Fraction
from collections import defaultdict

import numpy as np

from ks_flex_census import cache_save, cache_load
import branch_d6flex as bd6          # POINTS (read-only)
import branch_d6geo as bg6           # build_flex_rows, hdot_ri_zero (read-only, the CONVENTION)

T0 = time.time()

PRIMES = (998244353, 999999937)      # both > 1e6; the program's standard pair

CFG_D8 = dict(
    tag="d8flexcert", d=8, core_cache="d8galois_core_final",
    exp_V=311, exp_pairs=6128, exp_bases=56,
    primes=PRIMES,
    full_rankJ=True,                 # J is 12567 x 4976: full mod-p rank fits this box
    gate_full=True,                  # dense reference build of the FULL core fits at d=8
    gate_subcore=90,
)


# ======================================================================================
# Core loading + the SPARSE mirror of branch_d6geo.build_flex_rows (gate-checked below).
# ======================================================================================
def load_core(cfg):
    core = cache_load(cfg["core_cache"])
    assert core is not None, f"{cfg['core_cache']}.cache.json missing"
    core_syms = [tuple(v) for v in core["core_syms"]]
    core_bases = [tuple(b) for b in core["core_bases"]]
    assert len(core_syms) == cfg["exp_V"], (len(core_syms), cfg["exp_V"])
    assert len(core_bases) == cfg["exp_bases"], (len(core_bases), cfg["exp_bases"])
    exp_pairs = cfg["exp_pairs"]
    if "core_pairs" in core:
        assert len(core["core_pairs"]) == exp_pairs
        core_pairs = sorted(tuple(p) for p in core["core_pairs"])
    else:
        assert core.get("npairs") == exp_pairs
        core_pairs = None
    assert core.get("uncolorable") and core.get("all_critical") and core.get("complete")
    return core_syms, core_bases, core_pairs


def rays_at(core_syms, point_name):
    pt = bd6.POINTS[point_name]
    return [tuple(pt[s] for s in ray) for ray in core_syms]


def coord(i, c, real, d):
    return 2 * d * i + 2 * c + (0 if real else 1)


def sparse_flex_rows(rays_ri):
    """Sparse mirror of branch_d6geo.build_flex_rows: IDENTICAL row order and content (proved
    entry-by-entry by stage `gate`), rows stored as [(col, val), ...] with zeros omitted."""
    V = len(rays_ri)
    d = len(rays_ri[0])
    E = [(i, j) for i, j in combinations(range(V), 2)
         if bg6.hdot_ri_zero(rays_ri[i], rays_ri[j])]
    n = 2 * d * V

    def C(i, c, real):
        return 2 * d * i + 2 * c + (0 if real else 1)

    rows = []
    for i, j in E:
        re, im = [], []
        for c in range(d):
            Rei, Imi = rays_ri[i][c]
            Rej, Imj = rays_ri[j][c]
            if Rej: re.append((C(i, c, True), Rej))
            if Imj: re.append((C(i, c, False), Imj))
            if Rei: re.append((C(j, c, True), Rei))
            if Imi: re.append((C(j, c, False), Imi))
            if Imj: im.append((C(i, c, True), Imj))
            if Rej: im.append((C(i, c, False), -Rej))
            if Imi: im.append((C(j, c, True), -Imi))
            if Rei: im.append((C(j, c, False), Rei))
        rows.append(re)
        rows.append(im)
    for i in range(V):
        r = []
        for c in range(d):
            Re, Im = rays_ri[i][c]
            if Re: r.append((C(i, c, True), Re))
            if Im: r.append((C(i, c, False), Im))
        rows.append(r)

    triv = []
    for i in range(V):                                   # V per-ray-phase directions
        t = []
        for c in range(d):
            Re, Im = rays_ri[i][c]
            if Im: t.append((C(i, c, True), -Im))
            if Re: t.append((C(i, c, False), Re))
        triv.append(t)
    for a in range(d):                                   # d diagonal u(d) directions
        t = []
        for i in range(V):
            Re, Im = rays_ri[i][a]
            if Im: t.append((C(i, a, True), -Im))
            if Re: t.append((C(i, a, False), Re))
        triv.append(t)
    for a in range(d):                                   # d(d-1) off-diagonal u(d) directions
        for b in range(a + 1, d):
            t1, t2 = [], []
            for i in range(V):
                Ra, Ia = rays_ri[i][a]
                Rb, Ib = rays_ri[i][b]
                if Ib: t1.append((C(i, a, True), -Ib))
                if Rb: t1.append((C(i, a, False), Rb))
                if Ia: t1.append((C(i, b, True), -Ia))
                if Ra: t1.append((C(i, b, False), Ra))
                if Rb: t2.append((C(i, a, True), Rb))
                if Ib: t2.append((C(i, a, False), Ib))
                if Ra: t2.append((C(i, b, True), -Ra))
                if Ia: t2.append((C(i, b, False), -Ia))
            triv.append(t1)
            triv.append(t2)
    return dict(V=V, d=d, E=E, n=n, rows=rows, triv=triv)


def v_theta(core_syms, rays_ri):
    """v_theta = d/dtheta(rays): delta_{j,c} = i * e_{jc} * v_{j,c}, i.e. (Re,Im) -> (-e*Im, e*Re)
    with e = 1 on +-X entries and 0 elsewhere -- D10_AUDIT.md Sec.2.2's convention verbatim."""
    V = len(rays_ri)
    d = len(rays_ri[0])
    vt = [0] * (2 * d * V)
    for j, ray in enumerate(core_syms):
        for c, s in enumerate(ray):
            if s in ("X", "-X"):
                Re, Im = rays_ri[j][c]
                vt[coord(j, c, True, d)] = -Im
                vt[coord(j, c, False, d)] = Re
    return vt


# ======================================================================================
# Exact helpers (proof path: python ints / Fractions only).
# ======================================================================================
def sparse_dot(row, vec):
    return sum(v * vec[i] for i, v in row)


def densify(row, n):
    out = [0] * n
    for i, v in row:
        out[i] += v
    return out


def rank_fraction(rows):
    """Exact rank over Q by Fraction Gaussian elimination (forward-only).  Small matrices."""
    rows = [[Fraction(x) for x in r] for r in rows if any(r)]
    if not rows:
        return 0
    ncols = len(rows[0])
    rank = 0
    for col in range(ncols):
        piv = None
        for r in range(rank, len(rows)):
            if rows[r][col] != 0:
                piv = r
                break
        if piv is None:
            continue
        rows[rank], rows[piv] = rows[piv], rows[rank]
        prow = rows[rank]
        pv = prow[col]
        for r in range(rank + 1, len(rows)):
            f = rows[r][col]
            if f != 0:
                q = f / pv
                rows[r] = [a - q * b for a, b in zip(rows[r], prow)]
        rank += 1
        if rank == len(rows):
            break
    return rank


def scipy_J(fr):
    """J as an exact int64 scipy CSR (values are small integers, |entry| <= 13)."""
    import scipy.sparse as sps
    data, ri, ci = [], [], []
    for r, row in enumerate(fr["rows"]):
        for i, v in row:
            ri.append(r); ci.append(i); data.append(v)
    return sps.coo_matrix((data, (ri, ci)),
                          shape=(len(fr["rows"]), fr["n"]), dtype=np.int64).tocsr()


def flint_rank_sparse(rows, n, p):
    """rank over GF(p) of a sparsely-given integer matrix, via python-flint nmod_mat."""
    import flint
    M = flint.nmod_mat(len(rows), n, p)
    for r, row in enumerate(rows):
        for i, v in row:
            vv = v % p
            if vv:
                M[r, i] = vv
    return int(M.rank())


def flint_rank_coo(C, p):
    """rank over GF(p) of an exact-integer scipy sparse matrix."""
    import flint
    Cc = C.tocoo()
    M = flint.nmod_mat(C.shape[0], C.shape[1], p)
    for i, j, v in zip(Cc.row.tolist(), Cc.col.tolist(), Cc.data.tolist()):
        vv = v % p
        if vv:
            M[i, j] = vv
    return int(M.rank())


# ======================================================================================
# STAGE GATE -- the sparse builder is entry-by-entry IDENTICAL to branch_d6geo.build_flex_rows.
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
            rays_ri = rays_at(syms, ptn)
            ref = bg6.build_flex_rows(rays_ri)          # the published dense convention
            got = sparse_flex_rows(rays_ri)             # this file's sparse mirror
            ok = (ref["E"] == got["E"] and ref["n"] == got["n"]
                  and len(ref["rows"]) == len(got["rows"])
                  and len(ref["triv"]) == len(got["triv"]))
            assert ok, f"shape/edge mismatch on {label}@{ptn}"
            for rd, rs in zip(ref["rows"], got["rows"]):
                assert rd == densify(rs, got["n"]), f"J row mismatch on {label}@{ptn}"
            for td, ts in zip(ref["triv"], got["triv"]):
                assert td == densify(ts, got["n"]), f"T row mismatch on {label}@{ptn}"
            print(f"[gate] {label}@{ptn}: |E|={len(got['E'])} rows={len(got['rows'])} "
                  f"triv={len(got['triv'])} n={got['n']} -- IDENTICAL")
            checks.append([label, ptn, len(got["E"])])
            del ref, got
    out = dict(passed=True, checks=checks, secs=round(time.time() - t0, 2))
    cache_save(f"{cfg['tag']}_gate", out)
    print(f"[gate] PASS  ({out['secs']}s)")
    return out


# ======================================================================================
# STAGE CERT -- the exact flex >= 1 certificate (D10_AUDIT.md Sec.2.2 verbatim), both points.
# ======================================================================================
def cert_at_point(cfg, point_name, core_syms, core_bases, core_pairs):
    d = cfg["d"]
    t0 = time.time()
    rays_ri = rays_at(core_syms, point_name)
    fr = sparse_flex_rows(rays_ri)
    V, n = fr["V"], fr["n"]
    print(f"[cert:{point_name}] V={V} d={d} n={n} |E|={len(fr['E'])} "
          f"(expect {cfg['exp_pairs']}) Jrows={len(fr['rows'])} triv={len(fr['triv'])}")
    assert len(fr["E"]) == cfg["exp_pairs"], "point-degenerate: extra/missing orthogonal pairs"
    if core_pairs is not None:
        assert sorted(fr["E"]) == core_pairs, "edge set differs from the cached theta-identical set"
    assert len(fr["rows"]) == 2 * cfg["exp_pairs"] + V
    assert len(fr["triv"]) == V + d * d

    # ---- (1) J . v_theta == 0, EVERY row, exact python integers -------------------------
    vt = v_theta(core_syms, rays_ri)
    assert any(vt), "v_theta is identically zero?!"
    bad = sum(1 for row in fr["rows"] if sparse_dot(row, vt) != 0)
    print(f"[cert:{point_name}] (1) J.v_theta == 0 exactly: violations = {bad} of "
          f"{len(fr['rows'])} rows (edge rows {2*len(fr['E'])}, norm rows {V})")
    assert bad == 0

    # ---- (2) J . t == 0 for every gauge generator (exact int64, overflow-bounded) -------
    J = scipy_J(fr)
    maxJrowsum = int(np.abs(J).sum(axis=1).max())
    Tmat = np.zeros((len(fr["triv"]), n), dtype=np.int64)
    for r, trow in enumerate(fr["triv"]):
        for i, v in trow:
            Tmat[r, i] += v
    maxT = int(np.abs(Tmat).max())
    bound = maxJrowsum * maxT
    assert bound < 2 ** 62, "int64 overflow bound violated"
    prod = J @ Tmat.T                                    # exact int64
    nzero = int(np.count_nonzero(prod))
    print(f"[cert:{point_name}] (2) J.t == 0 for all {len(fr['triv'])} gauge generators: "
          f"nonzeros = {nzero} (overflow bound {bound} < 2^62)")
    assert nzero == 0
    del prod, J

    # ---- (3) v_theta NOT in span(T): exact Fraction rank on a d-ray restriction ---------
    # Restriction to a subset S of rays is a linear projection: if v_theta were in span(T)
    # globally, its restriction would be in span(T|_S).  So rank jump on ANY subset proves
    # global non-membership.  Try the cached bases first (deterministic), then random subsets.
    def restrict(vec_sparse_or_dense, S):
        cols = [coord(i, c, rl, d) for i in S for c in range(d) for rl in (True, False)]
        if isinstance(vec_sparse_or_dense, list) and vec_sparse_or_dense and \
                isinstance(vec_sparse_or_dense[0], tuple):
            dense = defaultdict(int)
            for i, v in vec_sparse_or_dense:
                dense[i] += v
            return [dense.get(cc, 0) for cc in cols]
        return [vec_sparse_or_dense[cc] for cc in cols]

    subset_used, r1, r2 = None, None, None
    candidates = [tuple(sorted(b)) for b in core_bases]
    rnd = random.Random(20260731)
    candidates += [tuple(sorted(rnd.sample(range(V), d))) for _ in range(200)]
    for S in candidates:
        TS = [restrict(t, S) for t in fr["triv"]]
        vtS = restrict(vt, S)
        if not any(vtS):
            continue
        ra = rank_fraction(TS)
        rb = rank_fraction(TS + [vtS])
        if rb == ra + 1:
            subset_used, r1, r2 = S, ra, rb
            break
    print(f"[cert:{point_name}] (3) restriction subset S = {subset_used}")
    print(f"[cert:{point_name}]     rank_Q(T|_S) = {r1}   rank_Q(T|_S + v_theta|_S) = {r2}  "
          f"(exact Fraction arithmetic; the +1 JUMP is the certificate; generic ceiling "
          f"{d + d*d - 1})")
    assert subset_used is not None and r2 == r1 + 1, \
        "NO SUBSET CERTIFIES NON-MEMBERSHIP -- v_theta may lie in the gauge tangent (FLEX FAILURE)"

    secs = round(time.time() - t0, 2)
    print(f"[cert:{point_name}] *** J.v_theta = 0 exactly AND v_theta NOT in span(T)  "
          f"=>  dim ker J >= rank(T) + 1  =>  FLEX >= 1 ***  ({secs}s)")
    import hashlib
    E_hash = hashlib.md5(repr(sorted(fr["E"])).encode()).hexdigest()
    return dict(point=point_name, V=V, n=n, E=len(fr["E"]), kernel_violations=0,
                gauge_product_nonzeros=0, subset=list(subset_used), rank_TS=r1,
                rank_TS_plus_vtheta=r2, flex_ge_1=True, secs=secs, E_hash=E_hash)


def stage_cert(cfg):
    print("=" * 100)
    print(f"[{cfg['tag']}] CERT -- exact flex>=1 certificate (D10_AUDIT Sec.2.2 method) at "
          f"x5 and x13")
    print("=" * 100)
    core_syms, core_bases, core_pairs = load_core(cfg)
    results = {}
    for ptn in ("x5", "x13"):                            # resumable: each point cached on finish
        res = cache_load(f"{cfg['tag']}_cert_{ptn}")
        if res and res.get("flex_ge_1") and "E_hash" in res:
            print(f"[cert:{ptn}] cached: rank_Q(T|_S)={res['rank_TS']} -> "
                  f"{res['rank_TS_plus_vtheta']} with v_theta => FLEX >= 1")
        else:
            res = cert_at_point(cfg, ptn, core_syms, core_bases, core_pairs)
            cache_save(f"{cfg['tag']}_cert_{ptn}", res)
        results[ptn] = res
    if core_pairs is None:                               # d=12: no cached pair list -- require
        assert results["x5"]["E_hash"] == results["x13"]["E_hash"], \
            "edge sets differ between x5 and x13"
        print(f"[cert] edge sets at x5 and x13 are SET-IDENTICAL "
              f"({results['x5']['E']} pairs, md5 match)")
    ok = all(r["flex_ge_1"] for r in results.values())
    print(f"[cert] OVERALL: {'PASS -- flex >= 1 certified at both points' if ok else 'FAIL'}")
    assert ok
    return results


# ======================================================================================
# STAGE GAUGE -- the gauge tangent dimension: exact ceiling + two-prime modular saturation.
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
        rays_ri = rays_at(core_syms, ptn)
        fr = sparse_flex_rows(rays_ri)
        V, n = fr["V"], fr["n"]
        ceiling = V + d * d - 1
        # exact dependency: sum(V phase rows) == sum(d diagonal u(d) rows), python ints
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
        ranks = {}
        for p in cfg["primes"]:
            ranks[str(p)] = flint_rank_sparse(fr["triv"], n, p)
        agree = len(set(ranks.values())) == 1
        rp = list(ranks.values())[0]
        exact = agree and rp == ceiling
        print(f"[gauge:{ptn}] rank_p(T) = {ranks}  agree={agree}  "
              f"{'SATURATES the exact ceiling => rank_Q(T) = ' + str(ceiling) + ' EXACTLY' if exact else 'below ceiling -- NOT saturated'}")
        assert agree
        out[ptn] = dict(generators=V + d * d, ceiling=ceiling, ranks=ranks,
                        exact=bool(exact), gauge_dim=rp)
    out["secs"] = round(time.time() - t0, 2)
    cache_save(f"{cfg['tag']}_gauge", out)
    print(f"[gauge] done ({out['secs']}s)")
    return out


# ======================================================================================
# STAGE RANKJ -- rank(J) mod two primes; squeeze against n - rank(T) - 1 for the exact nullity.
#
# rank_p(M) <= rank_Q(M) for EVERY prime p, and rank_Q(R.J) <= rank_Q(J) for EVERY integer R.
# The cert stage gives the opposite inequality rank_Q(J) <= n - rank(T) - 1.  So ANY modular
# rank of J (or of an exact-integer compression C = R.J) that HITS n - rank(T) - 1 pins
# rank_Q(J), the nullity, and flex = 1 exactly.  Two primes with agreement are reported (the
# program's standard).  Two independent engines are used:
#   * python-flint nmod_mat over the big primes (998244353, 999999937) -- direct, used on the
#     full J where it fits (d=8) and on the compressed C;
#   * a blocked float64 elimination over the small primes (1000003, 1048583), both > 1e6, whose
#     every intermediate value is PROVABLY < 2^53 (b*(p-1)^2 < 2^53 for panel width b <= 512),
#     hence EXACT integer arithmetic despite the float carrier; it checkpoints to /tmp between
#     panels, so arbitrarily large eliminations survive this box's 45 s execution windows.
#     The d=8 run cross-validates every engine against the full-J flint rank.
# ======================================================================================
SMALL_PRIMES = (1000003, 1048583)    # both prime, both > 1e6, (p-1)^2 * 512 < 2^53 (BLAS-exact)
COMP_ROUNDS = 6                      # every J-row enters 6 distinct C-rows (indep. coefficients)
COMP_SEED = 1007
TMP = "/tmp"


def build_C(cfg, ptn, target):
    """C = R.J, computed EXACTLY over Z (int64 sparse product, proven overflow bound), cached
    as .npy.  R assigns every J-row to COMP_ROUNDS distinct C-rows with random nonzero
    coefficients < 2^20 (deterministic seed).  For ANY R, rank(C) <= rank_Q(J)."""
    import scipy.sparse as sps
    m = target + 150
    path = os.path.join(TMP, f"{cfg['tag']}_C_{ptn}_m{m}_r{COMP_ROUNDS}_s{COMP_SEED}.npy")
    if os.path.exists(path):
        return np.load(path), m
    core_syms, _, _ = load_core(cfg)
    fr = sparse_flex_rows(rays_at(core_syms, ptn))
    J = scipy_J(fr)
    nrJ = J.shape[0]
    rng = random.Random(COMP_SEED)
    data, ri, ci = [], [], []
    for rd in range(COMP_ROUNDS):
        perm = list(range(nrJ))
        rng.shuffle(perm)
        for k, rowidx in enumerate(perm):
            data.append(rng.randrange(1, 1 << 20))
            ri.append(k % m)
            ci.append(rowidx)
    R = sps.coo_matrix((data, (ri, ci)), shape=(m, nrJ), dtype=np.int64).tocsr()
    maxJ = int(np.abs(J).max())
    bound = COMP_ROUNDS * (nrJ // m + 1) * (1 << 20) * maxJ
    assert bound < 2 ** 62, "int64 overflow bound violated in compression"
    C = np.asarray((R @ J).todense(), dtype=np.int64)    # exact over Z
    np.save(path, C)
    return C, m


def _panel_eliminate(A, b, p):
    """In-place panel step of exact mod-p elimination on a float64 carrier.  DEFERRED REDUCTION:
    entries of A may carry unreduced (but exact, |.| < 2^52) integer values from earlier dgemm
    updates; every value is reduced mod p immediately before it is USED (its column, or the
    pivot-row panel segment), and every new intermediate is < 2^52 + b*(p-1)^2 < 2^53, so all
    float64 arithmetic is exact integer arithmetic.  Returns (#pivots, inv_list, pivcols)."""
    m, n = A.shape
    pivcols, invs = [], []
    r = 0
    for c in range(b):
        col = A[r:, c]
        np.mod(col, p, out=col)                          # reduce the column before use
        nz = np.nonzero(col)[0]
        if nz.size == 0:
            continue                                     # rank-deficient column: skip
        i = r + int(nz[0])
        if i != r:
            A[[r, i]] = A[[i, r]]                        # full-row swap
        np.mod(A[r, c:b], p, out=A[r, c:b])              # reduce pivot row (panel part)
        inv = pow(int(A[r, c]), p - 2, p)
        invs.append(inv)
        np.mod(A[r, c:b] * inv, p, out=A[r, c:b])        # normalize pivot row (panel cols)
        colv = A[r + 1:, c].copy()                       # multipliers, reduced (pivot now unit)
        if colv.size and c + 1 < b:
            block = A[r + 1:, c + 1:b]
            block -= np.outer(colv, A[r, c + 1:b])       # adds < p^2 per column, <= b of them
            A[r + 1:, c] = colv                          # keep multipliers for the dgemm update
        pivcols.append(c)
        r += 1
        if r == m:
            break
    return r, invs, pivcols


def blas_rank_resumable(key, C_int, p, budget, b=64):
    """Exact rank of C_int over GF(p) (p < 2^21) by blocked elimination on a float64 carrier
    with deferred reduction; checkpoints the active submatrix to /tmp between panels (this box
    kills processes at ~45 s).  Returns (done, rank_so_far).  Exactness: float64 represents
    every integer of magnitude < 2^53; all values are kept < 2^52 by construction (a full
    reduction pass is inserted whenever the deferred bound approaches 2^52), and every product/
    sum formed is < 2^52 + b*(p-1)^2 < 2^53."""
    assert 2 ** 52 + 2 * b * (p - 1) ** 2 < 2 ** 53, "exactness bound violated"
    stp = os.path.join(TMP, f"{key}_p{p}_state.npy")
    mtp = os.path.join(TMP, f"{key}_p{p}_meta.json")
    import json as _json
    if os.path.exists(mtp):
        meta = _json.load(open(mtp))
        if meta.get("done"):
            return True, meta["rank"]
        A = np.load(stp)                                 # saved fully reduced
        r_acc = meta["r_acc"]
    else:
        A = np.mod(C_int, p).astype(np.float64)
        r_acc = 0
    t0 = time.time()
    defer = float(p)                                     # current bound on |entries| of A22
    step = 2.0 * b * float(p - 1) ** 2                   # growth per panel (factor-2 safety)
    while A.shape[0] > 0 and A.shape[1] > 0:
        if time.time() - t0 > budget:
            np.mod(A, p, out=A)                          # checkpoint fully reduced
            np.save(stp, A)
            _json.dump(dict(done=False, r_acc=r_acc), open(mtp, "w"))
            return False, r_acc
        if defer + step > 2.0 ** 52:
            np.mod(A, p, out=A)                          # rare global reduction pass
            defer = float(p)
        bb = min(b, A.shape[1])
        npiv, invs, pivcols = _panel_eliminate(A, bb, p)
        if npiv and bb < A.shape[1]:
            U = A[:npiv, bb:]                            # forward-substitute the pivot rows:
            np.mod(U, p, out=U)                          # reduce before use (deferred values)
            for i in range(npiv):                        # U[i] = inv_i*(U[i] - sum l_ij U[j])
                if i:
                    li = A[i, pivcols[:i]]
                    np.mod(U[i] - li @ U[:i], p, out=U[i])   # sums < b*(p-1)^2 + p: exact
                np.mod(U[i] * invs[i], p, out=U[i])
            L21 = A[npiv:, pivcols]                      # multipliers, reduced by construction
            A22 = A[npiv:, bb:]
            A22 -= L21 @ U                               # exact dgemm, adds < b*(p-1)^2
        defer += step
        r_acc += npiv
        A = np.ascontiguousarray(A[npiv:, bb:])
    _json.dump(dict(done=True, rank=r_acc), open(mtp, "w"))
    if os.path.exists(stp):
        os.remove(stp)
    return True, r_acc


def _finish_rankj_point(cfg, ptn, res):
    n, rankT, target = res["n"], res["rankT"], res["target"]
    vals = {**res.get("full", {}), **res.get("comp_flint", {}), **res.get("comp_blas", {})}
    rmax = max(vals.values()) if vals else -1
    res["rank_lower_bound"] = rmax
    two_prime_agree = (len(set(res.get("full", res.get("comp_blas", {})).values())) == 1
                       and len(vals) >= 2)
    if rmax == target and two_prime_agree:
        res["nullity_exact"] = rankT + 1
        res["flex_exact"] = 1
        print(f"[rankJ:{ptn}] *** SQUEEZE CLOSED: modular rank = {target} = n - rank(T) - 1 "
              f"(>=2 primes, agreement)  =>  rank_Q(J) = {target}, nullity = {rankT + 1}, "
              f"FLEX EXACTLY 1 ***")
    else:
        res["nullity_exact"] = None
        res["flex_exact"] = None
        res["flex_interval"] = [1, n - rmax - rankT]
        print(f"[rankJ:{ptn}] squeeze not closed: best modular lower bound {rmax} vs target "
              f"{target}; flex in [1, {n - rmax - rankT}]")
    res["done"] = True
    return res


def stage_rankj(cfg, points=("x5", "x13"), budget=32.0):
    """Resumable: every (point, engine, prime) unit is cached the moment it finishes; re-run the
    stage until it prints ALL UNITS DONE (this box kills processes at ~45 s)."""
    print("=" * 100)
    print(f"[{cfg['tag']}] RANKJ -- rank(J) over GF(p), squeezed for the exact nullity")
    print("=" * 100)
    t0 = time.time()
    core_syms, _, _ = load_core(cfg)
    gauge = cache_load(f"{cfg['tag']}_gauge")
    assert gauge is not None, "run stage gauge first"
    out = cache_load(f"{cfg['tag']}_rankJ") or {}
    out["primes_flint"] = list(cfg["primes"])
    out["primes_blas"] = list(SMALL_PRIMES)
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
        # ---- unit family 1: full J, flint, big primes (d=8 only: J fits) -----------------
        if cfg["full_rankJ"]:
            res.setdefault("full", {})
            for p in cfg["primes"]:
                if str(p) in res["full"]:
                    continue
                if time.time() - t0 > budget:
                    pending = True
                    break
                fr = sparse_flex_rows(rays_at(core_syms, ptn))
                t1 = time.time()
                res["full"][str(p)] = flint_rank_sparse(fr["rows"], fr["n"], p)
                print(f"[rankJ:{ptn}] FULL rank_p(J) mod {p} = {res['full'][str(p)]} "
                      f"({time.time()-t1:.1f}s)")
                out[ptn] = res
                cache_save(f"{cfg['tag']}_rankJ", out)
        # ---- unit family 2: compressed C = R.J, flint, big primes (d=8 validation) -------
        if cfg["full_rankJ"]:
            res.setdefault("comp_flint", {})
            for p in cfg["primes"]:
                if str(p) in res["comp_flint"]:
                    continue
                if time.time() - t0 > budget:
                    pending = True
                    break
                C, m = build_C(cfg, ptn, target)
                t1 = time.time()
                import scipy.sparse as sps
                res["comp_flint"][str(p)] = flint_rank_coo(sps.coo_matrix(C), p)
                print(f"[rankJ:{ptn}] compressed rank_p(R.J) mod {p} (m={m}, flint) = "
                      f"{res['comp_flint'][str(p)]} ({time.time()-t1:.1f}s)")
                out[ptn] = res
                cache_save(f"{cfg['tag']}_rankJ", out)
        # ---- unit family 3: compressed C = R.J, exact-BLAS engine, small primes ----------
        res.setdefault("comp_blas", {})
        for p in SMALL_PRIMES:
            if str(p) in res["comp_blas"]:
                continue
            left = budget - (time.time() - t0)
            if left < 6:
                pending = True
                break
            C, m = build_C(cfg, ptn, target)
            done, r = blas_rank_resumable(f"{cfg['tag']}_{ptn}", C, p, budget=left)
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
        # ---- close out the point ---------------------------------------------------------
        blas_ok = len(res["comp_blas"]) == len(SMALL_PRIMES)
        full_ok = (not cfg["full_rankJ"]) or len(res.get("full", {})) == len(cfg["primes"])
        if blas_ok and full_ok:
            if cfg["full_rankJ"]:
                fvals = set(res["full"].values())
                cvals = set(res["comp_flint"].values()) | set(res["comp_blas"].values())
                assert len(fvals) == 1 and cvals == fvals, \
                    "compression/engine cross-validation FAILED against the full flint rank"
                print(f"[rankJ:{ptn}] cross-validation: full flint, compressed flint and "
                      f"compressed exact-BLAS all agree = {fvals}")
            out[ptn] = _finish_rankj_point(cfg, ptn, res)
            cache_save(f"{cfg['tag']}_rankJ", out)
    if pending:
        print(f"[rankJ] BUDGET REACHED -- progress cached; re-run `rankj` to continue")
    else:
        print(f"[rankJ] ALL UNITS DONE")
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
    for ptn in ("x5", "x13"):
        c = cache_load(f"{cfg['tag']}_cert_{ptn}")
        if c:
            rows.append((f"cert@{ptn}: J.v_theta=0 exactly ({2*c['E']+c['V']} rows), "
                         f"rank_Q(T|_S)={c['rank_TS']} -> +v_theta {c['rank_TS_plus_vtheta']} "
                         f"=> FLEX >= 1", c["flex_ge_1"]))
    ga = cache_load(f"{cfg['tag']}_gauge")
    if ga:
        for ptn in ("x5", "x13"):
            e = ga[ptn]
            rows.append((f"gauge@{ptn}: dim = {e['gauge_dim']} "
                         f"(= V+d^2-1 = {e['ceiling']}, exact={e['exact']}, "
                         f"primes {list(e['ranks'].keys())})", e["exact"]))
    rj = cache_load(f"{cfg['tag']}_rankJ")
    if rj:
        for ptn in ("x5", "x13"):
            if ptn in rj and isinstance(rj[ptn], dict) and rj[ptn].get("done"):
                e = rj[ptn]
                if e.get("flex_exact") == 1:
                    rows.append((f"rankJ@{ptn}: rank_Q(J)={e['target']} nullity={e['nullity_exact']}"
                                 f" => FLEX EXACTLY 1 (two-prime agreement)", True))
                else:
                    rows.append((f"rankJ@{ptn}: modular lower bound {e['rank_lower_bound']} "
                                 f"(target {e['target']}), flex in {e.get('flex_interval')}",
                                 e.get("rank_lower_bound", -1) >= 0))
    allok = True
    for name, ok in rows:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
        allok &= bool(ok)
    print(f"\n  OVERALL: {'PASS' if allok else 'FAIL/INCOMPLETE'}")
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
        stage_rankj(cfg)
    elif which == "report":
        report(cfg)
    elif which == "all":
        stage_gate(cfg)
        stage_cert(cfg)
        stage_gauge(cfg)
        stage_rankj(cfg)
        report(cfg)
    else:
        print(f"unknown stage {which!r}")
        sys.exit(1)
    print(f"\n[{cfg['tag']} stage={which} total {time.time()-T0:.1f}s]")


if __name__ == "__main__":
    main(CFG_D8)
