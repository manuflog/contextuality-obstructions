#!/usr/bin/env python3
"""
nonrigid_ks_tower.py -- V57: per-rung NON-RIGIDITY statements (Trandafir-Cabello sense,
fixed-d rank-one direction) for the even-dimensional KS tower, d = 4, 6, 8, 10, 12.

CLAIM ASSEMBLED (one statement per rung):
    The published circle core at dimension d admits (at least) two rank-one realizations of the
    SAME orthogonality-and-completeness structure -- the two Gaussian-rational circle points
    x5 = (3+4i)/5 and x13 = (5+12i)/13 -- that are inequivalent under EVERY map of the form
    (unitary or antiunitary) x (per-ray phases) x (ray relabeling).  By Trandafir-Cabello's
    definition of rigidity (arXiv:2501.11640, Eq. (1): every realization, any D >= d, any rank,
    unitarily equivalent to canonical (x) identity), two mutually inequivalent rank-one
    realizations at D = d already contradict rigidity: if the set were rigid both would be
    equivalent to the reference set, hence to each other.  Verdict per rung:
        NON-RIGID in the Trandafir-Cabello sense (fixed-d rank-one direction:
        two inequivalent realizations exist); first-order degeneracy dimension
        modulo gauge = <flex>   with flex = 1, 6, 1, 1, 1 at d = 4, 6, 8, 10, 12.
    The antiunitary robustness matters because Xu-Saha-Bharti-Cabello's Result 1 equivalence
    (PRL 132, 140201) allows a complex-conjugated block: our separating invariant is preserved
    by antiunitaries too, so the two realizations are inequivalent even in that wider sense.

SEPARATING INVARIANT (newly computed here; exact, float-free):
    the multiset  { |<v_i, v_j>|^2 / (|v_i|^2 |v_j|^2) : i < j }  of normalized squared
    Gram moduli over ALL ray pairs, as exact Fractions from the Gaussian-integer
    representatives.  Unitaries and antiunitaries both preserve every |<u,v>|; per-ray phases
    and relabelings permute/fix the multiset.  Distinct multisets at x5 and x13 therefore
    certify inequivalence under the full (anti)unitary x phases x relabeling group.
    (Along the circle each such invariant is a polynomial in cos(theta); complex conjugation
    maps the member at theta to the member at -theta, so conjugate parameter pairs are
    GENUINELY antiunitarily equivalent and no invariant separates them.  x5 and x13 have
    cos(theta) = 3/5 vs 5/13 and are separated.)

WHAT IS ASSEMBLED FROM EXISTING CACHES vs NEWLY COMPUTED:
    d = 6, 8, 10, 12 (assembled; caches produced by branch_d{6,8,10,12}flexcert.py and the
    d{6,8,10,12} core builders, all read-only here):
      * d{d}flexcert_cert_x5 / _cert_x13 : the exact flex >= 1 (>= 6 at d=6) kernel
        certificates at both points, including the md5 EDGE-SET HASH of the orthogonality
        structure realized at each point -- equality of the two hashes is the recorded check
        that both points realize the SAME structure (the theta-identical Laurent vanishing
        itself is the cited theorem of the tower/d4 papers, not re-proved here);
      * d{d}flexcert_gauge : exact gauge-tangent dimension V + d^2 - 1 (mod-p rank hits the
        exact ceiling at two primes > 1e6);
      * d{d}flexcert_rankJ (+ d12flexcert_rankJ_flint) : mod-p rank of the constraint Jacobian
        hits the certificate target at every recorded prime/engine => exact nullity => exact
        first-order flex (two-sided squeeze);
      * core caches (d8galois_core_final, d10_core_final, d12_core_final): the recorded
        complete/uncolorable/all_critical flags.
    d = 4 (newly computed here, in the exact style of branch_d8flexcert.py, because no
    flexcert-style cache exists for the M9 core; the only cache consumed is the peeled core
    index list d4flex_M9_done):
      * the 89-ray M9 core is rebuilt deterministically (branch_d4flex.generic_symbolic_rays
        + cached indices), evaluated at both points, and its edge set (433 pairs), 4-clique
        bases (35), KS-uncolorability and per-ray criticality are re-verified from scratch;
      * a full two-sided flex certificate is run: J.v_theta = 0 exactly (all rows, python
        ints); J.t = 0 for all V + d^2 gauge generators (int64, overflow-bounded);
        v_theta not in span(T) by exact Fraction rank on a restriction (=> flex >= 1);
        mod-p rank of T hits the exact ceiling V + d^2 - 1 = 104 and mod-p rank of J hits
        n - 105 = 607 at both primes 998244353, 999999937 (=> flex <= 1).  Hence flex = 1
        exactly, reproducing the d=4 paper's certificate with the tower-paper convention.
    ALL rungs (newly computed here): the Gram-modulus multisets at x5/x13 and their
    comparison; independent SAT re-checks (CaDiCaL) of KS-uncolorability of every core, and
    of per-ray criticality at d = 4 and d = 6 (the d = 6 Galois-core cache records no
    criticality flag; d = 8, 10, 12 criticality is read from the recorded flags).

HONEST SCOPE (printed with every verdict):
    * Fixed d, rank one, modulo per-ray phases + u(d): that is the direction of
      Trandafir-Cabello rigidity our certificates address.  Nothing here computes their FULL
      degeneracy (any D >= d, any rank); non-rigidity transfers because their definition
      quantifies over MORE realizations, "flex = their degeneracy dimension" does NOT.
    * flex is the FIRST-ORDER kernel dimension modulo gauge.  At every rung the circle
      realizes >= 1 of it exactly; at d = 6 three further dimensions integrate to the exact
      grading 3-torus (tower paper, Theorem d6torus -- an immersion statement, imported by
      citation, never re-proved here), and the remaining directions are unobstructed through
      fourth order with integrability beyond open.
    * If any cache is missing or any recorded rank misses its target, this script SAYS SO and
      downgrades that rung's verdict; it never fabricates a certificate.

STAGES (CLI dispatch; results checkpoint to nonrigid_*.cache.json; no existing file is
modified, all pre-existing caches are opened read-only):
    python3 nonrigid_ks_tower.py d4        # the newly-computed exact d=4 certificate
    python3 nonrigid_ks_tower.py assemble  # d=6,8,10,12 from the flexcert caches
    python3 nonrigid_ks_tower.py gram      # separating-invariant multisets, all rungs
    python3 nonrigid_ks_tower.py sat       # SAT uncolorability (+criticality d=4,6) re-checks
    python3 nonrigid_ks_tower.py report    # per-rung verdicts (the V57 statement)
    python3 nonrigid_ks_tower.py all

EXPECTED FINAL OUTPUT (stage report):
    every rung PASS; verdict lines
        d= 4: NON-RIGID (TC sense, fixed-d rank-one direction) ... flex modulo gauge = 1
        d= 6: NON-RIGID (TC sense, fixed-d rank-one direction) ... flex modulo gauge = 6
        d= 8, 10, 12: same with flex = 1
    and "V57 VERDICT: PASS".
"""
import os, sys, time, hashlib
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from itertools import combinations
from fractions import Fraction
from collections import Counter

from ks_flex_census import cache_save, cache_load
import branch_d6flex as bd6              # POINTS (read-only)
import branch_d6geo as bg6               # hdot_ri_zero (read-only, the convention)
import branch_d8flexcert as bx8          # sparse_flex_rows, v_theta, rank_fraction, ... (read-only)
import branch_d4flex as bd4              # generic_symbolic_rays (read-only, deterministic)

T0 = time.time()
PRIMES = (998244353, 999999937)          # the program's standard pair, both > 1e6

POINT_COS = {"x5": Fraction(3, 5), "x13": Fraction(5, 13)}   # cos(theta) at each circle point

RUNGS = {
    6:  dict(tag="d6flexcert",  core="d6galois_core_final", V=280, E=3284,  bases=95,
             n=3360, gauge=315, rank_target=3039, flex=6,  flags_cached=False),
    8:  dict(tag="d8flexcert",  core="d8galois_core_final", V=311, E=6128,  bases=56,
             n=4976, gauge=374, rank_target=4601, flex=1,  flags_cached=True),
    10: dict(tag="d10flexcert", core="d10_core_final",      V=360, E=10872, bases=47,
             n=7200, gauge=459, rank_target=6740, flex=1,  flags_cached=True),
    12: dict(tag="d12flexcert", core="d12_core_final",      V=412, E=19885, bases=61,
             n=9888, gauge=555, rank_target=9332, flex=1,  flags_cached=True),
}
D4 = dict(V=89, E=433, bases=35, n=712, gauge=104, rank_target=607, flex=1)


# ======================================================================================
# Shared exact helpers.
# ======================================================================================
def rays_at(core_syms, point_name):
    pt = bd6.POINTS[point_name]
    return [tuple(pt[s] for s in ray) for ray in core_syms]


def hdot_ri(u, v):
    """Exact Hermitian dot conj(u).v over Gaussian-integer (Re,Im) pairs -> (Re, Im)."""
    re = im = 0
    for (a, b), (c, dd) in zip(u, v):
        re += a * c + b * dd
        im += a * dd - b * c
    return re, im


def gram_multiset(rays_ri):
    """Exact multiset of normalized squared Gram moduli |<u,v>|^2/(|u|^2|v|^2), all pairs."""
    norms = [sum(a * a + b * b for a, b in v) for v in rays_ri]
    ms = Counter()
    V = len(rays_ri)
    for i in range(V):
        ri = rays_ri[i]
        for j in range(i + 1, V):
            re, im = hdot_ri(ri, rays_ri[j])
            ms[Fraction(re * re + im * im, norms[i] * norms[j])] += 1
    return ms


def multiset_fingerprint(ms):
    items = sorted((str(k), v) for k, v in ms.items())
    return hashlib.md5(repr(items).encode()).hexdigest()


def load_d4_core():
    """Deterministic rebuild of the published 89-ray M9 core: the mechanism-independent
    symbolic pool of branch_d4flex (272 rays, fixed order) indexed by the cached peel."""
    idx = cache_load("d4flex_M9_done")
    assert idx is not None, "d4flex_M9_done.cache.json missing"
    pool = bd4.generic_symbolic_rays(4)
    assert len(pool) == 272, f"symbolic pool changed: {len(pool)} rays (expect 272)"
    core_syms = [tuple(pool[i]) for i in idx]
    assert len(core_syms) == D4["V"]
    return core_syms


def edges_and_bases(rays_ri, d):
    V = len(rays_ri)
    E = [(i, j) for i, j in combinations(range(V), 2) if bg6.hdot_ri_zero(rays_ri[i], rays_ri[j])]
    adj = [set() for _ in range(V)]
    for i, j in E:
        adj[i].add(j); adj[j].add(i)
    bases = []
    def extend(cands, cur):
        if len(cur) == d:
            bases.append(tuple(cur)); return
        if len(cur) + len(cands) < d:
            return
        cl = sorted(cands)
        for k, v in enumerate(cl):
            extend(set(cl[k + 1:]) & adj[v], cur + [v])
    for s in range(V):
        extend(set(x for x in adj[s] if x > s), [s])
    return E, bases


def sat_colorable(V, pairs, bases):
    from pysat.solvers import Cadical153
    s = Cadical153()
    for i, j in pairs:
        s.add_clause([-(i + 1), -(j + 1)])
    for b in bases:
        s.add_clause([x + 1 for x in b])
    return s.solve()


def sat_all_critical(V, pairs, bases):
    for r in range(V):
        cp = [(i, j) for i, j in pairs if r not in (i, j)]
        cb = [b for b in bases if r not in b]
        if not sat_colorable(V, cp, cb):
            return False, r
    return True, None


def flint_rank_dense(rows, ncols, p):
    import flint
    M = flint.nmod_mat(len(rows), ncols, p)
    for r, row in enumerate(rows):
        for i, v in row:
            vv = v % p
            if vv:
                M[r, i] = vv
    return int(M.rank())


# ======================================================================================
# STAGE d4 -- the newly-computed exact certificate for the M9 rung.
# ======================================================================================
def stage_d4():
    print("=" * 100)
    print("STAGE d4 -- exact structure + two-sided flex certificate for the 89-ray M9 core")
    print("      (newly computed; only the cached core INDEX list is consumed)")
    print("=" * 100)
    core_syms = load_d4_core()
    out = dict(V=D4["V"])
    per_point = {}
    E_ref = None
    for ptn in ("x5", "x13"):
        t0 = time.time()
        rays_ri = rays_at(core_syms, ptn)
        E, bases = edges_and_bases(rays_ri, 4)
        print(f"[d4:{ptn}] edge set: {len(E)} pairs (expect {D4['E']}); "
              f"4-clique bases: {len(bases)} (expect {D4['bases']})")
        assert len(E) == D4["E"] and len(bases) == D4["bases"]
        if E_ref is None:
            E_ref, bases_ref = E, bases
        else:
            assert E == E_ref and bases == bases_ref, \
                "edge/basis structure DIFFERS between the two points -- not one fixed graph"
        E_hash = hashlib.md5(repr(sorted(E)).encode()).hexdigest()

        fr = bx8.sparse_flex_rows(rays_ri)
        assert fr["E"] == E and fr["n"] == D4["n"]
        vt = bx8.v_theta(core_syms, rays_ri)
        bad = sum(1 for row in fr["rows"] if bx8.sparse_dot(row, vt) != 0)
        assert bad == 0, f"J.v_theta has {bad} nonzero rows"
        nz = 0
        for trow in fr["triv"]:
            tv = bx8.densify(trow, fr["n"])
            nz += sum(1 for row in fr["rows"] if bx8.sparse_dot(row, tv) != 0)
        assert nz == 0, f"J.T has {nz} nonzero products"
        print(f"[d4:{ptn}] (1) J.v_theta == 0 exactly (all {len(fr['rows'])} rows); "
              f"(2) J.t == 0 for all {len(fr['triv'])} gauge generators (exact ints)")

        # (3) v_theta not in span(T): Fraction rank on a restriction (=> flex >= 1)
        def restrict(vec, S):
            cols = [2 * 4 * i + 2 * c + (0 if rl else 1)
                    for i in S for c in range(4) for rl in (True, False)]
            return [vec[cc] for cc in cols]
        Tdense = [bx8.densify(t, fr["n"]) for t in fr["triv"]]
        sub = None
        import random as _r
        rnd = _r.Random(20260807)
        cands = [tuple(sorted(b)) for b in bases] + \
                [tuple(sorted(rnd.sample(range(D4["V"]), k))) for k in (8, 8, 12) for _ in (0,)]
        for S in cands:
            TS = [restrict(t, S) for t in Tdense]
            vtS = restrict(vt, S)
            if not any(vtS):
                continue
            ra = bx8.rank_fraction(TS)
            rb = bx8.rank_fraction(TS + [vtS])
            if rb == ra + 1:
                sub, r1, r2 = S, ra, rb
                break
        assert sub is not None, "no restriction certifies v_theta outside span(T)"
        print(f"[d4:{ptn}] (3) restriction S={list(sub)}: rank_Q(T|_S)={r1} -> +v_theta = {r2} "
              f"(exact Fractions; the +1 jump certifies flex >= 1)")

        # (4)+(5) two-sided squeeze: mod-p ranks at two primes > 1e6
        rT = {p: flint_rank_dense(fr["triv"], fr["n"], p) for p in PRIMES}
        rJ = {p: flint_rank_dense(fr["rows"], fr["n"], p) for p in PRIMES}
        print(f"[d4:{ptn}] (4) rank_p(T) = {rT} (exact ceiling V+d^2-1 = {D4['gauge']})")
        print(f"[d4:{ptn}] (5) rank_p(J) = {rJ} (certificate target n-{D4['gauge']}-1 = "
              f"{D4['rank_target']})")
        assert all(v == D4["gauge"] for v in rT.values()), "gauge rank misses the exact ceiling"
        assert all(v == D4["rank_target"] for v in rJ.values()), "rank(J) misses the target"
        nullity = D4["n"] - D4["rank_target"]
        flex = nullity - D4["gauge"]
        print(f"[d4:{ptn}] => rank_Q(J) = {D4['rank_target']} exactly, nullity {nullity}, "
              f"FLEX EXACTLY {flex}  ({time.time()-t0:.2f}s)")
        assert flex == D4["flex"]
        per_point[ptn] = dict(E_hash=E_hash, rank_T=D4["gauge"], rank_J=D4["rank_target"],
                              flex=flex)
    col = sat_colorable(D4["V"], E_ref, bases_ref)
    crit, badray = sat_all_critical(D4["V"], E_ref, bases_ref)
    print(f"[d4] SAT re-check: KS-uncolorable = {not col}; all {D4['V']} rays critical = {crit}")
    assert (not col) and crit
    out.update(points=per_point, E=D4["E"], bases=D4["bases"], uncolorable=True,
               all_critical=True, flex=D4["flex"],
               E_hash_equal=(per_point["x5"]["E_hash"] == per_point["x13"]["E_hash"]))
    assert out["E_hash_equal"]
    cache_save("nonrigid_d4cert", out)
    print(f"[d4] PASS: one fixed structure, realized at both points, flex exactly 1.")
    return out


# ======================================================================================
# STAGE assemble -- d = 6, 8, 10, 12 from the flexcert caches (read-only).
# ======================================================================================
def _ranks_recorded(rankj_point, extra=None):
    """Collect every recorded prime -> rank entry for one point ('full', 'comp_flint',
    'comp_blas'), plus an optional extra dict (the d12 flint cross-check file)."""
    found = {}
    for key in ("full", "comp_flint", "comp_blas"):
        for p, r in (rankj_point.get(key) or {}).items():
            found[f"{key}:{p}"] = int(r)
    for p, r in (extra or {}).items():
        found[f"xflint:{p}"] = int(r)
    return found


def stage_assemble():
    print("=" * 100)
    print("STAGE assemble -- rungs d = 6, 8, 10, 12 from the existing flexcert caches")
    print("=" * 100)
    out = {}
    ok_all = True
    for d, cfg in sorted(RUNGS.items()):
        tag = cfg["tag"]
        entry = dict(d=d, ok=True, notes=[])
        cx5 = cache_load(f"{tag}_cert_x5"); cx13 = cache_load(f"{tag}_cert_x13")
        gau = cache_load(f"{tag}_gauge");   rkj = cache_load(f"{tag}_rankJ")
        core = cache_load(cfg["core"])
        missing = [nm for nm, c in [(f"{tag}_cert_x5", cx5), (f"{tag}_cert_x13", cx13),
                                    (f"{tag}_gauge", gau), (f"{tag}_rankJ", rkj),
                                    (cfg["core"], core)] if c is None]
        if missing:
            entry["ok"] = False
            entry["notes"].append(f"MISSING caches: {missing} -- rung verdict DOWNGRADED to "
                                  f"'not verifiable from caches'")
            print(f"[d={d}] {entry['notes'][-1]}")
            out[d] = entry; ok_all = False
            continue

        # kernel certificates at both points + one fixed orthogonality structure
        for nm, c in (("x5", cx5), ("x13", cx13)):
            assert c["V"] == cfg["V"] and c["E"] == cfg["E"] and c["n"] == cfg["n"], \
                f"d={d}:{nm} cache counts differ from published"
            assert c["kernel_violations"] == 0 and c["gauge_product_nonzeros"] == 0
        hash_eq = cx5["E_hash"] == cx13["E_hash"]
        assert hash_eq, f"d={d}: edge-set hashes differ between the points"
        lower = cx5.get("k_cert") or (1 if cx5.get("flex_ge_1") else 0)
        lower13 = cx13.get("k_cert") or (1 if cx13.get("flex_ge_1") else 0)
        print(f"[d={d}] cert caches: V={cfg['V']} E={cfg['E']} n={cfg['n']}; kernel "
              f"certificates clean at BOTH points; E_hash equal = {hash_eq} "
              f"({cx5['E_hash'][:12]}...); flex lower bound {min(lower, lower13)}")

        # gauge dimension: exact ceiling hit at 2 primes, both points
        for nm in ("x5", "x13"):
            g = gau[nm]
            assert g["exact"] and g["gauge_dim"] == cfg["gauge"], f"d={d}:{nm} gauge not exact"
            assert all(int(r) == cfg["gauge"] for r in g["ranks"].values())
        print(f"[d={d}] gauge tangent: rank_Q(T) = {cfg['gauge']} = V + d^2 - 1 exactly "
              f"(mod-p ceiling hit at primes {sorted(gau['x5']['ranks'])})")

        # rank(J): every recorded prime/engine hits the target => exact nullity => exact flex
        extra = cache_load("d12flexcert_rankJ_flint") if d == 12 else None
        for nm in ("x5", "x13"):
            rec = _ranks_recorded(rkj[nm], (extra or {}).get(nm, {}).get("comp_flint"))
            assert rec, f"d={d}:{nm} no recorded ranks"
            bad = {k: v for k, v in rec.items() if v != cfg["rank_target"]}
            assert not bad, f"d={d}:{nm} ranks missing target: {bad}"
            nprimes = len({k.split(':')[1] for k in rec})
            print(f"[d={d}] rank(J) at {nm}: target {cfg['rank_target']} hit by all "
                  f"{len(rec)} recorded (engine, prime) entries ({nprimes} distinct primes)")
        nullity = cfg["n"] - cfg["rank_target"]
        flex = nullity - cfg["gauge"]
        assert flex == cfg["flex"] and flex >= min(lower, lower13)
        print(f"[d={d}] => nullity {nullity}, FLEX EXACTLY {flex} "
              f"(lower bound {min(lower, lower13)} from the kernel certificates)")

        # structural flags
        if cfg["flags_cached"]:
            flags = {k: core.get(k) for k in ("complete", "uncolorable", "all_critical")}
            assert all(flags.values()), f"d={d} core flags not all True: {flags}"
            print(f"[d={d}] core flags (from {cfg['core']}): {flags}")
            entry["flags_source"] = "cache"
        else:
            print(f"[d={d}] core cache records no flags -- uncolorability/criticality "
                  f"re-verified by SAT in stage `sat` (d=6)")
            entry["flags_source"] = "sat-recheck (stage sat)"
        entry.update(V=cfg["V"], E=cfg["E"], flex=flex, E_hash=cx5["E_hash"],
                     lower=min(lower, lower13))
        out[d] = entry
    cache_save("nonrigid_assemble", {str(k): v for k, v in out.items()})
    print(f"[assemble] {'PASS' if ok_all else 'PARTIAL'} ({time.time()-T0:.1f}s)")
    return out


# ======================================================================================
# STAGE gram -- the separating invariant at both points, every rung.
# ======================================================================================
def stage_gram():
    print("=" * 100)
    print("STAGE gram -- exact Gram-modulus multisets at x5 vs x13 (the separating invariant)")
    print("=" * 100)
    out = {}
    cores = {4: load_d4_core()}
    for d, cfg in RUNGS.items():
        c = cache_load(cfg["core"])
        assert c is not None, f"{cfg['core']} missing"
        cores[d] = [tuple(v) for v in c["core_syms"]]
    for d in sorted(cores):
        t0 = time.time()
        syms = cores[d]
        ms = {ptn: gram_multiset(rays_at(syms, ptn)) for ptn in ("x5", "x13")}
        equal = ms["x5"] == ms["x13"]
        sums = {ptn: sum(k * v for k, v in ms[ptn].items()) for ptn in ("x5", "x13")}
        diff = set(ms["x5"]) ^ set(ms["x13"])
        wit = sorted(diff)[:2]
        print(f"[gram d={d}] {len(syms)} rays, {len(syms)*(len(syms)-1)//2} pairs; "
              f"distinct values {len(ms['x5'])} vs {len(ms['x13'])}; multisets equal = {equal}")
        print(f"[gram d={d}]   sum of multiset: x5 = {sums['x5']}  x13 = {sums['x13']}"
              f"  (cos theta = {POINT_COS['x5']} vs {POINT_COS['x13']})")
        if wit:
            print(f"[gram d={d}]   witness values in one multiset only: "
                  f"{[str(w) for w in wit]}")
        assert not equal, (f"d={d}: Gram multisets IDENTICAL at the two points -- the "
                           f"separating invariant does NOT separate; rung verdict must be "
                           f"downgraded (do not claim inequivalence from this script)")
        out[str(d)] = dict(equal=False, n_distinct=(len(ms["x5"]), len(ms["x13"])),
                           sum_x5=str(sums["x5"]), sum_x13=str(sums["x13"]),
                           fp_x5=multiset_fingerprint(ms["x5"]),
                           fp_x13=multiset_fingerprint(ms["x13"]),
                           secs=round(time.time() - t0, 2))
        print(f"[gram d={d}]   => the two realizations are INEQUIVALENT under every "
              f"(anti)unitary x phases x relabeling ({out[str(d)]['secs']}s)")
    cache_save("nonrigid_gram", out)
    print("[gram] PASS: separation certified at every rung")
    return out


# ======================================================================================
# STAGE sat -- independent SAT re-checks (uncolorability everywhere; criticality d=4,6).
# ======================================================================================
def stage_sat():
    print("=" * 100)
    print("STAGE sat -- CaDiCaL re-checks of KS-uncolorability (+criticality at d=4,6)")
    print("=" * 100)
    out = {}
    # d=4 handled inside stage d4 as well; repeat here so `sat` is self-contained
    syms4 = load_d4_core()
    E4, B4 = edges_and_bases(rays_at(syms4, "x5"), 4)
    col = sat_colorable(len(syms4), E4, B4)
    crit, _ = sat_all_critical(len(syms4), E4, B4)
    print(f"[sat d=4] uncolorable = {not col}; all-critical = {crit}")
    assert not col and crit
    out["4"] = dict(uncolorable=True, all_critical=True, source="recomputed")
    for d, cfg in sorted(RUNGS.items()):
        c = cache_load(cfg["core"])
        pairs = [tuple(p) for p in c["core_pairs"]] if "core_pairs" in c else None
        if pairs is None:
            syms = [tuple(v) for v in c["core_syms"]]
            pairs, _ = edges_and_bases(rays_at(syms, "x5"), d)
        bases = [tuple(b) for b in c["core_bases"]]
        # HARDENING (gate fix 4): the cached bases must be genuine orthogonal d-cliques at
        # BOTH circle points, and every ray must lie in at least one basis.  Previously this
        # was imported by citation from the tower paper; now it is asserted in-certificate.
        covered = set()
        for ptn in ("x5", "x13"):
            rr = rays_at([tuple(v) for v in c["core_syms"]], ptn)
            eset = set(map(tuple, pairs))
            for b in bases:
                assert len(b) == d, f"d={d} {ptn}: basis {b} has size {len(b)} != {d}"
                for i, j in combinations(sorted(b), 2):
                    assert (i, j) in eset and bg6.hdot_ri_zero(rr[i], rr[j]), \
                        f"d={d} {ptn}: basis pair ({i},{j}) not orthogonal"
                covered.update(b)
        assert covered == set(range(cfg["V"])), \
            f"d={d}: {cfg['V'] - len(covered)} rays in no basis"
        print(f"[sat d={d}] bases hardened: {len(bases)} orthogonal {d}-cliques at both "
              f"points, all {cfg['V']} rays covered")
        col = sat_colorable(cfg["V"], pairs, bases)
        rec = dict(uncolorable=not col)
        print(f"[sat d={d}] uncolorable = {not col}", end="")
        assert not col
        if d == 6:
            t0 = time.time()
            crit, badray = sat_all_critical(cfg["V"], pairs, bases)
            print(f"; all-critical = {crit} ({time.time()-t0:.1f}s, {cfg['V']} removals)",
                  end="")
            assert crit, f"d=6 criticality FAILED at ray {badray}"
            rec["all_critical"] = True
            rec["source"] = "recomputed (no cached flag)"
        else:
            rec["all_critical"] = bool(c.get("all_critical"))
            rec["source"] = "uncolorability recomputed; criticality from cached flag"
        print()
        out[str(d)] = rec
    cache_save("nonrigid_sat", out)
    print("[sat] PASS")
    return out


# ======================================================================================
# STAGE report -- the V57 statement.
# ======================================================================================
def stage_report():
    print("=" * 100)
    print("V57 REPORT -- non-rigid Kochen-Specker cores, one verdict per rung")
    print("=" * 100)
    d4 = cache_load("nonrigid_d4cert")
    asm = cache_load("nonrigid_assemble")
    gram = cache_load("nonrigid_gram")
    sat = cache_load("nonrigid_sat")
    if not all([d4, asm, gram, sat]):
        print("[report] run stages d4, assemble, gram, sat first")
        return False
    ok = True
    for d in (4, 6, 8, 10, 12):
        if d == 4:
            flex, V, E = d4["flex"], d4["V"], d4["E"]
            structure_ok = d4["E_hash_equal"]; lower = 1
        else:
            e = asm.get(str(d))
            if not e or not e.get("ok", False):
                print(f"[d={d:2d}] NOT VERIFIABLE FROM CACHES -- verdict withheld"); ok = False
                continue
            flex, V, E, lower = e["flex"], e["V"], e["E"], e["lower"]
            structure_ok = True
        g = gram[str(d)]; s = sat[str(d)]
        sep = not g["equal"]
        good = structure_ok and sep and s["uncolorable"] and s["all_critical"]
        ok &= good
        print(f"[d={d:2d}] core: {V} rays / {E} orthogonal pairs; KS-uncolorable, all-critical "
              f"({s['source']})")
        print(f"       two realizations (x5, x13) of ONE structure; Gram-multiset separation "
              f"=> inequivalent under every unitary AND antiunitary (x phases, relabeling)")
        print(f"       VERDICT: non-rigid in the Trandafir-Cabello sense (fixed-d rank-one "
              f"direction: two inequivalent realizations exist);")
        print(f"                first-order degeneracy dimension modulo gauge = {flex} "
              f"(exact two-sided squeeze; kernel lower bound {lower})")
        if d == 6:
            print(f"       [cited, not recomputed] three of the 6 dimensions integrate to the "
                  f"exact grading 3-torus (tower paper thm:d6torus, an immersion statement); "
                  f"the remaining directions are unobstructed through fourth order, open beyond")
    print(f"\nSCOPE (binding): fixed d, rank one, modulo per-ray phases + u(d).  Nothing above "
          f"computes the full")
    print(f"Trandafir-Cabello degeneracy over D >= d / higher rank; non-rigidity transfers "
          f"a fortiori, dimensions do not.")
    print(f"\nV57 VERDICT: {'PASS' if ok else 'FAIL'}  ({time.time()-T0:.1f}s total)")
    return ok


STAGES = dict(d4=stage_d4, assemble=stage_assemble, gram=stage_gram, sat=stage_sat,
              report=stage_report)

if __name__ == "__main__":
    args = sys.argv[1:] or ["all"]
    if args == ["all"]:
        args = ["d4", "assemble", "gram", "sat", "report"]
    for a in args:
        STAGES[a]()
