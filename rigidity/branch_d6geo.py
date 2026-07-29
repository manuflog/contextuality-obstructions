#!/usr/bin/env python3
"""
branch_d6geo.py -- Branch D6-GEO: flex-tight d=6 core + WZ holonomy Galois group.

READ FIRST: D6_CIRCLE.md + branch_d6flex.py (the d=6 circle -- 46-ray/298-pair/17-basis
mechanism-stable KS-uncolorable core, exact flex=17 at generic points, LOOSE per the M10 lesson);
M9_GEOMETRY.md + branch_m9geo.py (the d=4 holonomy TEMPLATE: incidence-based tight-frame WZ
connection, exactly constant, char poly, Galois group S4); M10_SECOND_FLEX.md + branch_m10x.py
(M10's second flex direction, an exact SO(2) doublet rotation); IDEAS_C_PUREMATH.md Moonshot 1
(the claimed M10 second-flex holonomy: eigenvalues {0,0,+-2i/11}, Galois-trivial -- warm-up target).

STAGES (CLI dispatch, each checkpoints to d6geo_*.cache.json; stage1/stage1final are the
resumable/expensive ones -- re-run with larger budgets to continue a growth search in progress,
exactly like branch_d6flex.py's own stage3):
    python3 branch_d6geo.py stage0                    # M10 warm-up: verify {0,0,+-2i/11}, Galois
    python3 branch_d6geo.py stage1 [budget_s] [size_cap]   # grow the 46-ray core (SVD/mod-p rank);
                                                        #   resumable -- call again to keep growing
    python3 branch_d6geo.py stage1final [budget_s]    # certify the grown core (mod-p, both points)
    python3 branch_d6geo.py stage2        # WZ holonomy connection on the tightest core found
    python3 branch_d6geo.py stage3        # char poly + Galois group (the prize)
    python3 branch_d6geo.py stage4        # det W / Tr(S)
    python3 branch_d6geo.py all           # stage0 + stage2 + stage3 + stage4 (uses the cached
                                           #   stage1 core -- growth itself is run separately,
                                           #   see D6_GEOMETRY.md Stage 1 for this session's trace)

No existing file is modified. No git. Machinery reused UNMODIFIED (read-only imports): ks_flex_
census.cache_save/cache_load (to read branch_d6flex.py's own d6flex_*.cache.json checkpoints),
branch_d6flex.find_cliques (dimension-generic clique enumerator), branch_m10x.CORE_SYMS/IDX_X1/
IDX_X2 (the M10 21-ray critical core symbol list). Everything else (the mod-p flex-growth engine,
the incidence-frame connection builder generalized to non-monomial rays, the Galois-group
extraction) is new, self-contained code in this file.
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from itertools import combinations
from collections import Counter
from fractions import Fraction as F

import numpy as np
import sympy as sp
from sympy.polys.matrices import DomainMatrix

from ks_flex_census import cache_save, cache_load
import branch_d6flex as bd6
from sympy.polys.numberfields.galoisgroups import galois_group

HERE = os.path.dirname(os.path.abspath(__file__))
T0 = time.time()


def geo_save(name, obj): cache_save(f"d6geo_{name}", obj)
def geo_load(name): return cache_load(f"d6geo_{name}")


# ======================================================================================
# STAGE 0 -- WARM-UP: M10's second-flex holonomy generator, {0,0,+-2i/11}, Galois-trivial.
# Validates the holonomy-Galois pipeline on a known small case before trusting it at d=6.
# ======================================================================================
def m10_rebuild(t_re=0, t_im=2):
    """Independent rebuild of the 21-ray/77-pair/11-basis M10 critical core at x=2i, EXACT
       (sympy Gaussian integers), including the BASES (branch_m10x.py never computed these --
       it only needed the edge-based Jacobian, not cliques)."""
    from branch_m10x import CORE_SYMS, V as M10_V, D as M10_D, IDX_X1, IDX_X2
    t = t_re + t_im * sp.I

    def val(sym):
        return {"0": sp.Integer(0), "1": sp.Integer(1), "-1": sp.Integer(-1),
                "X": t, "-X": -t}[sym]

    rays = [tuple(val(c) for c in ray) for ray in CORE_SYMS]

    def hdot(u, v):
        return sp.expand(sum(sp.conjugate(u[c]) * v[c] for c in range(M10_D)))

    E = [(i, j) for i, j in combinations(range(M10_V), 2) if hdot(rays[i], rays[j]) == 0]
    assert len(E) == 77, f"expected 77 edges, got {len(E)}"
    adj = [set() for _ in range(M10_V)]
    for i, j in E:
        adj[i].add(j); adj[j].add(i)
    bases, complete = bd6.find_cliques(adj, M10_V, M10_D)
    assert complete and len(bases) == 11, f"expected 11 complete bases, got {len(bases)} complete={complete}"
    D = [sp.expand(hdot(r, r)) for r in rays]
    return dict(rays=rays, E=E, bases=bases, D=D, IDX_X1=IDX_X1, IDX_X2=IDX_X2, V=M10_V, dsize=M10_D)


def m10_connection():
    """Build the incidence-based tight-frame WZ connection A(theta) EXACTLY (sympy), following
       the SAME construction as M9_GEOMETRY.md Sec.3.1 (stack the dsize member rays of every
       basis, normalized v_j/sqrt(D_j), Nb bases -> E^dag E = Nb*I_dsize identically), but here
       the moving rays (19,20) are NOT Laurent monomials in theta (they are the SO(2) doublet
       rotation of M10_SECOND_FLEX.md: v19(theta)=cos(theta)*v19_0+sin(theta)*v20_0, v20(theta)=
       -sin(theta)*v19_0+cos(theta)*v20_0, all other 19 rays FIXED at t=2i) -- so instead of the
       Fourier-degree trick, A(theta) is built directly: only j in {19,20} have nonzero d(v_j)/
       d(theta), so A(theta)[c,d] = (1/Nb) * sum_{j in {19,20}} pcount_j * conj(v_j(theta)[c]) *
       d(v_j(theta)[d])/d(theta) / D_j  (every other ray contributes exactly 0, its derivative
       being identically 0)."""
    core = m10_rebuild()
    rays, E, bases, D = core["rays"], core["E"], core["bases"], core["D"]
    IDX_X1, IDX_X2, V, dsize = core["IDX_X1"], core["IDX_X2"], core["V"], core["dsize"]
    Nb = len(bases)
    pcount = Counter()
    for b in bases:
        for idx in b:
            pcount[idx] += 1
    p19, p20 = pcount[IDX_X1], pcount[IDX_X2]
    D19, D20 = D[IDX_X1], D[IDX_X2]
    assert D19 == D20, "rays 19,20 must have equal norm (M10_SECOND_FLEX.md fact (ii))"

    theta = sp.symbols("theta", real=True)
    v19_0, v20_0 = rays[IDX_X1], rays[IDX_X2]
    v19 = tuple(sp.cos(theta) * v19_0[c] + sp.sin(theta) * v20_0[c] for c in range(dsize))
    v20 = tuple(-sp.sin(theta) * v19_0[c] + sp.cos(theta) * v20_0[c] for c in range(dsize))
    dv19 = tuple(sp.diff(v19[c], theta) for c in range(dsize))
    dv20 = tuple(sp.diff(v20[c], theta) for c in range(dsize))

    A = sp.zeros(dsize, dsize)
    for c in range(dsize):
        for d in range(dsize):
            term = (p19 * sp.conjugate(v19[c]) * dv19[d] / D19
                    + p20 * sp.conjugate(v20[c]) * dv20[d] / D20) / Nb
            A[c, d] = sp.simplify(sp.expand_trig(sp.expand(term)))

    A0 = A.subs(theta, 0)
    diffcheck = sp.simplify(A - A0)
    is_constant = all(diffcheck[c, d] == 0 for c in range(dsize) for d in range(dsize))
    return dict(A=A, A0=A0, is_constant=is_constant, Nb=Nb, p19=p19, p20=p20,
                D19=D19, D20=D20, V=V, E=len(E), bases=len(bases))


def stage0_m10_warmup():
    print("=" * 100)
    print("STAGE 0 -- M10 WARM-UP: verify the second-flex holonomy generator {0,0,+-2i/11}")
    print("=" * 100)
    t0 = time.time()
    res = m10_connection()
    A0, Nb, p19, p20 = res["A0"], res["Nb"], res["p19"], res["p20"]
    print(f"[stage0] rebuilt 21-ray/77-pair/11-basis M10 core at x=2i: E={res['E']} bases={res['bases']} "
          f"(expect 77/11)")
    print(f"[stage0] pcount(ray19)={p19} pcount(ray20)={p20} Nb={Nb} D19={res['D19']} D20={res['D20']}")
    print(f"[stage0] A(theta) is EXACTLY CONSTANT (sympy simplify, symbolic theta): {res['is_constant']}")
    assert res["is_constant"], "connection is NOT constant -- pipeline assumption fails on M10"

    print("\n[stage0] A0 (the constant connection generator, exact):")
    for row in A0.tolist():
        print("   ", [str(sp.nsimplify(x)) for x in row])

    lam = sp.symbols("lambda")
    cp = sp.expand((A0 - lam * sp.eye(4)).det() * (-1) ** 4)
    cp = sp.expand(sp.together(cp))
    print(f"\n[stage0] characteristic polynomial of A0 (det(lambda*I - A0)):")
    print(f"    {sp.nsimplify(cp)}")
    # verify it's real/rational (I must cancel out identically for an anti-Hermitian matrix)
    cp_noI = cp.subs(sp.I, 0)
    has_I = sp.simplify(cp - cp_noI) != 0
    print(f"    contains residual sympy.I after full expansion: {has_I}  (expect False -- anti-Hermitian")
    print(f"    matrices always have REAL characteristic-polynomial coefficients)")

    rootdict = sp.roots(sp.Poly(cp, lam))  # {root: multiplicity}, WITH multiplicity (unlike solve)
    roots_full = []
    for r, mult in rootdict.items():
        roots_full += [sp.nsimplify(r)] * mult
    print(f"\n[stage0] roots with multiplicity (exact): {roots_full}")
    target = sorted([sp.nsimplify(0), sp.nsimplify(0), sp.nsimplify(sp.Rational(2, 11) * sp.I),
                      sp.nsimplify(-sp.Rational(2, 11) * sp.I)], key=lambda z: (sp.re(z), sp.im(z)))
    roots_sorted = sorted(roots_full, key=lambda z: (sp.re(z), sp.im(z)))
    match = (len(roots_sorted) == 4 and
             all(sp.simplify(a - b) == 0 for a, b in zip(roots_sorted, target)))
    print(f"[stage0] MATCHES claimed {{0,0,+2i/11,-2i/11}}: {match}")
    assert match, f"eigenvalues do not match the claimed {{0,0,+-2i/11}}"
    roots = list(rootdict.keys())

    # Galois group: substitute lambda = I*mu to get the REAL polynomial in mu (mu = eigenvalue of
    # the Hermitian operator -I*A0); its roots are exactly the rational numbers {0,0,2/11,-2/11},
    # so the splitting field is Q itself -- Galois group TRIVIAL.
    mu = sp.symbols("mu")
    cp_mu = sp.expand(cp.subs(lam, sp.I * mu))
    cp_mu = sp.nsimplify(sp.expand(cp_mu / sp.LC(sp.Poly(cp_mu, mu))))
    print(f"\n[stage0] substituting lambda=I*mu (Hermitian reduction): char poly in mu = {cp_mu}")
    mu_roots = sp.solve(sp.Eq(cp_mu, 0), mu)
    print(f"[stage0] mu-roots: {mu_roots}  (all rational => splitting field = Q => Galois group TRIVIAL)")
    all_rational = all(r.is_rational for r in mu_roots)
    print(f"[stage0] all roots rational: {all_rational}  =>  GALOIS GROUP = TRIVIAL (order 1)")
    assert all_rational

    print(f"\n[stage0] *** WARM-UP PASSED: M10 second-flex holonomy generator has eigenvalues "
          f"{{0,0,+2i/11,-2i/11}}, Galois group TRIVIAL. Pipeline validated on a known case. ***")
    print(f"[stage0] done in {time.time()-t0:.1f}s")

    out = dict(A0=[[str(x) for x in row] for row in A0.tolist()], charpoly=str(cp),
               roots=[str(r) for r in roots], mu_roots=[str(r) for r in mu_roots],
               galois_trivial=bool(all_rational), p19=p19, p20=p20, Nb=Nb)
    geo_save("stage0_m10warmup", out)
    return out


# ======================================================================================
# STAGE 1 -- GROW THE 46-RAY d=6 CORE TOWARD FLEX-TIGHT (mod-p rank for speed).
#
# The 46-ray/298-pair/17-basis core (D6_CIRCLE.md) has exact flex=17 -- per the M10 lesson this
# is LOOSE (1 modulus-motion + ~16 decoration from a SAT-minimal, basis-sparse core). Grow it by
# adding basis-participating rays from the 540-ray/1800-basis X-count==1-only ambient pool
# (branch_d6flex.py's stage3 working pool, `d6flex_stage3_best.cache.json` -- the SAME pool the
# 46-core was peeled from, so "keep" there is directly comparable to any superset we build here),
# recomputing flex via a fast mod-p adaptation of `branch_d6flex.exact_flex_hermitian_at_point`
# after each growth batch, until flex stabilizes or the budget/pool is exhausted.
# ======================================================================================
POINTS = bd6.POINTS  # {"x5": {"0":(0,0),"1":(5,0),...}, "x13": {...}} -- reused unmodified


def gf_rank(rows, ncols, p):
    """Fast mod-p rank via in-place Gaussian elimination (numpy, vectorized row ops) -- the same
       algorithm as sic_zoo.py's `_eliminate`, reimplemented self-contained here on PLAIN integer
       rows (not the Q(sqrt2) (a,b*s) pair representation sic_zoo uses -- our rows are already
       ordinary integers, since a concrete Gaussian-RATIONAL point like x=(3+4i)/5 makes every
       Re/Im component of the Hermitian-tangent system an ordinary integer, no ring extension
       needed)."""
    if not rows:
        return 0
    A = np.array(rows, dtype=np.int64) % p
    nr, nc = A.shape
    r = 0
    for c in range(nc):
        nz = np.nonzero(A[r:, c])[0]
        if nz.size == 0:
            continue
        i = r + int(nz[0])
        if i != r:
            A[[r, i]] = A[[i, r]]
        inv = pow(int(A[r, c]), p - 2, p)
        A[r] = (A[r] * inv) % p
        oth = np.nonzero(A[:, c])[0]
        oth = oth[oth != r]
        if oth.size:
            A[oth] = (A[oth] - np.outer(A[oth, c], A[r])) % p
        r += 1
        if r == nr:
            break
    return r


P_MODULUS = 998244353  # prime, ~1e9; p^2 ~1e18 < int64 max -- safe for a single multiply-mod


def ray_ri(sym_tuple, point):
    return tuple(point[c] for c in sym_tuple)


def hdot_ri_zero(u, v):
    dre = sum(u[c][0] * v[c][0] + u[c][1] * v[c][1] for c in range(len(u)))
    dim = sum(u[c][0] * v[c][1] - u[c][1] * v[c][0] for c in range(len(u)))
    return dre == 0 and dim == 0


def build_flex_rows(rays_ri):
    """Build the extended Hermitian-tangent Jacobian J (2 rows/edge + 1 norm row/ray) and the
       V+d^2-1 trivial-symmetry generator set T -- IDENTICAL construction/convention to
       branch_d6flex.exact_flex_hermitian_at_point, but plain-Python-int rows (no sympy), for
       either mod-p or float-SVD rank evaluation downstream."""
    V = len(rays_ri)
    d = len(rays_ri[0])
    E = [(i, j) for i, j in combinations(range(V), 2) if hdot_ri_zero(rays_ri[i], rays_ri[j])]
    n = 2 * d * V

    def coord(i, c, real): return 2 * d * i + 2 * c + (0 if real else 1)

    rows = []
    for i, j in E:
        re = [0] * n; im = [0] * n
        Rei, Imi = zip(*rays_ri[i]); Rej, Imj = zip(*rays_ri[j])
        for c in range(d):
            re[coord(i, c, True)] += Rej[c]; re[coord(i, c, False)] += Imj[c]
            re[coord(j, c, True)] += Rei[c]; re[coord(j, c, False)] += Imi[c]
            im[coord(i, c, True)] += Imj[c]; im[coord(i, c, False)] -= Rej[c]
            im[coord(j, c, True)] -= Imi[c]; im[coord(j, c, False)] += Rei[c]
        rows.append(re); rows.append(im)
    for i in range(V):
        r_ = [0] * n
        for c in range(d):
            r_[coord(i, c, True)] = rays_ri[i][c][0]; r_[coord(i, c, False)] = rays_ri[i][c][1]
        rows.append(r_)

    triv = []
    for i in range(V):
        t = [0] * n
        for c in range(d):
            t[coord(i, c, True)] = -rays_ri[i][c][1]; t[coord(i, c, False)] = rays_ri[i][c][0]
        triv.append(t)
    for a in range(d):
        t = [0] * n
        for i in range(V):
            t[coord(i, a, True)] = -rays_ri[i][a][1]; t[coord(i, a, False)] = rays_ri[i][a][0]
        triv.append(t)
    for a in range(d):
        for b in range(a + 1, d):
            t1 = [0] * n; t2 = [0] * n
            for i in range(V):
                t1[coord(i, a, True)] += -rays_ri[i][b][1]; t1[coord(i, a, False)] += rays_ri[i][b][0]
                t1[coord(i, b, True)] += -rays_ri[i][a][1]; t1[coord(i, b, False)] += rays_ri[i][a][0]
                t2[coord(i, a, True)] += rays_ri[i][b][0]; t2[coord(i, a, False)] += rays_ri[i][b][1]
                t2[coord(i, b, True)] += -rays_ri[i][a][0]; t2[coord(i, b, False)] += -rays_ri[i][a][1]
            triv.append(t1); triv.append(t2)
    return dict(V=V, d=d, E=E, n=n, rows=rows, triv=triv)


def modp_flex_certificate(rays_ri, p=P_MODULUS):
    """Mod-p adaptation of branch_d6flex.exact_flex_hermitian_at_point -- exact-representative
       rank (a generic prime gives the true rank with overwhelming probability) via fast mod-p
       Gaussian elimination (numpy), intentionally traded for speed vs sympy DomainMatrix/QQ."""
    fr = build_flex_rows(rays_ri)
    rankJ = gf_rank(fr["rows"], fr["n"], p)
    rankT = gf_rank(fr["triv"], fr["n"], p)
    ker = fr["n"] - rankJ
    flex = ker - rankT
    return dict(V=fr["V"], d=fr["d"], E=len(fr["E"]), n=fr["n"], rankJ=rankJ, ker=ker,
                rankT=rankT, flex=flex)


def svd_flex_certificate(rays_ri, tol=1e-7):
    """FAST approximate rank via float64 SVD (np.linalg.matrix_rank, LAPACK-backed, ~2x faster
       than the mod-p elimination above at this problem's sizes) -- used ONLY to steer the growth
       SEARCH (not as the final certificate: near-degenerate singular values could in principle
       mis-rank at the tolerance boundary). Cross-checked against gf_rank at every size tested in
       this branch (exact agreement, see D6_GEOMETRY.md Stage 1) before being trusted alone."""
    fr = build_flex_rows(rays_ri)
    AJ = np.array(fr["rows"], dtype=np.float64)
    AT = np.array(fr["triv"], dtype=np.float64)
    rankJ = int(np.linalg.matrix_rank(AJ, tol=tol)) if AJ.size else 0
    rankT = int(np.linalg.matrix_rank(AT, tol=tol)) if AT.size else 0
    ker = fr["n"] - rankJ
    flex = ker - rankT
    return dict(V=fr["V"], d=fr["d"], E=len(fr["E"]), n=fr["n"], rankJ=rankJ, ker=ker,
                rankT=rankT, flex=flex)


def load_d6flex_data():
    pool = cache_load("d6flex_pool_rays")
    best = cache_load("d6flex_stage3_best")
    core = cache_load("d6flex_core_final")
    assert pool is not None and best is not None and core is not None, \
        "branch_d6flex.py caches missing -- run that pipeline first (D6_CIRCLE.md)"
    return pool, best, core


def basis_completion_order(S, sub_bases):
    """For growth-priority: for every basis not fully contained in S, its 'missing count'
       (6 - |basis cap S|). Returns bases sorted by missing count ascending (bases closest to
       complete first)."""
    incomplete = []
    for b in sub_bases:
        missing = [x for x in b if x not in S]
        if missing:
            incomplete.append((len(missing), b, missing))
    incomplete.sort(key=lambda t: t[0])
    return incomplete


def grow_core(seed_budget_s=30.0, size_cap=220, p=P_MODULUS, point_name="x5", verbose=True,
              use_svd=True, progress_name="stage1_progress", max_batch=30):
    t0 = time.time()
    pool, best, core = load_d6flex_data()
    sub_idx, pairs, sub_bases = best["sub_idx"], best["pairs"], best["sub_bases"]
    sub_bases = [tuple(b) for b in sub_bases]
    V540 = len(sub_idx)
    point = POINTS[point_name]

    prog = geo_load(progress_name)
    if prog is not None:
        S = set(prog["S"])
        history = prog["history"]
        if verbose:
            print(f"[stage1] resuming from cached progress: |S|={len(S)}")
    else:
        S = set(sorted(best["keep"]))
        history = []
        if verbose:
            print(f"[stage1] starting from the 46-ray SAT-minimal core (local idx space of the "
                  f"540-ray pool)")

    def eval_S(S):
        Ssorted = sorted(S)
        syms = [tuple(pool[sub_idx[i]]) for i in Ssorted]
        rays_ri = [ray_ri(sym, point) for sym in syms]
        cert = (svd_flex_certificate(rays_ri) if use_svd else modp_flex_certificate(rays_ri, p=p))
        return cert

    if not history:
        cert0 = eval_S(S)
        history.append(dict(size=len(S), **cert0))
        if verbose:
            print(f"[stage1] |S|=46 (seed core): E={cert0['E']} rankJ={cert0['rankJ']} "
                  f"ker={cert0['ker']} rankT={cert0['rankT']} flex={cert0['flex']}  "
                  f"(expect flex=17, cross-check vs exact sympy result)")

    # Growth loop: repeatedly complete the closest-to-complete bases, in batches, re-evaluating
    # flex after each batch, until budget or size cap or 540-pool exhausted.
    batch_targets = None
    while time.time() - t0 < seed_budget_s and len(S) < min(size_cap, V540):
        incomplete = basis_completion_order(S, sub_bases)
        if not incomplete:
            break
        # take the single closest-to-complete tier (smallest missing count present)
        min_missing = incomplete[0][0]
        to_add = set()
        for missing, b, miss in incomplete:
            if missing != min_missing:
                break
            to_add |= set(miss)
        if not to_add:
            break
        # cap the PER-STEP batch size too (not just the total size_cap): a tier can in principle
        # include hundreds of rays at once (observed: a "missing=3" tier jump was large enough to
        # blow the per-call wall budget with no checkpoint in between) -- keep each growth step
        # small enough that eval_S (the expensive part) finishes well within one call.
        room = min(size_cap - len(S), max_batch)
        if room <= 0:
            break
        to_add = set(sorted(to_add)[:room])
        S |= to_add
        cert = eval_S(S)
        history.append(dict(size=len(S), **cert))
        if verbose:
            print(f"[stage1] +{len(to_add)} rays (completing missing={min_missing} bases) -> "
                  f"|S|={len(S)}: E={cert['E']} flex={cert['flex']}  (t={time.time()-t0:.1f}s)")
        geo_save(progress_name, dict(S=sorted(S), history=history))
        # stabilization criterion: flex unchanged over a WINDOW OF GROWTH (not just a couple of
        # tiny batches -- early tiers can have small missing-count batches that plateau briefly
        # before the next tier drops flex further) -- require >=25 rays of growth with no
        # improvement before declaring a genuine plateau.
        if len(history) >= 2:
            cur_flex = history[-1]["flex"]
            plateau_start = None
            for h in reversed(history):
                if h["flex"] == cur_flex:
                    plateau_start = h["size"]
                else:
                    break
            if plateau_start is not None and (len(S) - plateau_start) >= 80:
                if verbose:
                    print(f"[stage1] flex STABILIZED at {cur_flex} over a {len(S)-plateau_start}-ray "
                          f"growth window -- stopping early")
                break

    geo_save("stage1_progress", dict(S=sorted(S), history=history))
    return S, history


def stage1_grow(seed_budget_s=30.0, size_cap=220):
    print("=" * 100)
    print("STAGE 1 -- GROW THE d=6 CORE TOWARD FLEX-TIGHT (mod-p rank)")
    print("=" * 100)
    S, history = grow_core(seed_budget_s=seed_budget_s, size_cap=size_cap)
    best_h = min(history, key=lambda h: (h["flex"], h["size"]))
    print(f"\n[stage1] history: {[(h['size'], h['flex']) for h in history]}")
    print(f"[stage1] best so far: |S|={best_h['size']} flex={best_h['flex']}")
    print(f"[stage1] current |S|={len(S)} (call again to continue growing if not yet stabilized/"
          f"capped)")
    return S, history


def _core_syms_bases(S, pool, sub_idx, sub_bases):
    core_syms = [tuple(pool[sub_idx[i]]) for i in S]
    Sset = set(S)
    remap = {old: new for new, old in enumerate(S)}
    core_bases = [tuple(remap[x] for x in b) for b in sub_bases if all(x in Sset for x in b)]
    return core_syms, core_bases, remap, Sset


def stage1_finalize(jobs=None, budget_s=32.0):
    """Certify whatever core the growth search (Stage 1) settled on: extract its concrete
       syms/pairs/bases, re-run the EXACT-representative mod-p flex certificate (gf_rank, not the
       float-SVD search heuristic) at (point,prime) JOBS -- default both circle points x5/x13,
       ONE prime each (998244353); a second prime (999999937) can be added as an extra cross-
       check job. Resumable/checkpointed: each job is expensive (~20-35s at this core size), so
       jobs are processed one at a time per call, within `budget_s`, and cached results are
       reused across calls (matches branch_d6flex.stage4's own per-point-per-call splitting,
       done here for JOBS instead of points alone since mod-p rank is the bottleneck)."""
    print("=" * 100)
    print("STAGE 1 (finalize) -- certify the tightest core found")
    print("=" * 100)
    t0 = time.time()
    pool, best, core46 = load_d6flex_data()
    sub_idx, sub_bases = best["sub_idx"], [tuple(b) for b in best["sub_bases"]]
    prog = geo_load("stage1_progress")
    assert prog is not None, "run stage1 growth first"
    S = sorted(prog["S"])
    core_syms, core_bases, remap, Sset = _core_syms_bases(S, pool, sub_idx, sub_bases)
    E_abs = [(remap[i], remap[j]) for i, j in best["pairs"] if i in Sset and j in Sset]
    print(f"[stage1-final] core size |S|={len(S)}, bases fully contained: {len(core_bases)}, "
          f"pairs (abstract, theta-identical): {len(E_abs)}")

    if jobs is None:
        jobs = [("x5", 998244353), ("x13", 998244353)]

    done = geo_load("stage1_jobs") or {}
    for pt_name, p in jobs:
        key = f"{pt_name}_{p}"
        if key in done:
            print(f"[stage1-final] job {key} already cached: flex={done[key]['flex']}")
            continue
        if time.time() - t0 > budget_s:
            print(f"[stage1-final] budget reached, {key} deferred to a later call")
            continue
        point = POINTS[pt_name]
        rays_ri = [ray_ri(sym, point) for sym in core_syms]
        if f"nondeg_{pt_name}" not in done:
            E_pt = [(i, j) for i, j in combinations(range(len(core_syms)), 2)
                    if hdot_ri_zero(rays_ri[i], rays_ri[j])]
            match = sorted(E_abs) == sorted(E_pt)
            print(f"[stage1-final] {pt_name}: non-degeneracy -- |E_abstract|={len(E_abs)} "
                  f"|E_at_point|={len(E_pt)} EXACT MATCH={match}")
            done[f"nondeg_{pt_name}"] = match
        cert = modp_flex_certificate(rays_ri, p=p)
        print(f"[stage1-final] {key}: rankJ={cert['rankJ']} ker={cert['ker']} "
              f"rankT={cert['rankT']} flex={cert['flex']}  (t={time.time()-t0:.1f}s)")
        done[key] = cert
        geo_save("stage1_jobs", done)

    flexes = [done[f"{pt}_{p}"]["flex"] for pt, p in jobs if f"{pt}_{p}" in done]
    print(f"\n[stage1-final] jobs done: {[k for k in done if not k.startswith('nondeg')]}")
    if flexes:
        print(f"[stage1-final] flex values so far: {flexes}  "
              f"(all agree: {len(set(flexes)) == 1})")

    out = dict(core_syms=core_syms, core_bases=core_bases, E_abs=len(E_abs), size=len(core_syms),
               jobs_done=done)
    geo_save("stage1_core", out)
    return out


# ======================================================================================
# STAGE 2 -- THE WZ HOLONOMY on the flex-tight (tightest-found) d=6 core: 407 rays / 7767 pairs /
# 908 bases, EVERY ray with X-count exactly 1 (a genuine Laurent monomial m*X^e, e in {0,1}, in
# EVERY coordinate slot it uses) -- so, UNLIKE the M10 second-flex warm-up (whose moving rays were
# NOT monomial), the exact d=4 M9_GEOMETRY.py Fourier-degree trick PORTS DIRECTLY to d=6: build
# the incidence-based tight frame (stack each basis's dsize=6 normalized member rays, Ehat^dag
# Ehat = I_6 identically), and the connection A(theta) = A0 + A1*e^{i theta} + Am*e^{-i theta}
# via the SAME exact-Fraction closed form as branch_m9geo.exact_fourier -- generalized here only
# by making `dsize` a parameter (the d=4 code hardcoded range(4)).
# ======================================================================================
def _extract_em_d(core_syms, dsize):
    n = len(core_syms)
    e = [[0] * dsize for _ in range(n)]
    m = [[0] * dsize for _ in range(n)]
    present = [[False] * dsize for _ in range(n)]
    for j, ray in enumerate(core_syms):
        for c, sym in enumerate(ray):
            if sym == "0":
                continue
            present[j][c] = True
            if sym == "1": m[j][c] = 1; e[j][c] = 0
            elif sym == "-1": m[j][c] = -1; e[j][c] = 0
            elif sym == "X": m[j][c] = 1; e[j][c] = 1
            elif sym == "-X": m[j][c] = -1; e[j][c] = 1
    D = [sum(1 for c in range(dsize) if present[j][c]) for j in range(n)]
    return e, m, present, D


def exact_fourier_d(core_syms, bases, dsize):
    """EXACT (Fraction) Fourier structure A0+A1*z+Am/z of the rank-dsize incidence-frame WZ
       connection -- direct generalization of branch_m9geo.exact_fourier (identical formula,
       `dsize` no longer hardcoded to 4). Every ray here has AT MOST one nonzero coordinate with
       e=1 (X-count<=1, in fact ==1 by the growth restriction), same monomial structure M9 relies
       on, so the Fourier-degree decomposition (delta = e_jd - e_jc in {-1,0,1}) is EXACT, no
       approximation."""
    n = len(core_syms)
    e, m, present, D = _extract_em_d(core_syms, dsize)
    Nb = len(bases)
    pcount = Counter()
    for b in bases:
        for idx in b:
            pcount[idx] += 1
    A0 = [[F(0)] * dsize for _ in range(dsize)]
    A1 = [[F(0)] * dsize for _ in range(dsize)]
    Am = [[F(0)] * dsize for _ in range(dsize)]
    for c in range(dsize):
        for d in range(dsize):
            acc0 = acc1 = accm = F(0)
            for j in range(n):
                if not present[j][c] or not present[j][d]:
                    continue
                delta = e[j][d] - e[j][c]
                term = F(pcount[j] * m[j][c] * m[j][d] * e[j][d], D[j] * Nb)
                if delta == 0: acc0 += term
                elif delta == 1: acc1 += term
                elif delta == -1: accm += term
                else: raise RuntimeError(f"unexpected Fourier degree {delta}")
            A0[c][d] = acc0; A1[c][d] = acc1; Am[c][d] = accm
    return A0, A1, Am, pcount, Nb


def clear_denominators_d(Acoef, dsize):
    from math import gcd
    dens = [Acoef[c][d].denominator for c in range(dsize) for d in range(dsize)]
    L = 1
    for den in dens:
        L = L * den // gcd(L, den)
    M = [[int(Acoef[c][d] * L) for d in range(dsize)] for c in range(dsize)]
    return M, L


def stage2_holonomy():
    print("=" * 100)
    print("STAGE 2 -- THE d=6 WZ HOLONOMY on the tightest core found (407 rays/908 bases)")
    print("=" * 100)
    t0 = time.time()
    core = geo_load("stage1_core")
    assert core is not None, "run stage1/stage1final first"
    core_syms = [tuple(v) for v in core["core_syms"]]
    core_bases = [tuple(b) for b in core["core_bases"]]
    dsize = 6
    print(f"[stage2] core: {len(core_syms)} rays, {len(core_bases)} bases, dsize={dsize}")

    A0, A1, Am, pcount, Nb = exact_fourier_d(core_syms, core_bases, dsize)
    zeroA1 = all(A1[c][d] == 0 for c in range(dsize) for d in range(dsize))
    zeroAm = all(Am[c][d] == 0 for c in range(dsize) for d in range(dsize))
    print(f"[stage2] Nb={Nb} bases, incidence-frame connection A(theta)=A0+A1*e^i theta+Am*e^-i theta")
    print(f"[stage2] *** A1==0: {zeroA1}   Am==0: {zeroAm} *** "
          f"(constant connection iff both True -- the M9 d=4 headline fact, tested here at d=6)")

    print("\n[stage2] A0 (coefficient of i; true entry = i*A0):")
    for row in A0:
        print("   ", [str(x) for x in row])

    M, L = clear_denominators_d(A0, dsize)
    tr = sum(M[i][i] for i in range(dsize))
    print(f"\n[stage2] Common denominator L={L}. Integer matrix M := L*A0:")
    for row in M:
        print("    ", row)
    print(f"[stage2] Tr(M)={tr}   Tr(S)=Tr(M)/L={F(tr, L)}")

    out = dict(A0=[[str(x) for x in row] for row in A0], zeroA1=zeroA1, zeroAm=zeroAm,
               M=M, L=L, tr=tr, Nb=Nb, dsize=dsize)
    geo_save("stage2_connection", out)
    print(f"\n[stage2] done in {time.time()-t0:.1f}s")
    return out


# ======================================================================================
# STAGE 3 -- THE GALOIS GROUP (the prize): exact char poly of the constant d=6 connection
# generator M, factor over Q, Galois group via sympy.
# ======================================================================================
def stage3_galois():
    print("=" * 100)
    print("STAGE 3 -- THE d=6 HOLONOMY GALOIS GROUP")
    print("=" * 100)
    t0 = time.time()
    conn = geo_load("stage2_connection")
    assert conn is not None, "run stage2 first"
    M, L, dsize = conn["M"], conn["L"], conn["dsize"]
    assert conn["zeroA1"] and conn["zeroAm"], "connection is NOT constant -- cannot proceed"

    x = sp.symbols("x")
    Msp = sp.Matrix(M)
    cp = sp.expand(Msp.charpoly(x).as_expr())
    print(f"[stage3] exact characteristic polynomial of M (dsize={dsize}):")
    print(f"    {cp}")
    P = sp.Poly(cp, x, domain="QQ")
    print(f"[stage3] degree: {P.degree()}")
    factors = sp.factor_list(cp, x)
    print(f"[stage3] factorization over Q: {factors}")

    irred_factors = [f for f, mult in factors[1]]
    print(f"[stage3] irreducible factors: {len(irred_factors)}")
    for f in irred_factors:
        Pf = sp.Poly(f, x, domain="QQ")
        print(f"    factor (deg {Pf.degree()}): {f}")

    galois_groups = []
    for f in irred_factors:
        Pf = sp.Poly(f, x, domain="QQ")
        if Pf.degree() <= 1:
            print(f"    degree<=1 factor {f}: Galois group TRIVIAL (rational root)")
            galois_groups.append((f, "trivial (degree<=1)"))
            continue
        try:
            G, alt = galois_group(Pf, by_name=True)
            print(f"    factor {f}: Galois group = {G}  (subgroup of A_n: {alt})")
            galois_groups.append((f, str(G)))
        except Exception as ex:
            print(f"    factor {f}: galois_group computation FAILED: {ex}")
            galois_groups.append((f, f"FAILED: {ex}"))

    roots = sp.Poly(cp, x).nroots(n=40)
    roots = sorted(roots, key=lambda r: sp.re(r))
    print(f"\n[stage3] numeric roots (40-digit precision):")
    eigenphases = []
    for r in roots:
        val = sp.re(r) / L
        frac = val - sp.floor(val)
        eigenphases.append(frac)
        print(f"    lambda={sp.N(r,20)}   phi/(2pi)=lambda/{L} mod 1 = {sp.N(frac,20)}")

    out = dict(charpoly=str(cp), factors=[str(f) for f in irred_factors],
               galois_groups=[(str(f), g) for f, g in galois_groups],
               roots=[str(sp.N(r, 20)) for r in roots])
    geo_save("stage3_galois", out)
    print(f"\n[stage3] done in {time.time()-t0:.1f}s")
    return out


# ======================================================================================
# STAGE 4 -- det W / abelian layer: Tr(S) exactly.
# ======================================================================================
def stage4_abelian():
    print("=" * 100)
    print("STAGE 4 -- det W / abelian layer")
    print("=" * 100)
    conn = geo_load("stage2_connection")
    assert conn is not None, "run stage2 first"
    tr, L = conn["tr"], conn["L"]
    trS = F(tr, L)
    detW_phase = trS % 1
    print(f"[stage4] Tr(S) = {trS}")
    print(f"[stage4] det W(2pi) = exp(2pi i * Tr(S)) = exp(2pi i * {detW_phase})  "
          f"(a primitive {detW_phase.denominator}th root of unity, or trivial if denominator==1)")
    out = dict(trS=str(trS), detW_phase=str(detW_phase), denom=detW_phase.denominator)
    geo_save("stage4_abelian", out)
    return out


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "stage0"
    if which == "stage0":
        stage0_m10_warmup()
    elif which == "stage1":
        budget = float(sys.argv[2]) if len(sys.argv) > 2 else 30.0
        cap = int(sys.argv[3]) if len(sys.argv) > 3 else 220
        stage1_grow(seed_budget_s=budget, size_cap=cap)
    elif which == "stage1final":
        stage1_finalize(budget_s=float(sys.argv[2]) if len(sys.argv) > 2 else 32.0)
    elif which == "stage2":
        stage2_holonomy()
    elif which == "stage3":
        stage3_galois()
    elif which == "stage4":
        stage4_abelian()
    elif which == "all":
        stage0_m10_warmup()
        stage2_holonomy()
        stage3_galois()
        stage4_abelian()
    else:
        print(f"stage {which!r} not implemented yet")
        sys.exit(1)
    print(f"\n[branch_d6geo.py stage={which} total {time.time()-T0:.1f}s]")
