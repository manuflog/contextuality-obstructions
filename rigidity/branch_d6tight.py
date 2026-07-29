#!/usr/bin/env python3
"""
branch_d6tight.py -- Branch D6-TIGHT: DIAGNOSE the d=6 circle core's flex 17 and HUNT a
flex-tight (portable-flex = 1) core.

READ FIRST: D6_CIRCLE.md + branch_d6flex.py (the 46-ray/298-pair/17-basis mechanism-stable
KS-uncolorable critical core of the d=6 unimodular circle |x|^2=1, EXACT flex=17 at two generic
circle points); D6_GEOMETRY.md + branch_d6geo.py (the previous flex-tight attempt: greedy
basis-completion growth, floor 407 rays / flex 8); M10_SECOND_FLEX.md (the modulus-motion vs
DECORATION distinction, and the pool-portability test).

THE QUESTION. Is the pool-portable mechanism flex at d=6 exactly 1 = dim V (the unimodular
circle's own solution variety), with the observed 17 = 1 modulus-motion + 16 decoration?

THE TWO IDEAS THIS FILE ADDS (both new; neither is in branch_d6geo.py):

 (A) EXACT SINGLE-RAY EXTENSION TEST FOR PORTABILITY.  A core motion w is pool-portable only if,
     for EVERY ambient pool ray p, the enlarged configuration (core + p) still moves consistently
     -- i.e. there is a tangent u for p with  <w_i,v_p> + <v_i,u> = 0  for every core ray i _|_ p.
     Eliminating u exactly: solvable iff  b := (<w_i,v_p>)_i  lies in the column space of
     G_p : u |-> (<v_i,u>)_i.  So per pool ray p the portability constraints on w are exactly the
     rows of coker(G_p) . B_p, computed here without ever forming coker explicitly (Stage 3
     maintains the surviving subspace and, per ray, takes the nullspace of the small block matrix
     [B_p | G_p], projected back to the w-block).  Running this over ALL 7448 ambient rays gives a
     RIGOROUS UPPER BOUND on the portable flex (see the mod-p rigor note below).

 (B) THE EXACT FLEX BOOK-KEEPING LAW FOR RAY ADDITION (proved in D6_TIGHT.md Sec.4):
         flex(C + p) = flex(C) + 10 - 2*r_p - rho_p ,
     where r_p = dim_C span{v_i : i in N_C(p)} (<= d-1 = 5) and rho_p = rank of p's compatibility
     constraints restricted to ker(J_C).  Hence any ray with r_p = 5 ("saturating") NEVER increases
     flex and decreases it by exactly rho_p.  This replaces branch_d6geo.py's greedy
     basis-completion growth (which added low-degree rays and was observed to ADD decoration flex,
     17 -> 23) by a strictly monotone one, and explains why that search floored at 8.

MOD-p RIGOR NOTE (used throughout, and it points the RIGHT way).  For an integer matrix,
rank_p <= rank_Q (an r x r minor nonzero mod p is nonzero over Z).  Since
flex = n - rank(J) - rank(T), a mod-p evaluation gives  flex_p >= flex_Q: mod-p flex is an
*upper bound* on the true rational flex.  So "mod-p says flex <= 1" is a PROOF that the exact
rational flex is <= 1, and combined with the explicitly exhibited exact modulus motion (which is
verified in ker(J) over Z, giving flex >= 1) it pins flex = 1 EXACTLY.  Every headline number
below is either exact-rational (sympy QQ) or a mod-p bound in the rigorous direction, and both
are reported.

STAGES (CLI, each checkpointed to d6tight_*.cache.json):
    python3 branch_d6tight.py stage0     # SANITY GATE: d=4 M9 flex=1, d=4 M10 flex=2,
                                          #   d=6 46-ray flex=17 -- exact AND mod-p, PASS/FAIL
    python3 branch_d6tight.py stage1     # pool + core + stable adjacency + core-degree census
    python3 branch_d6tight.py stage2     # DECOMPOSE the 17: modulus / loose / twin / remainder
    python3 branch_d6tight.py stage3 [N] # PORTABILITY: single-ray extension over the whole pool
    python3 branch_d6tight.py stage4 [N] # HUNT: monotone saturating-ray growth (resumable)
    python3 branch_d6tight.py stage5     # certify the best core (2 primes + exact, SAT, both pts)
    python3 branch_d6tight.py stage6     # holonomy generator / char poly / Galois at the tight core
    python3 branch_d6tight.py all        # stage0..stage6 in one process

No existing file is modified.  Machinery reused UNMODIFIED (read-only imports):
ks_flex_census.cache_save/cache_load, branch_d6flex (POINTS, build_A0A1, find_cliques,
generic_symbolic_rays, exact_flex_hermitian_at_point, _sat_colorable), branch_d6geo
(exact_fourier_d / _extract_em_d for the holonomy connection, gf_rank as a cross-check engine).
"""
import os, sys, json, time
from itertools import combinations
from collections import Counter

import numpy as np
import sympy as sp
from sympy.polys.matrices import DomainMatrix

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ks_flex_census import cache_save, cache_load
import branch_d6flex as bd6

HERE = os.path.dirname(os.path.abspath(__file__))
DSIZE = 6
P1 = 998244353          # primes for the mod-p rank engine (rank_p <= rank_Q, see header)
P2 = 999999937
T0 = time.time()


def tsave(name, obj): cache_save(f"d6tight_{name}", obj)
def tload(name): return cache_load(f"d6tight_{name}")


# ==================================================================================================
# Exact linear algebra over GF(p) (numpy, vectorised) + exact rational rank (sympy QQ).
# ==================================================================================================
def gf_rref(A, p):
    """In-place RREF of an int64 array mod p.  Returns (R, pivot_cols)."""
    A = np.asarray(A, dtype=np.int64) % p
    nr, nc = A.shape
    piv = []
    r = 0
    for c in range(nc):
        if r >= nr:
            break
        nz = np.nonzero(A[r:, c])[0]
        if nz.size == 0:
            continue
        i = r + int(nz[0])
        if i != r:
            A[[r, i]] = A[[i, r]]
        A[r] = (A[r] * pow(int(A[r, c]), p - 2, p)) % p
        oth = np.nonzero(A[:, c])[0]
        oth = oth[oth != r]
        if oth.size:
            A[oth] = (A[oth] - np.outer(A[oth, c], A[r])) % p
        piv.append(c)
        r += 1
    return A[:r], piv


def gf_rank(rows, ncols, p=P1):
    if len(rows) == 0:
        return 0
    R, piv = gf_rref(np.array(rows, dtype=np.int64).reshape(len(rows), ncols), p)
    return len(piv)


def gf_nullspace(A, p=P1):
    """Basis (as ROWS) of the nullspace of A over GF(p)."""
    A = np.asarray(A, dtype=np.int64)
    if A.size == 0:
        return np.eye(A.shape[1], dtype=np.int64)
    R, piv = gf_rref(A, p)
    nc = A.shape[1]
    free = [c for c in range(nc) if c not in set(piv)]
    N = np.zeros((len(free), nc), dtype=np.int64)
    for k, fc in enumerate(free):
        N[k, fc] = 1
        for ri, pc in enumerate(piv):
            N[k, pc] = (-R[ri, fc]) % p
    return N


def exact_rank_qq(rows, ncols):
    dm = DomainMatrix.from_list_sympy(len(rows), ncols,
                                      [[sp.Integer(int(x)) for x in r] for r in rows]).convert_to(sp.QQ)
    return dm.rank()


# ==================================================================================================
# The Hermitian-tangent (flex) system.  IDENTICAL convention to
# branch_d6flex.exact_flex_hermitian_at_point / branch_d6geo.build_flex_rows -- verified by the
# Stage 0 gate, which reproduces three already-published flex numbers with THIS code.
#   rays_ri : list of tuples of (Re,Im) integer pairs.
#   E       : edge list; if None it is REBUILT AT THE POINT (the repo's non-degeneracy convention).
# ==================================================================================================
def hdot_ri(u, v):
    re = sum(u[c][0] * v[c][0] + u[c][1] * v[c][1] for c in range(len(u)))
    im = sum(u[c][0] * v[c][1] - u[c][1] * v[c][0] for c in range(len(u)))
    return re, im


def point_edges(rays_ri):
    V = len(rays_ri)
    return [(i, j) for i, j in combinations(range(V), 2) if hdot_ri(rays_ri[i], rays_ri[j]) == (0, 0)]


def build_rows(rays_ri, E=None):
    V, d = len(rays_ri), len(rays_ri[0])
    if E is None:
        E = point_edges(rays_ri)
    n = 2 * d * V

    def coord(i, c, real): return 2 * d * i + 2 * c + (0 if real else 1)

    rows = []
    for i, j in E:
        re = [0] * n; im = [0] * n
        for c in range(d):
            Rei, Imi = rays_ri[i][c]; Rej, Imj = rays_ri[j][c]
            re[coord(i, c, True)] += Rej; re[coord(i, c, False)] += Imj
            re[coord(j, c, True)] += Rei; re[coord(j, c, False)] += Imi
            im[coord(i, c, True)] += Imj; im[coord(i, c, False)] -= Rej
            im[coord(j, c, True)] -= Imi; im[coord(j, c, False)] += Rei
        rows.append(re); rows.append(im)
    for i in range(V):
        r_ = [0] * n
        for c in range(d):
            r_[coord(i, c, True)] = rays_ri[i][c][0]; r_[coord(i, c, False)] = rays_ri[i][c][1]
        rows.append(r_)

    triv = []
    for i in range(V):                                   # per-ray U(1) phases
        t = [0] * n
        for c in range(d):
            t[coord(i, c, True)] = -rays_ri[i][c][1]; t[coord(i, c, False)] = rays_ri[i][c][0]
        triv.append(t)
    for a in range(d):                                   # u(d) diagonal
        t = [0] * n
        for i in range(V):
            t[coord(i, a, True)] = -rays_ri[i][a][1]; t[coord(i, a, False)] = rays_ri[i][a][0]
        triv.append(t)
    for a in range(d):                                   # u(d) off-diagonal
        for b in range(a + 1, d):
            t1 = [0] * n; t2 = [0] * n
            for i in range(V):
                t1[coord(i, a, True)] += -rays_ri[i][b][1]; t1[coord(i, a, False)] += rays_ri[i][b][0]
                t1[coord(i, b, True)] += -rays_ri[i][a][1]; t1[coord(i, b, False)] += rays_ri[i][a][0]
                t2[coord(i, a, True)] += rays_ri[i][b][0]; t2[coord(i, a, False)] += rays_ri[i][b][1]
                t2[coord(i, b, True)] += -rays_ri[i][a][0]; t2[coord(i, b, False)] += -rays_ri[i][a][1]
            triv.append(t1); triv.append(t2)
    return dict(V=V, d=d, E=E, n=n, rows=rows, triv=triv)


def flex_modp(rays_ri, E=None, p=P1):
    fr = build_rows(rays_ri, E)
    rankJ = gf_rank(fr["rows"], fr["n"], p)
    rankT = gf_rank(fr["triv"], fr["n"], p)
    return dict(V=fr["V"], d=fr["d"], E=len(fr["E"]), n=fr["n"], rankJ=rankJ,
                ker=fr["n"] - rankJ, rankT=rankT, flex=fr["n"] - rankJ - rankT)


def flex_exact(rays_ri, E=None):
    fr = build_rows(rays_ri, E)
    rankJ = exact_rank_qq(fr["rows"], fr["n"])
    rankT = exact_rank_qq(fr["triv"], fr["n"])
    return dict(V=fr["V"], d=fr["d"], E=len(fr["E"]), n=fr["n"], rankJ=rankJ,
                ker=fr["n"] - rankJ, rankT=rankT, flex=fr["n"] - rankJ - rankT)


# ==================================================================================================
# STAGE 0 -- SANITY GATE.  Reproduce three ALREADY-PUBLISHED flex numbers with this file's own
# from-scratch code, exactly and mod-p:
#   d=4 M9   89-ray critical core (D4_FLEX_HUNT.md)          -> flex 1
#   d=4 M10  21-ray critical core (M10_SECOND_FLEX.md)       -> flex 2
#   d=6      46-ray critical core (D6_CIRCLE.md)             -> flex 17  (rankJ=454,ker=98,rankT=81)
# ==================================================================================================
D4_POINTS = {
    "M9":  {"0": (0, 0), "1": (5, 0), "-1": (-5, 0), "X": (3, 4), "-X": (-3, -4)},
    "M10": {"0": (0, 0), "1": (1, 0), "-1": (-1, 0), "X": (0, 2), "-X": (0, -2)},
}


def load_d6_core():
    core = cache_load("d6flex_core_final")
    assert core is not None, "d6flex_core_final.cache.json missing -- run branch_d6flex.py first"
    return [tuple(v) for v in core["core_syms"]], core["global_idx"]


def stage0_gate():
    print("=" * 96)
    print("STAGE 0 -- SANITY GATE (reproduce three published flex numbers with this file's code)")
    print("=" * 96)
    ok = True
    rays4 = bd6.generic_symbolic_rays(4)
    checks = []
    for key, expect in (("M9", 1), ("M10", 2)):
        idx = cache_load(f"d4flex_{key}_done")
        assert idx is not None, f"d4flex_{key}_done.cache.json missing"
        r = [tuple(D4_POINTS[key][c] for c in rays4[i]) for i in idx]
        cm = flex_modp(r); ce = flex_exact(r)
        good = (cm["flex"] == expect == ce["flex"])
        ok &= good
        print(f"[stage0] d=4 {key:4s} core V={cm['V']} E={cm['E']}: exact flex={ce['flex']} "
              f"(rankJ={ce['rankJ']} ker={ce['ker']} rankT={ce['rankT']}), mod-p flex={cm['flex']}"
              f"   expect {expect}  -> {'PASS' if good else 'FAIL'}")
        checks.append(dict(case=f"d4_{key}", expect=expect, exact=ce["flex"], modp=cm["flex"], pass_=good))

    csyms, _ = load_d6_core()
    r6 = [tuple(bd6.POINTS["x5"][c] for c in s) for s in csyms]
    cm = flex_modp(r6); ce = flex_exact(r6)
    good = (ce["flex"] == 17 == cm["flex"] and ce["rankJ"] == 454 and ce["ker"] == 98 and ce["rankT"] == 81)
    ok &= good
    print(f"[stage0] d=6 46-ray core V={ce['V']} E={ce['E']}: exact rankJ={ce['rankJ']} ker={ce['ker']} "
          f"rankT={ce['rankT']} flex={ce['flex']}, mod-p flex={cm['flex']}   expect 454/98/81/17 "
          f"-> {'PASS' if good else 'FAIL'}")
    checks.append(dict(case="d6_46ray", expect=17, exact=ce["flex"], modp=cm["flex"], pass_=good))
    print(f"[stage0] SANITY GATE: {'PASSED' if ok else 'FAILED'}  ({time.time()-T0:.1f}s)")
    tsave("stage0_gate", dict(passed=bool(ok), checks=checks))
    return ok


def stage0b_m10_twin():
    """Cross-check of the TWIN-ROTATION picture against the one place it is already published:
       M10 (d=4).  Predictions: exactly one twin pair, contributing 2 dimensions mod T -- i.e. ALL
       of M10's flex 2 -- and the *anti-Hermitian* i-multiple generator (w_i = i v_j, w_j = i v_i)
       lies in ker(J) and is (up to the factor 1/5) M10_SECOND_FLEX.md's own line direction T."""
    print("=" * 96)
    print("STAGE 0b -- twin-rotation cross-check on the published d=4 M10 core")
    print("=" * 96)
    rays4 = bd6.generic_symbolic_rays(4)
    idx = cache_load("d4flex_M10_done")
    r = [tuple(D4_POINTS["M10"][c] for c in rays4[i]) for i in idx]
    fr = build_rows(r); n, V = fr["n"], len(r)
    adj = [set() for _ in range(V)]
    for i, j in fr["E"]:
        adj[i].add(j); adj[j].add(i)
    tw = [(i, j) for i in range(V) for j in range(i + 1, V) if (adj[i] - {j}) == (adj[j] - {i})]
    rT = exact_rank_qq(fr["triv"], n)
    print(f"[stage0b] twin pairs: {tw}  syms: {[rays4[idx[k]] for k in tw[0]]}")
    ns = sub_nullspace_exact(fr["rows"], n, cols_of(tw[0][0], 4) + cols_of(tw[0][1], 4))
    dim = exact_rank_qq(fr["triv"] + ns, n) - rT
    i, j = tw[0]
    w = [0] * n
    for c in range(4):
        re, im = r[j][c]; w[8 * i + 2 * c] = -im; w[8 * i + 2 * c + 1] = re
        re, im = r[i][c]; w[8 * j + 2 * c] = -im; w[8 * j + 2 * c + 1] = re
    ok = residual_zero(fr["rows"], w)
    d1 = exact_rank_qq(fr["triv"] + [w], n) - rT
    print(f"[stage0b] pair-local kernel dim = {len(ns)} (= 2 phases + 2 rotations); dim mod T = {dim}"
          f"   [M10's TOTAL flex is 2 -- so the twin pair accounts for ALL of it]")
    print(f"[stage0b] anti-Hermitian i-multiple generator in ker(J) exactly: {ok}; dim mod T = {d1}")
    good = (len(tw) == 1 and dim == 2 and ok and d1 == 1)
    print(f"[stage0b] twin cross-check: {'PASS' if good else 'FAIL'}")
    tsave("stage0b_m10twin", dict(twins=[list(t) for t in tw], pair_local_dim=len(ns),
                                  dim_mod_T=dim, imult_in_ker=bool(ok), passed=bool(good)))
    return good


# ==================================================================================================
# STAGE 1 -- data: the 7448-ray ambient mechanism-stable pool, the 46-ray core, the (pool x core)
# stable-adjacency matrix, and the core-degree census that drives Stages 3/4.
# ==================================================================================================
def stable_between(symsA, symsB):
    """Mechanism-stable (Laurent-identical on the WHOLE circle |x|^2=1) orthogonality between two
       symbol lists -- branch_d6flex.stable_matrix_M9's three-integer-matrix-product test, applied
       to a rectangular pair of ray sets."""
    A0, A1 = bd6.build_A0A1(list(symsA), DSIZE)
    B0, B1 = bd6.build_A0A1(list(symsB), DSIZE)
    M0 = A0 @ B0.T + A1 @ B1.T
    Mp = A0 @ B1.T
    Mm = A1 @ B0.T
    return (M0 == 0) & (Mp == 0) & (Mm == 0)


def stage1_data():
    print("=" * 96)
    print("STAGE 1 -- ambient pool, core, stable adjacency, core-degree census")
    print("=" * 96)
    pool = [tuple(v) for v in cache_load("d6flex_pool_rays")]
    csyms, gidx = load_d6_core()
    assert [pool[i] for i in gidx] == csyms, "core/pool index provenance mismatch"
    S = stable_between(pool, csyms)
    deg = S.sum(axis=1).astype(int)
    deg[gidx] = -1
    hist = dict(sorted(Counter(deg.tolist()).items()))
    print(f"[stage1] pool={len(pool)} rays, core={len(csyms)} rays")
    print(f"[stage1] core-degree histogram over the ambient pool: {hist}")
    top = sorted(range(len(pool)), key=lambda i: -deg[i])[:12]
    print(f"[stage1] highest core-degree ambient rays: "
          f"{[(int(deg[i]), bd6.xcount(pool[i])) for i in top]}   (degree, X-count)")
    tsave("stage1_data", dict(core_global_idx=gidx, deg=deg.tolist(), hist={str(k): v for k, v in hist.items()}))
    return pool, csyms, gidx, deg


# ==================================================================================================
# STAGE 2 -- DECOMPOSE the 17-dimensional flex space of the 46-ray core.
#
# Every direction below is built in CLOSED FORM and then *verified* to lie in ker(J) by an exact
# integer residual check -- nothing is read off a numerically computed kernel basis.
#
#  (c) MODULUS MOTION      w_j = d/dtheta of ray j along x = e^{i theta}  (only the +-X entries move)
#  (a) LOOSE-RAY motion    the space of kernel vectors supported on ONE ray i.  Exactly
#                          dim = 12 - 2*r_i - 1  where r_i = dim_C span{v_j : j in N(i)}; one of
#                          those dimensions is ray i's trivial U(1) phase, so ray i contributes
#                          10 - 2*r_i genuinely loose dimensions.  r_i = 5 (saturated) => 0.
#  (b) TWIN-RAY rotation   kernel vectors supported on a PAIR {i,j}, modulo the two phases and the
#                          two rays' own loose spaces.  A pair with N(i)\{j} = N(j)\{i} carries a
#                          free 2-real-dimensional GL(1)-doublet rotation (M10_SECOND_FLEX.md's
#                          w_perp is exactly this at d=4).
#  (d) REMAINDER           17 - dim(span of all the above, modulo the trivial space T).
# ==================================================================================================
def cols_of(i, d=DSIZE):
    return list(range(2 * d * i, 2 * d * (i + 1)))


def modulus_motion(core_syms, point):
    """d/dtheta of the whole configuration along x=e^{i theta}: entry '1' is theta-independent,
       entry 'X' |-> x has derivative i*x.  With the repo's integer rescale ('1'->(5,0),
       'X'->(3,4) at x=(3+4i)/5) that is 'X' |-> i*(3+4i) = (-4,3)."""
    xr, xi = point["X"]
    dX = (-xi, xr)                              # i*(xr + i*xi)
    out = []
    for s in core_syms:
        w = []
        for c in s:
            if c == "X":
                w.append(dX)
            elif c == "-X":
                w.append((-dX[0], -dX[1]))
            else:
                w.append((0, 0))
        out.append(tuple(w))
    return [x for w in out for c in w for x in c]          # flattened, same coord convention


def residual_zero(rows, vec):
    v = np.array(vec, dtype=object)
    for r in rows:
        if int(np.dot(np.array(r, dtype=object), v)) != 0:
            return False
    return True


def c_rank_of_rays(rays_ri, idxs, d=DSIZE):
    """dim_C span{v_j : j in idxs}, exactly: = (real rank of the 2*|idxs| x 2d matrix whose rows
       are the real forms of v_j and i*v_j) / 2."""
    rows = []
    for j in idxs:
        rows.append([x for c in rays_ri[j] for x in c])
        rows.append([x for c in rays_ri[j] for x in (-c[1], c[0])])
    if not rows:
        return 0
    r = exact_rank_qq(rows, 2 * d)
    assert r % 2 == 0
    return r // 2


def sub_nullspace_exact(rows, n, cols):
    """Exact rational nullspace basis (as full-length-n integer vectors) of the system restricted
       to `cols` (all other coordinates forced to 0).  Rows that vanish on `cols` are dropped."""
    sub = []
    for r in rows:
        rr = [r[c] for c in cols]
        if any(rr):
            sub.append(rr)
    if not sub:
        basis = [[1 if k == t else 0 for k in range(len(cols))] for t in range(len(cols))]
    else:
        M = sp.Matrix(sub)
        basis = []
        for v in M.nullspace():
            den = sp.lcm([sp.nsimplify(x).q for x in v])
            basis.append([int(x * den) for x in v])
    out = []
    for b in basis:
        full = [0] * n
        for k, c in enumerate(cols):
            full[c] = b[k]
        out.append(full)
    return out


def stage2_decompose():
    print("=" * 96)
    print("STAGE 2 -- DECOMPOSITION of the 46-ray core's 17-dimensional flex space")
    print("=" * 96)
    csyms, gidx = load_d6_core()
    point = bd6.POINTS["x5"]
    rays = [tuple(point[c] for c in s) for s in csyms]
    fr = build_rows(rays)
    n, V = fr["n"], fr["V"]
    E = fr["E"]
    rows, triv = fr["rows"], fr["triv"]
    rankT = exact_rank_qq(triv, n)
    rankJ = exact_rank_qq(rows, n)
    flex = n - rankJ - rankT
    print(f"[stage2] V={V} E={len(E)} n={n} rankJ={rankJ} ker={n-rankJ} rankT={rankT} flex={flex}")
    assert flex == 17 and rankT == 81

    adj = [set() for _ in range(V)]
    for i, j in E:
        adj[i].add(j); adj[j].add(i)

    # ---- (c) the modulus motion -------------------------------------------------------------
    M = modulus_motion(csyms, point)
    inker = residual_zero(rows, M)
    dM = exact_rank_qq(triv + [M], n) - rankT
    print(f"[stage2] (c) MODULUS MOTION dv/dtheta: in ker(J) exactly = {inker}; "
          f"dim((M+T)/T) = {dM}   [expect True, 1]")
    assert inker and dM == 1

    # ---- (a) loose rays ---------------------------------------------------------------------
    loose_vecs, loose_report = [], []
    for i in range(V):
        r_i = c_rank_of_rays(rays, sorted(adj[i]))
        ns = sub_nullspace_exact(rows, n, cols_of(i))
        # dimension check against the closed form 12 - 2 r_i - 1
        assert len(ns) == 12 - 2 * r_i - 1, (i, len(ns), r_i)
        loose_report.append((i, len(adj[i]), r_i, 10 - 2 * r_i))
        loose_vecs += ns
    d_single = exact_rank_qq(triv + loose_vecs, n) - rankT
    nloose = [x for x in loose_report if x[3] > 0]
    print(f"[stage2] (a) LOOSE RAYS: {len(nloose)}/{V} rays have r_i < 5 (under-saturated "
          f"neighbourhood); per-ray (ray,deg,r_i,loosedim) = {nloose}")
    print(f"[stage2]     total single-ray-supported flex, modulo trivial: dim = {d_single}")

    # ---- (b) twin pairs ---------------------------------------------------------------------
    twins = []
    for i, j in combinations(range(V), 2):
        if (adj[i] - {j}) == (adj[j] - {i}):
            twins.append((i, j))
    pair_vecs = []
    twin_extra = {}
    for (i, j) in twins:
        ns = sub_nullspace_exact(rows, n, cols_of(i) + cols_of(j))
        pair_vecs += ns
        twin_extra[(i, j)] = len(ns)
    d_pair = exact_rank_qq(triv + loose_vecs + pair_vecs, n) - rankT
    print(f"[stage2] (b) TWIN PAIRS (identical orthogonality neighbourhoods): {len(twins)} pairs "
          f"{twins if len(twins) <= 14 else str(twins[:14]) + '...'}")
    print(f"[stage2]     pair-local kernel dims: {sorted(Counter(twin_extra.values()).items())}")
    print(f"[stage2]     dim((loose + twin + T)/T) = {d_pair}   (twin adds {d_pair - d_single})")

    # ---- (d) everything together ------------------------------------------------------------
    d_all = exact_rank_qq(triv + loose_vecs + pair_vecs + [M], n) - rankT
    print(f"[stage2] (d) dim((loose + twin + modulus + T)/T) = {d_all}  of  flex = {flex}   "
          f"=> UNEXPLAINED REMAINDER = {flex - d_all}")

    # twin classes (maximal sets of mutually-twin rays)
    par = list(range(V))
    def find(a):
        while par[a] != a:
            par[a] = par[par[a]]; a = par[a]
        return a
    for i, j in twins:
        a, b = find(i), find(j)
        if a != b:
            par[a] = b
    classes = {}
    for i in range(V):
        classes.setdefault(find(i), []).append(i)
    big = sorted([v for v in classes.values() if len(v) > 1], key=len, reverse=True)
    print(f"[stage2]     twin CLASSES of size>1: {[len(b) for b in big]}  -> {big}")

    res = dict(V=V, E=len(E), n=n, rankJ=rankJ, rankT=rankT, flex=flex,
               modulus_in_ker=bool(inker), dim_modulus=dM,
               loose_report=loose_report, dim_single=d_single,
               twins=[list(t) for t in twins], dim_pair=d_pair, dim_all=d_all,
               remainder=flex - d_all, twin_classes=big)
    tsave("stage2_decomp", res)
    print(f"[stage2] done ({time.time()-T0:.1f}s)")
    return res


def stage2b_blocks(p=P1):
    """What are the 8 DELOCALISED remainder directions?  Scan the coordinate-block structure: for a
       coordinate subset B, the rays whose support lies inside B can be moved by a unitary of that
       block without disturbing any ray whose support misses B.  Reports, for every B, the
       dimension of the kernel supported on R_B = {rays with support inside B}, modulo trivial."""
    print("=" * 96)
    print("STAGE 2b -- coordinate-BLOCK structure behind the delocalised remainder")
    print("=" * 96)
    csyms, _ = load_d6_core()
    point = bd6.POINTS["x5"]
    rays = [tuple(point[c] for c in s) for s in csyms]
    fr = build_rows(rays)
    n, V, rows, triv = fr["n"], fr["V"], fr["rows"], fr["triv"]
    rankT = gf_rank(triv, n, p)
    supp = [frozenset(c for c in range(DSIZE) if s[c] != "0") for s in csyms]
    print(f"[stage2b] coordinate-support multiset of the 46 rays: "
          f"{sorted(Counter(tuple(sorted(s)) for s in supp).items(), key=lambda kv: -kv[1])[:10]}")
    best = []
    allvecs = []
    for k in range(2, DSIZE):
        for B in combinations(range(DSIZE), k):
            Bs = frozenset(B)
            RB = [i for i in range(V) if supp[i] <= Bs]
            if len(RB) < 2:
                continue
            free = all(not (supp[i] & Bs) for i in range(V) if i not in RB)
            cols = [c for i in RB for c in cols_of(i)]
            sub = [[r[c] for c in cols] for r in rows]
            sub = [r for r in sub if any(r)]
            NS = gf_nullspace(np.array(sub, dtype=np.int64), p) if sub else None
            vecs = []
            for b in NS:
                full = [0] * n
                for kk, c in enumerate(cols):
                    full[c] = int(b[kk])
                vecs.append(full)
            dim = gf_rank(triv + vecs, n, p) - rankT if vecs else 0
            if dim > 0:
                best.append((tuple(B), len(RB), free, dim))
                allvecs += vecs
    best.sort(key=lambda t: -t[3])
    print(f"[stage2b] coordinate blocks carrying local flex (B, |R_B|, block-free, dim mod T):")
    for b in best[:16]:
        print(f"          {b}")
    tot = gf_rank(triv + allvecs, n, p) - rankT
    print(f"[stage2b] TOTAL dim of all coordinate-block-supported flex, modulo trivial = {tot} "
          f"(of flex 17)")
    tsave("stage2b_blocks", dict(blocks=[[list(b[0]), b[1], bool(b[2]), b[3]] for b in best],
                                 total=int(tot)))
    return best, tot


# ==================================================================================================
# STAGE 3 -- POOL-PORTABILITY: the single-ray extension test, run over the WHOLE 7448-ray ambient
# mechanism-stable pool.
#
# A core motion w extends to the enlarged configuration (core + p) iff there is a tangent u for p
# with  <w_i,v_p> + <v_i,u> = 0  for every core ray i in N(p).  Writing b_i = <w_i,v_p> and
# G_p : u |-> (<v_i,u>)_i, this is solvable iff b in range(G_p), i.e. iff  L . b = 0  for every L in
# the left-nullspace of G_p (equivalently: for every C-linear relation sum_i c_i v_i = 0 among p's
# core neighbours, sum_i conj(c_i) <w_i,v_p> = 0).  Portability to the whole pool implies
# portability to core+{p} for each p separately, so intersecting over all p gives a rigorous UPPER
# BOUND on the portable flex.  The trivial space T is contained in every one of these subspaces
# (per-ray phases and global u(6) motions extend to the pool verbatim), so
#     portable flex = dim(surviving subspace) - rank(T).
# ==================================================================================================
def modmul(A, B, p, chunk=4):
    """Matrix product mod p with chunked accumulation (keeps every partial sum < 2^63)."""
    out = np.zeros((A.shape[0], B.shape[1]), dtype=np.int64)
    for s in range(0, A.shape[1], chunk):
        out = (out + A[:, s:s + chunk] @ B[s:s + chunk]) % p
    return out


def _pool_arrays(pool, point):
    PRe = np.zeros((len(pool), DSIZE), dtype=np.int64)
    PIm = np.zeros((len(pool), DSIZE), dtype=np.int64)
    for k, s in enumerate(pool):
        for c, sym in enumerate(s):
            re, im = point[sym]
            PRe[k, c] = re; PIm[k, c] = im
    return PRe, PIm


def stage3_portability(p=P1, verbose=True, restrict=None):
    print("=" * 96)
    print("STAGE 3 -- POOL-PORTABILITY of the 17 flex directions (single-ray extension test)")
    print("=" * 96)
    pool = [tuple(v) for v in cache_load("d6flex_pool_rays")]
    csyms, gidx = load_d6_core()
    point = bd6.POINTS["x5"]
    rays = [tuple(point[c] for c in s) for s in csyms]
    fr = build_rows(rays)
    n, V = fr["n"], fr["V"]
    K0 = gf_nullspace(np.array(fr["rows"], dtype=np.int64), p)       # (98 x n)
    k0 = K0.shape[0]
    rankT = gf_rank(fr["triv"], n, p)
    print(f"[stage3] kernel dim (mod p) = {k0}, rankT = {rankT}, flex = {k0 - rankT}")
    assert k0 == 98 and rankT == 81

    KR = K0.reshape(k0, V, DSIZE, 2)[:, :, :, 0].copy() % p          # (k0,V,d) real parts
    KI = K0.reshape(k0, V, DSIZE, 2)[:, :, :, 1].copy() % p
    PRe, PIm = _pool_arrays(pool, point)
    Sadj = stable_between(pool, csyms)
    CRe = PRe[gidx]; CIm = PIm[gidx]

    constraints = []
    n_relrays = 0
    degeneracy_flags = []
    t0 = time.time()
    order = sorted(range(len(pool)), key=lambda i: -int(Sadj[i].sum()))
    for pi in order:
        if pi in set(gidx):
            continue
        if restrict is not None and bd6.xcount(pool[pi]) != restrict:
            continue
        N = np.nonzero(Sadj[pi])[0]
        if N.size < 2:
            continue
        pr, pim = PRe[pi] % p, PIm[pi] % p
        # G: (2|N|) x 12 real matrix of u |-> <v_i,u> ; rows alternate Re, Im
        G = np.zeros((2 * N.size, 2 * DSIZE), dtype=np.int64)
        for a, i in enumerate(N):
            for c in range(DSIZE):
                G[2 * a, 2 * c] = CRe[i, c]; G[2 * a, 2 * c + 1] = CIm[i, c]
                G[2 * a + 1, 2 * c] = -CIm[i, c]; G[2 * a + 1, 2 * c + 1] = CRe[i, c]
        rG = gf_rank(G, 2 * DSIZE, p)
        if rG >= 2 * N.size:
            continue                     # no relations among the neighbours -> no constraint
        # exactness guard: rank_p <= rank_Q <= 2*min(|N|,d-1); if rank_p already attains the cap,
        # rank_Q = rank_p is FORCED (no mod-p degeneracy possible).
        if rG != 2 * min(int(N.size), DSIZE - 1):
            degeneracy_flags.append((int(pi), int(N.size), int(rG)))
        L = gf_nullspace(G.T, p)                                     # (2|N| - rG) x 2|N|
        # B: (2|N|) x k0 -- b_i(w_k) = <w_k|_i , v_p>
        B = np.zeros((2 * N.size, k0), dtype=np.int64)
        for a, i in enumerate(N):
            wr, wi = KR[:, i, :], KI[:, i, :]
            B[2 * a] = (modmul(wr, pr.reshape(-1, 1), p).ravel() + modmul(wi, pim.reshape(-1, 1), p).ravel()) % p
            B[2 * a + 1] = (modmul(wr, pim.reshape(-1, 1), p).ravel() - modmul(wi, pr.reshape(-1, 1), p).ravel()) % p
        rowsC = modmul(L % p, B, p)
        if rowsC.size:
            constraints.append(rowsC)
            n_relrays += 1
    C = np.vstack(constraints) if constraints else np.zeros((0, k0), dtype=np.int64)
    print(f"[stage3] ambient rays contributing relations: {n_relrays}; constraint rows: {C.shape[0]} "
          f"({time.time()-t0:.1f}s)")
    if degeneracy_flags:
        print(f"[stage3] WARNING: {len(degeneracy_flags)} rays where mod-p rank(G) did not attain "
              f"the structural cap 2*min(|N|,5) -- exactness not automatic there: "
              f"{degeneracy_flags[:8]}")
    rC = gf_rank(C, k0, p) if C.shape[0] else 0
    surv = k0 - rC
    print(f"[stage3] rank of all portability constraints on ker(J): {rC}")
    print(f"[stage3] surviving subspace dim = {surv}  (contains T, dim {rankT})")
    print(f"[stage3] *** POOL-PORTABLE FLEX = {surv - rankT} ***")

    # which named directions survive?  (modulus, twins)
    dec = tload("stage2_decomp")
    named = {}
    M = modulus_motion(csyms, point)
    # express a vector of ker(J) in the K0 basis: solve K0^T a = M over GF(p)
    def in_kernel_coords(vec):
        A = np.concatenate([K0.T % p, (np.array(vec, dtype=np.int64) % p).reshape(-1, 1)], axis=1)
        R, piv = gf_rref(A, p)
        assert k0 not in piv, "vector not in ker(J) mod p"
        a = np.zeros(k0, dtype=np.int64)
        for ri, pc in enumerate(piv):
            a[pc] = R[ri, k0] % p
        return a
    aM = in_kernel_coords(M)
    resM = modmul(C, aM.reshape(-1, 1), p).ravel() if C.shape[0] else np.zeros(1, dtype=np.int64)
    named["modulus"] = bool(np.all(resM % p == 0))
    print(f"[stage3] modulus motion survives every portability constraint: {named['modulus']} "
          f"(it must -- it is the family's own motion)")

    tsave(f"stage3_portability{'' if restrict is None else '_x%d' % restrict}",
          dict(p=int(p), restrict=restrict, kernel_dim=int(k0), rankT=int(rankT),
                                     constraint_rows=int(C.shape[0]), rank_constraints=int(rC),
                                     surviving=int(surv), portable_flex=int(surv - rankT),
                                     n_relation_rays=int(n_relrays),
                                     degeneracy_flags=degeneracy_flags[:50],
                                     modulus_survives=named["modulus"]))
    print(f"[stage3] done ({time.time()-T0:.1f}s)")
    return surv - rankT


# ==================================================================================================
# STAGE 4 -- THE HUNT.  Monotone growth by SATURATING rays.
#
# THE BOOK-KEEPING LAW (D6_TIGHT.md Sec.4).  Adding one ambient ray p to a core C changes the
# certificate by
#     d(ker) = 11 - 2 r_p - rho_p ,      d(rank T) = 1 ,      d(flex) = 10 - 2 r_p - rho_p
# with r_p = dim_C span{v_i : i in N_C(p)} <= d-1 = 5 and rho_p = rank of p's portability
# constraints restricted to ker(J_C).  A ray with r_p = 5 ("SATURATING") therefore never increases
# the flex and lowers it by exactly rho_p -- so a greedy sequence of saturating rays gives a
# MONOTONE search, and since the total available cut is flex - 1 = 16, at most 16 added rays can
# ever be needed.  (branch_d6geo.py's basis-completion growth added low-r_p rays and was observed
# to ADD flex, 17 -> 23, before flooring at 8 with 407 rays; this is the fix.)
# ==================================================================================================
def _ray_constraint_rows(pi, Sadj, gidx, CRe, CIm, PRe, PIm, KR, KI, k0, p):
    """(rows, r_p, |N|) -- the portability-constraint rows ray pi imposes on ker(J_C), in the fixed
       K0 kernel basis, plus its saturation rank r_p."""
    N = np.nonzero(Sadj[pi])[0]
    if N.size == 0:
        return None, 0, 0
    G = np.zeros((2 * N.size, 2 * DSIZE), dtype=np.int64)
    for a, i in enumerate(N):
        for c in range(DSIZE):
            G[2 * a, 2 * c] = CRe[i, c]; G[2 * a, 2 * c + 1] = CIm[i, c]
            G[2 * a + 1, 2 * c] = -CIm[i, c]; G[2 * a + 1, 2 * c + 1] = CRe[i, c]
    rG = gf_rank(G, 2 * DSIZE, p)
    r_p = rG // 2
    if rG >= 2 * N.size:
        return np.zeros((0, k0), dtype=np.int64), r_p, int(N.size)
    L = gf_nullspace(G.T, p)
    pr, pim = PRe[pi] % p, PIm[pi] % p
    B = np.zeros((2 * N.size, k0), dtype=np.int64)
    for a, i in enumerate(N):
        wr, wi = KR[:, i, :], KI[:, i, :]
        B[2 * a] = (modmul(wr, pr.reshape(-1, 1), p).ravel() + modmul(wi, pim.reshape(-1, 1), p).ravel()) % p
        B[2 * a + 1] = (modmul(wr, pim.reshape(-1, 1), p).ravel() - modmul(wi, pr.reshape(-1, 1), p).ravel()) % p
    return modmul(L % p, B, p), r_p, int(N.size)


def stage4_hunt(variant="any", p=P1):
    """Greedy: pick SATURATING (r_p = 5) ambient rays maximising the rank their portability
       constraints add to the accumulated set, until the accumulated rank reaches flex-1 = 16
       (=> flex 1) or no candidate adds anything.  variant='x1' restricts candidates to X-count==1
       rays (keeping D6_CIRCLE.md's 'every ray theta-parametrized' character); variant='any' uses
       the whole 7448-ray pool."""
    print("=" * 96
          )
    print(f"STAGE 4 -- HUNT for a flex-tight core (variant={variant})")
    print("=" * 96)
    pool = [tuple(v) for v in cache_load("d6flex_pool_rays")]
    csyms, gidx = load_d6_core()
    point = bd6.POINTS["x5"]
    rays = [tuple(point[c] for c in s) for s in csyms]
    fr = build_rows(rays)
    n, V = fr["n"], fr["V"]
    K0 = gf_nullspace(np.array(fr["rows"], dtype=np.int64), p)
    k0 = K0.shape[0]
    KR = K0.reshape(k0, V, DSIZE, 2)[:, :, :, 0].copy() % p
    KI = K0.reshape(k0, V, DSIZE, 2)[:, :, :, 1].copy() % p
    PRe, PIm = _pool_arrays(pool, point)
    Sadj = stable_between(pool, csyms)
    CRe, CIm = PRe[gidx], PIm[gidx]
    deg = Sadj.sum(axis=1).astype(int)
    coreset = set(gidx)

    cands = [i for i in range(len(pool)) if i not in coreset and deg[i] >= 6
             and (variant != "x1" or bd6.xcount(pool[i]) == 1)]
    print(f"[stage4] candidate ambient rays (core-degree>=6{', X-count==1' if variant=='x1' else ''}): "
          f"{len(cands)}")
    rowsC, rp, satur = {}, {}, {}
    t0 = time.time()
    for pi in cands:
        R, r_p, Nn = _ray_constraint_rows(pi, Sadj, gidx, CRe, CIm, PRe, PIm, KR, KI, k0, p)
        rowsC[pi] = R; rp[pi] = r_p; satur[pi] = (r_p == DSIZE - 1)
    nsat = sum(1 for pi in cands if satur[pi] and rowsC[pi].shape[0] > 0)
    print(f"[stage4] saturating (r_p=5) candidates with >=1 relation: {nsat}   ({time.time()-t0:.1f}s)")

    # greedy: maintain an RREF basis of the accumulated constraint span
    basis = np.zeros((0, k0), dtype=np.int64)
    chosen, trace = [], []
    target = 16
    while True:
        best, best_gain, best_rows = None, 0, None
        for pi in cands:
            if pi in chosen or not satur[pi] or rowsC[pi].shape[0] == 0:
                continue
            cat = np.vstack([basis, rowsC[pi]])
            g = gf_rank(cat, k0, p) - basis.shape[0]
            if g > best_gain:
                best, best_gain, best_rows = pi, g, cat
        if best is None:
            break
        R, _ = gf_rref(best_rows, p)
        basis = R
        chosen.append(best)
        trace.append((int(best), int(deg[best]), int(bd6.xcount(pool[best])), int(best_gain),
                      int(17 - basis.shape[0])))
        print(f"[stage4]  + ray {best} (core-deg {deg[best]}, X-count {bd6.xcount(pool[best])}) "
              f"kills {best_gain} decoration dims -> predicted flex = {17 - basis.shape[0]}  "
              f"(|core| = {V + len(chosen)})")
        if basis.shape[0] >= target:
            break
    pred = 17 - basis.shape[0]
    print(f"[stage4] chosen {len(chosen)} saturating rays; accumulated cut = {basis.shape[0]}; "
          f"PREDICTED flex = {pred}")

    new_syms = csyms + [pool[i] for i in chosen]
    new_gidx = list(gidx) + [int(i) for i in chosen]
    tsave(f"stage4_hunt_{variant}", dict(chosen=[int(c) for c in chosen], trace=trace,
                                         predicted_flex=int(pred), global_idx=new_gidx,
                                         core_syms=[list(s) for s in new_syms]))

    # immediate mod-p verification of the enlarged core's actual flex
    r2 = [tuple(point[c] for c in s) for s in new_syms]
    cert = flex_modp(r2, p=p)
    print(f"[stage4] mod-p certificate of the enlarged {cert['V']}-ray core: E={cert['E']} "
          f"rankJ={cert['rankJ']} ker={cert['ker']} rankT={cert['rankT']} => flex={cert['flex']}")
    print(f"[stage4] rankT == V+d^2-1 = {cert['V']+35}: {cert['rankT'] == cert['V']+35}")
    tsave(f"stage4_cert_{variant}", cert)
    print(f"[stage4] done ({time.time()-T0:.1f}s)")
    return new_syms, cert


# ==================================================================================================
# STAGE 5 -- CERTIFY the tight core: mechanism-stability, non-degeneracy at two independent generic
# circle points, KS-uncolorability (SAT), EXACT rational flex (sympy QQ) at both points, and the
# lower bound flex >= 1 via the explicitly exhibited modulus motion.
# ==================================================================================================
def certify_core(new_syms, tag, do_exact=True, points=("x5", "x13")):
    print(f"[stage5:{tag}] core: {len(new_syms)} rays, X-count histogram "
          f"{dict(sorted(Counter(bd6.xcount(s) for s in new_syms).items()))}, support-size "
          f"{dict(sorted(Counter(sum(1 for c in s if c != '0') for s in new_syms).items()))}")
    stab = stable_between(new_syms, new_syms)
    np.fill_diagonal(stab, False)
    V = len(new_syms)
    E_abs = sorted((i, j) for i, j in combinations(range(V), 2) if stab[i, j])
    adj = [set(np.nonzero(stab[i])[0].tolist()) for i in range(V)]
    bases, complete = bd6.find_cliques(adj, V, DSIZE)
    col = bd6._sat_colorable(V, E_abs, bases)
    print(f"[stage5:{tag}] mechanism-stable pairs (theta-identical) = {len(E_abs)}, "
          f"bases = {len(bases)} (complete enum={complete}), KS-uncolorable = {not col}")
    out = dict(V=V, pairs=len(E_abs), bases=len(bases), uncolorable=bool(not col),
               core_syms=[list(s) for s in new_syms])
    prev = tload(f"stage5_certified_{tag}") or {}
    for name in points:
        point = bd6.POINTS[name]
        r = [tuple(point[c] for c in s) for s in new_syms]
        Ep = sorted(point_edges(r))
        match = (Ep == E_abs)
        print(f"[stage5:{tag}] {name}: non-degeneracy -- pairs at the point = {len(Ep)} "
              f"(abstract {len(E_abs)}), EXACT MATCH = {match}")
        cm = flex_modp(r, p=P1); cm2 = flex_modp(r, p=P2)
        M = modulus_motion(new_syms, point)
        lower = residual_zero(build_rows(r)["rows"], M)
        line = (f"[stage5:{tag}] {name}: mod-p flex = {cm['flex']} (p={P1}) / {cm2['flex']} (p={P2}); "
                f"rankJ={cm['rankJ']} ker={cm['ker']} rankT={cm['rankT']} (V+d^2-1={V+35}); "
                f"modulus motion exactly in ker(J) = {lower}")
        print(line)
        rec = dict(match=bool(match), modp1=cm, modp2=cm2, modulus_in_ker=bool(lower))
        if do_exact:
            ce = flex_exact(r)
            print(f"[stage5:{tag}] {name}: EXACT rational flex = {ce['flex']} "
                  f"(rankJ={ce['rankJ']} ker={ce['ker']} rankT={ce['rankT']})")
            rec["exact"] = ce
        out[name] = rec
        tsave(f"stage5_certified_{tag}", {**prev, **out})
    for name in ("x5", "x13"):                      # keep results from earlier per-point calls
        if name not in out and name in prev:
            out[name] = prev[name]
    return out


def stage5_certify(variant="any", do_exact=True, points=("x5", "x13")):
    print("=" * 96)
    print(f"STAGE 5 -- CERTIFICATION of the tight core (variant={variant}, points={points})")
    print("=" * 96)
    h = tload(f"stage4_hunt_{variant}") if variant != "coupled" else tload("stage7_coupled")
    assert h is not None, f"run stage4 {variant} first"
    new_syms = [tuple(s) for s in h["core_syms"]]
    out = certify_core(new_syms, variant, do_exact=do_exact, points=points)
    out["chosen"] = h.get("chosen", []); out["global_idx"] = h["global_idx"]
    tsave(f"stage5_certified_{variant}", out)
    print(f"[stage5] done ({time.time()-T0:.1f}s)")
    return out


# ==================================================================================================
# STAGE 6 -- the UNDECORATED holonomy generator: the incidence-frame Wilczek-Zee connection of a
# flex-tight core, its characteristic polynomial and Galois group.  Reuses branch_d6geo's exact
# Fraction Fourier machinery UNMODIFIED (it is already X-count-generic).
# ==================================================================================================
def holonomy_report(core_syms, bases, tag):
    from branch_d6geo import exact_fourier_d, clear_denominators_d
    A0, A1, Am, pcount, Nb = exact_fourier_d(list(core_syms), [tuple(b) for b in bases], DSIZE)
    z1 = all(A1[c][d] == 0 for c in range(DSIZE) for d in range(DSIZE))
    zm = all(Am[c][d] == 0 for c in range(DSIZE) for d in range(DSIZE))
    offdiag0 = all(A0[c][d] == 0 for c in range(DSIZE) for d in range(DSIZE) if c != d)
    M, L = clear_denominators_d(A0, DSIZE)
    print(f"[stage6:{tag}] Nb={Nb} bases; A1==0: {z1}, Am==0: {zm} (constant connection); "
          f"A0 diagonal: {offdiag0}")
    print(f"[stage6:{tag}] L={L}, M=L*A0 =")
    for r in M:
        print(f"          {r}")
    x = sp.symbols("x")
    cp = sp.expand(sp.Matrix(M).charpoly(x).as_expr())
    print(f"[stage6:{tag}] trace(M) = {sum(M[c][c] for c in range(DSIZE))}, Tr(S)=Tr(M)/L")
    print(f"[stage6:{tag}] char poly = {cp}")
    fl = sp.factor_list(cp, x)
    print(f"[stage6:{tag}] factorization over Q: {fl}")
    from sympy.polys.numberfields.galoisgroups import galois_group
    ggs = []
    for f, mult in fl[1]:
        Pf = sp.Poly(f, x, domain="QQ")
        if Pf.degree() <= 1:
            ggs.append((str(f), "trivial (deg<=1)"))
            continue
        try:
            G, alt = galois_group(Pf, by_name=True)
            ggs.append((str(f), str(G)))
        except Exception as ex:
            ggs.append((str(f), f"FAILED: {ex}"))
    print(f"[stage6:{tag}] Galois groups of the irreducible factors: {ggs}")
    return dict(L=L, M=M, constant=bool(z1 and zm), diagonal=bool(offdiag0), Nb=int(Nb),
                trace=int(sum(M[c][c] for c in range(DSIZE))), charpoly=str(cp),
                factors=[str(f) for f, _ in fl[1]], galois=ggs)


def stage6_holonomy(variant="any"):
    print("=" * 96)
    print(f"STAGE 6 -- UNDECORATED HOLONOMY GENERATOR of the flex-tight core (variant={variant})")
    print("=" * 96)
    cert = (tload(f"stage5_certified_{variant}") or tload(f"stage7_{variant}"))
    assert cert is not None, f"run stage5 {variant} first"
    syms = [tuple(s) for s in cert["core_syms"]]
    stab = stable_between(syms, syms); np.fill_diagonal(stab, False)
    adj = [set(np.nonzero(stab[i])[0].tolist()) for i in range(len(syms))]
    bases, complete = bd6.find_cliques(adj, len(syms), DSIZE)
    bp = sorted(set(x for b in bases for x in b))
    print(f"[stage6] {len(syms)} rays, {len(bases)} bases (complete={complete}); "
          f"basis-participating rays: {len(bp)}; X-count histogram of the BASIS-PARTICIPATING part: "
          f"{dict(sorted(Counter(bd6.xcount(syms[i]) for i in bp).items()))}")
    rep = holonomy_report(syms, bases, variant)
    rep["n_rays"] = len(syms); rep["n_bases"] = len(bases); rep["n_basis_participating"] = len(bp)
    tsave(f"stage6_holonomy_{variant}", rep)
    return rep


# ==================================================================================================
# STAGE 7 -- COUPLED flex-tight core.  Once flex = 1 has been reached, the book-keeping law says
# ANY further SATURATING ray (r_p = 5) keeps flex <= 1, and flex >= 1 always (the modulus motion).
# So flex STAYS EXACTLY 1 under arbitrary saturating growth -- which lets us enrich the BASIS
# structure (the only thing the WZ connection sees) freely, aiming at a coupled (non-diagonal,
# high-X-count) basis-participating set while keeping the core flex-tight.
# ==================================================================================================
def _saturating_basis_closing(pool, syms, minx, p, exclude, want=None):
    """Ambient rays that (i) are SATURATING w.r.t. the current core (r_p = d-1, so adding them
       cannot raise the flex), (ii) have X-count >= minx, and (iii) close at least one new basis
       with the current core."""
    CRe, CIm = _pool_arrays(syms, bd6.POINTS["x5"])
    Sadj = stable_between(pool, syms)
    deg = Sadj.sum(axis=1).astype(int)
    stabC = stable_between(syms, syms); np.fill_diagonal(stabC, False)
    adjC = [set(np.nonzero(stabC[i])[0].tolist()) for i in range(len(syms))]
    out = []
    for pi in sorted(range(len(pool)), key=lambda i: -deg[i]):
        if pi in exclude or deg[pi] < DSIZE - 1 or bd6.xcount(pool[pi]) < minx:
            continue
        N = np.nonzero(Sadj[pi])[0]
        rows = []
        for i in N:
            rows.append([x for c in range(DSIZE) for x in (CRe[i, c], CIm[i, c])])
            rows.append([x for c in range(DSIZE) for x in (-CIm[i, c], CRe[i, c])])
        if gf_rank(rows, 2 * DSIZE, p) != 2 * (DSIZE - 1):
            continue
        sub = {int(i) for i in N}
        adjsub = [adjC[i] & sub for i in range(len(syms))]
        found = [False]

        def ext(cands, cur):
            if found[0]:
                return
            if len(cur) == DSIZE - 1:
                found[0] = True; return
            if len(cur) + len(cands) < DSIZE - 1:
                return
            cl = sorted(cands)
            for k, v in enumerate(cl):
                ext(set(x for x in cl[k + 1:] if x in adjsub[v]), cur + [v])
                if found[0]:
                    return
        ext(set(sorted(sub)), [])
        if found[0]:
            out.append(pi)
            if want and len(out) >= want:
                break
    return out


def stage7_iterate(minx=2, rounds=6, cap=130, p=P1):
    """Iterate the saturating/basis-closing enrichment: each round the core grows, so MORE ambient
       rays become saturating and basis-closing.  flex stays EXACTLY 1 throughout by the
       book-keeping law (verified mod-p each round)."""
    print("=" * 96)
    print(f"STAGE 7-ITER -- iterated coupled enrichment (X-count>={minx}, cap={cap})")
    print("=" * 96)
    pool = [tuple(v) for v in cache_load("d6flex_pool_rays")]
    st = tload("stage7_iter") or tload("stage7_coupled") or tload("stage5_certified_any")
    syms = [tuple(s) for s in st["core_syms"]]
    gidx = list(st["global_idx"])
    point = bd6.POINTS["x5"]
    for rnd in range(rounds):
        if len(syms) >= cap:
            break
        cand = _saturating_basis_closing(pool, syms, minx, p, set(gidx),
                                         want=max(1, cap - len(syms)))
        if not cand:
            print(f"[stage7-iter] round {rnd}: no further saturating basis-closing candidates")
            break
        cand = cand[:cap - len(syms)]
        syms = syms + [pool[i] for i in cand]
        gidx += [int(i) for i in cand]
        cert = flex_modp([tuple(point[c] for c in s) for s in syms], p=p)
        print(f"[stage7-iter] round {rnd}: +{len(cand)} rays -> V={cert['V']} E={cert['E']} "
              f"rankT={cert['rankT']} (V+35={cert['V']+35}) mod-p flex={cert['flex']}")
        tsave("stage7_iter", dict(global_idx=gidx, core_syms=[list(s) for s in syms], cert=cert))
        if cert["flex"] != 1:
            print("[stage7-iter] *** flex left 1 -- stopping (should not happen for saturating rays)")
            break
    return syms


def stage7_bulk(minx=2, K=50, p=P1, from_key="stage7_iter", maxx=DSIZE, save="stage7_bulk"):
    """Bulk enrichment: add the K highest-core-degree SATURATING ambient rays with X-count >= minx
       WITHOUT requiring each to close a basis by itself (new bases can be closed jointly by
       several added rays).  Saturation still guarantees flex <= 1, hence = 1."""
    print("=" * 96)
    print(f"STAGE 7-BULK -- add {K} saturating X-count>={minx} rays (joint basis closure allowed)")
    print("=" * 96)
    pool = [tuple(v) for v in cache_load("d6flex_pool_rays")]
    st = tload(from_key) or tload("stage5_certified_any")
    syms = [tuple(s) for s in st["core_syms"]]
    gidx = list(st["global_idx"])
    point = bd6.POINTS["x5"]
    CRe, CIm = _pool_arrays(syms, point)
    Sadj = stable_between(pool, syms)
    deg = Sadj.sum(axis=1).astype(int)
    chosen = []
    for pi in sorted(range(len(pool)), key=lambda i: -deg[i]):
        if pi in set(gidx) or deg[pi] < DSIZE - 1 or not (minx <= bd6.xcount(pool[pi]) <= maxx):
            continue
        N = np.nonzero(Sadj[pi])[0]
        rows = []
        for i in N:
            rows.append([x for c in range(DSIZE) for x in (CRe[i, c], CIm[i, c])])
            rows.append([x for c in range(DSIZE) for x in (-CIm[i, c], CRe[i, c])])
        if gf_rank(rows, 2 * DSIZE, p) != 2 * (DSIZE - 1):
            continue
        chosen.append(pi)
        if len(chosen) >= K:
            break
    syms = syms + [pool[i] for i in chosen]
    gidx += [int(i) for i in chosen]
    cert = flex_modp([tuple(point[c] for c in s) for s in syms], p=p)
    print(f"[stage7-bulk] +{len(chosen)} rays -> V={cert['V']} E={cert['E']} rankJ={cert['rankJ']} "
          f"rankT={cert['rankT']} (V+35={cert['V']+35}) mod-p flex={cert['flex']}")
    tsave(save, dict(global_idx=gidx, core_syms=[list(s) for s in syms], cert=cert))
    return syms, cert


def stage7_coupled(minx=2, cap=400, p=P1):
    print("=" * 96)
    print(f"STAGE 7 -- COUPLED flex-tight core (add basis-forming saturating rays, X-count>={minx})")
    print("=" * 96)
    pool = [tuple(v) for v in cache_load("d6flex_pool_rays")]
    base = tload("stage5_certified_any")
    assert base is not None, "run stage5 any first"
    syms = [tuple(s) for s in base["core_syms"]]
    gidx = base["global_idx"]
    point = bd6.POINTS["x5"]
    CRe, CIm = _pool_arrays(syms, point)
    PRe, PIm = _pool_arrays(pool, point)
    Sadj = stable_between(pool, syms)
    deg = Sadj.sum(axis=1).astype(int)
    coreset = set(gidx)

    # saturating candidates (r_p = 5) with X-count>=minx that CLOSE at least one new basis
    stabC = stable_between(syms, syms); np.fill_diagonal(stabC, False)
    adjC = [set(np.nonzero(stabC[i])[0].tolist()) for i in range(len(syms))]
    chosen = []
    t0 = time.time()
    for pi in sorted(range(len(pool)), key=lambda i: -deg[i]):
        if pi in coreset or deg[pi] < DSIZE - 1 or bd6.xcount(pool[pi]) < minx:
            continue
        N = np.nonzero(Sadj[pi])[0]
        rows = []
        for i in N:
            rows.append([x for c in range(DSIZE) for x in (CRe[i, c], CIm[i, c])])
            rows.append([x for c in range(DSIZE) for x in (-CIm[i, c], CRe[i, c])])
        if gf_rank(rows, 2 * DSIZE, p) != 2 * (DSIZE - 1):
            continue                                    # not saturating -> would ADD flex
        sub = {int(i) for i in N}
        adjsub = [adjC[i] & sub for i in range(len(syms))]
        # does p close a basis?  need a (d-1)-clique inside N
        found = False
        Nl = sorted(sub)
        def ext(cands, cur):
            nonlocal found
            if found:
                return
            if len(cur) == DSIZE - 1:
                found = True; return
            if len(cur) + len(cands) < DSIZE - 1:
                return
            cl = sorted(cands)
            for k, v in enumerate(cl):
                ext(set(x for x in cl[k + 1:] if x in adjsub[v]), cur + [v])
                if found:
                    return
        ext(set(Nl), [])
        if found:
            chosen.append(pi)
        if len(chosen) >= cap:
            break
    print(f"[stage7] saturating, basis-closing candidates with X-count>={minx}: {len(chosen)} "
          f"({time.time()-t0:.1f}s)")
    new_syms = syms + [pool[i] for i in chosen]
    cert = flex_modp([tuple(point[c] for c in s) for s in new_syms], p=p)
    print(f"[stage7] enlarged core {cert['V']} rays: E={cert['E']} rankJ={cert['rankJ']} "
          f"ker={cert['ker']} rankT={cert['rankT']} (V+35={cert['V']+35}) => mod-p flex={cert['flex']}")
    tsave("stage7_coupled", dict(chosen=[int(c) for c in chosen], cert=cert,
                                 global_idx=list(gidx) + [int(c) for c in chosen],
                                 core_syms=[list(s) for s in new_syms]))
    return new_syms, cert


def stage7b_certify_coupled(do_exact=True):
    h = tload("stage7_coupled")
    assert h is not None, "run stage7 first"
    syms = [tuple(s) for s in h["core_syms"]]
    out = certify_core(syms, "coupled", do_exact=do_exact)
    out["global_idx"] = h["global_idx"]
    tsave("stage5_certified_coupled", out)
    return out


# ==================================================================================================
# STAGE 8 -- SELF-CHECK: re-read every checkpoint and assert the headline numbers.  PASS/FAIL.
# ==================================================================================================
def stage8_summary():
    print("=" * 96)
    print("STAGE 8 -- SELF-CHECK of every headline number of branch D6-TIGHT")
    print("=" * 96)
    checks = []

    def chk(name, cond, detail=""):
        checks.append((name, bool(cond), detail))

    g = tload("stage0_gate")
    chk("stage0 sanity gate (d4 M9 flex 1, d4 M10 flex 2, d6 46-ray flex 17)", g and g["passed"])
    g2 = tload("stage0b_m10twin")
    if g2:
        chk("stage0b twin cross-check on d=4 M10 (1 twin pair = all of M10's flex 2)", g2["passed"])
    d = tload("stage2_decomp")
    if d:
        chk("46-ray core exact certificate rankJ=454 ker=98 rankT=81 flex=17",
            d["rankJ"] == 454 and d["rankT"] == 81 and d["flex"] == 17)
        chk("modulus motion exactly in ker(J), 1-dimensional mod trivial",
            d["modulus_in_ker"] and d["dim_modulus"] == 1)
        chk("loose-ray decoration = 0 (every core ray has a saturated neighbourhood r_i=5)",
            d["dim_single"] == 0 and all(x[2] == 5 for x in d["loose_report"]))
        chk("4 twin pairs contributing exactly 8 decoration dimensions",
            len(d["twins"]) == 4 and d["dim_pair"] == 8)
        chk("modulus + twin = 9, delocalised remainder = 8", d["dim_all"] == 9 and d["remainder"] == 8)
    b = tload("stage2b_blocks")
    if b:
        chk("all 17 flex directions are coordinate-block supported", b["total"] == 17)
    s3 = tload("stage3_portability")
    if s3:
        chk("POOL-PORTABLE FLEX = 1 over the full 7448-ray ambient pool", s3["portable_flex"] == 1)
        chk("the surviving portable direction is the modulus motion", s3["modulus_survives"])
    s3x = tload("stage3_portability_x1")
    if s3x:
        chk("X-count==1-only ambient pool gives portable flex 7 (the previous branch's floor)",
            s3x["portable_flex"] == 7)
    for tag, exp in (("any", (50, 341, 17)), ("coupled", (52, 370, 21))):
        c = tload(f"stage5_certified_{tag}")
        if not c:
            continue
        chk(f"[{tag}] core {exp[0]} rays / {exp[1]} pairs / {exp[2]} bases, KS-uncolorable",
            (c["V"], c["pairs"], c["bases"]) == exp and c["uncolorable"])
        for pt in ("x5", "x13"):
            if pt in c:
                chk(f"[{tag}] {pt}: non-degenerate, EXACT rational flex = 1, modulus in ker(J)",
                    c[pt]["match"] and c[pt].get("exact", {}).get("flex") == 1
                    and c[pt]["modulus_in_ker"])
    h = tload("stage6_holonomy_any")
    if h:
        chk("[any] holonomy generator constant, diagonal, L=17, M=diag(6,7,3,10,10,3)",
            h["constant"] and h["diagonal"] and h["L"] == 17
            and [h["M"][i][i] for i in range(6)] == [6, 7, 3, 10, 10, 3])
    hc = tload("stage6_holonomy_coupled")
    if hc:
        chk("[coupled] holonomy constant, NON-diagonal, L=21, trace 51, a cubic factor with S3",
            hc["constant"] and not hc["diagonal"] and hc["L"] == 21 and hc["trace"] == 51
            and any("S3" in g for _, g in hc["galois"]))
    npass = sum(1 for _, ok, _ in checks if ok)
    for name, ok, det in checks:
        print(f"   [{'PASS' if ok else 'FAIL'}] {name}{(' -- ' + det) if det else ''}")
    print(f"[stage8] {npass}/{len(checks)} checks passed -> "
          f"{'ALL PASS' if npass == len(checks) else 'FAILURES PRESENT'}")
    tsave("stage8_summary", dict(checks=[[n, o] for n, o, _ in checks], passed=npass,
                                 total=len(checks)))
    return npass == len(checks)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which == "all":
        stage0_gate(); stage0b_m10_twin(); stage1_data(); stage2_decompose(); stage2b_blocks()
        stage3_portability(); stage3_portability(restrict=1)
        stage4_hunt("any"); stage5_certify("any"); stage6_holonomy("any")
        stage4_hunt("x1")
        stage7_coupled(); stage5_certify("coupled"); stage6_holonomy("coupled")
        sys.exit(0 if stage8_summary() else 1)
    if which == "stage0":
        sys.exit(0 if stage0_gate() else 1)
    elif which == "stage0b":
        sys.exit(0 if stage0b_m10_twin() else 1)
    elif which == "stage1":
        stage1_data()
    elif which == "stage2":
        stage2_decompose()
    elif which == "stage2b":
        stage2b_blocks()
    elif which == "stage3":
        _r = None
        if len(sys.argv) > 2 and sys.argv[2].startswith("x"):
            _r = int(sys.argv[2][1:])
        stage3_portability(p=P1, restrict=_r)
    elif which == "stage4":
        stage4_hunt(variant=sys.argv[2] if len(sys.argv) > 2 else "any")
    elif which == "stage5":
        _v = sys.argv[2] if len(sys.argv) > 2 else "any"
        _pts = (sys.argv[3],) if len(sys.argv) > 3 and sys.argv[3] in bd6.POINTS else ("x5", "x13")
        stage5_certify(variant=_v, points=_pts,
                       do_exact=("modp" not in sys.argv[3:]))
    elif which == "stage6":
        stage6_holonomy(variant=sys.argv[2] if len(sys.argv) > 2 else "any")
    elif which == "stage7":
        stage7_coupled(minx=int(sys.argv[2]) if len(sys.argv) > 2 else 2,
                       cap=int(sys.argv[3]) if len(sys.argv) > 3 else 400)
    elif which == "stage7bulk":
        stage7_bulk(minx=int(sys.argv[2]) if len(sys.argv) > 2 else 2,
                    K=int(sys.argv[3]) if len(sys.argv) > 3 else 50,
                    from_key=sys.argv[4] if len(sys.argv) > 4 else "stage7_iter",
                    maxx=int(sys.argv[5]) if len(sys.argv) > 5 else DSIZE,
                    save=sys.argv[6] if len(sys.argv) > 6 else "stage7_bulk")
    elif which == "stage7iter":
        stage7_iterate(minx=int(sys.argv[2]) if len(sys.argv) > 2 else 2,
                       rounds=int(sys.argv[3]) if len(sys.argv) > 3 else 6,
                       cap=int(sys.argv[4]) if len(sys.argv) > 4 else 130)
    elif which == "stage8":
        sys.exit(0 if stage8_summary() else 1)
    elif which == "stage7b":
        stage7b_certify_coupled(do_exact=(len(sys.argv) <= 2 or sys.argv[2] != "modp"))
    else:
        print(f"unknown stage {which!r}")
        sys.exit(1)
