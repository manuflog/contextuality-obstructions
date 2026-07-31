#!/usr/bin/env python3
# line30_flex.py -- TASK 3: exact flex certificate for the 30-degree-line core at t = 2,
# following branch_d8flexcert.py's convention (= branch_d6geo.build_flex_rows):
#
#   variables = (Re, Im) of every coordinate of every ray  (n = 2*d*V, d = 3);
#   J rows    = 2 rows per orthogonal pair (Re/Im of linearized Hermitian dot)
#               + 1 norm row per ray;
#   T (gauge) = V per-ray-phase directions + d^2 = 9 global u(3) directions
#               (3 diagonal + 6 off-diagonal anti-Hermitian generators), |T| = V + 9,
#               with the single exact dependency sum(phase rows) == sum(diag rows)
#               =>  rank_Q(T) <= V + d^2 - 1 = V + 8.
#
# Entries at t=2 live in Q(sqrt3, i); (Re, Im) parts live in K = Q(sqrt3), represented
# exactly as Fraction pairs (p, q) = p + q*sqrt3.  NO FLOATS in the certificate path.
#
# The t-direction tangent (modulus deformation, norm-preserving representative):
#   ray i has a_i unit entries and b_i X/Y entries, |r_i(t)|^2 = a_i + b_i t^2;
#   R_i(t) = (|r_i(2)| / |r_i(t)|) * r_i(t)  keeps every norm constant and keeps every
#   STABLE edge exactly orthogonal for all t, so  v_t := dR_i/dt|_{t=2}  must satisfy
#   J.v_t = 0 exactly.  Explicitly  v_i = r_i'(2) - mu_i r_i(2),  mu_i = 2 b_i /
#   (a_i + 4 b_i),  r_i'(2) = (entry/2 on X/Y entries, 0 on unit entries).
#
#   stage cert  : (0) every core pair is a LINE-STABLE edge (else STOP);
#                 (1) J.v_t = 0 exactly, every row, over K;
#                 (2) J.t = 0 exactly for all V+9 gauge generators, over K;
#                 (3) exact dependency sum(phase)==sum(diag);
#                 (4) restriction certificate: subset S with rank_K(T|_S) + 1 =
#                     rank_K(T|_S + v_t|_S)  (exact Gaussian elimination over K)
#                     => v_t NOT in span(T) => FLEX >= 1;
#                 (5) modular corroboration: rank_p(T) = V+8 (saturates the exact
#                     ceiling => rank_Q(T) = V+8 EXACTLY) and rank_p(T+v_t) = V+9
#                     => v_t not in span_Q(T), independently; two primes, sqrt3
#                     realized as a square root of 3 mod p.
#   stage rankj : rank_p(J) squeezed against n - rank(T) - 1 (two primes) => if it hits,
#                 nullity = rank(T)+1 exactly and FLEX EXACTLY 1.
#
# Cache: line30_flex.cache.json.
import json, os, sys, time
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from itertools import combinations
from fractions import Fraction
import random

from line30_ring import (hdot_pol, pzero, kadd, ksub, kmul, kneg, kscale, kzero,
                         KZERO, krank)

T0 = time.time()
HERE = os.path.dirname(os.path.abspath(__file__))
D = 3


def jload(name):
    p = os.path.join(HERE, name)
    return json.load(open(p)) if os.path.exists(p) else None


pool = jload("line30_stable_pool.cache.json")
STABLE_RAYS = [tuple(r) for r in pool["rays"]]
STABLE_E = {tuple(e) for e in pool["pairs"]}
STABLE_T = [tuple(t) for t in pool["triads"]]

TARGET = sys.argv[2] if len(sys.argv) > 2 else "core"
if TARGET == "core":
    core = jload("line30_core.cache.json")
    assert core and core.get("complete") and core.get("uncolorable")
    CORE_SYMS = [tuple(s) for s in core["core_syms"]]
    # core_pairs / core_triads are stored in GLOBAL stable-pool indices; relabel local.
    _g2l = {g: l for l, g in enumerate(core["keep"])}
    CORE_PAIRS = sorted(tuple(sorted((_g2l[a], _g2l[b]))) for a, b in core["core_pairs"])
    CORE_TRIADS = [sorted(_g2l[v] for v in t) for t in core["core_triads"]]
else:                                   # the FULL stable pool as the deforming object
    CORE_SYMS = STABLE_RAYS
    CORE_PAIRS = sorted(STABLE_E)
    CORE_TRIADS = [list(t) for t in STABLE_T]
V = len(CORE_SYMS)
N = 2 * D * V
CACHE_NAME = f"line30_flex_{TARGET}.cache.json"


# --------------------------------------------------------------------------------------
# rays at rational t as (Re, Im) pairs of K elements
# --------------------------------------------------------------------------------------
def entry_K(sym, t):
    """(Re, Im) of the alphabet entry at x = t*zeta, each a K element."""
    th = Fraction(t) / 2
    table = {
        "0":  ((Fraction(0), Fraction(0)), (Fraction(0), Fraction(0))),
        "1":  ((Fraction(1), Fraction(0)), (Fraction(0), Fraction(0))),
        "X":  ((Fraction(0), th), (th, Fraction(0))),          # t*(sqrt3 + i)/2
        "Y":  ((Fraction(0), th), (-th, Fraction(0))),         # t*(sqrt3 - i)/2
    }
    if sym.startswith("-"):
        (re, im) = table[sym[1:]]
        return (kneg(re), kneg(im))
    return table[sym]


def rays_K(t):
    return [[entry_K(s, t) for s in ray] for ray in CORE_SYMS]


def coord(i, c, real):
    return 2 * D * i + 2 * c + (0 if real else 1)


def build_rows_K(rays, E):
    """branch_d6geo.build_flex_rows convention verbatim, K-valued, sparse rows."""
    rows = []
    for i, j in E:
        re, im = [], []
        for c in range(D):
            Rei, Imi = rays[i][c]
            Rej, Imj = rays[j][c]
            if not kzero(Rej): re.append((coord(i, c, True), Rej))
            if not kzero(Imj): re.append((coord(i, c, False), Imj))
            if not kzero(Rei): re.append((coord(j, c, True), Rei))
            if not kzero(Imi): re.append((coord(j, c, False), Imi))
            if not kzero(Imj): im.append((coord(i, c, True), Imj))
            if not kzero(Rej): im.append((coord(i, c, False), kneg(Rej)))
            if not kzero(Imi): im.append((coord(j, c, True), kneg(Imi)))
            if not kzero(Rei): im.append((coord(j, c, False), Rei))
        rows.append(re)
        rows.append(im)
    for i in range(V):
        r = []
        for c in range(D):
            Re, Im = rays[i][c]
            if not kzero(Re): r.append((coord(i, c, True), Re))
            if not kzero(Im): r.append((coord(i, c, False), Im))
        rows.append(r)

    triv = []
    for i in range(V):                                   # V per-ray phases
        t = []
        for c in range(D):
            Re, Im = rays[i][c]
            if not kzero(Im): t.append((coord(i, c, True), kneg(Im)))
            if not kzero(Re): t.append((coord(i, c, False), Re))
        triv.append(t)
    for a in range(D):                                   # 3 diagonal u(3)
        t = []
        for i in range(V):
            Re, Im = rays[i][a]
            if not kzero(Im): t.append((coord(i, a, True), kneg(Im)))
            if not kzero(Re): t.append((coord(i, a, False), Re))
        triv.append(t)
    for a in range(D):                                   # 6 off-diagonal u(3)
        for b in range(a + 1, D):
            t1, t2 = [], []
            for i in range(V):
                Ra, Ia = rays[i][a]
                Rb, Ib = rays[i][b]
                if not kzero(Ib): t1.append((coord(i, a, True), kneg(Ib)))
                if not kzero(Rb): t1.append((coord(i, a, False), Rb))
                if not kzero(Ia): t1.append((coord(i, b, True), kneg(Ia)))
                if not kzero(Ra): t1.append((coord(i, b, False), Ra))
                if not kzero(Rb): t2.append((coord(i, a, True), Rb))
                if not kzero(Ib): t2.append((coord(i, a, False), Ib))
                if not kzero(Ra): t2.append((coord(i, b, True), kneg(Ra)))
                if not kzero(Ia): t2.append((coord(i, b, False), kneg(Ia)))
            triv.append(t1)
            triv.append(t2)
    return rows, triv


def v_t_at(t):
    """v_t = d/dt of the norm-preserving representatives, exact, K-valued dense."""
    vt = [KZERO] * N
    tF = Fraction(t)
    for i, ray in enumerate(CORE_SYMS):
        a = sum(1 for s in ray if s in ("1", "-1"))
        b = sum(1 for s in ray if s in ("X", "-X", "Y", "-Y"))
        mu = Fraction(b) * tF / (a + b * tF * tF)   # |r|'/|r| = t*b/(a + b t^2)
        for c, s in enumerate(ray):
            Re, Im = entry_K(s, t)
            if s in ("X", "-X", "Y", "-Y"):
                fac = Fraction(1) / tF - mu                    # entry/t - mu*entry
                vt[coord(i, c, True)] = kscale(fac, Re)
                vt[coord(i, c, False)] = kscale(fac, Im)
            elif s != "0":
                vt[coord(i, c, True)] = kscale(-mu, Re)
                vt[coord(i, c, False)] = kscale(-mu, Im)
    return vt


def sdot_K(row, vec):
    s = KZERO
    for i, v in row:
        s = kadd(s, kmul(v, vec[i]))
    return s


# --------------------------------------------------------------------------------------
# modular arithmetic: K -> GF(p) with sqrt3 |-> tonelli(3, p); two primes with (3|p)=1
# --------------------------------------------------------------------------------------
def tonelli(a, p):
    assert pow(a, (p - 1) // 2, p) == 1
    if p % 4 == 3:
        return pow(a, (p + 1) // 4, p)
    q, s = p - 1, 0
    while q % 2 == 0:
        q //= 2; s += 1
    z = 2
    while pow(z, (p - 1) // 2, p) != p - 1:
        z += 1
    m, c, tt, r = s, pow(z, q, p), pow(a, q, p), pow(a, (q + 1) // 2, p)
    while tt != 1:
        i, t2 = 0, tt
        while t2 != 1:
            t2 = t2 * t2 % p; i += 1
        b = pow(c, 1 << (m - i - 1), p)
        m, c, tt, r = i, b * b % p, tt * b * b % p, r * b % p
    return r


def pick_primes():
    out = []
    for p in (998244353, 999999937, 1000000007, 1000000009, 1004535809):
        if pow(3, (p - 1) // 2, p) == 1:
            out.append(p)
        if len(out) == 2:
            break
    assert len(out) == 2, "need two primes with 3 a QR"
    return out


def K_modp(x, p, s3):
    pn, qn = x
    return (pn.numerator * pow(pn.denominator, -1, p) +
            qn.numerator * pow(qn.denominator, -1, p) * s3) % p


def rank_modp_dense(rows_dense, p):
    """exact rank over GF(p), numpy int64 elimination (all values < p < 2^30, products
    < p^2 < 2^60: int64-exact)."""
    import numpy as np
    A = np.array(rows_dense, dtype=np.int64) % p
    m, n = A.shape
    r = 0
    for c in range(n):
        nz = np.nonzero(A[r:, c])[0]
        if nz.size == 0:
            continue
        i = r + int(nz[0])
        if i != r:
            A[[r, i]] = A[[i, r]]
        inv = pow(int(A[r, c]), p - 2, p)
        A[r, c:] = (A[r, c:] * inv) % p
        col = A[r + 1:, c].copy()
        if col.size:
            A[r + 1:, c:] = (A[r + 1:, c:] - np.outer(col, A[r, c:])) % p
        r += 1
        if r == m:
            break
    return r


def densify_K_modp(sparse_rows, p, s3):
    out = []
    for row in sparse_rows:
        d = [0] * N
        for i, v in row:
            d[i] = K_modp(v, p, s3)
        out.append(d)
    return out


# --------------------------------------------------------------------------------------
def stage_cert():
    # (0) every core pair is a line-stable edge
    stable_idx = {r: i for i, r in enumerate(STABLE_RAYS)}
    E = CORE_PAIRS
    for i, j in E:
        gi, gj = stable_idx[CORE_SYMS[i]], stable_idx[CORE_SYMS[j]]
        key = (min(gi, gj), max(gi, gj))
        assert key in STABLE_E, f"core pair {(i, j)} is NOT a line-stable edge -- STOP"
        assert pzero(hdot_pol(CORE_SYMS[i], CORE_SYMS[j]))     # belt and braces
    print(f"[cert] (0) all {len(E)} core pairs vanish IDENTICALLY in t (line-stable)")

    rays = rays_K(2)
    rows, triv = build_rows_K(rays, E)
    assert len(rows) == 2 * len(E) + V and len(triv) == V + 9
    print(f"[cert] target={TARGET} V={V} d=3 n={N} |E|={len(E)} Jrows={len(rows)} "
          f"triv={len(triv)}")

    # (1) J.v_t = 0 exactly over K
    vt = v_t_at(2)
    nmoving = sum(1 for i in range(V)
                  if any(not kzero(vt[coord(i, c, rl)]) for c in range(D)
                         for rl in (True, False)))
    print(f"[cert] v_t: {nmoving} of {V} rays move (nonzero v_t block); "
          f"mixed-degree rays are the only movers")
    if not any(not kzero(x) for x in vt):
        print(f"[cert] *** v_t == 0 IDENTICALLY on target={TARGET}: every ray is "
              f"projectively t-CONSTANT; the t-deformation acts TRIVIALLY (v_t = 0 is "
              f"in span(T) degenerately) => NO FLEX on this target ***")
        json.dump(dict(V=V, n=N, E=len(E), vt_zero=True, flex_ge_1=False),
                  open(os.path.join(HERE, CACHE_NAME), "w"))
        return
    bad = sum(1 for row in rows if not kzero(sdot_K(row, vt)))
    print(f"[cert] (1) J.v_t == 0 exactly over Q(sqrt3): violations = {bad} of {len(rows)} "
          f"rows (edge rows {2*len(E)}, norm rows {V})")
    assert bad == 0

    # (2) J.t = 0 for every gauge generator, exactly over K
    nz = 0
    for tgen in triv:
        tv = [KZERO] * N
        for i, v in tgen:
            tv[i] = kadd(tv[i], v)
        nz += sum(1 for row in rows if not kzero(sdot_K(row, tv)))
    print(f"[cert] (2) J.t == 0 for all {len(triv)} gauge generators: nonzero dots = {nz}")
    assert nz == 0

    # (3) exact dependency: sum(phase rows) == sum(diag rows)
    acc = [KZERO] * N
    for tgen in triv[:V]:
        for i, v in tgen:
            acc[i] = kadd(acc[i], v)
    for tgen in triv[V:V + 3]:
        for i, v in tgen:
            acc[i] = ksub(acc[i], v)
    dep = all(kzero(x) for x in acc)
    print(f"[cert] (3) exact dependency sum(phase)==sum(diag): {dep}  "
          f"=> rank_Q(T) <= V + 8 = {V + 8}")
    assert dep

    # (4) restriction certificate over K: rank jump on a witness subset
    def restrict(vec_dense_or_sparse, S):
        cols = [coord(i, c, rl) for i in S for c in range(D) for rl in (True, False)]
        if vec_dense_or_sparse and isinstance(vec_dense_or_sparse[0], tuple) and \
                len(vec_dense_or_sparse[0]) == 2 and isinstance(vec_dense_or_sparse[0][0], int):
            dd = {}
            for i, v in vec_dense_or_sparse:
                dd[i] = kadd(dd.get(i, KZERO), v)
            return [dd.get(cc, KZERO) for cc in cols]
        return [vec_dense_or_sparse[cc] for cc in cols]

    subset_used, r1, r2 = None, None, None
    cands = [tuple(t) for t in CORE_TRIADS]
    rnd = random.Random(20260731)
    cands += [tuple(sorted(rnd.sample(range(V), 3))) for _ in range(200)]
    for S in cands:
        TS = [restrict(t, S) for t in triv]
        vtS = restrict(vt, S)
        if all(kzero(x) for x in vtS):
            continue
        ra = krank(TS)
        rb = krank(TS + [vtS])
        if rb == ra + 1:
            subset_used, r1, r2 = S, ra, rb
            break
    print(f"[cert] (4) witness subset S = {subset_used}: rank_K(T|_S) = {r1} -> "
          f"rank_K(T|_S + v_t|_S) = {r2}  (exact Q(sqrt3) elimination; generic ceiling "
          f"{3 + 9 - 1} = 11)")
    assert subset_used is not None and r2 == r1 + 1, \
        "NO SUBSET CERTIFIES NON-MEMBERSHIP -- v_t may be gauge-trivial"

    # (5) modular corroboration, two primes
    primes = pick_primes()
    ranks_T, ranks_Tv = {}, {}
    for p in primes:
        s3 = tonelli(3, p)
        assert s3 * s3 % p == 3 % p
        Td = densify_K_modp(triv, p, s3)
        ranks_T[p] = rank_modp_dense(Td, p)
        vd = [K_modp(x, p, s3) for x in vt]
        ranks_Tv[p] = rank_modp_dense(Td + [vd], p)
    print(f"[cert] (5) rank_p(T) = {ranks_T}  (ceiling V+8 = {V + 8}; saturation => "
          f"rank_Q(T) = {V + 8} EXACTLY)")
    print(f"[cert]     rank_p(T + v_t) = {ranks_Tv}  (= V+9 = {V + 9} => v_t NOT in "
          f"span_Q(T), independent second proof)")
    sat_T = all(r == V + 8 for r in ranks_T.values())
    jump = all(r == V + 9 for r in ranks_Tv.values())
    assert sat_T and jump
    print(f"[cert] *** J.v_t = 0 exactly AND v_t NOT in span(T)  =>  dim ker J >= "
          f"rank(T) + 1 = {V + 9}  =>  FLEX >= 1 ***   ({time.time()-T0:.1f}s)")
    json.dump(dict(V=V, n=N, E=len(E), gauge_dim=V + 8, dep_ok=True, jrows=len(rows),
                   rank_T_modp={str(k): v for k, v in ranks_T.items()},
                   rank_Tv_modp={str(k): v for k, v in ranks_Tv.items()},
                   subset=list(subset_used), rank_TS=r1, rank_TS_plus=r2,
                   vt_zero=False, moving_rays=nmoving, flex_ge_1=True),
              open(os.path.join(HERE, CACHE_NAME), "w"))


def stage_rankj():
    E = CORE_PAIRS
    rays = rays_K(2)
    rows, triv = build_rows_K(rays, E)
    rankT = V + 8
    target = N - rankT - 1
    primes = pick_primes()
    got = {}
    for p in primes:
        s3 = tonelli(3, p)
        Jd = densify_K_modp(rows, p, s3)
        got[p] = rank_modp_dense(Jd, p)
    print(f"[rankJ] n={N} rank(T)={rankT} => certificate ceiling rank_Q(J) <= {target}")
    print(f"[rankJ] rank_p(J) = {got}")
    if all(r == target for r in got.values()):
        print(f"[rankJ] *** SQUEEZE CLOSED: rank_Q(J) = {target}, nullity = {rankT + 1}, "
              f"FLEX EXACTLY 1 ***")
    else:
        lo = max(got.values())
        print(f"[rankJ] squeeze open: flex in [1, {N - lo - rankT}]")
    c = jload(CACHE_NAME) or {}
    c["rank_J_modp"] = {str(k): v for k, v in got.items()}
    c["rank_J_target"] = target
    c["flex_exact_1"] = bool(all(r == target for r in got.values()))
    json.dump(c, open(os.path.join(HERE, CACHE_NAME), "w"))


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "cert"
    dict(cert=stage_cert, rankj=stage_rankj)[which]()
