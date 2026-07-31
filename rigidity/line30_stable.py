#!/usr/bin/env python3
# line30_stable.py -- TASK 1: the LINE-STABLE orthogonality hypergraph of the alphabet
# {0, +-1, +-x, +-x*}, x = t*zeta, zeta = e^{i pi/6}, t a FORMAL positive variable.
#
#   stage pool    : rays (cross product == 0 IDENTICALLY in t), edges (Hermitian dot == 0
#                   IDENTICALLY in t), triads.  Exact Z[zeta][t].
#   stage spec    : EXACT specialization at t = 2, 3, 1/3, 1: ray merges + extra edges of
#                   the specialized pool vs the stable graph (t=1 is the expected-special
#                   control).
#   stage rebuild : independent from-scratch rebuild of the specialized pools (t=2 must
#                   reproduce sqrt3i_check's 145 rays / 390 pairs / 30 triads).
#   stage locus   : for every non-edge pair, the set {t>0 : <u,v>(t)=0} = common positive
#                   roots of two real quadratics over Q(sqrt3); exact gcd + exact root
#                   signs => the COMPLETE finite list of special moduli.  Same for merges.
#
# Exact arithmetic only (python ints / Fractions).  Caches: line30_stable*.cache.json.
# This box kills processes at ~45 s: run stages one at a time.
import json, os, sys, time
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from itertools import combinations, product
from fractions import Fraction

from line30_ring import (Z0, SYMS, hdot_pol, cross_pol, pzero, peval_scaled,
                         build_stable_pool, z_reim2, ksub, kmul, kneg, kzero,
                         kinv, ksign)

T0 = time.time()
HERE = os.path.dirname(os.path.abspath(__file__))


def cpath(tag):
    return os.path.join(HERE, f"line30_stable_{tag}.cache.json")


def save(tag, obj):
    json.dump(obj, open(cpath(tag), "w"))


def load(tag):
    p = cpath(tag)
    return json.load(open(p)) if os.path.exists(p) else None


def get_pool():
    c = load("pool")
    assert c, "run stage pool first"
    rays = [tuple(r) for r in c["rays"]]
    E = [tuple(e) for e in c["pairs"]]
    T = [tuple(t) for t in c["triads"]]
    return rays, E, T


# ---------------------------------------------------------------------------------------
def stage_pool():
    rays, E, T = build_stable_pool()
    print(f"[pool] LINE-STABLE pool: {len(rays)} rays / {len(E)} pairs / {len(T)} triads"
          f"   ({time.time()-T0:.1f}s)")
    save("pool", dict(rays=[list(r) for r in rays], pairs=[list(e) for e in E],
                      triads=[list(t) for t in T]))


# ---------------------------------------------------------------------------------------
def stage_spec():
    rays, E, T = get_pool()
    n = len(rays)
    Es = set(E)
    out = {}
    for (p, q) in [(2, 1), (3, 1), (1, 3), (1, 1)]:
        merges, extra, missing = [], [], []
        for i, j in combinations(range(n), 2):
            if all(peval_scaled(P, p, q) == Z0 for P in cross_pol(rays[i], rays[j])):
                merges.append((i, j))
            z = (peval_scaled(hdot_pol(rays[i], rays[j]), p, q) == Z0)
            st = (i, j) in Es
            if z and not st:
                extra.append((i, j))
            if st and not z:
                missing.append((i, j))
        tag = f"t={p}/{q}" if q != 1 else f"t={p}"
        print(f"[spec {tag}] ray merges: {len(merges)}   extra edges: {len(extra)}   "
              f"missing stable edges: {len(missing)}")
        assert not missing, f"a stable edge FAILED to specialize at {tag} -- impossible"
        if extra:
            print(f"           extras (first 8): {[(rays[i], rays[j]) for i, j in extra[:8]]}")
        if merges:
            print(f"           merges (first 8): {[(rays[i], rays[j]) for i, j in merges[:8]]}")
        out[tag] = dict(merges=len(merges), extra=len(extra),
                        extra_pairs=[[list(rays[i]), list(rays[j])] for i, j in extra],
                        merge_pairs=[[list(rays[i]), list(rays[j])] for i, j in merges])
    save("spec", out)
    print(f"[spec] done ({time.time()-T0:.1f}s)")


# ---------------------------------------------------------------------------------------
def rebuild_at(p, q):
    def same_at(u, v):
        return all(peval_scaled(P, p, q) == Z0 for P in cross_pol(u, v))
    R = []
    for vec in product(SYMS, repeat=3):
        if all(s == "0" for s in vec):
            continue
        if not any(same_at(vec, r) for r in R):
            R.append(vec)
    m = len(R)
    EE = [(i, j) for i in range(m) for j in range(i+1, m)
          if peval_scaled(hdot_pol(R[i], R[j]), p, q) == Z0]
    EEs = set(EE)
    TT = [t for t in combinations(range(m), 3)
          if all((a, b) in EEs for a, b in combinations(t, 2))]
    return len(R), len(EE), len(TT)


def stage_rebuild():
    out = load("rebuild") or {}
    todo = [("t=2", 2, 1), ("t=3", 3, 1), ("t=1/3", 1, 3), ("t=1", 1, 1)]
    for tag, p, q in todo:
        if tag in out:
            print(f"[rebuild {tag}] cached: {out[tag]}")
            continue
        r = rebuild_at(p, q)
        out[tag] = list(r)
        note = "  (sqrt3i_check reference: 145/390/30)" if tag == "t=2" else \
               "  (t=1: |x|=1, EXPECTED special)" if tag == "t=1" else ""
        print(f"[rebuild {tag}] from-scratch pool: {r[0]} rays / {r[1]} pairs / "
              f"{r[2]} triads{note}")
        save("rebuild", out)
        if time.time() - T0 > 34:
            print("[rebuild] budget -- re-run to continue")
            return
    print(f"[rebuild] all done ({time.time()-T0:.1f}s)")


# ---------------------------------------------------------------------------------------
# SPECIAL-LOCUS machinery: quadratics over K = Q(sqrt3), exact gcd, exact root signs.
# ---------------------------------------------------------------------------------------
def pol_to_K_quads(P):
    re, im = [], []
    for k in range(3):
        (ur, vr), (ui, vi) = z_reim2(P[k])
        re.append((Fraction(ur), Fraction(vr)))
        im.append((Fraction(ui), Fraction(vi)))
    return re, im


def kpoly_trim(c):
    c = list(c)
    while c and kzero(c[-1]):
        c.pop()
    return c


def kpoly_mod(A, B):
    A = kpoly_trim(A); B = kpoly_trim(B)
    binv = kinv(B[-1])
    while len(A) >= len(B) and A:
        f = kmul(A[-1], binv)
        off = len(A) - len(B)
        for i in range(len(B)):
            A[off + i] = ksub(A[off + i], kmul(f, B[i]))
        A = kpoly_trim(A)
    return A


def kpoly_gcd(A, B):
    A = kpoly_trim(A); B = kpoly_trim(B)
    while B:
        A, B = B, kpoly_mod(A, B)
    return A


def positive_roots_exact(G):
    G = kpoly_trim(G)
    if not G or len(G) == 1:
        return []
    if len(G) == 2:                       # linear: c1 t + c0
        r = kneg(kmul(G[0], kinv(G[1])))
        return [("K", r)] if ksign(r) > 0 else []
    c0, c1, c2 = G
    four = (Fraction(4), Fraction(0))
    disc = ksub(kmul(c1, c1), kmul(four, kmul(c2, c0)))
    sd = ksign(disc)
    if sd < 0:
        return []
    out = []
    A = kneg(c1)
    C = kmul((Fraction(2), Fraction(0)), c2)
    sC = ksign(C)
    for s in (+1, -1):
        if sd == 0 and s == -1:
            continue
        sA = ksign(A)
        A2mB = ksub(kmul(A, A), disc)
        sA2mB = ksign(A2mB)
        if s == +1:
            if sA > 0 or (sA == 0 and sd > 0):
                sroot = 1
            elif sA == 0 and sd == 0:
                sroot = 0
            else:                          # sA < 0: sign = sign(disc - A^2)
                sroot = -sA2mB
        else:
            if sA < 0 or (sA == 0 and sd > 0):
                sroot = -1
            elif sA == 0 and sd == 0:
                sroot = 0
            else:                          # sA > 0: sign = sign(A^2 - disc)
                sroot = sA2mB
        if sroot != 0 and sroot == sC:
            out.append(("quad", c2, c1, c0, s))
    return out


def root_str(desc):
    from math import sqrt
    if desc[0] == "K":
        p, q = desc[1]
        val = float(p) + float(q) * sqrt(3.0)
        return f"({p}{'+' if q >= 0 else ''}{q}*sqrt3) ~ {val:.6f}"
    _, c2, c1, c0, s = desc
    f = lambda k: float(k[0]) + float(k[1]) * sqrt(3.0)
    a, b, c = f(c2), f(c1), f(c0)
    d = b*b - 4*a*c
    val = (-b + s * sqrt(max(d, 0.0))) / (2*a)
    return (f"[{'+' if s>0 else '-'}]root of ({c2[0]}+{c2[1]}s3)t^2+({c1[0]}+{c1[1]}s3)t"
            f"+({c0[0]}+{c0[1]}s3) ~ {val:.6f}")


def vanish_locus(polys):
    G = None
    for P in polys:
        re, im = pol_to_K_quads(P)
        for Q in (re, im):
            Q = kpoly_trim(Q)
            if not Q:
                continue
            G = Q if G is None else kpoly_gcd(G, Q)
            if len(kpoly_trim(G)) == 1:
                return []
    assert G is not None
    return positive_roots_exact(G)


def stage_locus():
    rays, E, T = get_pool()
    n = len(rays)
    Es = set(E)
    st = load("locus_state") or dict(next_i=0, edge_special={}, merge_special={})
    edge_special = {k: v for k, v in st["edge_special"].items()}
    merge_special = {k: v for k, v in st["merge_special"].items()}
    i0 = st["next_i"]
    for i in range(i0, n):
        for j in range(i + 1, n):
            if (i, j) not in Es:
                P = hdot_pol(rays[i], rays[j])
                assert not pzero(P)
                for desc in vanish_locus([P]):
                    edge_special.setdefault(root_str(desc), []).append([i, j])
            cps = cross_pol(rays[i], rays[j])
            assert not all(pzero(P) for P in cps)
            nz = [P for P in cps if not pzero(P)]
            for desc in vanish_locus(nz):
                merge_special.setdefault(root_str(desc), []).append([i, j])
        if time.time() - T0 > 32:
            save("locus_state", dict(next_i=i + 1, edge_special=edge_special,
                                     merge_special=merge_special))
            print(f"[locus] checkpoint at row {i+1}/{n} -- re-run to continue "
                  f"({time.time()-T0:.1f}s)")
            return
    save("locus_state", dict(next_i=n, edge_special=edge_special,
                             merge_special=merge_special))
    print(f"[locus] sweep COMPLETE ({time.time()-T0:.1f}s)")
    print(f"[locus] EXTRA-EDGE special values of t>0 ({len(edge_special)} distinct):")
    for k in sorted(edge_special):
        print(f"        t = {k}   ({len(edge_special[k])} pairs)")
    print(f"[locus] RAY-MERGE special values of t>0 ({len(merge_special)} distinct):")
    for k in sorted(merge_special):
        print(f"        t = {k}   ({len(merge_special[k])} pairs)")
    save("locus", dict(edge_special_values={k: len(v) for k, v in edge_special.items()},
                       merge_special_values={k: len(v) for k, v in merge_special.items()}))


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "pool"
    dict(pool=stage_pool, spec=stage_spec, rebuild=stage_rebuild,
         locus=stage_locus)[which]()
