# V46 - SHARP VALUATION OF THE LOCAL TRIVIALIZER + THE EVEN/ODD INVERSION.
# Session 2026-07-10. Corrects and strengthens the local-classicality valuation story.
#
# THEOREM (parity lemma, even d). For commuting u,v (symplectic form <u,v> = 0 mod d),
#   parity(c(u,v)) = <u,v> mod 2 = 0  (d even),
# where W(u)W(v) = tau^{c(u,v)} W(u+v mod d). Hence every commuting closed context has
# EVEN tau-level phase: products of commuting contexts are ALWAYS omega-powers at even
# d - integrality is automatic, not a gauge choice. ("PM is tau-odd" was an artifact of
# a label-layout mismatch in V44's pinned families, fixed and pinned there.)
#
# THEOREM (mu_d valuation, even d). Every commuting Weyl label group G at even d admits
# a trivializer lambda with values in mu_d (the omega-grid: no tau steps at all).
# Proof chain: (i) parity lemma => c = 2c' with c' a SYMMETRIC Z_d-valued 2-cocycle
# (symmetry from <u,v> = 0 mod d); (ii) symmetric classes = Ext(G, Z_d), additive over
# a cyclic decomposition and detected by cyclic transgressions E' = sum_j c'(jg, g);
# (iii) E' = a(a-2)q(g)/2 - dK with a = ord(g): divisible by a in both parity cases
# (a even: a/2*(a-2) integer multiple of a/... a odd: q even forced), so every cyclic
# class vanishes and the total class is 0 over Z_d.
#
# THEOREM (odd d). The same transgression argument at level 2d gives lambda in mu_{2d};
# this is SHARP: the d=3 group <(1,1)> requires mu_{2d} (complete search below) - i.e.
# genuine tau-steps are an ODD-d phenomenon, inverting the previous story. (The former
# claim "half tau-steps in general necessary" at even d is RETRACTED; mu_{4d} remains a
# correct but non-sharp upper bound. State-sector tau-necessity - V33/V34,
# d4_moment_facets - is a DIFFERENT theorem and is unaffected.)
#
# Expected: "V46 PASS" in under a minute.
import numpy as np, itertools, json
from weyl import build

def make_c(d, m, W):
    Winv = {}
    def c_of(a, b):
        M = W(a) @ W(b); sk = tuple((a + b) % d)
        if sk not in Winv: Winv[sk] = np.linalg.inv(W(np.array(sk)))
        R = M @ Winv[sk]
        ph = np.angle(R[0, 0]); sc = ph / (np.pi / d)
        assert abs(sc - round(sc)) < 1e-6
        return int(round(sc)) % (2 * d)
    return c_of

def symform(a, b, d, m):
    return int(sum(a[2*i]*b[2*i+1] - a[2*i+1]*b[2*i] for i in range(m))) % d

def part_parity():
    for d, m in [(2, 2), (4, 2), (6, 1), (8, 1), (10, 1), (12, 1)]:
        X, Z, w, tau, W, N = build(d, m)
        c_of = make_c(d, m, W)
        rng = np.random.default_rng(3)
        allv = [np.array(v) for v in itertools.product(range(d), repeat=2*m) if any(v)]
        n = 0
        while n < 120:
            u = allv[int(rng.integers(0, len(allv)))]; v = allv[int(rng.integers(0, len(allv)))]
            if symform(u, v, d, m): continue
            assert c_of(u, v) % 2 == 0, f"PARITY FAIL d={d} u={u} v={v}"
            n += 1
        print(f"  even d={d}: 120 commuting pairs, c(u,v) even: PASS")
    # odd-d contrast: commuting pairs with ODD c exist
    d, m = 3, 1
    X, Z, w, tau, W, N = build(d, m)
    c_of = make_c(d, m, W)
    found = None
    for u, v in itertools.product([np.array(x) for x in itertools.product(range(3), repeat=2) if any(x)], repeat=2):
        if symform(u, v, 3, 1): continue
        if c_of(u, v) % 2 == 1: found = (tuple(u), tuple(v), c_of(u, v)); break
    assert found, "no odd-c commuting pair at d=3?"
    print(f"  odd d=3 contrast: commuting pair {found[0]},{found[1]} has ODD c={found[2]}: PASS")

def walk_search(d, m, gens, W, grid):
    u, v = np.array(gens[0]), np.array(gens[1])
    G = {}
    for a, b in itertools.product(range(d), range(d)):
        g = tuple((a*u + b*v) % d)
        if any(g): G.setdefault(g, (a, b))
    elems = list(G)
    c_of = make_c(d, m, W)
    C = {(g, h): c_of(np.array(g), np.array(h)) for g, h in itertools.product(elems, elems)}
    n4 = 4 * d
    gu, gv = tuple(u % d), tuple(v % d)
    for k1, k2 in itertools.product(grid, grid):
        k = {}
        for a in range(d):
            for b in range(d):
                if a == 0 and b == 0: continue
                g = tuple((a*u + b*v) % d)
                if not any(g) or g in k: continue
                if b > 0:
                    prev = tuple((a*u + (b-1)*v) % d); kgen, gen = k2, gv
                else:
                    prev = tuple(((a-1)*u) % d); kgen, gen = k1, gu
                base = 0 if not any(prev) else k[prev]
                cc = C[(prev, gen)] if any(prev) else 0
                k[g] = (base + kgen - 2*cc) % n4
        if not all(kk in grid for kk in k.values()): continue
        good = all((k[g] + k[h] - (0 if not any(tuple((np.array(g)+np.array(h))%d)) else
                    k[tuple((np.array(g)+np.array(h))%d)]) - 2*C[(g, h)]) % n4 == 0
                   for g, h in itertools.product(elems, elems))
        if good: return True, len(elems)
    return False, len(elems)

def part_mud_even():
    # d=2: all six PM contexts (per-site layout); d=4: cert4 generator pairs; d=6,8,12 random
    X, Z, w, tau, W, N = build(2, 2)
    PM = [[(1,0,0,0),(0,0,1,0)], [(0,0,0,1),(0,1,0,0)], [(1,0,0,1),(0,1,1,0)],
          [(1,0,0,0),(0,0,0,1)], [(0,0,1,0),(0,1,0,0)], [(1,0,1,0),(0,1,0,1)]]
    for gens in PM:
        ok, sz = walk_search(2, 2, gens, W, set(range(0, 8, 4)))
        assert ok, f"mu_d FAIL at PM {gens}"
    print("  d=2: all 6 PM contexts trivialize over mu_d: PASS")
    fam4 = [[tuple(v) for v in it["ctx"]] for it in json.load(open("cert4_min.json"))["items"]]
    X4, Z4, w4, tau4, W4, N4 = build(4, 2)
    for Cx in fam4:
        ok, sz = walk_search(4, 2, [Cx[0], Cx[1]], W4, set(range(0, 16, 4)))
        assert ok, f"mu_d FAIL at cert4 {Cx}"
    print("  d=4: all 6 cert4 contexts trivialize over mu_d: PASS")
    for d in (6, 8, 12):
        Xd, Zd, wd, taud, Wd, Nd = build(d, 1)
        rng = np.random.default_rng(11)
        allv = [np.array(x) for x in itertools.product(range(d), repeat=2) if any(x)]
        done = 0
        while done < 5:
            u = allv[int(rng.integers(0, len(allv)))]; v = allv[int(rng.integers(0, len(allv)))]
            if symform(u, v, d, 1): continue
            ok, sz = walk_search(d, 1, [tuple(u), tuple(v)], Wd, set(range(0, 4*d, 4)))
            if sz < 3: continue
            assert ok, f"mu_d FAIL d={d} <{u},{v}>"
            done += 1
        print(f"  d={d}: 5/5 abelian groups trivialize over mu_d: PASS")

def part_odd_sharp():
    X3, Z3, w3, tau3, W3, N3 = build(3, 1)
    # mu_{2d} always works at d=3 (sample); mu_d fails for <(1,1)> - sharp
    for gens in [((1,0),(2,0)), ((0,1),(0,2)), ((1,1),(2,2)), ((1,2),(2,1))]:
        ok, sz = walk_search(3, 1, list(gens), W3, set(range(0, 12, 2)))
        assert ok, f"mu_2d FAIL at d=3 {gens}"
    okd, _ = walk_search(3, 1, [(1,1),(2,2)], W3, set(range(0, 12, 4)))
    assert not okd, "expected mu_d to FAIL for d=3 <(1,1)>"
    print("  d=3: mu_2d suffices (4/4 groups); mu_d provably fails for <(1,1)> (complete search): SHARP")

def part_layout_regression():
    # The former BLOCK-layout PM labels are non-commuting operator triples under
    # weyl.build's per-site W: pinned as the source of the "tau-odd PM" artifact.
    X, Z, w, tau, W, N = build(2, 2)
    bad = [(1,0,0,0), (0,1,0,0)]   # block-layout "X1,X2" = per-site X(x)I, Z(x)I
    A, B = W(np.array(bad[0])), W(np.array(bad[1]))
    assert not np.allclose(A@B, B@A, atol=1e-10), "regression: expected non-commuting"
    print("  layout regression: block-layout 'X1,X2' are non-commuting under per-site W (artifact pinned)")

if __name__ == "__main__":
    print("(1) parity lemma: even d => c even on commuting pairs (integrality automatic)")
    part_parity()
    print("(2) mu_d valuation at even d (complete searches)")
    part_mud_even()
    print("(3) odd d: mu_2d sufficient and sharp")
    part_odd_sharp()
    print("(4) V44 layout artifact regression")
    part_layout_regression()
    print("V46 PASS")
