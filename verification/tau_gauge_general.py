# V45 - THE CONSTRUCTIVE q-PARITY GAUGE, AND THE TAU-GENERAL FORM OF V44.
# Session 2026-07-10 cross-check findings, consolidated. Three certified facts:
#
# (1) q-PARITY IDENTITY (constructive gauge lemma). For every closed commuting triple
#     C at any d (convention weyl.py: W(v) = tau^{-q(v)} X^a Z^b, q(v) = sum_i a_i b_i),
#     the tau-level context phase satisfies  st(C) == sum_{v in C} q(v)  (mod 2), i.e.
#     s_tau = A.qbar (mod 2) with qbar = q mod 2. PROOF (general, from Paper B Thm 1's
#     telescope): c(u,v) = -q(u) - q(v) + 2 beta(u,v) + q(u+v) gives, telescoping any
#     closed word, st(C) = -sum q(v) + 2 B(C), B integer. Hence the GF(2) gauge system
#     A a = s_tau of V44 is ALWAYS solvable with the EXPLICIT solution a = qbar --
#     no double annihilation needed, and the integral gauge of note v3 Thm 3 is
#     CANONICAL: s := (st - A.qbar)/2 mod d. This is the tau-general statement of the
#     detection equivalence (the roadmap item): gauge-free input st, canonical s,
#     kernel pairings u.s gauge-invariant.
#
# (2) DETECTION EQUIVALENCE AT d=6 (beyond the certified d=2,4): spectrum confinement
#     u.s in {0,3} for every Z_6 kernel vector, double-annihilator equivalence with
#     brute-force solvability, and d/2-witness existence, on random closed-triple
#     families at m=2.
#
# (3) mu_{4d} VALUATION (note v3 Thm 1 / V23): instance evidence EXTENDED to
#     d = 6, 8, 12 (every abelian group <u,v> tested has a mu_{4d} trivializer;
#     complete search over the two free generator values). PINNED NEGATIVE: the
#     closed-form candidate lambda0 = -2q + quadratic refinement FAILS (its residual
#     is not Z_2-bilinear at d=3 and d=4), so the general mu_{4d} valuation has no
#     one-line proof; it remains instance-verified (d=2,3,4 in V23; d=6,8,12 here).
#     U(1)-triviality itself is general (Ext(G,U(1)) = 0).
#
# Expected: "V45 PASS" after ~1-2 min.
import numpy as np, itertools
from weyl import build

def qint(v, m): return sum(int(v[2*i]) * int(v[2*i+1]) for i in range(m))

def make_sym(m):
    def sym_d(a, b, d): return int(sum(a[2*i]*b[2*i+1] - a[2*i+1]*b[2*i] for i in range(m))) % d
    return sym_d

# ---------- (1) q-parity identity ----------
def part1():
    tot = 0
    for d, m in [(2, 2), (4, 2), (6, 1), (6, 2), (8, 1), (10, 1), (12, 1)]:
        X, Z, w, tau, W, _ = build(d, m)
        rng = np.random.default_rng(7)
        allv = [np.array(v) for v in itertools.product(range(d), repeat=2*m) if any(v)]
        sym = make_sym(m)
        n = 0
        for _ in range(4000):
            u = allv[int(rng.integers(0, len(allv)))]; v = allv[int(rng.integers(0, len(allv)))]
            w3 = (-(u + v)) % d
            if not w3.any(): continue
            if sym(u, v, d) or sym(u, w3, d) or sym(v, w3, d): continue
            M = W(u) @ W(v) @ W(w3)
            tr = np.trace(M); ph = np.angle(tr/abs(tr)); sc = ph/(np.pi/d)
            assert abs(sc - round(sc)) < 1e-6
            st = int(round(sc)) % (2*d)
            assert st % 2 == (qint(u, m) + qint(v, m) + qint(w3, m)) % 2, \
                f"q-parity FAIL d={d} u={u} v={v}"
            n += 1
            if n >= 150: break
        tot += n
    print(f"(1) q-parity identity st == A.qbar (mod 2): {tot} closed triples, d=2..12: PASS")

# ---------- (2) equivalence at d=6 ----------
def part2():
    d, m = 6, 2
    X, Z, w, tau, W, _ = build(d, m)
    rng = np.random.default_rng(20260710)
    allv = [np.array(v) for v in itertools.product(range(d), repeat=2*m) if any(v)]
    sym = make_sym(m)
    def st_of(trip):
        M = W(trip[0]) @ W(trip[1]) @ W(trip[2])
        tr = np.trace(M); ph = np.angle(tr/abs(tr)); sc = ph/(np.pi/d)
        assert abs(sc - round(sc)) < 1e-6
        return int(round(sc)) % (2*d)
    def kernel_left(A):
        U = np.array(list(itertools.product(range(d), repeat=A.shape[0])), dtype=np.int64)
        return U[((U @ A) % d == 0).all(axis=1)]
    def solvable_bf(A, s):
        G = np.array(list(itertools.product(range(d), repeat=A.shape[1])), dtype=np.int64)
        return bool((((G @ A.T) % d == s % d).all(axis=1)).any())
    checked = uns = 0
    for trial in range(1500):
        t0 = None
        for _ in range(200):
            u = allv[int(rng.integers(0, len(allv)))]; v = allv[int(rng.integers(0, len(allv)))]
            if (u == v).all() or sym(u, v, d): continue
            w3 = (-(u + v)) % d
            if not w3.any() or sym(u, w3, d) or sym(v, w3, d): continue
            t0 = [tuple(u), tuple(v), tuple(w3)]; break
        if t0 is None or len(set(t0)) < 3: continue
        labset = set(t0)
        for _ in range(30):
            if len(labset) >= 7: break
            base = list(labset)
            u = np.array(base[int(rng.integers(0, len(base)))])
            v = allv[int(rng.integers(0, len(allv)))] if rng.random() < 0.5 else \
                np.array(base[int(rng.integers(0, len(base)))])
            if (u == np.array(v)).all() or sym(u, np.array(v), d): continue
            w3 = (-(u + np.array(v))) % d
            if not w3.any() or sym(u, w3, d) or sym(np.array(v), w3, d): continue
            t = [tuple(u), tuple(v), tuple(w3)]
            if len(set(t)) < 3: continue
            if len(labset | set(t)) > 7: continue
            labset |= set(t)
        labels = [np.array(v) for v in sorted(labset)]
        idx = {tuple(v): i for i, v in enumerate(labels)}
        ctxs = []
        for i, j in itertools.combinations(range(len(labels)), 2):
            v3 = tuple((-(labels[i] + labels[j])) % d)
            if v3 in idx:
                k = idx[v3]
                if k in (i, j): continue
                trip = [labels[i], labels[j], labels[k]]
                if any(sym(a, b, d) for a, b in itertools.combinations(trip, 2)): continue
                key = tuple(sorted((i, j, k)))
                if key not in [tuple(sorted(x)) for x in ctxs]: ctxs.append((i, j, k))
        if len(ctxs) < 3 or len(ctxs) > 8 or len(labels) > 6: continue
        A = np.zeros((len(ctxs), len(labels)), int); st = []
        for r, (i, j, k) in enumerate(ctxs):
            A[r, i] += 1; A[r, j] += 1; A[r, k] += 1
            st.append(st_of([labels[i], labels[j], labels[k]]))
        st = np.array(st)
        qbar = np.array([qint(v, m) % 2 for v in labels])
        assert ((st - A @ qbar) % 2 == 0).all(), "q-parity gauge failed"
        s = ((st - A @ qbar) // 2) % d
        vals = set(int((u @ s) % d) for u in kernel_left(A))
        solv = solvable_bf(A, s)
        assert vals.issubset({0, 3}), f"SPECTRUM VIOLATION {sorted(vals)}"
        assert solv == (vals <= {0}), f"DA MISMATCH solv={solv} vals={sorted(vals)}"
        if not solv:
            uns += 1; assert 3 in vals
        checked += 1
    assert checked >= 15 and uns >= 1, f"insufficient coverage ({checked}/{uns})"
    print(f"(2) d=6 m=2 equivalence: {checked} families ({uns} unsolvable, witnessed): PASS")

# ---------- (3) mu_{4d} sufficiency at d=6,8,12 ----------
def part3():
    for d, m in [(6, 1), (8, 1), (12, 1)]:
        X, Z, w, tau, W, N = build(d, m)
        rng = np.random.default_rng(5)
        allv = [np.array(v) for v in itertools.product(range(d), repeat=2*m) if any(v)]
        sym = make_sym(m)
        Winv = {}
        def c_of(a, b):
            M = W(a) @ W(b); sk = tuple((a + b) % d)
            if sk not in Winv: Winv[sk] = np.linalg.inv(W(np.array(sk)))
            R = M @ Winv[sk]
            ph = np.angle(R[0, 0]); sc = ph / (np.pi / d)
            assert abs(sc - round(sc)) < 1e-6
            return int(round(sc)) % (2 * d)
        done = 0
        while done < 8:
            u = allv[int(rng.integers(0, len(allv)))]; v = allv[int(rng.integers(0, len(allv)))]
            if sym(u, v, d): continue
            G = {}
            for a, b in itertools.product(range(d), range(d)):
                g = tuple((a*u + b*v) % d)
                if any(g): G.setdefault(g, (a, b))
            elems = list(G)
            if len(elems) < 4 or len(elems) > 40: continue
            if any(sym(np.array(g), np.array(h), d) for g, h in itertools.combinations(elems, 2)):
                continue
            C = {(g, h): c_of(np.array(g), np.array(h))
                 for g, h in itertools.product(elems, elems)}
            n4 = 4 * d
            gu, gv = tuple(u % d), tuple(v % d)
            found = None
            for k1, k2 in itertools.product(range(n4), range(n4)):
                # coboundary recursion k(prev+gen) = k(prev)+k(gen)-2c(prev,gen);
                # any true solution obeys it along the walk => search is complete.
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
                good = True
                for g, h in itertools.product(elems, elems):
                    sk = tuple((np.array(g) + np.array(h)) % d)
                    ks = 0 if not any(sk) else k[sk]
                    if (k[g] + k[h] - ks - 2*C[(g, h)]) % n4 != 0:
                        good = False; break
                if good: found = (k1, k2); break
            assert found is not None, f"NO mu_4d TRIVIALIZER at d={d}: u={u} v={v}"
            done += 1
        print(f"(3) d={d}: 8/8 abelian groups <u,v> admit mu_{{4d}} trivializers: PASS")

if __name__ == "__main__":
    part1(); part2(); part3()
    print("V45 PASS")
