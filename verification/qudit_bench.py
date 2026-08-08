# V58 - QUDIT BENCHMARKING SUITE, STAGE 1: witness inventory + exact robustness
# margins + compatibility-assumption ledger + honest scope. (M3 of QC_BRIDGE.md.)
#
# CLAIM. (i) INVENTORY (stage 1) + stage cert4bound (stage 2): the program possesses
#   exactly TWO state-independent operator witnesses with certified deterministic
#   bounds. (1) The Peres-Mermin witness on two qubits (Hilbert dim 4): quantum
#   value 6 (every state, incl. maximally mixed), noncontextual bound 4 (brute
#   force over all 512 sign assignments). (2) NEW, stage cert4bound: the paperB
#   cos-functional on the six cert4 contexts (Weyl d=4, m=2, Hilbert dim 16):
#   quantum value 6 (every state - each context correlator is a scalar identity),
#   deterministic bound 4 by exhaustion over ALL 4^9 = 262,144 assignments (and
#   the same bound on the 32,768 spectra-constrained ones) - the program's first
#   operator-witness bound above d=2. The Weyl certificates cert8/cert16 (values
#   d/2 = 4/8 at d=8/16, m=2) remain exact COCYCLE obstructions with no derived
#   bound (8^100-scale exhaustion is infeasible). The d=4 census's 61 facet classes (23,256
#   facets; tier-2 sub-orbit: 21,504 exact {0,+-1}-lifts with integer bounds {6,7})
#   are STATE-DEPENDENT witnesses (CF(rho)>0 <=> facet violated, V43). At single-qudit
#   d=6,8,10,12 the program has KS-colorability obstructions only (V57); NO operator
#   witness with a deterministic bound exists there today.
# (ii) MARGINS, all exact rational arithmetic: PM has v_mixed = 6 = v_Q, so the
#   state-depolarizing threshold eps* = (v_Q-bound)/(v_Q-v_mixed) is UNDEFINED (0/0):
#   state noise never degrades a state-independent witness. Under the uniform context-
#   correlator visibility model S(eta)=6*eta the exact threshold is eta*=2/3, i.e.
#   tolerated effective noise eps* = 1/3. The ibm_fez calibration point S = 4.6125 =
#   369/80 gives eps_eff = 1 - S/6 = 37/160 exactly, margin to threshold 49/480.
#   For the census facets the maximally mixed state's moment vector is computed
#   EXACTLY (monomial Weyl traces over Z[zeta_8]); every one of the 23,256 facets is
#   satisfied at it (=> CF(mixed)=0, certified), and the per-class mixed-state margin
#   M0 is the exact denominator datum in eps*(rho) = V(rho)/(V(rho)+M0).
# PROOF SKETCH. All operator algebra is done in an exact monomial representation:
#   a Weyl word is (permutation, exponent vector) over tau = e^{i*pi/d}, multiplied
#   by integer arithmetic on exponents mod 2d; scalars, traces and eigenvalue sets
#   are decided by exact reduction modulo Phi_{2d}(x) = x^d + 1 (2d a power of two
#   for every dimension used here: d = 2,4,8,16). No float enters any certificate
#   path; numpy is used only for exact int64 linear algebra on small integers.
#   Deterministic bounds by exhaustive enumeration (512 PM sign assignments; 64
#   census vertices rebuilt from scratch). Caches are read ONLY
#   (cert*_min.json, fiber16_ctxs.json, cert16_lambda.npy, d4_facet_classes.npz,
#   d4_tier2_orbit_data.npz); no file is written or modified.
# STAGES (CLI):  python3 qudit_bench.py [inventory|margins|cert4bound|ledger|report|all]
#   (default: all). EXPECTED FINAL OUTPUT LINE:  V58 PASS
import sys, os, json, itertools
from fractions import Fraction
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)

# ----------------------------------------------------------------------------------
# Exact monomial algebra over tau = e^{i pi / d}, exponents mod 2d.
# A monomial matrix M is (n, mod, perm, exps): M|j> = tau^{exps[j]} |perm[j]>.
# Convention pinned by weyl.py / verify_cert8.py / verify_cert16.py:
#   X|j> = |j+1 mod d>,  Z|j> = w^j |j>  (w = tau^2),
#   W(v) = tau^{-sum a_i b_i} kron_i X^{a_i} Z^{b_i}.
# ----------------------------------------------------------------------------------

def m_id(n, mod):
    return (n, mod, tuple(range(n)), (0,) * n)

def m_mul(A, B):
    nA, modA, pA, eA = A; nB, modB, pB, eB = B
    assert nA == nB and modA == modB
    perm = tuple(pA[pB[j]] for j in range(nA))
    exps = tuple((eB[j] + eA[pB[j]]) % modA for j in range(nA))
    return (nA, modA, perm, exps)

def m_kron(A, B):
    nA, mod, pA, eA = A; nB, modB, pB, eB = B
    assert mod == modB
    n = nA * nB
    perm = [0] * n; exps = [0] * n
    for jA in range(nA):
        for jB in range(nB):
            J = jA * nB + jB
            perm[J] = pA[jA] * nB + pB[jB]
            exps[J] = (eA[jA] + eB[jB]) % mod
    return (n, mod, tuple(perm), tuple(exps))

def m_phase(A, k):
    n, mod, p, e = A
    return (n, mod, p, tuple((x + k) % mod for x in e))

def m_pow(A, k):
    R = m_id(A[0], A[1])
    for _ in range(k):
        R = m_mul(R, A)
    return R

def m_inv(A):
    n, mod, p, e = A
    perm = [0] * n; exps = [0] * n
    for j in range(n):
        perm[p[j]] = j
        exps[p[j]] = (-e[j]) % mod
    return (n, mod, tuple(perm), tuple(exps))

def m_scalar(A):
    """Return exponent s with A = tau^s * I, else None."""
    n, mod, p, e = A
    if p != tuple(range(n)): return None
    if any(x != e[0] for x in e): return None
    return e[0]

def m_trace_coeffs(A, d):
    """Exact trace as integer coefficient vector over the Z-basis 1,tau,...,tau^{d-1}
    of Z[tau], using tau^d = -1 (valid: 2d is a power of two for every d used)."""
    n, mod, p, e = A
    assert mod == 2 * d and (2 * d) & (2 * d - 1) == 0, "Phi_{2d}=x^d+1 needs 2d a power of 2"
    c = [0] * d
    for j in range(n):
        if p[j] == j:
            k = e[j] % (2 * d)
            if k < d: c[k] += 1
            else:     c[k - d] -= 1
    return c

def weyl_builder(d, m):
    mod = 2 * d
    X = (d, mod, tuple((j + 1) % d for j in range(d)), (0,) * d)
    Z = (d, mod, tuple(range(d)), tuple((2 * j) % mod for j in range(d)))
    def T1(a, b):
        return m_mul(m_pow(X, a % d), m_pow(Z, b % d))
    def W(v):
        v = [int(x) % d for x in v]
        M = T1(v[0], v[1])
        for i in range(1, m):
            M = m_kron(M, T1(v[2 * i], v[2 * i + 1]))
        q = sum(v[2 * i] * v[2 * i + 1] for i in range(m))
        return m_phase(M, (-q) % mod)
    return W

def eig_omega_set(A, d):
    """Exact eigenphase set {j : omega^j in spec(A)} for a monomial A with A^d = I.
    Per permutation cycle of length L and phase exponent E, the eigenvalues are the
    j in Z_d with 2 j L == E (mod 2d); asserts exactly L solutions per cycle."""
    n, mod, p, e = A
    assert mod == 2 * d
    assert m_scalar(m_pow(A, d)) == 0, "operator order does not divide d"
    seen = [False] * n
    js = set()
    for j0 in range(n):
        if seen[j0]: continue
        L = 0; E = 0; j = j0
        while not seen[j]:
            seen[j] = True
            E = (E + e[j]) % mod
            j = p[j]; L += 1
        sols = [j for j in range(d) if (2 * j * L - E) % mod == 0]
        assert len(sols) == L, "eigenvalue count mismatch on a cycle"
        js.update(sols)
    return sorted(js)

# ----------------------------------------------------------------------------------
# PM block (two qubits, Hilbert dim 4): full exact re-derivation.
# ----------------------------------------------------------------------------------

_PM = None
def pm_certify():
    global _PM
    if _PM is not None: return _PM
    d = 2
    W = weyl_builder(d, 2)
    obs = dict(XI=(1,0,0,0), IX=(0,0,1,0), XX=(1,0,1,0),
               IZ=(0,0,0,1), ZI=(0,1,0,0), ZZ=(0,1,0,1),
               XZ=(1,0,0,1), ZX=(0,1,1,0), YY=(1,1,1,1))
    names = list(obs)
    fam = [['XI','IX','XX'], ['IZ','ZI','ZZ'], ['XZ','ZX','YY'],
           ['XI','IZ','XZ'], ['IX','ZI','ZX'], ['XX','ZZ','YY']]
    M = {nm: W(obs[nm]) for nm in names}
    # each observable squares to I (exactly)
    for nm in names:
        assert m_scalar(m_mul(M[nm], M[nm])) == 0, nm
    # commute/anticommute structure: each operator commutes with exactly its 4
    # row/column partners and anticommutes with the remaining 4
    partners = {nm: set() for nm in names}
    for C in fam:
        for a in C:
            partners[a].update(x for x in C if x != a)
    for i, a in enumerate(names):
        for b in names[i+1:]:
            AB, BA = m_mul(M[a], M[b]), m_mul(M[b], M[a])
            if AB == BA:
                assert b in partners[a], (a, b, "commute but not partners")
            else:
                assert AB == m_phase(BA, 2), (a, b, "neither commute nor anticommute")
                assert b not in partners[a], (a, b, "partners but anticommute")
    # context products: rows +I,+I,+I, columns +I,+I,-I  (tau=i, tau^2=-1)
    signs = []
    for C in fam:
        P = m_id(4, 4)
        for nm in C: P = m_mul(P, M[nm])
        s = m_scalar(P)
        assert s in (0, 2), (C, s)
        signs.append(+1 if s == 0 else -1)
    assert signs == [1, 1, 1, 1, 1, -1], signs
    # six-sign product = -1 (the d=2 obstruction, value d/2 = 1)
    prod = 1
    for s in signs: prod *= s
    assert prod == -1
    # quantum value: the SIGNED sum of context-product operators is 6*I exactly,
    # so S(rho) = 6 for EVERY state; and |<+-I>| = 1 caps each term, so 6 = max.
    v_Q = Fraction(6)
    # v_mixed from exact traces: Tr(context product)/4 = sign
    v_mixed = Fraction(0)
    wit_sign = [1, 1, 1, 1, 1, -1]        # witness S = R1+R2+R3+C1+C2-C3
    for k, C in enumerate(fam):
        P = m_id(4, 4)
        for nm in C: P = m_mul(P, M[nm])
        c = m_trace_coeffs(P, 2)          # basis 1, tau=i ; Tr = c[0] + c[1] i
        assert c[1] == 0
        v_mixed += Fraction(wit_sign[k] * c[0], 4)
    assert v_mixed == 6
    # deterministic bound: brute force over ALL 512 sign assignments
    best = -10; attain = 0; histo = Counter()
    for bits in itertools.product((1, -1), repeat=9):
        g = dict(zip(names, bits))
        vals = []
        for C in fam:
            r = 1
            for nm in C: r *= g[nm]
            vals.append(r)
        Sdet = vals[0] + vals[1] + vals[2] + vals[3] + vals[4] - vals[5]
        histo[Sdet] += 1
        if Sdet > best: best, attain = Sdet, 1
        elif Sdet == best: attain += 1
    assert best == 4 and histo[6] == 0 and histo[5] == 0
    _PM = dict(v_Q=v_Q, bound=Fraction(4), v_mixed=v_mixed, n_attain=attain,
               signs=signs, histo=dict(sorted(histo.items())))
    return _PM

# ----------------------------------------------------------------------------------
# Weyl cocycle certificates cert4 / cert8 / cert16: exact re-verification.
# ----------------------------------------------------------------------------------

def cert_verify(d, items, expect_value):
    """items: list of (ctx list-of-labels, lam). Verifies pairwise commutation,
    scalar context products in mu_d (omega powers), the multiplicity condition
    lambda^T A == 0 mod d, and the certificate value  sum lam*s == expect (mod d)."""
    W = weyl_builder(d, 2)
    mod = 2 * d
    mult = Counter(); total = 0
    svals = []
    for ctx, lam in items:
        Ws = [W(v) for v in ctx]
        for i in range(len(Ws)):
            for j in range(i + 1, len(Ws)):
                assert m_mul(Ws[i], Ws[j]) == m_mul(Ws[j], Ws[i]), ("noncommuting", ctx)
        P = m_id(d * d, mod)
        for A in Ws: P = m_mul(P, A)
        e = m_scalar(P)
        assert e is not None and e % 2 == 0, ("context product not an omega scalar", ctx)
        s = (e // 2) % d
        svals.append(s)
        for v in ctx: mult[tuple(v)] += lam
        total = (total + lam * s) % d
    assert all(c % d == 0 for c in mult.values()), "multiplicity condition fails"
    assert total == expect_value % d, (total, expect_value)
    return svals

_CERTS = None
def certs_certify():
    global _CERTS
    if _CERTS is not None: return _CERTS
    out = {}
    c4 = json.load(open("cert4_min.json"))
    it4 = [([tuple(v) for v in it["ctx"]], int(it["lam"])) for it in c4["items"]]
    s4 = cert_verify(4, it4, int(c4["value"]))
    out["cert4"] = dict(d=4, contexts=len(it4),
                        nobs=len({v for C, _ in it4 for v in C}),
                        value=int(c4["value"]), svals=s4)
    c8 = json.load(open("cert8_min.json"))
    it8 = [([tuple(v) for v in it["ctx"]], int(it["lam"])) for it in c8["items"]]
    s8 = cert_verify(8, it8, int(c8["value"]))
    for s, it in zip(s8, c8["items"]):        # cross-check stored s labels
        assert s == int(it["s"]) % 8
    out["cert8"] = dict(d=8, contexts=len(it8),
                        nobs=len({v for C, _ in it8 for v in C}),
                        value=int(c8["value"]), svals=s8)
    ctxs = [[tuple(v) for v in C] for C in json.load(open("fiber16_ctxs.json"))]
    lam = np.load("cert16_lambda.npy")
    it16 = [(ctxs[i], int(lam[i]) % 16) for i in range(len(ctxs)) if int(lam[i]) % 16]
    s16 = cert_verify(16, it16, 8)
    out["cert16"] = dict(d=16, contexts=len(it16),
                         nobs=len({v for C, _ in it16 for v in C}),
                         value=8, svals=s16)
    _CERTS = out
    return out

# ----------------------------------------------------------------------------------
# d=4 census block: exact vertex rebuild + facet re-verification + exact
# maximally-mixed moment vector.
# ----------------------------------------------------------------------------------

RE_IM = {0: (1, 0), 1: (0, -1), 2: (-1, 0), 3: (0, 1)}   # exp(-i*pi/2*k)

_CENSUS = None
def census_certify():
    global _CENSUS
    if _CENSUS is not None: return _CENSUS
    d = 4
    W = weyl_builder(d, 2)
    fam = [[tuple(v) for v in it["ctx"]] for it in json.load(open("cert4_min.json"))["items"]]
    obs = sorted({t for C in fam for t in C}); oi = {t: k for k, t in enumerate(obs)}
    # exact context phases s(C): product = omega^{s} I
    S = []
    for C in fam:
        P = m_id(16, 8)
        for v in C: P = m_mul(P, W(v))
        e = m_scalar(P)
        assert e is not None and e % 2 == 0
        S.append((e // 2) % 4)
    CTX = fam[1:6]                                        # DROP = 0 (deleted context)
    # exact spectra (as omega-exponent sets), and W(v)^4 = I for every observable
    spec = {v: eig_omega_set(W(v), 4) for v in obs}
    # deterministic assignments: exhaustive over Z_4^9 with the context and
    # spectral-support constraints (exact integer arithmetic)
    A = np.zeros((5, 9), dtype=np.int64)
    for r, C in enumerate(CTX):
        for v in C: A[r, oi[v]] += 1
    rhs = np.array([S[i] for i in range(6) if i != 0], dtype=np.int64)
    grid = np.array(list(itertools.product(range(4), repeat=9)), dtype=np.int64)
    ok = ((grid @ A.T) % 4 == rhs % 4).all(axis=1)
    for v in obs:
        ok &= np.isin(grid[:, oi[v]], spec[v])
    L = grid[ok]
    assert len(L) == 64, len(L)
    # vertex moment vectors in the 150 character coordinates, exact integers
    V150 = []
    for row in L:
        vec = []
        for C in CTX:
            j1, j2 = int(row[oi[C[0]]]), int(row[oi[C[1]]])
            for a in range(4):
                for b in range(4):
                    if (a, b) == (0, 0): continue
                    re, im = RE_IM[(a * j1 + b * j2) % 4]
                    vec += [re, im]
        V150.append(vec)
    V150 = np.array(V150, dtype=np.int64)
    assert V150.shape == (64, 150) and set(np.unique(V150)) <= {-1, 0, 1}
    # exact maximally-mixed moment vector: char (a,b) of context (v1,v2) is
    # Tr(W1^{-a} W2^{-b})/16 -- computed exactly; nonzero only on scalar words.
    y_mixed = []
    for C in CTX:
        W1, W2 = W(C[0]), W(C[1])
        iW1, iW2 = m_inv(W1), m_inv(W2)
        for a in range(4):
            for b in range(4):
                if (a, b) == (0, 0): continue
                op = m_mul(m_pow(iW1, a), m_pow(iW2, b))
                e = m_scalar(op)
                if e is None:
                    assert m_trace_coeffs(op, 4) == [0, 0, 0, 0], "nonscalar word with trace != 0"
                    y_mixed += [Fraction(0), Fraction(0)]
                else:
                    assert e % 2 == 0, "scalar context word with odd tau exponent"
                    re, im = {0: (1, 0), 1: (0, 1), 2: (-1, 0), 3: (0, -1)}[(e // 2) % 4]
                    y_mixed += [Fraction(re), Fraction(im)]
    assert all(f.denominator == 1 for f in y_mixed)
    y_mixed = np.array([int(f) for f in y_mixed], dtype=np.int64)
    # facet caches (read-only) and full exact re-verification
    dd = np.load("d4_facet_classes.npz")
    rows, full, P = dd["rows"].astype(np.int64), dd["full"].astype(np.int64), dd["P"].astype(np.int64)
    Y = V150[:, P]
    assert rows.shape == (61, 37) and full.shape == (23256, 34)
    # 61 class representatives: cB + cA.y >= 0 on all 64 vertices, min 0, tight counts
    tight_ok = 0
    for r in rows:
        cB, cA, tight, orbit = int(r[0]), r[1:34], int(r[34]), int(r[36])
        f = cA @ Y.T + cB
        assert f.min() == 0 and (f >= 0).all()
        assert int((f == 0).sum()) == tight
        tight_ok += 1
    assert tight_ok == 61 and int(rows[:, 36].sum()) == 23256
    bound_counts = Counter(int(b) for b in rows[:, 0])
    assert set(bound_counts) <= {1, 2, 3, 4}
    # all 23,256 facets: valid and supporting
    fv = full[:, :33] @ Y.T + full[:, 33][:, None]
    assert fv.min() == 0 and (fv >= 0).all() and (fv.min(axis=1) == 0).all()
    # tier-2 sub-orbit: exact {0,+-1} lifts with integer bounds {6,7}
    t2 = np.load("d4_tier2_orbit_data.npz")
    enc, B = t2["enc"].astype(np.int64), t2["B"].astype(np.int64)
    assert (enc % 4 == 0).all()
    lam = enc // 4
    assert set(np.unique(lam)) <= {-1, 0, 1}
    t2v = lam @ V150.T
    assert (t2v.max(axis=1) == B).all() and set(np.unique(B)) == {6, 7}
    t2_counts = Counter(int(b) for b in B)
    # mixed state inside the polytope: EVERY facet satisfied at y_mixed => CF=0
    fm = full[:, :33] @ y_mixed[P] + full[:, 33]
    assert (fm >= 0).all() and fm.min() > 0, "mixed state must be strictly inside"
    m0_class = rows[:, 1:34] @ y_mixed[P] + rows[:, 0]        # per-class mixed margin
    m0_t2 = B - lam @ y_mixed                                  # tier-2 mixed margin
    assert (m0_class > 0).all() and (m0_t2 > 0).all()
    _CENSUS = dict(S=S, nvert=64, rows=rows, bound_counts=dict(sorted(bound_counts.items())),
                   t2_counts=dict(sorted(t2_counts.items())), n_t2=len(B),
                   y_mixed_nonzero=int((y_mixed != 0).sum()),
                   m0_class=m0_class, m0_t2=m0_t2, cB=rows[:, 0].copy())
    return _CENSUS

# ----------------------------------------------------------------------------------
# Stages
# ----------------------------------------------------------------------------------

def stage_inventory():
    print("=" * 78)
    print("V58 STAGE inventory - state-independent-contextuality witness inventory")
    print("  (every 'recomputed here' row is re-derived in exact arithmetic in this run)")
    print("=" * 78)
    pm = pm_certify()
    ce = certs_certify()
    cs = census_certify()
    print("dim | witness | type | quantum value | deterministic bound | pinned | certified by")
    print("-" * 78)
    print("  4 (2 qubits, Weyl d=2 m=2) | Peres-Mermin S=R1+R2+R3+C1+C2-C3 | "
          "SIC operator witness")
    print(f"    | v_Q = {pm['v_Q']} (EVERY state: signed context products are +I exactly)")
    print(f"    | bound = {pm['bound']} (exhaustive: all 512 sign assignments; "
          f"{pm['n_attain']} attain 4; none reach 5 or 6; distribution {pm['histo']})")
    print("    | pinned: papers/paperB.tex (Hardware section: 'noncontextual bound 4 and")
    print("    |   quantum value 6'); hardware/README.md | certified: RECOMPUTED HERE +")
    print("    |   pm_gauge_invariance.py (V10), pm_local_system.py (V54)")
    print("    | calibration point: ibm_fez holonomy job S = 4.6125 +- 0.0173 -- a")
    print("    |   CALIBRATION POINT, not a result (QC_BRIDGE.md M3 charter).")
    print("-" * 78)
    for k in ("cert4", "cert8", "cert16"):
        c = ce[k]
        print(f"{c['d']*c['d']:>3} (Weyl d={c['d']} m=2) | {k}: {c['contexts']} contexts, "
              f"{c['nobs']} observables | Weyl-COCYCLE certificate (state-independent)")
        print(f"    | cocycle value S = {c['value']} = d/2 (omega-exponent; recomputed here")
        print("    |   exactly: commutation, scalar products, multiplicity mod d, value)")
        if k == "cert4":
            print("    | deterministic bound: 4 - DERIVED IN THIS SCRIPT (stage cert4bound):")
            print("    |   paperB's cos-functional S=sum_C cos(2pi(s_hat_C - s(C))/d) on the")
            print("    |   six cert4 contexts has classical max 4 (exhaustive, exact) vs")
            print("    |   quantum value 6 -> a calibrated SIC operator witness at d=4.")
        else:
            print("    | deterministic bound: NONE PINNED - this is an algebraic obstruction")
            print("    |   certificate, NOT an inequality witness. paperB's cos-functional")
            print("    |   S=sum_C cos(2pi(s_hat_C - s(C))/d) is stated with 'a classical bound")
            print("    |   below the quantum value' as an EXISTENCE claim; no bound value is")
            print("    |   computed anywhere in the repository for d >= 8. We do not invent one.")
        src = {"cert4": "cert4_min.json", "cert8": "cert8_min.json (verify_cert8.py)",
               "cert16": "fiber16_ctxs.json + cert16_lambda.npy (verify_cert16.py)"}[k]
        print(f"    | pinned: {src} | certified: RECOMPUTED HERE (exact monomial algebra)")
    print("-" * 78)
    print(" 16 (Weyl d=4 m=2) | d=4 facet census: 61 classes / 23,256 facets | "
          "STATE-DEPENDENT facets")
    print("    | these witness contextuality OF STATES (CF(rho)>0 <=> facet violated,")
    print("    |   V43 classifier theorem); they are NOT state-independent witnesses.")
    print(f"    | deterministic bounds (33 faithful coords, integer-primitive): "
          f"{cs['bound_counts']} (bound: #classes)")
    print(f"    | tier-2 sub-orbit: {cs['n_t2']} exact {{0,+-1}}-lifts in 150 character")
    print(f"    |   coords with integer bounds {cs['t2_counts']} (bound: #facets) --")
    print("    |   ALL re-verified here on the 64 exactly-rebuilt vertices (max = bound).")
    print("    | quantum value: NOT PINNED per class - no exact quantum maximum of any")
    print("    |   facet functional is certified in this repository (V48 records are")
    print("    |   float LP arbitration, not exact certificates).")
    print("    | pinned: d4_facet_classes.npz, d4_tier2_orbit_data.npz | certified:")
    print("    |   RECOMPUTED HERE (vertices, validity, tightness, orbit total 23,256)")
    print("-" * 78)
    print("d=4,6,8,10,12 (single qudit) | KS circle cores | KS-COLORABILITY obstruction")
    print("    | quantum value / deterministic bound: N/A - a colorability obstruction is")
    print("    |   not an inequality; it has no noise parameter and no operator form here.")
    print("    | pinned: rigidity/branch_d{6,8,10,12}flexcert.py caches + the d=4 M9 core")
    print("    | certified: CITED - V57 nonrigid_ks_tower.py (SAT/CaDiCaL uncolorability +")
    print("    |   criticality re-checks). NOT re-run in this script (external SAT dep).")
    print("-" * 78)
    print("d=6, 10, 12 (Weyl, m=2) | operator witness with deterministic bound: NONE.")
    print("    | What exists instead: paperB Thm 'attain' asserts value-d/2 attainment at")
    print("    |   every even d (ROZF lower-bound construction, cited, not instantiated");
    print("    |   here); V45 exhibits d=6 value-3 families for in-script random seeds")
    print("    |   (tau_gauge_general.py) - no canonical pinned d=6 certificate file.")
    print("    | Honest verdict: at d=6..12 the program possesses NO operator witness")
    print("    |   with a deterministic classical bound today. Stated, not papered over.")
    print("-" * 78)
    print("paperB/paperC on d/2 holonomy values as witnesses: paperB frames the value d/2")
    print("  as a 'structural fingerprint' with a state-independent measurement protocol")
    print("  (the cos-functional) whose classical bound is asserted to exist for even d;")
    print("  it is computed at d=2 (PM: 4 vs 6) and, in this script's cert4bound stage,")
    print("  at d=4 (cert4: 4 vs 6). paperC's d/2 appears as the class value and the")
    print("  ghost rotation (gamma - s* = d/2); both are exact algebraic invariants,")
    print("  benchmarkable through the d=2 and (now) d=4 operator witnesses above.")
    print("V58 inventory PASS")

def stage_margins():
    print("=" * 78)
    print("V58 STAGE margins - exact robustness margins (all Fractions)")
    print("=" * 78)
    pm = pm_certify()
    v_Q, bound, v_mix = pm["v_Q"], pm["bound"], pm["v_mixed"]
    print("[PM operator witness, two qubits]")
    print(f"  v_Q = {v_Q}, bound = {bound}, v_mixed = {v_mix} (computed: signed sum of")
    print("    exact traces Tr(+-I)/4; NOT assumed 0).")
    print("  state-depolarizing threshold eps* = (v_Q - bound)/(v_Q - v_mixed):")
    print(f"    denominator v_Q - v_mixed = {v_Q - v_mix} -> eps* UNDEFINED. A state-")
    print("    independent witness is degraded by NO amount of state depolarization;")
    print("    quoting a finite state-noise threshold here would be dishonest.")
    print("  measurement-noise model (the one that binds, per paperB: 'only measurement")
    print("    imperfection enters'): uniform context-correlator visibility eta,")
    print("    S(eta) = 6*eta. Exact threshold:")
    print(f"    eta* = bound/v_Q = {bound/v_Q},  tolerated effective noise eps* = "
          f"{1 - bound/v_Q} exactly.")
    S = Fraction("4.6125"); SE = Fraction("0.0173")
    eps_eff = 1 - S / 6
    print("[ibm_fez calibration point - benchmarking frame]")
    print(f"  measured S = 4.6125 = {S} (exact rational on the pinned decimal), "
          f"SE = {SE} (shot noise)")
    print(f"  eps_eff = (v_Q - S)/(v_Q - v_rand) = 1 - S/6 = {eps_eff} = {float(eps_eff)}")
    print("    (v_rand = 0 is the RANDOM-OUTCOME endpoint of the visibility model, not")
    print("     a state value - the mixed STATE gives 6, see above)")
    print(f"  margin to threshold: 1/3 - {eps_eff} = {Fraction(1,3) - eps_eff} "
          f"(~{float(Fraction(1,3) - eps_eff):.4f})")
    print(f"  fraction of quantum-classical gap retained: (S-4)/(6-4) = {(S-4)/2}")
    print(f"  shot-noise band on eps_eff: [{1-(S+SE)/6}, {1-(S-SE)/6}]")
    print("  CAVEATS (hardware/README.md + COMPATIBILITY.md, attached in full spirit):")
    print("   - This sigma-level is shot-noise-only under the implemented witness model;")
    print("     it is NOT a loophole-free contextuality certification. Calibration drift,")
    print("     gate/model systematics, measurement disturbance, inconsistent marginals,")
    print("     look-elsewhere/backend selection are NOT included.")
    print("   - Nondisturbance/compatibility is UNTESTED: the stored artifacts contain no")
    print("     per-observable marginals (COMPATIBILITY.md par.4).")
    print("   - The pinned pm_S = 4.6125 is NOT independently reproducible from its own")
    print("     stored counts (empty pub metadata; naive recomputation 4.5662, gap +0.0463")
    print("     ~ 2.7x SE; COMPATIBILITY.md par.3b). eps_eff above is exact arithmetic on")
    print("     a pinned but not re-derivable input. Calibration point, never a result.")
    cs = census_certify()
    print("[d=4 census facets - state-dependent margins, exact]")
    print(f"  maximally mixed moment vector: computed exactly (monomial traces);")
    print(f"    {cs['y_mixed_nonzero']} of 150 coordinates nonzero (torsion characters -")
    print("    scalar Weyl words; 'v_mixed = 0' would have been WRONG coordinate-wise).")
    print("  CF(maximally mixed) = 0 CERTIFIED: all 23,256 facets satisfied strictly at")
    print("    y_mixed (complete H-representation, V43 => membership is a theorem).")
    m0c = Counter((int(cb), int(m0)) for cb, m0 in zip(cs["cB"], cs["m0_class"]))
    print("  per-class mixed-state margin M0 = cB + cA.y_mixed (exact integers):")
    print(f"    (cB, M0): #classes = {dict(sorted(m0c.items()))}")
    print("  tier-2 {0,+-1} lifts: mixed-state margin M0 = B - lam.y_mixed:")
    t2ctr = Counter((int(b), int(m)) for b, m in zip(
        np.load("d4_tier2_orbit_data.npz")["B"].astype(np.int64), cs["m0_t2"]))
    print(f"    (B, M0): #facets = {dict(sorted(t2ctr.items()))}")
    print("  depolarizing threshold for a violating state rho with exact violation V>0:")
    print("    eps*(rho) = V / (V + M0)   (moments are linear in rho; mixing endpoint")
    print("    is the exact y_mixed above). M0 is the certified denominator datum.")
    print("  NOT CERTIFIED (named states): no exact quantum-state facet violation V is")
    print("    pinned in the repository (V48's violating states are float LP records).")
    print("    Exact eps*(rho) for named states needs a Z[zeta_8] state-moment layer -")
    print("    stage-2 work, listed in arxiv/moonshots/QUDIT_BENCH.md, not claimed here.")
    print("V58 margins PASS")

_C4B = None
def cert4bound_certify():
    """cert4 cos-functional: calibrated state-independent inequality at d=4, m=2.
    CLAIM. Let C_1..C_6 be the cert4 contexts (cert4_min.json; each a triple of
      pairwise-commuting Weyl observables on (C^4)^{x2}, product = omega^{s(C)} I,
      s = (0,2,2,0,2,0), omega = i). paperB's noncontextual functional
        S(g) = sum_C cos(2*pi*(s_hat_C(g) - s(C))/4),
        s_hat_C(g) = sum_{v in C} g(v) mod 4   (g assigns an omega-exponent
        outcome to each of the 9 observables),
      satisfies: (a) quantum value v_Q = 6 for EVERY state (each context correlator
      Re<omega^{-s(C)} prod_C> equals 1 identically - the product IS the scalar,
      re-verified here as an operator identity; and 6 is the maximum since each
      term is Re of a unitary's expectation, <= 1); (b) deterministic bound 4:
      max S(g) = 4 over ALL 4^9 = 262,144 assignments g in Z_4^9 (9,216 attain)
      AND over the 32,768 spectra-constrained assignments g(v) in spec(W(v))
      (1,664 attain) - identical bound on both spaces, so the KS spectral
      constraint is not load-bearing; (c) S(g) is always even, with value
      distribution (ambient) {-4:9216, -2:61440, 0:120832, 2:61440, 4:9216};
      no assignment reaches 5 or 6. Strict gap v_Q - bound = 2.
    PROOF. Exact throughout: at d=4 the functional's values cos(2*pi*k/4) are the
      INTEGERS {1, 0, -1, 0} for k = 0..3, so the decision path is pure integer
      arithmetic (a fortiori exact in Q(zeta_8); floats appear nowhere). Operator
      facts by the exact monomial algebra above. Analytic cap, re-derived
      independently by the exhaustion: for ANY g in Z_4^9 the multiplicity
      identity gives sum_C lam_C s_hat_C = sum_v (lam^T A)_v g(v) == 0 mod 4,
      hence sum_C lam_C (s_hat_C - s(C)) == -2 == 2 mod 4 with every lam_C odd
      (lam = 1,3,3,1,3,1); so either some delta_C is odd - then an even number
      >= 2 are odd, contributing >= 2 zero terms - or all delta_C are even with
      an odd count of delta_C = 2, contributing a -1 term; either way S <= 4.
      The exhaustion (exact int64 linear algebra) certifies attainment and the
      full distribution.
    EXPECTED. bound = 4, v_Q = v_mixed = 6, eta* = 2/3, eps* = 1/3."""
    global _C4B
    if _C4B is not None: return _C4B
    d = 4
    W = weyl_builder(d, 2)
    c4 = json.load(open("cert4_min.json"))
    items = [([tuple(v) for v in it["ctx"]], int(it["lam"])) for it in c4["items"]]
    svals = cert_verify(4, items, int(c4["value"]))     # full certificate re-check
    fam = [C for C, _ in items]; lams = [l for _, l in items]
    obs = sorted({v for C in fam for v in C}); oi = {v: k for k, v in enumerate(obs)}
    assert len(fam) == 6 and len(obs) == 9
    assert all(l % 2 == 1 for l in lams), "analytic cap needs every lam odd"
    # (a) operator identity: every context product IS omega^{s(C)} I, so the
    # context correlator is 1 for EVERY state; each term <= 1 caps v_Q at 6.
    for C, s in zip(fam, svals):
        P = m_id(16, 8)
        for v in C: P = m_mul(P, W(v))
        assert m_scalar(P) == (2 * s) % 8
    # exact spectra (omega-exponent sets; also asserts W(v)^4 = I)
    spec = {v: eig_omega_set(W(v), 4) for v in obs}
    # (b) exhaustion, exact int64: S(g) = sum_C COS[(s_hat_C - s_C) mod 4]
    COS = np.array([1, 0, -1, 0], dtype=np.int64)   # cos(2 pi k / 4), exact
    Actx = np.zeros((6, 9), dtype=np.int64)
    for r, C in enumerate(fam):
        for v in C: Actx[r, oi[v]] += 1
    grid = np.array(list(itertools.product(range(4), repeat=9)), dtype=np.int64)
    delta = (grid @ Actx.T - np.array(svals, dtype=np.int64)) % 4
    Sg = COS[delta].sum(axis=1)
    ok = np.ones(len(grid), dtype=bool)
    for v in obs:
        ok &= np.isin(grid[:, oi[v]], spec[v])
    assert int(ok.sum()) == 32768, "spectra-constrained space size"
    h_amb = Counter(int(x) for x in Sg)
    h_spc = Counter(int(x) for x in Sg[ok])
    assert Sg.max() == 4 and Sg[ok].max() == 4, "deterministic bound"
    assert h_amb[5] == 0 and h_amb[6] == 0
    assert (Sg % 2 == 0).all(), "parity of the value distribution"
    assert h_amb == {-4: 9216, -2: 61440, 0: 120832, 2: 61440, 4: 9216}
    assert h_spc == {-4: 1664, -2: 7680, 0: 14080, 2: 7680, 4: 1664}
    v_Q = Fraction(6); bound = Fraction(4); v_mixed = Fraction(6)
    _C4B = dict(v_Q=v_Q, bound=bound, v_mixed=v_mixed, svals=svals, lams=lams,
                n_amb=len(grid), n_spc=int(ok.sum()),
                att_amb=h_amb[4], att_spc=h_spc[4],
                h_amb=dict(sorted(h_amb.items())), h_spc=dict(sorted(h_spc.items())),
                eta=bound / v_Q, eps=1 - bound / v_Q)
    return _C4B

def stage_cert4bound():
    print("=" * 78)
    print("V58 STAGE cert4bound - calibrated inequality from the cert4 Weyl certificate")
    print("=" * 78)
    c = cert4bound_certify()
    print("[functional] paperB cos-functional on the six cert4 contexts (d=4, m=2,")
    print("  Hilbert dim 16): S(g) = sum_C cos(2pi(s_hat_C - s(C))/4), s_hat_C = sum of")
    print(f"  assigned omega-exponents in C mod 4; exact context phases s = {c['svals']}")
    print(f"  (products = omega^s I, re-verified as operator identities), lam = {c['lams']}.")
    print("  At d=4 all functional values are INTEGERS (cos(2pi k/4) in {1,0,-1}):")
    print("  the entire decision path is exact integer arithmetic, floats nowhere.")
    print(f"[quantum]   v_Q = {c['v_Q']} for EVERY state incl. maximally mixed (each context")
    print("  correlator is identically 1 - scalar operator identity; each term <= 1, so")
    print(f"  6 is the quantum maximum). v_mixed = {c['v_mixed']} = v_Q.")
    print(f"[classical] bound = {c['bound']} by EXHAUSTION: ambient Z_4^9 ({c['n_amb']} assignments,")
    print(f"  {c['att_amb']} attain 4, none reach 5 or 6, distribution {c['h_amb']});")
    print(f"  spectra-constrained ({c['n_spc']} assignments with g(v) in spec(W(v)),")
    print(f"  {c['att_spc']} attain 4, distribution {c['h_spc']}).")
    print("  Same bound on both spaces -> the KS spectral constraint is not load-bearing.")
    print("  Analytic cross-check (from the certificate itself): sum_C lam_C s_hat_C == 0")
    print("  mod 4 for ANY assignment, so sum lam_C delta_C == 2 mod 4 with all lam odd;")
    print("  hence >= 2 zero terms or >= 1 minus-one term -> S <= 4. Exhaustion agrees.")
    print(f"[gap]       v_Q - bound = {c['v_Q'] - c['bound']} > 0: STRICT quantum-classical gap - the")
    print("  program's first operator-witness bound above d=2 (stage-2 item 1 of")
    print("  QUDIT_BENCH.md, now closed).")
    print("[margins - V58 conventions, all Fractions]")
    print(f"  state-depolarizing threshold eps* = (v_Q - bound)/(v_Q - v_mixed): denominator")
    print(f"  {c['v_Q'] - c['v_mixed']} -> UNDEFINED, exactly as PM: state noise never degrades a")
    print("  state-independent witness; the binding model is measurement noise.")
    print(f"  uniform context-correlator visibility S(eta) = 6*eta: threshold eta* =")
    print(f"  bound/v_Q = {c['eta']}, tolerated effective noise eps* = {c['eps']} exactly -")
    print("  the SAME constants as PM (6, 4, 2/3, 1/3), now at d=4.")
    print("  calibration point: NONE - no d=4 hardware run exists in this repository;")
    print("  no number is invented for one.")
    print("[protocol caveat - paperB, applies verbatim] a single joint measurement of a")
    print("  context returns omega^{s(C)} identically (algebraic identity) and certifies")
    print("  nothing; the bound is operational only under physically separate (sequential/")
    print("  ancilla) readout of the three observables, where the ledger stage's")
    print("  assumptions (A1)-(A4) apply unchanged.")
    print("V58 cert4bound PASS")

def stage_ledger():
    print("=" * 78)
    print("V58 STAGE ledger - compatibility-assumption ledger (sequential benchmarks)")
    print("=" * 78)
    print("A sequential-measurement contextuality BENCHMARK assumes:")
    print(" (A1) COMPATIBILITY of context partners: the three observables of a context,")
    print("      as implemented, are jointly measurable (ideal: commuting projective")
    print("      measurements realized by the sequential Hadamard-test readout).")
    print(" (A2) NONDISTURBANCE: earlier measurements in the sequence do not alter the")
    print("      statistics of later ones beyond the Lueders update (no signaling in time).")
    print(" (A3) CONTEXT-INDEPENDENT IMPLEMENTATION: a shared observable is realized by")
    print("      the same operation in both contexts containing it.")
    print(" (A4) Stationarity/iid across shots and no drift within a job.")
    print("What breaks them: sequential crosstalk and readout backaction (A1/A2), context-")
    print("  dependent transpilation or qubit re-selection (A3), calibration drift (A4).")
    print("The COST when they fail: S > 4 no longer certifies contextuality - a disturbing")
    print("  device can inflate the classical bound to 4 + Delta (contextuality-by-default")
    print("  / Kujala-Dzhafarov style); Delta is NOT computable from our stored data")
    print("  because no per-observable marginals were persisted. What survives without")
    print("  (A1)-(A3) is only the benchmarking reading: S is a figure of merit for how")
    print("  well the device implements the ideal witness model - which is exactly the")
    print("  frame this suite adopts (QC_BRIDGE.md M3: benchmarking, never a test).")
    print("Loophole-aware precedents (the bar we do NOT meet and do not claim to meet):")
    print("  - Um et al., Phys. Rev. Applied 13, 034077 (2020): sequential contextuality")
    print("    on a trapped ion with the compatibility/disturbance budget quantified and")
    print("    fed into the certified-randomness analysis.")
    print("  - Wang et al., Sci. Adv. 8, eabk1660 (2022): loophole-aware (detection +")
    print("    compatibility) Kochen-Specker contextuality test on trapped ions.")
    print("  (Citations as pinned in QC_BRIDGE.md; author lists to be re-verified against")
    print("   the journal record at write-up, per the house J23/J24 rule.)")
    print("Our own data's status (hardware/COMPATIBILITY.md - the ledger's local rows):")
    print("  - Nondisturbance UNTESTED: neither stored artifact contains per-observable")
    print("    marginals; the diagnostic 'same observable's marginal across its two")
    print("    contexts' is NOT COMPUTABLE from existing files (par.4).")
    print("  - Order-reversal, repeatability, disturbance-corrected bound: require NEW")
    print("    experiments (par.5). None performed.")
    print("  - Between-device homogeneity FAILS: chi^2 = 265.2 on 2 dof, Birge ratio 11.5;")
    print("    the defensible combined significance is ~7 sigma, not the shot-noise 80.")
    print("  - The fez holonomy pm_S = 4.6125 is not reproducible from its stored counts")
    print("    (empty metadata; par.3b).")
    print("Ledger verdict: every number this suite attaches to hardware is CONDITIONAL on")
    print("  (A1)-(A4), the assumptions are untested on our own data, and the suite's")
    print("  claims are framed so that they survive that fact (benchmarking constants,")
    print("  not certification).")
    print("V58 ledger PASS")

def stage_report():
    print("=" * 78)
    print("V58 STAGE report - honest scope")
    print("=" * 78)
    print("WHAT THIS SUITE IS:")
    print("  - Exact-arithmetic benchmarking constants: the PM deterministic bound 4 and")
    print("    quantum value 6 re-proved by exhaustion + exact operator algebra; exact")
    print("    visibility threshold 1/3; exact eps_eff = 37/160 for the fez calibration")
    print("    point; the complete d=4 facet catalogue re-verified (61 classes, 23,256")
    print("    facets, tier-2 {0,+-1} lifts with bounds {6,7}) with exact mixed-state")
    print("    margins - facet-COMPLETE robustness data, which per the M3 literature")
    print("    scout nobody in the literature ships.")
    print("  - A cocycle-certificate tower (cert4/cert8/cert16, values d/2) re-verified")
    print("    in float-free arithmetic; for cert4 the stage cert4bound derives the")
    print("    calibrated inequality exactly (quantum 6 every state, classical bound 4")
    print("    by full 4^9 exhaustion, eps* = 1/3) - the first operator-witness bound")
    print("    above d=2 in the program.")
    print("WHAT THIS SUITE IS NOT:")
    print("  - NOT self-testing: our own V57 shows the program's KS cores are NON-RIGID")
    print("    at every rung d=4..12, so configuration self-testing claims from these")
    print("    sets would be false, not merely unproved.")
    print("  - NOT a loophole-free contextuality test: nondisturbance is untested on our")
    print("    data (see ledger stage).")
    print("  - NOT device certification in the Xu-Saha-Bharti-Cabello sense (PRL 132,")
    print("    140201): their criterion requires rigid KS sets; ours are not.")
    print("  - NOT a foundational test of quantum mechanics, and the ibm_fez S = 4.6125")
    print("    is a calibration point for the benchmarking pipeline, never a result.")
    print("  - No operator witness with a deterministic bound exists in this program at")
    print("    d = 6..12 today; the gap list and stage-2 candidates are in")
    print("    arxiv/moonshots/QUDIT_BENCH.md.")
    print("V58 report PASS")

STAGES = dict(inventory=stage_inventory, margins=stage_margins,
              cert4bound=stage_cert4bound, ledger=stage_ledger, report=stage_report)

def main():
    arg = sys.argv[1] if len(sys.argv) > 1 else "all"
    if arg == "all":
        for name in ("inventory", "margins", "cert4bound", "ledger", "report"):
            STAGES[name]()
        print("V58 PASS")
    elif arg in STAGES:
        STAGES[arg]()
        print("V58 PASS")
    else:
        print(f"usage: python3 qudit_bench.py [{'|'.join(STAGES)}|all]"); sys.exit(2)

if __name__ == "__main__":
    main()
