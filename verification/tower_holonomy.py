# V47 - TOWER CONJECTURE (note v3 Conj. 2) RESOLVED BY CASES. See INDEX.md entry.
# Expected final line: 'c2_tower_probe done.' with A: all PASS, C: pinned d=3,7 naive-section
# counterexamples + Weil cure, E: attained groups (exhaustive iff d=2 mod 4 or d=2).
#!/usr/bin/env python3
# c2_tower_probe.py -- probe of CONJECTURE 2 ("Tower", note_v3_local_validity.tex):
#   "Loop holonomies of Weyl families at local dimension d take values in mu_{2d};
#    the 2-adic tower of paper B is exhaustive."
# Convention (verification/weyl.py, per-site labels):
#   W(v) = tau^{-q(v)} kron_i X^{a_i} Z^{b_i},  tau = e^{i pi/d},  q(v) = sum_i a_i b_i.
# Invariant levels (V12 wf_loop_holonomy.py): ray-level (Pancharatnam) vs
#   DETERMINANT-SECTION holonomy of SU-closed context loops. Pinned base case d=2: -i.
# Parts (select by argv[1], default "all"):
#   A: operator-level holonomies of closed Weyl words (cocycle level), d=2,3,4,5,6,8; m=1,2
#   B: exact determinant table of the canonical section gates (X, Z, W(v), F, S, M_a)
#   C: det-section holonomies of SU-closed loops: friend loops T_Z->T_X->T_XZ->T_Z
#      (50 random SU closers each) + 200 random Clifford-word loops per d
#   D: odd d: naive DFT section vs Weil (metaplectic, Gauss-sum-normalized) section
#   E: attained-subgroup summary and tower-exhaustiveness verdict
#   F: (exploratory) ray-level total Pancharatnam product of the friend loop
import numpy as np, sys

# ---------- infrastructure ----------
def gates(d):
    w = np.exp(2j*np.pi/d); tau = np.exp(1j*np.pi/d)
    X = np.roll(np.eye(d), 1, axis=0)
    Z = np.diag([w**k for k in range(d)])
    F = np.array([[w**(j*k) for k in range(d)] for j in range(d)]) / np.sqrt(d)
    if d % 2 == 0:
        S = np.diag([tau**(k*k) for k in range(d)])          # tau-grid gate (even d)
    else:
        S = np.diag([w**((k*(k+1))//2) for k in range(d)])   # omega-grid gate (odd d)
    return w, tau, X, Z, F, S

def W_op(v, d, m):
    w, tau, X, Z, _, _ = gates(d)
    M = np.linalg.matrix_power(X, int(v[0]) % d) @ np.linalg.matrix_power(Z, int(v[1]) % d)
    for i in range(1, m):
        M = np.kron(M, np.linalg.matrix_power(X, int(v[2*i]) % d) @ np.linalg.matrix_power(Z, int(v[2*i+1]) % d))
    q = sum(int(v[2*i]) * int(v[2*i+1]) for i in range(m))
    return tau**(-q) * M

def Mmult(a, d):
    P = np.zeros((d, d))
    for k in range(d): P[(a*k) % d, k] = 1
    return P

def gridc(z, d, tol=1e-6):
    """classify unit-modulus z on the tau-grid: returns (class, tau-exponent mod 2d)"""
    assert abs(abs(z) - 1) < 1e-6, f"|z|={abs(z)}"
    e = (np.angle(z) * d / np.pi) % (2*d)
    r = round(e)
    if min(abs(e - r), abs(e - r - 2*d), abs(e - r + 2*d)) < tol:
        r %= 2*d
        return ("mu_d" if r % 2 == 0 else "mu_2d*", r)     # mu_2d* = mu_2d \ mu_d
    r2 = round(2*e)
    if min(abs(2*e - r2), abs(2*e - r2 - 4*d), abs(2*e - r2 + 4*d)) < tol:
        return ("mu_4d*", (r2 % (4*d)) / 2)                 # mu_4d \ mu_2d
    return ("OFF-GRID", e)

def in_mu2d(cls): return cls in ("mu_d", "mu_2d*")

# ---------- Part A: operator-level (cocycle) holonomies of closed Weyl words ----------
def partA():
    print("== A: closed Weyl words: product = tau^c * I, membership of tau^c ==")
    all_ok, attain = True, {}
    for d in (2, 3, 4, 5, 6, 8):
        for m in (1, 2):
            rng = np.random.default_rng(7); N = d**m
            exps = set()
            for _ in range(150):
                k = int(rng.integers(3, 7))
                vs = [rng.integers(0, d, 2*m) for _ in range(k-1)]
                vs.append((-sum(vs)) % d)
                M = np.eye(N, dtype=complex)
                for v in vs: M = M @ W_op(v, d, m)
                z = M[0, 0]
                assert np.allclose(M, z*np.eye(N), atol=1e-8), "closed word not scalar"
                cls, e = gridc(z, d)
                if cls == "OFF-GRID" or cls == "mu_4d*": all_ok = False
                exps.add(int(e) if cls != "OFF-GRID" else e)
            # explicit odd-exponent attempt: [X, Z, -(X+Z)] word
            vs2 = [np.array([1, 0] + [0, 0]*(m-1)), np.array([0, 1] + [0, 0]*(m-1))]
            vs2.append((-vs2[0] - vs2[1]) % d)
            M = np.eye(N, dtype=complex)
            for v in vs2: M = M @ W_op(v, d, m)
            cls2, e2 = gridc(M[0, 0], d)
            odd = sorted(e for e in exps if isinstance(e, int) and e % 2 == 1)
            attain[(d, m)] = (sorted(e for e in exps if isinstance(e, int)), int(e2))
            print(f"A d={d} m={m}: 150 closed words all tau^Z: "
                  f"{'PASS' if all(isinstance(e,int) for e in exps) else 'FAIL'}; "
                  f"odd exponents seen: {odd[:6]}{'...' if len(odd)>6 else ''}; "
                  f"XZ-word exponent {e2} ({cls2})")
    print(f"A VERDICT: operator-level holonomies always tau^(integer) i.e. in mu_2d: "
          f"{'THEOREM-CONSISTENT (all PASS)' if all_ok else 'FAIL'}")
    return attain

# ---------- Part B: determinant table of the canonical section ----------
def partB():
    print("== B: exact determinants of canonical gates (tau-exponent mod 2d) ==")
    for d in (2, 3, 4, 5, 6, 7, 8, 12, 16):
        w, tau, X, Z, F, S = gates(d)
        # sanity: S X S^dagger proportional to XZ (S is a legitimate context switch T_X -> T_XZ)
        R = S @ X @ S.conj().T; XZ = X @ Z
        rat = R[1, 0] / XZ[1, 0]
        assert np.allclose(R, rat * XZ, atol=1e-8), f"S gate wrong at d={d}"
        dX = gridc(np.linalg.det(X), d); dZ = gridc(np.linalg.det(Z), d)
        dF = gridc(np.linalg.det(F), d); dS = gridc(np.linalg.det(S), d)
        sigma = d*(d-1)*(2*d-1)//6
        pred_S = ("even-d tau^sigma, sigma=%d mod %d = %d" % (sigma, 2*d, sigma % (2*d))) if d % 2 == 0 else "omega-grid"
        # Weyl dets, and multiplicative gates
        rng = np.random.default_rng(1); wcls = set()
        for _ in range(25):
            v = rng.integers(0, d, 2)
            wcls.add(gridc(np.linalg.det(W_op(v, d, 1)), d))
        mcls = set()
        for a in range(1, d):
            if np.gcd(a, d) == 1: mcls.add(gridc(np.linalg.det(Mmult(a, d)), d))
        print(f"B d={d}: det X={dX}, det Z={dZ}, det F={dF} [mu_4 check: {dF[1]*2 % d == 0 or d%2==1}], "
              f"det S={dS} ({pred_S}); det W(v) in {sorted(wcls)}; det M_a in {sorted(mcls)}")

# ---------- Part C: det-section holonomies of SU-closed context loops ----------
def friend_holonomy(d, n_closers=50, seed=0):
    w, tau, X, Z, F, S = gates(d)
    B = S @ F                       # columns = eigenbasis of S X S^dag ~ XZ  (context T_XZ)
    rng = np.random.default_rng(seed); vals = set()
    for _ in range(n_closers):
        D = np.diag(np.exp(1j*rng.uniform(0, 2*np.pi, d)))
        P = np.eye(d)[rng.permutation(d)]
        C = D @ P @ B.conj().T
        C = C / np.linalg.det(C)**(1.0/d)          # SU(d) closer
        assert abs(np.linalg.det(C) - 1) < 1e-8
        L = C @ S @ F                               # SU-closed loop T_Z->T_X->T_XZ->T_Z
        nz = (np.abs(L) > 1e-8)
        assert (nz.sum(axis=0) == 1).all() and (nz.sum(axis=1) == 1).all(), "loop not closed (L not monomial)"
        vals.add(complex(np.round(np.linalg.det(L), 9)))
    assert len(vals) == 1, f"closer-dependence at d={d}: {vals}"
    return vals.pop()

def partC():
    print("== C: det-section holonomies, SU closers ==")
    res = {}
    for d in (2, 3, 4, 5, 6, 7, 8):
        hol = friend_holonomy(d)
        cls, e = gridc(hol, d)
        res[d] = (cls, e)
        flag = "IN mu_2d" if in_mu2d(cls) else "**NOT in mu_2d -- COUNTEREXAMPLE**"
        print(f"C d={d}: friend loop T_Z->T_X->T_XZ->T_Z, 50/50 SU closers agree: "
              f"hol = {hol:.6f} = tau^{e} ({cls}) -> {flag}")
    # random Clifford-word loops (leg = word in canonical gates, closed by SU closer):
    # holonomy = det(word); collect attained exponents per d
    print("-- random Clifford-word loops (200 words/d, length<=8) --")
    att = {}
    for d in (2, 3, 4, 5, 6, 8):
        w, tau, X, Z, F, S = gates(d)
        pool = [X, Z, F, S] + [Mmult(a, d) for a in range(2, d) if np.gcd(a, d) == 1]
        rng = np.random.default_rng(3); exps = set(); okall = True
        for _ in range(200):
            L = int(rng.integers(1, 9))
            Mw = np.eye(d, dtype=complex)
            for _ in range(L): Mw = Mw @ pool[int(rng.integers(0, len(pool)))]
            cls, e = gridc(np.linalg.det(Mw), d)
            exps.add((cls, e))
            if d % 2 == 0 and not in_mu2d(cls): okall = False
        ex_int = sorted(2*e for c, e in exps)   # half-integer-safe: exponents in tau^(1/2) units
        g = int(np.gcd.reduce([int(round(x)) for x in ex_int if x] + [4*d]))
        att[d] = g
        bad = sorted((c, e) for c, e in exps if not in_mu2d(c))
        print(f"C d={d}: 200 word-loops: attained subgroup = <tau^{g/2}> (order {int(4*d//g)}); "
              f"violations of mu_2d: {bad if bad else 'none'}"
              + ("" if d % 2 == 1 or okall else "  ** EVEN-d FAIL **"))
    return res, att

# ---------- Part D: odd d, naive DFT section vs Weil section ----------
def partD():
    print("== D: odd d: naive vs Weil (metaplectic) section ==")
    for d in (3, 5, 7):
        w, tau, X, Z, F, S = gates(d)
        g = sum(w**(k*k) for k in range(d))          # quadratic Gauss sum
        zeta = np.sqrt(d) / g                         # |zeta| = 1
        FW = zeta * F                                 # Weil-normalized Fourier
        assert np.allclose(np.linalg.matrix_power(FW, 4), np.eye(d), atol=1e-8), "FW^4 != I"
        dF = gridc(np.linalg.det(F), d); dFW = gridc(np.linalg.det(FW), d)
        # friend holonomy in both sections
        holN = np.linalg.det(S @ F); holW = np.linalg.det(S @ FW)
        cN = gridc(holN, d); cW = gridc(holW, d)
        print(f"D d={d}: Gauss phase g/sqrt(d) = {g/np.sqrt(d):.4f}; det F = tau^{dF[1]} ({dF[0]}), "
              f"det F_Weil = tau^{dFW[1]} ({dFW[0]})")
        print(f"D d={d}: friend hol NAIVE = tau^{cN[1]} ({cN[0]}) -> "
              f"{'in mu_2d' if in_mu2d(cN[0]) else '**VIOLATES mu_2d**'}; "
              f"friend hol WEIL = tau^{cW[1]} ({cW[0]}) -> "
              f"{'in mu_2d' if in_mu2d(cW[0]) else '**VIOLATES mu_2d**'}")

# ---------- Part E: attained subgroups / tower verdict ----------
def partE():
    print("== E: attained det-holonomy subgroup per even d (canonical tau-grid section) ==")
    for d in (2, 4, 6, 8, 12, 16):
        w, tau, X, Z, F, S = gates(d)
        exps = []
        for G in [X, Z, F, S] + [Mmult(a, d) for a in range(2, d) if np.gcd(a, d) == 1]:
            cls, e = gridc(np.linalg.det(G), d)
            assert cls in ("mu_d", "mu_2d*"), f"even-d generator off tau-grid at d={d}"
            exps.append(int(e))
        g = int(np.gcd.reduce(exps + [2*d]))
        order = 2*d // g if g else 1
        sigma = d*(d-1)*(2*d-1)//6
        tag = ("EXHAUSTIVE (= mu_2d)" if order == 2*d else
               f"NOT exhaustive: attained = mu_{order} < mu_{2*d}")
        print(f"E d={d}: generator tau-exponents {sorted(set(exps))} (sigma(d)={sigma} mod {2*d} = {sigma%(2*d)}), "
              f"attained = <tau^{g}> = mu_{order}; bound mu_{2*d}: holds; {tag}")
    print("E note: every generator det is realized by an explicit SU-closed loop "
          "(self-loops at T_Z for X,Z,S,M_a; the 2-context loop T_Z->T_X->T_Z for F), "
          "so 'attained' is exact, not just an upper bound.")

# ---------- Part F: exploratory ray level ----------
def partF():
    print("== F (exploratory): total Pancharatnam product of the friend loop ==")
    for d in (2, 3, 4, 5, 6, 8):
        w, tau, X, Z, F, S = gates(d)
        B0 = np.eye(d); B1 = F; B2 = S @ F
        p = 1 + 0j; ok = True
        for k in range(d):
            t = np.vdot(B0[:, k], B1[:, k]) * np.vdot(B1[:, k], B2[:, k]) * np.vdot(B2[:, k], B0[:, k])
            if abs(t) < 1e-12: ok = False; break
            p *= t / abs(t)
        if not ok:
            print(f"F d={d}: strand overlap vanishes; product undefined"); continue
        cls, e = gridc(p, d)
        print(f"F d={d}: total ray-level product = {p:.6f} = tau^{e} ({cls})")

if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("A", "all"): partA()
    if which in ("B", "all"): partB()
    if which in ("C", "all"): partC()
    if which in ("D", "all"): partD()
    if which in ("E", "all"): partE()
    if which in ("F", "all"): partF()
    print("c2_tower_probe done.")
