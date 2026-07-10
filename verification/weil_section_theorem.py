# V51 - ODD-d WEIL-SECTION TOWER THEOREM (closes V47's open case): every SU-closed
# loop holonomy in the Weil-normalized odd-d section lies in mu_{2d}; generator dets:
# det F_W = (-1)^floor(d/4) (Schur x Gauss), det M_a = Jacobi symbol (Zolotarev),
# det S in mu_d, det W(v) = (-1)^{ab}. Naive-DFT violation iff d = 3 (mod 4), proven.
# Weil attained group: mu_6 iff 3|d else mu_2 (exhaustive iff d=3).
# Expected: 'cw_weil_probe PASS' (<30 s).
#!/usr/bin/env python3
# cw_weil_probe.py -- machine verification of the ODD-d WEIL-SECTION TOWER-HOLONOMY
# THEOREM, closing the "general proof open" item of V47 (verification/tower_holonomy.py,
# Part D / INDEX.md entry V47(iii)).
#
# THEOREM (odd d, Weil section). Let d be odd, omega = e^{2 pi i/d}, tau = e^{i pi/d},
# and let the canonical odd-d section gates on C^d be
#   W(v) = tau^{-ab} X^a Z^b            (v=(a,b), representatives 0..d-1; weyl.py convention),
#   F_W  = (sqrt(d)/g_d) F,  F_{jk} = omega^{jk}/sqrt(d),  g_d = sum_k omega^{k^2},
#   S    = diag(omega^{k(k+1)/2}),
#   M_a  : e_k -> e_{ak mod d},  gcd(a,d)=1.
# Then the determinant of every gate lies in mu_{2d} = mu_2 x mu_d; hence every SU-closed
# context-loop holonomy (det of the loop word; the SU closer contributes det = 1) lies in
# mu_{2d}, for ALL odd d.
#
# GENERATOR TABLE (each line verified below at d = 3,5,7,9,11,13,15):
#   (1) det X = (-1)^{d-1} = +1  (d-cycle, d odd);
#       det Z = omega^{d(d-1)/2} = +1  ((d-1)/2 integer => exponent = 0 mod d).
#   (2) det W(v) = tau^{-abd} det(X)^a det(Z)^b = (e^{i pi})^{-ab} = (-1)^{ab}  in mu_2.
#       (Representative note: a -> a+d flips (-1)^{ab} for odd b, consistent with
#        W((a+d,b)) = (-1)^b W((a,b)); every value is in mu_2 regardless.)
#   (3) det S = omega^{sum_{k<d} k(k+1)/2} = omega^{binom(d+1,3)} = omega^{(d-1)d(d+1)/6}.
#       Reduced mod d:  = 1 if 3 does not divide d  (8 and 3 both divide d^2-1),
#                       = omega^{(d/3)(d-1)/2 mod d} if 3 | d.   In mu_d either way.
#   (4) det F = i^{d(d-1)/2}   [Schur's evaluation of the DFT determinant:
#       det(omega^{jk})_{jk} = prod_{j<k}(omega^k - omega^j), each factor
#       = 2 i sin(pi(k-j)/d) e^{i pi (j+k)/d}, the total e^{i pi (j+k)/d} phase is
#       e^{i pi d(d-1)^2/(2d)} = 1 for odd d, sines positive => phase i^{d(d-1)/2}].
#       COROLLARY (naive-section dichotomy, proves V47's empirical finding for all odd d):
#       i^{d(d-1)/2} in mu_{2d} iff (d-1)/2 even iff d = 1 (mod 4); at d = 3 (mod 4) the
#       naive DFT section violates the mu_{2d} bound (d=3: det S det F = omega * (-i) =
#       e^{i pi/6}; d=7: det F = i, both pinned by V47).
#   (5) g_d = sqrt(d) for d = 1 (mod 4), i sqrt(d) for d = 3 (mod 4)  [Gauss].
#   (6) det F_W = (sqrt(d)/g_d)^d det F = (-1)^{floor(d/4)}  in mu_2:
#       d = 4m+1: zeta = 1,  det F_W = i^{(4m+1)2m}    = (-1)^m;
#       d = 4m+3: zeta = -i, det F_W = i * i^{(2m+3)}  = (-1)^m.
#   (7) det M_a = sign(k -> ak mod d) = Jacobi symbol (a|d)  in mu_2
#       [Zolotarev's lemma, odd-composite extension due to Frobenius].
# ASSEMBLY: det is multiplicative, det(SU closer) = 1, and mu_2 * mu_d = mu_{2d} for odd d
# (gcd(2,d)=1), so every SU-closed loop holonomy is a product of the values above => mu_{2d}.
#
# ATTAINED-GROUP COROLLARY (also verified): -1 is attained (W((1,1)) self-loop), so the
# attained holonomy group is mu_2 x <det S> = mu_6 if 3 | d, mu_2 otherwise; the Weil-section
# bound is EXHAUSTIVE at odd d iff d = 3 (mu_6 = mu_{2d}).
#
# Expected final line: 'cw_weil_probe PASS' (< 30 s).
import numpy as np

DS = (3, 5, 7, 9, 11, 13, 15)
TOL = 1e-9

def gates(d):
    w = np.exp(2j*np.pi/d); tau = np.exp(1j*np.pi/d)
    X = np.roll(np.eye(d), 1, axis=0)
    Z = np.diag([w**k for k in range(d)])
    F = np.array([[w**(j*k) for k in range(d)] for j in range(d)]) / np.sqrt(d)
    S = np.diag([w**((k*(k+1))//2) for k in range(d)])
    g = sum(w**(k*k) for k in range(d))
    FW = (np.sqrt(d)/g) * F
    return w, tau, X, Z, F, S, g, FW

def Mmult(a, d):
    P = np.zeros((d, d))
    for k in range(d): P[(a*k) % d, k] = 1
    return P

def jacobi(a, n):
    """Jacobi symbol (a|n), n odd positive."""
    assert n > 0 and n % 2 == 1
    a %= n; r = 1
    while a:
        while a % 2 == 0:
            a //= 2
            if n % 8 in (3, 5): r = -r
        a, n = n, a
        if a % 4 == 3 and n % 4 == 3: r = -r
        a %= n
    return r if n == 1 else 0

def in_mu(z, n, tol=1e-6):
    # power amplifies float error by ~n, so membership uses a looser tolerance
    # than the exact-value comparisons (which stay at TOL)
    return abs(z**n - 1) < tol

def close(x, y, tol=TOL):
    return abs(x - y) < tol

def check(cond, msg):
    if not cond: raise AssertionError(msg)

def run():
    all_pinned = {}
    for d in DS:
        w, tau, X, Z, F, S, g, FW = gates(d)

        # (1) det X, det Z
        dX, dZ = np.linalg.det(X), np.linalg.det(Z)
        check(close(dX, 1), f"d={d}: det X != 1")
        check(close(dZ, 1), f"d={d}: det Z != 1")

        # (2) det W(v) = (-1)^{ab} for ALL representatives v in {0..d-1}^2
        Xp = [np.linalg.matrix_power(X, a) for a in range(d)]
        Zp = [np.linalg.matrix_power(Z, b) for b in range(d)]
        for a in range(d):
            for b in range(d):
                dW = np.linalg.det(tau**(-(a*b)) * (Xp[a] @ Zp[b]))
                check(close(dW, (-1)**(a*b)),
                      f"d={d}: det W({a},{b}) = {dW:.6f} != (-1)^(ab)")
                check(in_mu(dW, 2*d), f"d={d}: det W({a},{b}) not in mu_2d")

        # (3) det S = omega^{C(d+1,3)}; reduced residue
        T = (d-1)*d*(d+1)//6
        dS = np.linalg.det(S)
        check(close(dS, w**T), f"d={d}: det S != omega^C(d+1,3)")
        red = 0 if d % 3 else ((d//3)*((d-1)//2)) % d
        check(close(dS, w**red), f"d={d}: det S reduced residue wrong")
        check(in_mu(dS, d), f"d={d}: det S not in mu_d")

        # (4) det F = i^{d(d-1)/2}; naive dichotomy
        dF = np.linalg.det(F)
        eF = (d*(d-1)//2) % 4
        check(close(dF, 1j**eF), f"d={d}: det F = {dF:.6f} != i^(d(d-1)/2)")
        naive_ok = in_mu(dF, 2*d)
        check(naive_ok == (d % 4 == 1),
              f"d={d}: naive dichotomy fails (in mu_2d: {naive_ok})")

        # (5) Gauss sum
        gpred = np.sqrt(d) * (1 if d % 4 == 1 else 1j)
        check(close(g, gpred), f"d={d}: Gauss sum g = {g:.6f} != {gpred:.6f}")

        # (6) det F_W = (-1)^{floor(d/4)}
        dFW = np.linalg.det(FW)
        check(close(dFW, (-1)**(d//4)), f"d={d}: det F_W = {dFW:.6f} != (-1)^floor(d/4)")
        check(close(dFW, (np.sqrt(d)/g)**d * dF), f"d={d}: det F_W factorization broken")
        check(in_mu(dFW, 2), f"d={d}: det F_W not in mu_2")
        # F_W is a legitimate section gate: exact order 4
        check(np.allclose(np.linalg.matrix_power(FW, 4), np.eye(d), atol=1e-8),
              f"d={d}: F_W^4 != I")

        # (7) det M_a = Jacobi(a|d) for every unit a (Zolotarev/Frobenius)
        for a in range(1, d):
            if np.gcd(a, d) != 1: continue
            dM = np.linalg.det(Mmult(a, d))
            check(close(dM, jacobi(a, d)), f"d={d}: det M_{a} != (a|d)")

        # (8) friend loop T_Z->T_X->T_XZ->T_Z in the WEIL section, explicit SU closers
        #     (the V47 Part D loop, now at all seven d): hol = det(S F_W) in mu_{2d},
        #     closer-independent, and equal to the generator-table prediction.
        rng = np.random.default_rng(0); vals = set()
        B = S @ FW
        for _ in range(25):
            D = np.diag(np.exp(1j*rng.uniform(0, 2*np.pi, d)))
            P = np.eye(d)[rng.permutation(d)]
            C = D @ P @ B.conj().T
            C = C / np.linalg.det(C)**(1.0/d)
            check(abs(np.linalg.det(C) - 1) < 1e-8, f"d={d}: closer not SU")
            L = C @ S @ FW
            nz = np.abs(L) > 1e-8
            check((nz.sum(axis=0) == 1).all() and (nz.sum(axis=1) == 1).all(),
                  f"d={d}: friend loop not closed")
            vals.add(complex(np.round(np.linalg.det(L), 9)))
        check(len(vals) == 1, f"d={d}: closer-dependence {vals}")
        hol = vals.pop()
        pred = w**red * (-1)**(d//4)          # det S * det F_W
        check(close(hol, pred), f"d={d}: friend hol {hol:.6f} != predicted {pred:.6f}")
        check(in_mu(hol, 2*d), f"d={d}: friend hol not in mu_2d")

        # (9) 300 random loop words in the FULL Weil generating set {W(v), F_W, S, M_a}:
        #     det(word) in mu_{2d} and equal to the product of per-gate closed forms.
        units = [a for a in range(2, d) if np.gcd(a, d) == 1]
        for t in range(300):
            Lw = int(rng.integers(1, 9))
            Mw = np.eye(d, dtype=complex); predw = 1.0 + 0j
            for _ in range(Lw):
                c = int(rng.integers(0, 4))
                if c == 0:
                    a, b = int(rng.integers(0, d)), int(rng.integers(0, d))
                    Mw = Mw @ (tau**(-(a*b)) * (Xp[a] @ Zp[b])); predw *= (-1)**(a*b)
                elif c == 1:
                    Mw = Mw @ FW; predw *= (-1)**(d//4)
                elif c == 2:
                    Mw = Mw @ S; predw *= w**red
                else:
                    a = units[int(rng.integers(0, len(units)))] if units else 1
                    Mw = Mw @ Mmult(a, d); predw *= jacobi(a, d)
            dw = np.linalg.det(Mw)
            check(close(dw, predw, 1e-7), f"d={d}: word det != product of closed forms")
            check(in_mu(dw, 2*d, 1e-7), f"d={d}: word det not in mu_2d")

        # attained-group corollary
        ordS = d // np.gcd(red, d) if red else 1
        attained = 2 * ordS            # mu_2 x <omega^red>, coprime orders
        exhaustive = (attained == 2*d)
        all_pinned[d] = (eF, red, (-1)**(d//4), hol, attained, exhaustive)
        print(f"d={d:2d}: det F = i^{eF}, g/sqrt(d) = {g/np.sqrt(d):.3f}, "
              f"det F_W = {(-1)**(d//4):+d}, det S = omega^{red}, "
              f"friend hol (Weil) = {hol:.6f} in mu_{2*d}: True; "
              f"naive bound {'HOLDS' if d % 4 == 1 else 'VIOLATED (as proven)'}; "
              f"attained = mu_{attained}"
              f"{' = mu_2d (EXHAUSTIVE)' if exhaustive else f' < mu_{2*d}'}")
    return all_pinned

if __name__ == "__main__":
    run()
    print("theorem verified: all Weil-section generator dets in mu_2d at d=3,5,7,9,11,13,15;")
    print("naive-section dichotomy (violation iff d=3 mod 4) verified; "
          "attained group mu_6 iff 3|d else mu_2.")
    print("cw_weil_probe PASS")
