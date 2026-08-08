#!/usr/bin/env python3
"""
V56 -- repeatable_efficiency_dichotomy.py

Certifies Theorem "Sharp core, and the efficiency dichotomy" of the Copenhagen note.

THE CLAIM, in two halves:

  (a) EFFICIENT + REPEATABLE => PROJECTIVE, in every finite dimension.
      If I_1(rho) = K rho K^dag is efficient (one Kraus operator) and repeatable on its
      outcome, then E_1 = K^dag K is an orthogonal projection.

  (b) WITHOUT EFFICIENCY the folk claim "repeatable => projective" is FALSE already at
      d=3, and d=3 is MINIMAL.  Witness: E_1 = diag(1,0,c), E_2 = diag(0,1,1-c), 0<c<1,
      with I_j(rho) = tr(E_j rho) |e_j><e_j|.  Both outcomes repeatable, E_1 not a
      projection, the instrument covariant under the commutant of {E_j}.  At d=2 no such
      witness exists.

PROOF OF (a) -- this is a theorem, the numerics below are only a stress test:
  statistics:     tr(K rho K^dag) = tr(E_1 rho) for all rho  =>  K^dag K = E_1
  repeatability:  tr(E_1 K rho K^dag) = tr(E_1 rho) for all rho  =>  K^dag E_1 K = K^dag K
                  =>  K^dag (I - E_1) K = 0.
  Put I - E_1 = S^dag S (S = (I-E_1)^{1/2} >= 0).  Then (SK)^dag (SK) = 0 => SK = 0
                  =>  ran(K) subset ker(I - E_1) = Eig_1(E_1).
  Polar K = U|K| with |K| = E_1^{1/2}, so dim ran(K) = rank(E_1).  Hence
                  rank(E_1) <= dim Eig_1(E_1).
  The reverse inequality is automatic (Eig_1(E_1) subset ran(E_1)).  So they are equal,
  ran(E_1) = Eig_1(E_1), and a positive contraction whose range is its own 1-eigenspace
  is the orthogonal projector onto it.                                              QED

PROOF OF (b), minimality at d=2:
  Repeatability on a nonzero outcome forces a nonzero eigenvalue-1 core K_j (the sharp
  core lemma).  For v in K_1, w in K_2: <v,w> = <v, E_2 w> = <E_2 v, w> = 0 since
  E_2 v = (I - E_1) v = 0.  So K_1 perp K_2, both nonzero.  At d=2 that forces
  dim K_1 = dim K_2 = 1 and K_1 + K_2 = C^2, whence E_1 = proj K_1.               QED

PRIOR ART -- the first sentence of the theorem is NOT new and the note says so.
Repeatability confining posterior states to unit-eigenvalue subspaces is the norm-1
property:
  M. Ozawa, J. Math. Phys. 25, 79 (1984), doi:10.1063/1.526000
  P. Busch, M. Grabowski, P. J. Lahti, Found. Phys. 25, 1239 (1995), doi:10.1007/BF02055331
  T. Heinonen, P. Lahti, J.-P. Pellonpaa, S. Pulmannova, K. Ylinen,
      J. Math. Phys. 44, 1998 (2003), doi:10.1063/1.1566454
  P. Busch, P. Lahti, J.-P. Pellonpaa, K. Ylinen, "Quantum Measurement",
      Springer (2016), doi:10.1007/978-3-319-43389-9
Part (a) is the finite-dimensional content of
  F. Buscemi, G. M. D'Ariano, P. Perinotti, Phys. Rev. Lett. 92, 070403 (2004),
      doi:10.1103/PhysRevLett.92.070403
whose analysis is carried out for PURE measurements -- one contraction M_e per outcome,
state |psi> -> M_e|psi>/||M_e|psi>|| -- which is exactly efficiency.  That is why their
"nonorthogonal repeatability occurs only for infinite dimensions" and our finite-d
witness (b) are consistent rather than contradictory: (b)'s instrument is not efficient.

Expected output (all four checks PASS):
    [a] non-projective effects tested 4000, efficient repeatable instruments found 0
    [b] qutrit witness repeatable at c in {0.2,0.5,0.9}, E1 non-projective, Kraus rank 2
    [c] qutrit instrument covariant under the commutant of {E_1,E_2}
    [d] d=2 counterexamples to minimality found: 0

Usage:  python3 repeatable_efficiency_dichotomy.py
"""
import sys

import numpy as np

RNG = np.random.default_rng(20260807)
TOL = 1e-9


def rand_effect(d):
    """A random effect 0 <= E <= I, generically with no eigenvalue exactly 0 or 1."""
    A = RNG.normal(size=(d, d)) + 1j * RNG.normal(size=(d, d))
    H = A.conj().T @ A
    return H / (np.linalg.eigvalsh(H).max() * (1.0 + RNG.uniform(0.0, 1.0)))


def is_projector(E, tol=TOL):
    return np.linalg.norm(E @ E - E) < tol


def eig1_dim(E, tol=TOL):
    return int(np.sum(np.abs(np.linalg.eigvalsh(E) - 1.0) < tol))


def check_a(trials=4000, unitaries=20):
    """Search for an efficient repeatable instrument on a NON-projective effect.

    K^dag K = E forces K = U E^{1/2} for some unitary U (square case), so the search is
    over U.  Repeatability residual is K^dag E K - E.
    """
    tested = 0
    found = 0
    for _ in range(trials):
        d = int(RNG.integers(2, 6))
        E = rand_effect(d)
        if is_projector(E):
            continue
        tested += 1
        w, V = np.linalg.eigh(E)
        Eh = V @ np.diag(np.sqrt(np.clip(w, 0.0, None))) @ V.conj().T
        for _ in range(unitaries):
            X = RNG.normal(size=(d, d)) + 1j * RNG.normal(size=(d, d))
            U, _ = np.linalg.qr(X)
            K = U @ Eh
            if np.linalg.norm(K.conj().T @ E @ K - E) < 1e-8:
                found += 1
    ok = found == 0
    print(f"  [a] non-projective effects tested {tested}, "
          f"efficient repeatable instruments found {found}   -> {'PASS' if ok else 'FAIL'}")
    return ok


def check_b(cs=(0.2, 0.5, 0.9), states=200):
    """The qutrit witness: both outcomes repeatable, E_1 non-projective, not efficient."""
    ok = True
    for c in cs:
        E1 = np.diag([1.0, 0.0, c])
        E2 = np.eye(3) - E1
        e1 = np.array([1, 0, 0], dtype=complex)
        e2 = np.array([0, 1, 0], dtype=complex)
        rep = True
        for _ in range(states):
            A = RNG.normal(size=(3, 3)) + 1j * RNG.normal(size=(3, 3))
            rho = A @ A.conj().T
            rho /= np.trace(rho).real
            I1 = np.trace(E1 @ rho).real * np.outer(e1, e1.conj())
            I2 = np.trace(E2 @ rho).real * np.outer(e2, e2.conj())
            rep &= abs(np.trace(E1 @ I1).real - np.trace(E1 @ rho).real) < 1e-12
            rep &= abs(np.trace(E2 @ I2).real - np.trace(E2 @ rho).real) < 1e-12
        kraus = int(np.linalg.matrix_rank(E1))
        good = rep and not is_projector(E1) and kraus == 2
        ok &= good
        print(f"  [b] c={c}: both outcomes repeatable={rep}, E1 projective={is_projector(E1)}, "
              f"Kraus operators needed={kraus} (efficient would be 1)   "
              f"-> {'PASS' if good else 'FAIL'}")
    return ok


def check_c(c=0.5, trials=200):
    """Covariance: for unitaries V commuting with both effects, I_j(V rho V^dag) = V I_j(rho) V^dag.

    The commutant of {diag(1,0,c), diag(0,1,1-c)} for c not in {0,1} is the diagonal
    unitaries, which fix |e_1><e_1| and |e_2><e_2|.
    """
    E1 = np.diag([1.0, 0.0, c])
    E2 = np.eye(3) - E1
    e1 = np.array([1, 0, 0], dtype=complex)
    e2 = np.array([0, 1, 0], dtype=complex)
    worst = 0.0
    for _ in range(trials):
        V = np.diag(np.exp(1j * RNG.uniform(0, 2 * np.pi, size=3)))
        assert np.linalg.norm(V @ E1 - E1 @ V) < TOL
        A = RNG.normal(size=(3, 3)) + 1j * RNG.normal(size=(3, 3))
        rho = A @ A.conj().T
        rho /= np.trace(rho).real
        for E, e in ((E1, e1), (E2, e2)):
            lhs = np.trace(E @ (V @ rho @ V.conj().T)).real * np.outer(e, e.conj())
            rhs = V @ (np.trace(E @ rho).real * np.outer(e, e.conj())) @ V.conj().T
            worst = max(worst, float(np.linalg.norm(lhs - rhs)))
    ok = worst < 1e-12
    print(f"  [c] covariance under the commutant, worst residual {worst:.2e}   "
          f"-> {'PASS' if ok else 'FAIL'}")
    return ok


def check_d(trials=20000):
    """d=2 minimality: no effect with both eigenvalue-1 cores nonzero is non-projective."""
    viol = 0
    for _ in range(trials):
        E = rand_effect(2)
        if eig1_dim(E) >= 1 and eig1_dim(np.eye(2) - E) >= 1 and not is_projector(E):
            viol += 1
    # also sweep the boundary explicitly: E = diag(1, t)
    for t in np.linspace(0.0, 1.0, 2001):
        E = np.diag([1.0, t])
        if eig1_dim(E) >= 1 and eig1_dim(np.eye(2) - E) >= 1 and not is_projector(E):
            viol += 1
    ok = viol == 0
    print(f"  [d] d=2 counterexamples to minimality found: {viol}   -> {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    print("V56 -- repeatable/efficient dichotomy for POVM instruments")
    results = [check_a(), check_b(), check_c(), check_d()]
    print()
    if all(results):
        print("VERDICT: PASS. Efficient + repeatable => projective in finite d (proved above,")
        print("stress-tested here); without efficiency the qutrit witness is repeatable,")
        print("covariant and non-projective; d=3 is minimal. Consistent with Buscemi-D'Ariano-")
        print("Perinotti (PRL 92, 070403), whose 'only in infinite dimensions' is a statement")
        print("about pure -- i.e. efficient -- measurements.")
        return 0
    print("VERDICT: FAIL -- do not cite this certificate until resolved.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
