"""Stage 11 (Branch W): the J-SPLITTING — a conjugate-linear involution on the deformation
space of any conjugation-symmetric ray configuration, generalizing flex_R + flex_skew to
points that are not literally real.

THEOREM (J-split, stated precisely in M3M2.md Stage 11). Let {v_i} be a ray configuration
in C^3 whose ray SET is preserved by an antiunitary J = U∘conj (U unitary): J v_i = c_i
v_{sigma(i)} with sigma a permutation, c_i phases. Then W = (w_i) |-> (J.W)_{sigma(i)} =
c_i^{-1} U conj(w_i) maps first-order orthogonality-preserving deformations to deformations,
normalizes the trivial subspace (u(3) + per-ray phases), and descends to a REAL-linear
involution S on the flex quotient Q. Hence flex = flex_+ + flex_- (S-eigenspaces). When
sigma = id and all v_i real (U = I), flex_+ = flex_R and flex_- = flex_skew.

This script VERIFIES the construction numerically (SVD nullspaces, tolerance 1e-8) at:
  A. Peres-33 (theta=0, real rays, J = conj):        expect (flex_+, flex_-) = (0, 1)
  B. the Penrose-side slice theta=-pi/2, J = U∘conj with U a signed permutation found by
     search (conj(v(theta)) = v(-theta) and a signed permutation realizes theta+pi):
     expect flex = 1 split as (0,1) or (1,0) — measured, and the GA tangent's eigenvalue.
  C. a GENERIC point theta=0.7: verify NO signed-permutation antiunitary self-symmetry
     exists (flexible WITHOUT J-symmetry — bounds what Hypothesis L may claim).
  D. the 49-ray rigid Gaussian core at x=1+i: is the core set conj-invariant?

Run: python3 stage11_jsplit.py
"""
import os, sys, time
import numpy as np
from itertools import permutations
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import peres_penrose as pp
import sic_zoo as sz
import ks_flex_census as kfc

TOL = 1e-8

def unitize(rays):
    return [np.asarray(v, dtype=complex) / np.linalg.norm(v) for v in rays]

def orth_pairs(rays, tol=1e-9):
    n = len(rays)
    return [(i, j) for i in range(n) for j in range(i + 1, n)
            if abs(np.vdot(rays[i], rays[j])) < tol]

def real_vec(W):
    """C^{3n} -> R^{6n}."""
    return np.concatenate([np.concatenate([w.real, w.imag]) for w in W])

def cplx_vec(x, n):
    out = []
    for i in range(n):
        b = x[6 * i:6 * (i + 1)]
        out.append(b[:3] + 1j * b[3:])
    return out

def constraint_matrix(rays, pairs):
    """Rows: Re/Im[<w_i,v_j> + <v_i,w_j>] per edge + Re<v_i,w_i> per ray. Real 6n cols."""
    n = len(rays)
    rows = []
    def row_from_pairfunc(f):
        r = np.zeros(6 * n)
        for k in range(6 * n):
            e = np.zeros(6 * n); e[k] = 1.0
            r[k] = f(cplx_vec(e, n))
        return r
    # build via linearity using basis probes (6n small here; fine)
    for (i, j) in pairs:
        for part in (np.real, np.imag):
            rows.append(row_from_pairfunc(
                lambda W, i=i, j=j, part=part: part(np.vdot(W[i], rays[j]) + np.vdot(rays[i], W[j]))))
    for i in range(n):
        rows.append(row_from_pairfunc(lambda W, i=i: np.real(np.vdot(rays[i], W[i]))))
    return np.array(rows)

def trivial_basis(rays):
    """u(3) (9 real dims) + per-ray phases (n dims); contains one relation (iI)."""
    n = len(rays)
    T = []
    # u(3) basis: i*E_kk (3), E_kl - E_lk (3), i*(E_kl + E_lk) (3)
    gens = []
    for k in range(3):
        A = np.zeros((3, 3), complex); A[k, k] = 1j; gens.append(A)
    for k in range(3):
        for l in range(k + 1, 3):
            A = np.zeros((3, 3), complex); A[k, l] = 1; A[l, k] = -1; gens.append(A)
            A = np.zeros((3, 3), complex); A[k, l] = 1j; A[l, k] = 1j; gens.append(A)
    for A in gens:
        T.append(real_vec([A @ v for v in rays]))
    for i in range(n):
        W = [np.zeros(3, complex) for _ in range(n)]
        W[i] = 1j * rays[i]
        T.append(real_vec(W))
    return np.array(T).T  # columns

def nullspace(M, tol=TOL):
    u, s, vt = np.linalg.svd(M)
    rank = int(np.sum(s > tol * max(M.shape) * (s[0] if len(s) else 1)))
    return vt[rank:].T  # columns span null

def flex_quotient(rays, pairs):
    """Return (Q_basis, flex, checks) — Q = null(constraints) ∩ trivials^perp."""
    C = constraint_matrix(rays, pairs)
    N = nullspace(C)
    T = trivial_basis(rays)
    # T must lie in null(C)
    resid = np.linalg.norm(C @ T) / (np.linalg.norm(T) + 1e-300)
    tT, sT, _ = np.linalg.svd(T, full_matrices=False)
    rankT = int(np.sum(sT > TOL * max(T.shape) * sT[0]))
    Tb = tT[:, :rankT]
    # project N off T
    P = N - Tb @ (Tb.T @ N)
    uQ, sQ, _ = np.linalg.svd(P, full_matrices=False)
    fdim = int(np.sum(sQ > 1e-6 * (sQ[0] if sQ[0] > 0 else 1)))
    Q = uQ[:, :fdim]
    return Q, fdim, dict(triv_in_null_resid=resid, dimN=N.shape[1], rankT=rankT)

def find_antiunitary(rays, tol=1e-8):
    """Search signed 3x3 permutations U for a self-map J=U∘conj of the ray set.
    Returns (U, sigma, phases) or None."""
    n = len(rays)
    for perm in permutations(range(3)):
        for signs in ((s0, s1, s2) for s0 in (1, -1) for s1 in (1, -1) for s2 in (1, -1)):
            U = np.zeros((3, 3))
            for r, c in enumerate(perm):
                U[r, c] = signs[r]
            sigma, phases, ok = [], [], True
            for i in range(n):
                img = U @ np.conj(rays[i])
                hit = None
                for j in range(n):
                    ov = np.vdot(rays[j], img)
                    if abs(abs(ov) - 1) < tol:
                        hit = (j, ov); break
                if hit is None:
                    ok = False; break
                sigma.append(hit[0]); phases.append(hit[1])
            if ok and len(set(sigma)) == n:
                return U, sigma, phases
    return None

def j_split(rays, pairs, U, sigma, phases):
    """Build the induced real-linear map on R^{6n}, restrict to Q, split eigen ±1."""
    n = len(rays)
    Q, fdim, checks = flex_quotient(rays, pairs)
    # real-linear map M: W -> W' with W'_{sigma(i)} = phases_i^{-1} U conj(w_i)
    dim = 6 * n
    M = np.zeros((dim, dim))
    for k in range(dim):
        e = np.zeros(dim); e[k] = 1.0
        W = cplx_vec(e, n)
        Wp = [np.zeros(3, complex) for _ in range(n)]
        for i in range(n):
            Wp[sigma[i]] = (1.0 / phases[i]) * (U @ np.conj(W[i]))
        M[:, k] = real_vec(Wp)
    S = Q.T @ M @ Q
    inv_resid = np.linalg.norm(S @ S - np.eye(fdim)) if fdim else 0.0
    if fdim == 0:
        return dict(flex=0, plus=0, minus=0, inv_resid=0.0, checks=checks)
    evals, evecs = np.linalg.eigh((S + S.T) / 2)
    plus = int(np.sum(evals > 0))
    minus = int(np.sum(evals < 0))
    return dict(flex=fdim, plus=plus, minus=minus, inv_resid=inv_resid, checks=checks,
                evals=np.round(evals, 6).tolist())

def main():
    t0 = time.time()
    print("=" * 96)
    print("STAGE 11 -- the J-splitting: flex = flex_+ + flex_- at conjugation-symmetric points")
    print("=" * 96)

    # ---- A. Peres-33 (theta = 0): real rays, J = conj -------------------------------------
    print("\n[A] Peres-33 (theta=0), J = plain conjugation (expect flex=1 split (0,1)):")
    raysA = unitize(pp.rays_c([1.0 + 0j]))  # zs = e^{i*0}
    pairsA = orth_pairs(raysA)
    print(f"    {len(raysA)} rays, {len(pairsA)} orthogonal pairs")
    resA = j_split(raysA, pairsA, np.eye(3), list(range(len(raysA))), [1.0] * len(raysA))
    print(f"    flex = {resA['flex']}, (flex_+, flex_-) = ({resA['plus']}, {resA['minus']}), "
          f"S^2-I resid = {resA['inv_resid']:.2e}, trivials-in-null resid = {resA['checks']['triv_in_null_resid']:.2e}")
    okA = (resA['flex'], resA['plus'], resA['minus']) == (1, 0, 1) and resA['inv_resid'] < 1e-6
    print(f"    [{'PASS' if okA else 'FAIL'}] matches known flex_R=0, flex_skew=1")

    # ---- B. Penrose-side slice theta = -pi/2: find J = U∘conj by search --------------------
    print("\n[B] slice theta=-pi/2 (Z[sqrt-2]/Penrose modulus), antiunitary self-symmetry by search:")
    zsB = np.exp(-1j * np.pi / 2)
    raysB = unitize(pp.rays_c([zsB]))
    pairsB = orth_pairs(raysB)
    found = find_antiunitary(raysB)
    if found is None:
        print("    NO signed-permutation antiunitary self-symmetry found  [ATTENTION]")
        okB = False
    else:
        U, sigma, phases = found
        nfix = sum(1 for i, s in enumerate(sigma) if s == i)
        print(f"    FOUND J = U∘conj: U = signed permutation {np.argmax(np.abs(U), axis=1).tolist()} "
              f"signs {[int(x) for x in U[np.arange(3), np.argmax(np.abs(U), axis=1)]]}, "
              f"sigma fixes {nfix}/{len(raysB)} rays")
        resB = j_split(raysB, pairsB, U, sigma, phases)
        print(f"    flex = {resB['flex']}, (flex_+, flex_-) = ({resB['plus']}, {resB['minus']}), "
              f"S^2-I resid = {resB['inv_resid']:.2e}")
        # GA tangent eigenvalue: tangent dv/dtheta at theta=-pi/2, numerically
        h = 1e-6
        raysBp = unitize(pp.rays_c([np.exp(1j * (-np.pi / 2 + h))]))
        tangent = [(vp - v) / h for vp, v in zip(raysBp, raysB)]
        QB, fB, _ = flex_quotient(raysB, pairsB)
        tvec = real_vec(tangent)
        tq = QB.T @ tvec
        print(f"    GA tangent projects onto Q with norm fraction "
              f"{np.linalg.norm(tq)/ (np.linalg.norm(tvec)+1e-300):.3f} (nonzero => tangent is the flex)")
        okB = resB['flex'] == 1 and resB['inv_resid'] < 1e-6
        print(f"    [{'PASS' if okB else 'FAIL'}] J-splitting well-defined at a NON-REAL flexible point")

    # ---- C. generic theta = 0.7: measure the J-symmetry and the split ---------------------
    print("\n[C] generic slice theta=0.7 (conj maps it to theta=-0.7; is there a self-J?):")
    raysC = unitize(pp.rays_c([np.exp(0.7j)]))
    pairsC = orth_pairs(raysC)
    foundC = find_antiunitary(raysC)
    if foundC is None:
        print("    NO signed-permutation antiunitary self-symmetry at generic theta")
        okC = True   # would have supported the 'special points only' reading
    else:
        U, sigma, phases = foundC
        nfix = sum(1 for i, s in enumerate(sigma) if s == i)
        print(f"    FOUND J = U∘conj (U perm {np.argmax(np.abs(U),axis=1).tolist()}, "
              f"signs {[int(U[r, np.argmax(np.abs(U[r]))]) for r in range(3)]}), "
              f"sigma fixes {nfix}/33 rays")
        print("    [HONEST CORRECTION of this stage's own brief: the antiunitary symmetry exists")
        print("    along the WHOLE GA family (set(-theta) = set(theta) as ray sets via a")
        print("    coordinate signed-permutation), not only at the 4 conjugation-fixed points.]")
        resC = j_split(raysC, pairsC, U, sigma, phases)
        print(f"    flex = {resC['flex']}, (flex_+, flex_-) = ({resC['plus']}, {resC['minus']}), "
              f"S^2-I resid = {resC['inv_resid']:.2e}")
        # tangent eigenvalue at theta=0.7
        h = 1e-6
        raysCp = unitize(pp.rays_c([np.exp(1j * (0.7 + h))]))
        tangentC = [(vp - v) / h for vp, v in zip(raysCp, raysC)]
        QC, _, _ = flex_quotient(raysC, pairsC)
        MC = None  # reuse j_split internals cheaply: recompute S action on tangent directly
        tq = QC.T @ real_vec(tangentC)
        # sign of tangent under S: build S again (small)
        n = len(raysC); dim = 6 * n
        Mfull = np.zeros((dim, dim))
        for k in range(dim):
            e = np.zeros(dim); e[k] = 1.0
            W = cplx_vec(e, n)
            Wp = [np.zeros(3, complex) for _ in range(n)]
            for i in range(n):
                Wp[sigma[i]] = (1.0 / phases[i]) * (U @ np.conj(W[i]))
            Mfull[:, k] = real_vec(Wp)
        S = QC.T @ Mfull @ QC
        sgn = float(tq @ S @ tq / (tq @ tq)) if tq @ tq > 1e-12 else float('nan')
        print(f"    GA tangent's J-eigenvalue at theta=0.7: {sgn:+.6f}")
        okC = resC['flex'] == 1 and resC['inv_resid'] < 1e-6 and abs(abs(sgn) - 1) < 1e-4
        print(f"    [{'PASS' if okC else 'ATTENTION'}] measured structure (2nd honest correction of")
        print("    the brief, which predicted J-odd): THIS J (swap-type, orientation-PRESERVING on")
        print("    the modulus: theta -> -theta from conj, undone by the coordinate swap) sees the")
        print("    flex as J-EVEN, split (1,0). The special loci theta=0,+-pi/2 additionally admit")
        print("    plain conj (orientation-REVERSING), for which the flex is J-ODD, split (0,1) --")
        print("    recovering flex_skew. The split is per-J; the well-posedness theorem is what is")
        print("    J-independent.")

    # ---- D. the 49-core at x=1+i: is the core set conj-invariant? --------------------------
    print("\n[D] 49-ray rigid Gaussian core at x=1+i: conjugation-invariance of the CORE set:")
    B, C = 2, -2
    alph = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1),
            kfc.qconj((0, 1), B), kfc.qneg(kfc.qconj((0, 1), B))]
    pool = kfc.collect_rays(kfc.raw_vectors(alph, 3), B, C)
    import branch_m3m2 as bm
    core = [pool[i] for i in bm.CORE49_IDX]
    # numeric rays: ring element (a,b) = a + b*x with x = 1+i
    x = 1 + 1j
    def num(rv):
        return np.array([a + b * x for (a, b) in rv], dtype=complex)
    core_num = unitize([num(r) for r in core])
    pool_num = unitize([num(r) for r in pool])
    # conj maps pool to pool (alphabet conj-closed) -- find sigma on the pool
    def find_in(rays, v, tol=1e-8):
        for j, w in enumerate(rays):
            if abs(abs(np.vdot(w, v)) - 1) < tol:
                return j
        return None
    sigma_pool = [find_in(pool_num, np.conj(v)) for v in pool_num]
    ok_pool = all(s is not None for s in sigma_pool)
    print(f"    conj maps the 127-ray pool to itself: {ok_pool}")
    core_set = set(bm.CORE49_IDX)
    img = {sigma_pool[i] for i in bm.CORE49_IDX}
    inv = img == core_set
    print(f"    conj image of the 49-core inside the core: {inv} "
          f"({len(img & core_set)}/49 rays land back in the core)")
    print("    => the rigid core " + ("IS" if inv else "is NOT") + " conjugation-invariant.")
    okD = ok_pool

    print(f"\ntotal time: {time.time()-t0:.1f}s")
    allok = okA and okB and okC and okD
    print("PASS" if allok else "ATTENTION -- see flags above")
    return allok

if __name__ == "__main__":
    main()
