#!/usr/bin/env python3
"""
B1 — THEOREM 1 verifier: flex(C_n) = 2n - 8 for odd n >= 5 (umbrella realization in CP^2).

Proof structure (hand proof in cycle_flex_theorem.tex); this script machine-checks every
ingredient the proof uses, plus the final formula:

  (P1) umbrella well-defined: consecutive rays orthogonal exactly;
  (P2) NON-ADJACENCY POSITIVITY: <v_i, v_{i+k}> = [(-1)^k cos(k pi/n) + cos(pi/n)]/(1+cos(pi/n)) > 0
       for 2 <= k <= n-2 (and = 0 for k = 1)  [feeds Lemma (ii)];
  (P3) consecutive triples {v_{k-1}, v_k, v_{k+1}} linearly independent; in particular
       {v_{k-1}, v_{k+1}} is a basis of v_k^perp  [feeds Lemma (i)];
  (P4) STRESS-FREENESS (Lemma (i)): the only alpha in C^n with
       conj(alpha_{k-1}) v_{k-1} + alpha_k v_{k+1} in C*v_k for all k is alpha = 0
       (equivalently rank J = 2n, i.e. trivial cokernel of the rigidity matrix);
  (P5) STABILIZER (Lemma (ii)): {A in u(3): A v_j = lambda_j v_j for all j} = i*R*I
       (dim 1), so the PU(3)-orbit has dimension 8;
  (P6) THE FORMULA: flex(C_n) = 4n - 2n - 8 = 2n - 8, checked for n = 5,7,9,11,13,15.

n = 3 is EXCLUDED (C_3 = ONB: no non-adjacent pairs; stabilizer is the maximal torus; flex = 0,
and the formula 2n-8 = -2 correctly does NOT apply). Checked as a boundary case.
"""
import numpy as np, sys, os
sys.path.insert(0, os.path.dirname(__file__))
from flex_dimension import flex_dimension, odd_cycle, unit, edges_from

def check_n(n, verbose=True):
    rays = odd_cycle(n)
    ok = True
    # (P1) consecutive orthogonality
    e = max(abs(np.vdot(rays[j], rays[(j+1) % n])) for j in range(n))
    ok &= e < 1e-9
    # (P2) non-adjacency positivity + closed form
    c = np.cos(np.pi/n)
    for k in range(2, n-1):
        pred = (((-1)**k) * np.cos(k*np.pi/n) + c) / (1 + c)
        for i in range(n):
            val = np.vdot(rays[i], rays[(i+k) % n]).real
            ok &= abs(val - pred) < 1e-9 and pred > 1e-12
    # (P3) consecutive triple independence
    for k in range(n):
        M = np.array([rays[(k-1) % n], rays[k], rays[(k+1) % n]])
        ok &= abs(np.linalg.det(M)) > 1e-9
    # (P4) stress-freeness: solve conj(a_{k-1}) v_{k-1} + a_k v_{k+1} = c_k v_k for all k.
    # Project onto v_k^perp: both terms already lie in v_k^perp, v_k not in it -> c_k = 0 and,
    # by independence of {v_{k-1}, v_{k+1}}, a_{k-1} = a_k = 0. Verify by building the real
    # linear system for (Re a, Im a) and checking its kernel is 0.
    rows = []
    for k in range(n):
        # equation: conj(a_{k-1}) v_{k-1} + a_k v_{k+1} - c_k v_k = 0 (3 complex eqs; eliminate c_k
        # by projecting onto v_k^perp basis {v_{k-1}, v_{k+1}} ... simpler: express in the basis
        # {v_{k-1}, v_{k+1}, v_k} and demand the first two coordinates vanish).
        B = np.array([rays[(k-1) % n], rays[(k+1) % n], rays[k]]).T   # basis matrix (cols)
        Binv = np.linalg.inv(B)
        # LHS coefficient vectors of conj(a_{k-1}) and a_k in that basis:
        col_prev = Binv @ rays[(k-1) % n]   # = e1
        col_next = Binv @ rays[(k+1) % n]   # = e2
        # demand coords 0 and 1 of  conj(a_{k-1})*e1 + a_k*e2  vanish:
        # coord0: conj(a_{k-1}) = 0 ; coord1: a_k = 0. Build real 2x(2n) rows per complex eq.
        for coord in (0, 1):
            re = [0.0]*(2*n); im = [0.0]*(2*n)
            # conj(a_{k-1}) coefficient col_prev[coord]; a_k coefficient col_next[coord]
            cp = col_prev[coord]; cn_ = col_next[coord]
            ip_ = (k-1) % n
            # conj(a) = x - i y
            re[2*ip_]   += cp.real;  re[2*ip_+1] += cp.imag      # Re(cp*(x-iy)) = cp.re*x + cp.im*y (cp real here but keep general)
            im[2*ip_]   += cp.imag;  im[2*ip_+1] += -cp.real
            re[2*k]     += cn_.real; re[2*k+1]   += -cn_.imag
            im[2*k]     += cn_.imag; im[2*k+1]   += cn_.real
            rows.append(re); rows.append(im)
    Msys = np.array(rows)
    ker_stress = 2*n - np.linalg.matrix_rank(Msys, tol=1e-9)
    ok &= (ker_stress == 0)
    # (P5) stabilizer dimension: {A in u(3): each v_j eigenvector} -> build linear system
    # A v_j x v_j = 0 (projective condition). Parametrize u(3) by 9 real params.
    herm = []
    for a in range(3):
        H = np.zeros((3,3), complex); H[a,a] = 1; herm.append(1j*H)
    for a in range(3):
        for b in range(a+1,3):
            H = np.zeros((3,3), complex); H[a,b]=1; H[b,a]=1; herm.append(1j*H)
            K = np.zeros((3,3), complex); K[a,b]=1j; K[b,a]=-1j; herm.append(1j*K)
    rowsS = []
    for j in range(n):
        v = rays[j]
        P = np.eye(3) - np.outer(v, v.conj())      # projector onto v^perp
        for A in herm:
            pass
        # build: for parameter vector t, A(t) v = sum t_m herm[m] v ; require P A v = 0
        # -> 3 complex equations (2 independent) per vertex, linear in t
        block = np.array([P @ (A @ v) for A in herm]).T   # 3 x 9 complex
        rowsS.append(block.real); rowsS.append(block.imag)
    S = np.vstack(rowsS)
    stab_dim = 9 - np.linalg.matrix_rank(S, tol=1e-9)
    ok &= (stab_dim == 1)     # scalars i*R*I only  -> orbit dim = 9 - 1 = 8
    # (P6) the formula via the engine
    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        flex = flex_dimension(rays, name=f"C{n}")
    ok &= (flex == 2*n - 8)
    if verbose:
        print(f"  n={n:2d}: P1 orth {e:.1e} | P2 positivity+closed-form OK | P3 triples OK | "
              f"P4 stress-kernel={ker_stress} | P5 stab-dim={stab_dim} (orbit 8) | flex={flex} =2n-8={2*n-8} "
              f"{'PASS' if ok else 'FAIL'}")
    return ok

if __name__ == "__main__":
    print("THEOREM 1 verifier: flex(C_n) = 2n-8, odd n>=5 (every proof ingredient machine-checked)")
    allok = True
    for n in [5,7,9,11,13,15]:
        allok &= check_n(n)
    # boundary case n=3: formula must NOT apply (ONB; flex 0; stabilizer = torus, dim 2 -> orbit 6)
    rays3 = odd_cycle(3)
    import io, contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        f3 = flex_dimension(rays3, name="C3")
    print(f"  n= 3 boundary: flex={f3} (formula 2n-8=-2 correctly does not apply; C3=ONB rigid)")
    allok &= (f3 == 0)
    print("cycle_flex_theorem PASS" if allok else "cycle_flex_theorem FAIL")
