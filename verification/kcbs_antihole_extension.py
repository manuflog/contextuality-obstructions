# V50 - C7/C9/ANTIHOLE EXTENSION OF V49: the twisted spectral detector generalizes;
# the 'torsion shadow' is the reflection identity 2I-G (an alpha=2 antihole-family law,
# not an odd-cycle law). See INDEX.md. Expected: 'C7 PROBE PASS' (~2 s).
# c7_probe.py -- V50 candidate: C7 / C9 / antihole extension of V49 (kcbs_holonomy_home.py).
# Companion to the V49 pentagon result. Expected final line: 'C7 PROBE PASS'.
# Usage: python3 c7_probe.py [A B C D E F]   (default: all parts; each part < 40 s)
#
# Part A (SCENARIO + NO-GO TRANSFER, C7): 29 independent sets (Lucas L7); the
#   C7-optimal model (standard d=3 odd-cycle realization, Lovasz/CSW; classical bound
#   (n-1)/2 = 3, quantum max n cos(pi/n)/(1+cos(pi/n)) = theta(C7), cf. Araujo-Quintino-
#   Budroni-Cunha-Cabello PRA 88, 022118 (2013); Cabello-Severini-Winter PRL 112, 040401
#   (2014); Lovasz IEEE IT-25 (1979)) has FULL support identical to a noncontextual
#   mixture and is not logically contextual, yet CF = 2(S-3) > 0: both V49 no-gos
#   (support-presheaf and scenario-complex cohomology) transfer verbatim.
#
# Part B (FACET UNIQUENESS, C7): CF(x) = 2*max(0, S-3) exactly, S = sum_k <P_k>, on
#   random qutrit states AND on random rank-1 realizations in d = 3,4,5,7 (C7 is
#   t-perfect: STAB(C7) = nonneg + edge + single odd-cycle facet; the LP confirms the
#   cycle facet is the unique quantum-operative one, with ABM normalization
#   (n/2 - (n-1)/2) = 1/2, hence the factor 2). Consequence: the realization admits a
#   contextual state iff lambda_max(G) > 3, since max_psi S = lambda_max(sum P) =
#   lambda_max(G) -- a SINGLE twisted-adjacency detector still characterizes CF > 0.
#
# Part C (DETECTOR + H^1 GEOMETRY, C7): the non-orthogonality graph is the circulant
#   C7(2,3), H^1 rank = 14 - 7 + 1 = 8 (vs 1 at the pentagon). Pinned:
#   (i) identity lambda_max(sum P) = 1 + lambda_max(holonomy-twisted weighted adjacency);
#   (ii) quantum max theta(C7) at the TRIVIAL class, PSD-critical (rank 3);
#   (iii) PERRON BOUND (proof: lambda_max(B) <= rho(B) <= rho(|B|), Perron-Frobenius):
#   at ANY fixed magnitudes the trivial class maximizes the detector -- holonomy can
#   only DECREASE quantum reach; checked on 200 random draws;
#   (iv) the Hessian of the detector over the 8 holonomy coordinates at the optimum is
#   NEGATIVE DEFINITE of FULL RANK 8 (eigenvalues in [-0.147, -0.011]): every H^1
#   coordinate matters, none is flat -- CF depends on the whole class, not a summary;
#   (v) HOLONOMY DECIDES at fixed magnitudes, as at C5 -- with a pinned twist: on the
#   UNIFORM (circulant-magnitude) slice the bounded search finds NO realizable
#   noncontextual class (best 3.061 > 3; 0/1064 random realizations even reach the
#   contextual trivial-PSD corner; existence open), while at pinned NON-UNIFORM
#   magnitudes a feasibility search converges (restart 0): trivial class lam_max =
#   3.032 CONTEXTUAL vs a twisted class lam_max = 2.909 noncontextual, both PSD with
#   margin 0.013, both realized in C^7, both LP-certified. The C5 single-angle window
#   |phi| < pi/2 does not survive as such; the decision is a joint condition in the
#   8-torus.
#
# Part D (TORSION SHADOW DEMYSTIFIED): the full (-1)-twist of ANY unit-diagonal Gram
#   G = I + A is I - A = 2I - G: its spectrum is the exact reflection 2 - spec(G)
#   (one-line identity). Hence the twist value is 2 - lambda_min(G) = 2 at ANY
#   PSD-critical point of ANY scenario -- V49's pentagon "torsion shadow" is this
#   identity, and its coincidence with the classical bound is the arithmetic accident
#   alpha(C5) = 2. At C7 (alpha = 3): sweep of ALL 256 order-2 classes of
#   H^1(C7(2,3); Z2) at the optimal magnitudes -- NO class lands on the classical
#   bound 3; the all-(-1) class (co-tree pattern 11110101) is the UNIQUE class at 2
#   and is the MINIMUM of the sweep; range [2, 3.152183]; 21 classes keep the
#   detector > 3; the trivial class is the ONLY one that stays PSD.
#
# Part E (C9, MINIMAL): CF = 2(S-4) at the C9-optimal point (LP over the 76 independent
#   sets, Lucas L9) and on 60 random states (dev 2e-16); full twist = 2 - lambda_min
#   = 2 exactly again (classical bound is 4); no circulant order-2 twist lands on 4.
#   Verdict: 'full twist = 2' is the odd-cycle law; 'twist = classical bound' was the
#   pentagon accident.
#
# Part F (WHERE THE SHADOW SURVIVES: the alpha = 2 family): for the C7 ANTIHOLE
#   scenario (exclusivity graph = complement of C7, alpha = 2, non-orthogonality graph
#   = C7 with H^1 rank 1 -- the true pentagon analogue), a contextual rank-1
#   realization is found in d = 5 at lambda_max = 2.109916 = theta(antiC7) =
#   7/theta(C7) = 1 + sec(pi/7) to displayed precision, at TRIVIAL holonomy (Perron),
#   LP-certified CF = (2/3)(S-2) > 0; rank(G) <= 5 < 7 forces lambda_min(G) = 0, so
#   the (-1)-twist value is EXACTLY 2 = alpha = classical bound, and the twist is not
#   realizable (lambda_min(2I-G) = 2 - theta < 0), exactly as V49 Part C(iii): the
#   torsion-shadow law generalizes along the ANTIHOLE (alpha = 2) family, not the
#   cycle family. C5, being self-complementary, is the unique member of both -- the
#   'pentagon accident' resolved as family membership.
#
# GENERAL FORMULATION (proposal; conjectural parts flagged). Data: an exclusivity
#   graph X (vertices = rank-1 events, contexts = edges/cliques), non-orthogonality
#   graph N = complement of X. THEOREM-LEVEL (V49 + this probe + standard results):
#   rank-1 realizations of X mod vertex gauge = (magnitude assignment m on E(N),
#   holonomy class h in H^1(N; U(1))) subject to PSD-realizability of
#   G(m,h) = I + A(m,h); for any state psi in the realization, S(psi) = sum_k <P_k>
#   <= lambda_max(G(m,h)) with equality at the top eigenstate; the trivial class
#   maximizes lambda_max at fixed m (Perron), and sup over realizable (m, trivial)
#   equals theta(X) (Lovasz via CSW). CONJECTURE 1 (verified here for C5, C7, C9;
#   t-perfectness of odd cycles is the mechanism): when STAB(X)'s only quantum-
#   operative facets are the odd-hole inequalities, a realization admits a contextual
#   state iff lambda_max(G(m,h)) > alpha(X), and CF = (S - alpha)/(NSmax - alpha) --
#   ONE holonomy-twisted spectral number decides. For general X (clique/antihole/wheel
#   facets) the honest object is the facet-indexed family of twisted variational
#   functionals; whether a single spectral detector survives is OPEN. CONJECTURE 2
#   (two torsion layers, the program's dictionary): the SAME local system L(m,h) on N
#   carries both sectors -- (U(1) layer, state-dependent, KCBS-type) h has generic
#   order, is invisible to support cohomology (V49 no-gos), and acts as a THRESHOLD
#   modulator of lambda_max: contextuality is a positivity/cone event, decided by
#   where the class sits relative to the PSD wall; (torsion layer, state-independent,
#   AvN-type) when operator relations force h to a nontrivial TORSION class (the
#   Weyl/parity sector of papers B-D, gamma - s* = d/2 ghost-tower law), the exit from
#   the NC polytope happens for EVERY state and 'the class IS the obstruction'. The
#   reflection identity is the bridge instance: the order-2 twist h -> -h acts on
#   spectra as lambda -> 2 - lambda, sending quantum-critical (PSD-critical) points to
#   the classical bound precisely in the alpha = 2 family. Status: dictionary is a
#   proposal; no theorem yet identifies the AvN Z_d class with a holonomy class of a
#   rank-1 local system -- flagged OPEN.
#
# Runtime: full run ~4 s on a laptop; every part well under 40 s.
import sys, itertools
import numpy as np
from scipy.optimize import linprog, minimize

rng = np.random.default_rng(11)
OK = True
def check(name, cond):
    global OK
    print(("  PASS " if cond else "  FAIL ") + name)
    OK = OK and bool(cond)

# ---------- generic scenario machinery (contexts = exclusive pairs) ----------
OUT = [(1,0),(0,1),(0,0)]

def indsets(nv, edges):
    return [a for a in itertools.product([0,1], repeat=nv)
            if all(a[i]*a[j] == 0 for (i,j) in edges)]

def build_M(contexts, sets):
    M = np.zeros((3*len(contexts), len(sets)))
    for col, a in enumerate(sets):
        for c, (i,j) in enumerate(contexts):
            M[3*c + OUT.index((a[i], a[j])), col] = 1.0
    return M

def CF_factory(contexts, M):
    ones = np.ones(M.shape[1])
    def CF(x):
        e = []
        for (i,j) in contexts:
            e += [x[i], x[j], 1.0 - x[i] - x[j]]
        e = np.maximum(np.array(e), 0.0)
        r = linprog(-ones, A_ub=M, b_ub=e, bounds=(0, None), method="highs")
        assert r.status == 0
        return 1.0 + r.fun
    return CF

def cn_vectors(n):
    """standard odd-n-cycle optimal realization in R^3 (repo convention at n=5)."""
    c = np.cos(np.pi/n); ct2 = c/(1+c); st = np.sqrt(1-ct2)
    return np.array([[st*np.cos(np.pi*k*(n-1)/n), st*np.sin(np.pi*k*(n-1)/n),
                      np.sqrt(ct2)] for k in range(n)])

def realize(G):
    """rows of W: vectors in C^n with <v_i, v_j> = G[i,j] (needs G PSD)."""
    lam, U = np.linalg.eigh(G)
    assert lam.min() > -1e-9
    return (U*np.sqrt(np.clip(lam, 0, None))).conj()

def top_state_x(W):
    """x_k = |<v_k|psi>|^2 at the top eigenstate of sum_k P_k."""
    Sm = W.T @ W.conj()
    _, U2 = np.linalg.eigh(Sm)
    psi = U2[:, -1]
    return np.abs(W.conj() @ psi)**2

# ---------- C7 scenario ----------
CTX7 = [(k, (k+1) % 7) for k in range(7)]
SETS7 = indsets(7, CTX7)
M7 = build_M(CTX7, SETS7)
CF7 = CF_factory(CTX7, M7)
V7 = cn_vectors(7); G0 = V7 @ V7.T
c7 = np.cos(np.pi/7); TH7 = 7*c7/(1+c7)                      # theta(C7) = 3.31766...
E2 = [(k, (k+2) % 7) for k in range(7)]
E3 = [(k, (k+3) % 7) for k in range(7)]
EDGES = E2 + E3                                               # non-orth graph C7(2,3)
TREE = [(0,2),(2,4),(4,6),(6,1),(1,3),(3,5)]                  # spanning tree (path)
COTREE = [e for e in EDGES if e not in TREE]                  # 8 fundamental cycles
MAG0 = {e: G0[e[0], e[1]] for e in EDGES}                     # optimal magnitudes > 0
TREE_ORDER = [0,2,4,6,1,3,5]

def gram7(mag, phase=None):
    G = np.eye(7, dtype=complex)
    for e in EDGES:
        i, j = e
        w = mag[e]*np.exp(1j*(0.0 if phase is None else phase.get(e, 0.0)))
        G[i, j] = w; G[j, i] = np.conj(w)
    return G

def fund_hols(G):
    """the 8 fundamental-cycle Bargmann invariants (gauge-invariant)."""
    pos = {v: i for i, v in enumerate(TREE_ORDER)}
    hols = []
    for (i, j) in COTREE:
        a, b = pos[i], pos[j]
        path = TREE_ORDER[min(a,b):max(a,b)+1]
        if a > b:
            path = path[::-1]
        pr = 1.0 + 0j
        for u, v in zip(path, path[1:]):
            pr *= G[u, v]
        pr *= G[j, i]
        hols.append(pr)
    return np.array(hols)

def proj_out(v, constraints):
    """project v onto the orthocomplement of span(constraints) (proper Gram-Schmidt)."""
    basis = []
    for u in constraints:
        w = u.copy()
        for b in basis:
            w = w - (b.conj() @ w)*b
        nw = np.linalg.norm(w)
        if nw > 1e-12:
            basis.append(w/nw)
    for b in basis:
        v = v - (b.conj() @ v)*b
    return v

def rand_realization(d):
    """random rank-1 realization of C7 in C^d (contexts orthogonal by construction)."""
    for _ in range(60):
        Z = rng.normal(size=(7, d)) + 1j*rng.normal(size=(7, d))
        W = np.zeros((7, d), complex); good = True
        for k in range(7):
            prev = ([k-1] if k >= 1 else []) + ([0] if k == 6 else [])
            v = proj_out(Z[k].copy(), [W[p] for p in prev])
            nv = np.linalg.norm(v)
            if nv < 1e-8:
                good = False; break
            W[k] = v/nv
        if good:
            return W
    return None

# ---------- Part A ----------
def partA():
    print("== Part A: C7 scenario, no-go transfer, optimal point ==")
    check(f"|independent sets of C7| = {len(SETS7)} = 29 (Lucas L7)", len(SETS7) == 29)
    orth = max(abs(V7[i] @ V7[j]) for (i, j) in CTX7)
    x = (V7 @ np.array([0, 0, 1.0]))**2
    S = x.sum()
    check(f"standard d=3 realization: orth {orth:.1e}; S_max = {S:.9f} = "
          "7cos(pi/7)/(1+cos(pi/7)) = theta(C7)",
          orth < 1e-12 and abs(S - TH7) < 1e-12)
    e_q = np.maximum(np.concatenate([[x[i], x[j], 1-x[i]-x[j]] for (i, j) in CTX7]), 0)
    e_nc = M7 @ (np.ones(len(SETS7))/len(SETS7))
    check("optimal model FULL support, IDENTICAL support presheaf to a NC mixture",
          e_q.min() > 1e-6 and e_nc.min() > 1e-6
          and np.array_equal(e_q > 1e-12, e_nc > 1e-12))
    ext = all(any(a[i] == o[0] and a[j] == o[1] for a in SETS7)
              for (i, j) in CTX7 for o in OUT)
    check("every allowed section extends globally (NOT logically contextual)", ext)
    cf = CF7(x)
    check(f"yet CF(optimal) = {cf:.9f} = 2(S-3): both V49 no-gos transfer to C7",
          abs(cf - 2*(S-3)) < 1e-9)

# ---------- Part B ----------
def partB():
    print("== Part B: facet uniqueness -- CF = 2*max(0, S-3) ==")
    dev = 0.0; nctx = 0
    for t in range(240):
        if t % 3 == 0:
            psi = rng.normal(size=3) + 1j*rng.normal(size=3)
            psi /= np.linalg.norm(psi)
            x = np.abs(V7.astype(complex) @ psi)**2
        elif t % 3 == 1:
            psi = rng.normal(size=3); psi /= np.linalg.norm(psi)
            x = (V7 @ psi)**2
        else:
            A = rng.normal(size=(3, 3)) + 1j*rng.normal(size=(3, 3))
            rho = A @ A.conj().T; rho /= np.trace(rho).real
            x = np.array([np.real(v.conj() @ rho @ v) for v in V7.astype(complex)])
        cf = CF7(x); pred = 2*max(0.0, x.sum() - 3)
        dev = max(dev, abs(cf - pred)); nctx += cf > 1e-9
    check(f"240 random qutrit states (standard realization): max|CF - 2max(0,S-3)| = "
          f"{dev:.2e} ({nctx}/240 contextual)", dev < 1e-8)
    devr = 0.0; lmax_all = 0.0; mism = 0; nreal = 0
    for t in range(120):
        W = rand_realization((3, 4, 5, 7)[t % 4])
        if W is None:
            continue
        nreal += 1
        G = W.conj() @ W.T
        lam = np.linalg.eigvalsh(G); lmax_all = max(lmax_all, lam[-1])
        xt = top_state_x(W)
        cft = CF7(xt); devr = max(devr, abs(cft - 2*max(0.0, xt.sum() - 3)))
        if abs(lam[-1] - 3) > 1e-6 and (lam[-1] > 3) != (cft > 1e-8):
            mism += 1
        psi = rng.normal(size=W.shape[1]) + 1j*rng.normal(size=W.shape[1])
        psi /= np.linalg.norm(psi)
        xr = np.abs(W.conj() @ psi)**2
        devr = max(devr, abs(CF7(xr) - 2*max(0.0, xr.sum() - 3)))
    check(f"{nreal} random rank-1 realizations (d in 3,4,5,7), top+random states: "
          f"max|CF - 2max(0,S-3)| = {devr:.2e}", devr < 1e-8 and nreal >= 110)
    check(f"lambda_max(G) <= theta(C7) on all samples (max seen {lmax_all:.6f})",
          lmax_all <= TH7 + 1e-7)
    check("single detector characterizes: (lambda_max(G)>3) == (CF(top state)>0), "
          f"{mism} mismatches", mism == 0)
    print("  => the odd-cycle facet is the unique quantum-operative facet at C7 too;")
    print("     realization admits a contextual state iff 1 + lambda_max(twisted adj) > 3.")

# ---------- Part C ----------
def partC():
    print("== Part C: twisted spectral detector on C7; the 8 holonomy coordinates ==")
    check("H^1(C7(2,3); U(1)) rank = |E|-|V|+1 = 14-7+1 = 8 (vs 1 at C5)",
          len(EDGES) - 7 + 1 == 8 and len(COTREE) == 8)
    lamG = np.linalg.eigvalsh(G0)
    lamA = np.linalg.eigvalsh(G0 - np.eye(7))
    check(f"identity lam_max(sum P) = 1 + lam_max(twisted adjacency) = {lamG[-1]:.9f} "
          "= theta(C7), at TRIVIAL holonomy",
          abs(lamG[-1] - 1 - lamA[-1]) < 1e-12 and abs(lamG[-1] - TH7) < 1e-12)
    check(f"optimal point PSD-critical: rank 3, lam_min = {lamG[0]:.2e} "
          "(theta-body boundary)", abs(lamG[0]) < 1e-12 and np.sum(lamG > 1e-9) == 3)
    # gauge invariance of the fundamental holonomies
    ph = {COTREE[0]: 0.7, COTREE[3]: -1.1, COTREE[6]: 2.0}
    Gt = gram7(MAG0, ph)
    D = np.diag(np.exp(1j*rng.uniform(0, 2*np.pi, 7)))
    hdev = np.max(np.abs(fund_hols(D @ Gt @ D.conj().T) - fund_hols(Gt)))
    sdev = np.max(np.abs(np.linalg.eigvalsh(D @ Gt @ D.conj().T)
                         - np.linalg.eigvalsh(Gt)))
    check(f"8 fundamental Bargmann invariants and spectrum gauge-invariant "
          f"(dev {hdev:.1e}, {sdev:.1e})", hdev < 1e-12 and sdev < 1e-12)
    # Perron bound: trivial class maximizes the detector at any fixed magnitudes
    worst = -np.inf
    for _ in range(200):
        mag = {e: rng.uniform(0.05, 0.85) for e in EDGES}
        phr = {e: rng.uniform(-np.pi, np.pi) for e in EDGES}
        worst = max(worst, np.linalg.eigvalsh(gram7(mag, phr))[-1]
                    - np.linalg.eigvalsh(gram7(mag))[-1])
    check(f"PERRON BOUND: lam_max(m,h) <= lam_max(m,trivial) for all h "
          f"(200 draws, max excess {worst:.1e}; proof: rho(B) <= rho(|B|))",
          worst < 1e-10)
    # Hessian over the 8 holonomy coordinates at the optimum
    def lmax_p(p):
        return np.linalg.eigvalsh(gram7(MAG0, dict(zip(COTREE, p))))[-1]
    h = 1e-3; f0 = lmax_p(np.zeros(8)); H = np.zeros((8, 8))
    for i in range(8):
        ei = np.zeros(8); ei[i] = h
        H[i, i] = (lmax_p(ei) + lmax_p(-ei) - 2*f0)/h**2
        for j in range(i+1, 8):
            ej = np.zeros(8); ej[j] = h
            H[i, j] = H[j, i] = (lmax_p(ei+ej) - lmax_p(ei-ej) - lmax_p(-ei+ej)
                                 + lmax_p(-ei-ej))/(4*h**2)
    ev = np.linalg.eigvalsh(H)
    check(f"detector Hessian over the 8 H^1 coordinates at the optimum: eigenvalues in "
          f"[{ev[0]:.4f}, {ev[-1]:.4f}] -- NEGATIVE DEFINITE, RANK 8: all coordinates "
          "matter, none flat", ev[-1] < -1e-4)
    # generic interior point: first derivatives nonzero in every coordinate
    mag8 = {e: 0.8*MAG0[e] for e in EDGES}
    def lm8(p):
        return np.linalg.eigvalsh(gram7(mag8, dict(zip(COTREE, p))))[-1]
    p0 = 0.15*rng.standard_normal(8); gmin = np.inf
    for i in range(8):
        ei = np.zeros(8); ei[i] = 1e-5
        gmin = min(gmin, abs(lm8(p0+ei) - lm8(p0-ei))/2e-5)
    check(f"generic interior point: dlam/dphi_i != 0 for every coordinate "
          f"(min |grad| = {gmin:.2e})", gmin > 1e-6)
    # holonomy decides at fixed magnitudes.
    # PINNED NEGATIVE first: on the UNIFORM (circulant-magnitude) slice the C5-style
    # window seems to collapse -- 0/1064 random rank-1 realizations even reach
    # {trivial class PSD and lam_max > 3}, and bounded penalty descent at
    # (g2,g3) = (0.73,0.33) stalls ABOVE 3 while keeping PSD:
    g2b, g3b = 0.73, 0.33
    magb = {e: (g2b if e in E2 else g3b) for e in EDGES}
    lt = np.linalg.eigvalsh(gram7(magb))
    check(f"uniform slice (g2,g3)=({g2b},{g3b}): trivial class realizable "
          f"(lam_min={lt[0]:.4f}) and contextual (lam_max={lt[-1]:.4f} > 3)",
          lt[0] > 0.02 and lt[-1] > 3.05)
    rl = np.random.default_rng(3)
    def lam_pair(p, mag):
        lam = np.linalg.eigvalsh(gram7(mag, dict(zip(COTREE, p))))
        return lam[-1], lam[0]
    best_u = np.inf
    for r in range(4):
        res = minimize(lambda p: (lambda a, b: a + 2000*max(0.0, 0.02 - b)**2)
                       (*lam_pair(p, magb)), rl.uniform(-1.5, 1.5, 8),
                       method="Nelder-Mead", options={"maxfev": 4000})
        a, b = lam_pair(res.x, magb)
        if b > 0.015:
            best_u = min(best_u, a)
    check(f"uniform slice, pinned search budget: NO realizable noncontextual class "
          f"found (best lam_max = {best_u:.4f} > 3; existence left open)", best_u > 3)
    # POSITIVE: non-uniform magnitudes restore the C5 phenomenon. Joint feasibility
    # search over (14 magnitudes, 8 holonomies), pinned seed, converges at restart 0:
    def feas(z):
        m, p = z[:14], z[14:]
        if np.any(m < 0.01) or np.any(m > 0.95):
            return 10.0
        a0, b0 = lam_pair(np.zeros(8), dict(zip(EDGES, m)))
        a1, b1 = lam_pair(p, dict(zip(EDGES, m)))
        r = lambda x: max(0.0, x)
        return (r(3.03 - a0)**2 + r(0.012 - b0)**2
                + r(0.012 - b1)**2 + r(a1 - 2.97)**2)
    m0 = np.array([0.71]*7 + [0.33]*7)
    sol = None
    for r in range(40):
        z0 = np.concatenate([m0 + rl.uniform(-0.08, 0.08, 14),
                             rl.uniform(-1.2, 1.2, 8)])
        res = minimize(feas, z0, method="Nelder-Mead",
                       options={"maxfev": 8000, "xatol": 1e-9, "fatol": 1e-14})
        if res.fun < 1e-16:
            sol = res.x; break
    check(f"HOLONOMY DECIDES at fixed (non-uniform) magnitudes: feasibility search "
          f"converged (restart {r}, penalty 0)", sol is not None)
    if sol is not None:
        mfix = dict(zip(EDGES, sol[:14]))
        for (pvec, want) in [(np.zeros(8), True), (sol[14:], False)]:
            G = gram7(mfix, dict(zip(COTREE, pvec)))
            lam = np.linalg.eigvalsh(G)
            W = realize(G)
            orth = max(abs(np.vdot(W[i], W[j])) for (i, j) in CTX7)
            mdev = max(abs(abs(np.vdot(W[e[0]], W[e[1]])) - mfix[e]) for e in EDGES)
            xt = top_state_x(W)
            cf = CF7(xt)
            pred = 2*max(0.0, lam[-1] - 3)
            check(f"  {'trivial' if want else 'twisted'} class, same magnitudes, "
                  f"realized in C^7 (orth {orth:.1e}, mag dev {mdev:.1e}): "
                  f"lam_max={lam[-1]:.4f}, lam_min={lam[0]:.4f}, CF={cf:.4f} -- "
                  f"{'CONTEXTUAL' if want else 'noncontextual'}",
                  orth < 1e-8 and lam[0] > 0.01 and abs(cf - pred) < 1e-8
                  and (cf > 1e-3) == want)
        hols = np.angle(fund_hols(gram7(mfix, dict(zip(COTREE, sol[14:])))))
        print("  noncontextual class fundamental holonomies (rad): "
              + np.array2string(hols, precision=3))
        print("  => the H^1 class decides at C7 as at C5, but the decision is a joint")
        print("     condition in the 8-torus and (pinned) was only found OFF the")
        print("     uniform-magnitude slice -- a genuine departure from the pentagon.")

# ---------- Part D ----------
def partD():
    print("== Part D: torsion shadow -- ALL order-2 twists of the C7-optimal point ==")
    # reflection identity: the full (-1)-twist of any unit-diagonal Gram G = I + A
    # is I - A = 2I - G, so spec(twist) = 2 - spec(G). One-line proof; spot check:
    B = rng.normal(size=(7, 7)) + 1j*rng.normal(size=(7, 7))
    A = (B + B.conj().T)/2; np.fill_diagonal(A, 0)
    dev = np.max(np.abs(np.linalg.eigvalsh(np.eye(7) - A)
                        - (2 - np.linalg.eigvalsh(np.eye(7) + A)[::-1])))
    check(f"REFLECTION IDENTITY: full (-1)-twist is G -> 2I - G, spec reflects "
          f"(random check dev {dev:.1e}; algebraic proof: I - A = 2I - G)", dev < 1e-12)
    lam0 = np.linalg.eigvalsh(G0)
    check(f"C7-optimal: full-twist value = 2 - lam_min(G) = {2-lam0[0]:.12f} = 2 "
          "EXACTLY (rank-3 PSD-criticality) -- NOT the classical bound 3: the C5 "
          "coincidence is the accident alpha(C5)=2", abs(lam0[0]) < 1e-12)
    # sweep all 256 order-2 classes (co-tree gauge: tree edges fixed +1)
    A0 = G0 - np.eye(7)
    vals = np.empty(256); lmins = np.empty(256)
    for b in range(256):
        A = A0.copy()
        for i, (u, v) in enumerate(COTREE):
            if (b >> i) & 1:
                A[u, v] *= -1; A[v, u] *= -1
        lam = np.linalg.eigvalsh(np.eye(7) + A)
        vals[b] = lam[-1]; lmins[b] = lam[0]
    # locate the canonical all-(-1) class in co-tree gauge
    tau = np.ones(7)
    for i, v in enumerate(TREE_ORDER):
        tau[v] = (-1)**i                       # gauge making tree edges +1
    bm = 0
    for i, (u, v) in enumerate(COTREE):
        if tau[u]*tau[v]*(-1) < 0:
            bm |= (1 << i)
    check(f"canonical all-(-1) class = co-tree pattern {bm:08b}: value "
          f"{vals[bm]:.12f} = 2 exactly", abs(vals[bm] - 2) < 1e-11)
    hits3 = int(np.sum(np.abs(vals - 3) < 1e-9))
    hits2 = int(np.sum(np.abs(vals - 2) < 1e-9))
    nt = vals[1:]
    check(f"NO order-2 class lands on the classical bound 3 (|val-3|<1e-9: {hits3}); "
          f"exact value 2 hit by {hits2} class(es)", hits3 == 0)
    nctx = int(np.sum(nt > 3 + 1e-9))
    nreal = int(np.sum(lmins[1:] > -1e-9))
    print(f"  255 nontrivial classes: range [{nt.min():.6f}, {nt.max():.6f}], "
          f"{nctx} keep detector > 3, {nreal} remain PSD at these magnitudes.")
    check("the all-(-1) class attains the MINIMUM of the detector over all 256 "
          "order-2 classes (min = 2, unique)",
          int(np.argmin(vals)) == bm and abs(vals.min() - 2) < 1e-11)
    check("at the optimal magnitudes the trivial class is the ONLY realizable "
          "order-2 class", nreal == 0)

# ---------- Part E ----------
def partE():
    print("== Part E: C9 minimal probe (odd-cycle law vs pentagon accident) ==")
    CTX9 = [(k, (k+1) % 9) for k in range(9)]
    SETS9 = indsets(9, CTX9)
    M9 = build_M(CTX9, SETS9); CF9 = CF_factory(CTX9, M9)
    V9 = cn_vectors(9); G9 = V9 @ V9.T
    c9 = np.cos(np.pi/9); TH9 = 9*c9/(1+c9)
    check(f"|independent sets of C9| = {len(SETS9)} = 76 (Lucas L9); "
          "H^1(C9(2,3,4)) rank = 27-9+1 = 19", len(SETS9) == 76)
    orth = max(abs(V9[i] @ V9[j]) for (i, j) in CTX9)
    x = (V9 @ np.array([0, 0, 1.0]))**2
    S = x.sum(); cf = CF9(x)
    check(f"C9-optimal (orth {orth:.1e}): S = {S:.9f} = theta(C9); CF = {cf:.9f} "
          "= 2(S-4)", orth < 1e-12 and abs(S - TH9) < 1e-12
          and abs(cf - 2*(S - 4)) < 1e-9)
    dev = 0.0
    for t in range(60):
        psi = rng.normal(size=3) + 1j*rng.normal(size=3)
        psi /= np.linalg.norm(psi)
        xr = np.abs(V9.astype(complex) @ psi)**2
        dev = max(dev, abs(CF9(xr) - 2*max(0.0, xr.sum() - 4)))
    check(f"60 random states: max|CF - 2max(0,S-4)| = {dev:.2e} "
          "(cycle facet unique at C9 too)", dev < 1e-8)
    lam9 = np.linalg.eigvalsh(G9)
    check(f"full (-1)-twist = 2 - lam_min = {2-lam9[0]:.12f} = 2 exactly; classical "
          "bound is 4: 'full twist = 2' is the odd-cycle law, "
          "'twist = classical bound' was the pentagon accident", abs(lam9[0]) < 1e-12)
    # circulant order-2 twists (sign per step class): none on the classical bound 4
    g = {s: G9[0, s] for s in (2, 3, 4)}
    vals = []
    for s2 in (1, -1):
        for s3 in (1, -1):
            for s4 in (1, -1):
                A = np.zeros((9, 9))
                sg = {2: s2, 3: s3, 4: s4}
                for k in range(9):
                    for s in (2, 3, 4):
                        A[k, (k+s) % 9] = A[(k+s) % 9, k] = sg[s]*g[s]
                vals.append(np.linalg.eigvalsh(np.eye(9) + A)[-1])
    vals = np.array(vals)
    check(f"8 circulant order-2 twists of the C9-optimal point: none = 4 "
          f"(range [{vals.min():.6f}, {vals.max():.6f}])",
          np.all(np.abs(vals - 4) > 1e-6))

# ---------- Part F ----------
def partF():
    print("== Part F: the alpha=2 family -- C7 antihole, where the shadow survives ==")
    dist = lambda i, j: min((i-j) % 7, (j-i) % 7)
    EA = [(i, j) for i in range(7) for j in range(i+1, 7) if dist(i, j) in (2, 3)]
    SETSA = indsets(7, EA)
    MA = build_M(EA, SETSA); CFA = CF_factory(EA, MA)
    thA = 7/TH7                                            # theta(antiC7) = 1+sec(pi/7)
    check(f"antihole scenario: 14 contexts, |indep sets| = {len(SETSA)} = 15, "
          "alpha = 2; non-orth graph = C7, H^1 rank = 7-7+1 = 1 (pentagon-like)",
          len(SETSA) == 15 and len(EA) == 14)
    def build(params):
        Z = params.reshape(7, 5, 2)
        Zc = Z[..., 0] + 1j*Z[..., 1]
        V = np.zeros((7, 5), complex)
        for k in range(7):
            v = proj_out(Zc[k].copy(),
                         [V[p] for p in range(k) if dist(p, k) in (2, 3)])
            nv = np.linalg.norm(v)
            if nv < 1e-9:
                return None
            V[k] = v/nv
        return V
    def negobj(params):
        V = build(params)
        if V is None:
            return 10.0
        return -np.linalg.eigvalsh(V.conj() @ V.T)[-1]
    best, bval = None, np.inf
    for r in range(6):
        res = minimize(negobj, rng.standard_normal(70), method="L-BFGS-B",
                       options={"maxiter": 250})
        if res.fun < bval:
            best, bval = res.x, res.fun
    W = build(best)
    G = W.conj() @ W.T
    lam = np.linalg.eigvalsh(G)
    orth = max(abs(np.vdot(W[i], W[j])) for (i, j) in EA)
    check(f"contextual antihole realization found in d=5: lam_max = {lam[-1]:.6f} > 2 "
          f"(theta(antiC7) = {thA:.6f}; orth {orth:.1e})",
          lam[-1] > 2.01 and lam[-1] <= thA + 1e-6 and orth < 1e-10)
    xt = top_state_x(W)
    cf = CFA(xt)
    check(f"LP-certified: S(top) = {xt.sum():.6f} > 2 = classical bound, "
          f"CF = {cf:.6f} > 0", cf > 1e-6 and abs(xt.sum() - lam[-1]) < 1e-9)
    check(f"CF = (2/3)(S-2) at this point (ABM normalization: NS max 7/2, "
          f"classical 2), dev {abs(cf - (2/3)*(xt.sum()-2)):.1e} -- cycle facet "
          "operative here too (single point, not a census)",
          abs(cf - (2/3)*(xt.sum() - 2)) < 1e-8)
    hol = np.prod([G[k, (k+1) % 7] for k in range(7)])
    print(f"  single U(1) holonomy of the realization: |h|={abs(hol):.3e}, "
          f"arg={np.angle(hol):+.4f} rad (Perron: optimum at trivial class)")
    check(f"TORSION SHADOW SURVIVES: rank(G) <= 5 < 7 => lam_min = {lam[0]:.2e} = 0 "
          f"=> (-1)-twist value = 2 - lam_min = {2-lam[0]:.12f} = classical bound "
          "alpha = 2 EXACTLY (the C5 law lives in the alpha=2 antihole family)",
          abs(lam[0]) < 1e-9)
    check(f"and the twist is NOT realizable: lam_min(2I-G) = {2-lam[-1]:.6f} < 0 "
          "(exactly as V49 Part C(iii))", 2 - lam[-1] < -1e-6)

PARTS = {"A": partA, "B": partB, "C": partC, "D": partD, "E": partE, "F": partF}

if __name__ == "__main__":
    sel = [s.upper() for s in sys.argv[1:]] or list("ABCDEF")
    for s in sel:
        PARTS[s]()
    print("C7 PROBE " + ("PASS" if OK else "FAIL"))
