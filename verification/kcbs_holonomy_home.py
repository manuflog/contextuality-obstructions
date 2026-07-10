# V49 - KCBS COHOMOLOGICAL HOME: no-gos + U(1)-holonomy-twisted spectral detector.
# See INDEX.md entry and the paper C open problem. Expected: 'C3 PROBE PASS'.
# c3_kcbs_probe.py -- Paper C Open Problem 1: the cohomological home of KCBS-type
# (state-dependent, class-invisible) contextuality. Companion to c3_kcbs_memo.md.
#
# Part A (NO-GO, possibilistic collapse): the KCBS-optimal quantum model and an explicit
#   noncontextual model have IDENTICAL support presheaves, and the model is not even
#   logically contextual (every allowed local section extends to a global assignment).
#   => every invariant of the support presheaf -- the Abramsky-Brandenburger Cech
#   obstruction in every degree, and Caru's higher obstructions -- takes the same value
#   on a contextual and a noncontextual model, hence provably cannot characterize CF>0.
#
# Part B (FACET UNIQUENESS): on the pentagon the noncontextual polytope is affinely
#   STAB(C5); the single odd-cycle facet is the only quantum-operative one, so
#   CF(rho) = 2*max(0, S(rho)-2), S = sum_k <P_k>.  Verified by LP against the
#   11-vertex polytope on random pure and mixed qutrit states.
#
# Part C (CANDIDATE, twisted spectral detector): a rank-1 realization of the pentagon,
#   modulo the U(1)^5 vertex gauge, is classified by (pentagram Gram magnitudes,
#   Bargmann holonomy) in C^1_+ x H^1(C5*;U(1)).  Identity:
#       lambda_max(sum_k P_k) = 1 + lambda_max(holonomy-twisted weighted adjacency).
#   Tests: (i) the identity; (ii) KCBS sits at holonomy +1, PSD-critical, value sqrt5;
#   (iii) EXACT coincidence: same magnitudes, holonomy -1 => lambda_max = 2 = classical
#   bound exactly (the anti-holonomy pentagon is exactly classical-critical);
#   (iv) HOLONOMY DECIDES: at fixed magnitudes g=0.52, phi=1.0 is contextual and
#   phi=1.5 is not -- both realized by explicit vectors in d=5, CF checked by LP.
#
# Part D (CONTROL): the noncontextual rank-1 realization u_k=(e_k+e_{k+2})/sqrt2 has
#   the same exclusivity graph C5, holonomy +1, g=1/2, detector value exactly 2, CF=0.
import numpy as np, itertools
from scipy.optimize import linprog

rng = np.random.default_rng(7)
ok = True
def check(name, cond):
    global ok
    print(("  PASS " if cond else "  FAIL ") + name)
    ok = ok and cond

# ---------- pentagon scenario ----------
CTX = [(k, (k+1) % 5) for k in range(5)]           # contexts = exclusive pairs
OUT = [(1,0), (0,1), (0,0)]                         # allowed outcomes per context
INDSETS = [a for a in itertools.product([0,1], repeat=5)
           if all(a[k]*a[(k+1)%5] == 0 for k in range(5))]   # 11 independent sets

def det_model(a):
    e = np.zeros((5,3))
    for c,(i,j) in enumerate(CTX):
        e[c, OUT.index((a[i], a[j]))] = 1.0
    return e.ravel()

M = np.array([det_model(a) for a in INDSETS]).T     # 15 x 11

def model_from_x(x):
    e = np.zeros((5,3))
    for c,(i,j) in enumerate(CTX):
        e[c] = [x[i], x[j], 1-x[i]-x[j]]
    return e.ravel()

def CF_lp(x):
    """contextual fraction on the pentagon via ABM LP (max NC weight)."""
    e = model_from_x(x)
    r = linprog(-np.ones(M.shape[1]), A_ub=M, b_ub=e, bounds=(0,None), method="highs")
    assert r.status == 0
    return 1.0 + r.fun

# ---------- KCBS realization (repo convention, d=3) ----------
c = np.cos(np.pi/5); ct2 = c/(1+c); st = np.sqrt(1-ct2)
V = np.array([[st*np.cos(4*np.pi*k/5), st*np.sin(4*np.pi*k/5), np.sqrt(ct2)]
              for k in range(5)])
psi_opt = np.array([0,0,1.0])
x_opt = (V @ psi_opt)**2

print("== Part A: possibilistic collapse no-go ==")
e_q  = model_from_x(x_opt)
e_nc = M @ (np.ones(11)/11)                          # uniform NC mixture
check("KCBS-optimal model has FULL support (all 15 probs > 0)", e_q.min() > 1e-6)
check("NC uniform-mixture model has full support too", e_nc.min() > 1e-6)
check("support presheaves IDENTICAL (same possibilistic model)",
      np.array_equal(e_q > 1e-12, e_nc > 1e-12))
ext = all(any(a[i]==o[0] and a[j]==o[1] for a in INDSETS)
          for (i,j) in CTX for o in OUT)
check("every allowed section extends globally (NOT logically contextual)", ext)
cf0 = CF_lp(x_opt)
check(f"yet CF(optimal) = {cf0:.5f} = 2*sqrt5-4 (contextual)",
      abs(cf0 - (2*np.sqrt(5)-4)) < 1e-9)
print("  => any support-presheaf invariant (AB Cech, all degrees; Caru higher)")
print("     is constant across {contextual KCBS-optimal, noncontextual mixture}: NO-GO.")

print("== Part B: unique operative facet; CF = 2*max(0, S-2) ==")
dev = 0.0; nctx = 0
for t in range(300):
    if t % 3 == 0:
        psi = rng.normal(size=3) + 1j*rng.normal(size=3); psi /= np.linalg.norm(psi)
        x = np.abs(V.astype(complex) @ psi)**2
    elif t % 3 == 1:
        psi = rng.normal(size=3); psi /= np.linalg.norm(psi); x = (V @ psi)**2
    else:
        A = rng.normal(size=(3,3)) + 1j*rng.normal(size=(3,3))
        rho = A @ A.conj().T; rho /= np.trace(rho).real
        x = np.array([np.real(v.conj() @ rho @ v) for v in V.astype(complex)])
    cf = CF_lp(x); pred = 2*max(0.0, x.sum()-2)
    dev = max(dev, abs(cf-pred)); nctx += cf > 1e-9
check(f"300 random qutrit states: max |CF_LP - 2*max(0,S-2)| = {dev:.2e}", dev < 1e-8)
print(f"  ({nctx}/300 contextual; odd-cycle facet is the unique operative facet)")

print("== Part C: twisted spectral detector ==")
PENT = [(0,2),(1,3),(2,4),(3,0),(4,1)]              # pentagram (complement) edges
CYC  = [0,2,4,1,3]                                   # pentagram as a 5-cycle

def gram(g, phi):
    """uniform pentagram magnitudes g, all holonomy phi on edge (3,0)."""
    G = np.eye(5, dtype=complex)
    for (i,j) in PENT:
        w = g*np.exp(1j*phi) if (i,j) == (3,0) else g
        G[i,j] = w; G[j,i] = np.conj(w)
    return G

def bargmann(G):
    return np.prod([G[CYC[i], CYC[(i+1)%5]] for i in range(5)])

# (i) identity on KCBS: lambda_max(Gram) = 1 + lambda_max(twisted adjacency)
GK = V @ V.T
gk = GK[0,2]
lamG = np.linalg.eigvalsh(GK)
lamA = np.linalg.eigvalsh(GK - np.eye(5))
check(f"KCBS Gram: pentagram weight g = {gk:.6f} = (sqrt5-1)/2, holonomy +1",
      abs(gk-(np.sqrt(5)-1)/2) < 1e-12 and bargmann(GK) > 0)
check(f"identity lam_max(sum P) = 1 + lam_max(twisted C5) = {lamG[-1]:.6f} = sqrt5",
      abs(lamG[-1] - (1+lamA[-1])) < 1e-12 and abs(lamG[-1]-np.sqrt(5)) < 1e-12)
check(f"KCBS point is PSD-critical (lam_min = {lamG[0]:.2e}): theta-body boundary",
      abs(lamG[0]) < 1e-12)

# (iii) exact anti-holonomy coincidence at the SAME magnitudes
lam_pi = np.linalg.eigvalsh(gram(gk, np.pi))
check(f"holonomy -1 at same magnitudes: lam_max = {lam_pi[-1]:.15f} = 2 EXACTLY "
      "(classical bound)", abs(lam_pi[-1]-2) < 1e-12)
check(f"  (and it is NOT realizable: lam_min = {lam_pi[0]:.6f} < 0)", lam_pi[0] < -1e-6)

# (iv) holonomy decides at fixed magnitudes, with explicit d=5 realizations
def realize(g, phi):
    G = gram(g, phi)
    lam, U = np.linalg.eigh(G)
    assert lam.min() > -1e-10
    W = (U * np.sqrt(np.clip(lam,0,None))).conj()    # rows k: vector v_k, <v_i,v_j>=G[i,j]
    return G, W

for phi, want in [(1.0, True), (1.5, False)]:
    G, W = realize(0.52, phi)
    orth = max(abs(np.vdot(W[i], W[j])) for (i,j) in CTX)
    mags = [abs(np.vdot(W[i], W[j])) for (i,j) in PENT]
    hol  = np.angle(bargmann(G))
    lam, U = np.linalg.eigh(G)
    S = lam[-1]
    # top eigenstate of sum P_k in the d=5 realization, via Gram machinery:
    # x_k = |<v_k|psi>|^2 with psi the top eigenvector -> x = |sqrt(lam)*U[:,-1]|^2-ish;
    # directly: sum_k P_k has same nonzero spectrum as G; moments of top eigenstate:
    P = np.array([np.outer(W[k], W[k].conj()) for k in range(5)])
    lam2, U2 = np.linalg.eigh(P.sum(axis=0))
    psi = U2[:,-1]
    x = np.array([np.real(psi.conj() @ P[k] @ psi) for k in range(5)])
    cf = CF_lp(x); pred = 2*max(0.0, S-2)
    check(f"g=0.52, phi={phi}: realized in d=5 (orth {orth:.1e}, |g| dev "
          f"{max(abs(m-0.52) for m in mags):.1e}, hol {hol:.3f}); "
          f"S={S:.4f}, CF={cf:.4f} -- {'CONTEXTUAL' if want else 'noncontextual'}",
          orth < 1e-8 and abs(cf-pred) < 1e-8 and ((cf > 1e-3) == want))
print("  => magnitudes alone do NOT determine the state sector; the holonomy class does.")

# critical holonomy curve: max realizable detector value as a function of phi
phis = np.linspace(0, np.pi, 601); best = []
for phi in phis:
    ang = (phi + 2*np.pi*np.arange(5))/5
    gmax = 1/(2*np.abs(np.cos(ang).min()))           # PSD boundary for uniform g
    best.append(1 + 2*gmax*np.cos(ang).max())
best = np.array(best)
phic = phis[np.searchsorted(-(best-2), 0.0)] if best[-1] < 2 else np.pi
i_c = int(np.argmin(np.abs(best-2)))
check(f"max detector over realizable uniform-g family: {best[0]:.6f} at phi=0 "
      f"(= Lovasz theta(C5) = sqrt5)", abs(best[0]-np.sqrt(5)) < 1e-9)
check(f"contextuality window: detector > 2 iff |phi| < phi_c ~ {phis[i_c]:.4f} rad; "
      "monotone decreasing", np.all(np.diff(best) < 1e-9) and best[-1] < 2)

print("== Part D: control -- noncontextual realization, same exclusivity graph ==")
E = np.eye(5)
Wnc = np.array([(E[k]+E[(k+2)%5])/np.sqrt(2) for k in range(5)])
orth = max(abs(Wnc[i] @ Wnc[j]) for (i,j) in CTX)
Gnc = Wnc @ Wnc.T
lam_nc = np.linalg.eigvalsh(Gnc)
lam2, U2 = np.linalg.eigh(sum(np.outer(w,w) for w in Wnc))
xnc = np.array([ (Wnc[k] @ U2[:,-1])**2 for k in range(5)])
check(f"u_k=(e_k+e_k+2)/sqrt2: C5 exclusivity (orth {orth:.1e}), g=1/2, hol +1, "
      f"detector = {lam_nc[-1]:.10f} = 2 exactly, CF(top)={CF_lp(xnc):.1e}",
      orth < 1e-12 and abs(Gnc[0,2]-0.5) < 1e-12 and abs(lam_nc[-1]-2) < 1e-9
      and CF_lp(xnc) < 1e-8)

print("C3 PROBE " + ("PASS" if ok else "FAIL"))
