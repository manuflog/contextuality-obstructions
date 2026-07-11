# V54 - TWO-LAYER UNIFICATION, PM MILESTONE (open problem (a)): the V49/V50 rank-1
# local system on the Peres 24-ray graph is KS-rigid (flexes 0 vs C5's 2), holonomy
# pure torsion; the AvN -1 = forced holonomy invariant (transversal parity theorem;
# hexagon holonomy law; h(z_obs) = -1); tight-frame degeneracy = state-independence.
# Expected: 'CU UNIFY PROBE PASS' (~3 s). See INDEX.md.
#!/usr/bin/env python3
# cu_unify_probe.py -- TWO-LAYER UNIFICATION probe (open problem (a), note v3; V50 Conj. 2).
# Milestone: realize the Peres-Mermin (-1) AvN obstruction as an invariant of the SAME
# rank-1 local system (Gram magnitudes m on the non-orthogonality graph N + U(1)
# holonomy class h in H^1(N;U(1))) that V49/V50 use for KCBS/odd cycles/antiholes.
#
# Configuration: the 24 joint eigenrays of the 6 corrected-layout PM contexts
# (V44 PMC lists, per-site labels, weyl.py convention W(v)=tau^{-q} kron X^a Z^b).
#
# Part A: build the 24 rays; contexts commute; context signs eps_C = (+++,++-),
#         prod eps_C = -1 (the AvN class); rays REAL; Peres 24-ray census
#         (4 of type e_i, 12 of type (e_i+-e_j)/sqrt2, 8 of type (+-1,+-1,+-1,+-1)/2).
# Part B: graph structure. Exclusivity graph X 9-regular (108 edges), N = complement
#         minus context cliques structure; sharing context pairs meet in 2+2 MUB blocks
#         (|<p,q>| = 1/sqrt2, block = shared-observable eigenvalue); non-sharing pairs
#         fully unbiased (all 1/2). TRANSVERSAL PARITY THEOREM: for EVERY choice of one
#         ray per context, the number of shared-observable ray pairs that are ORTHOGONAL
#         is ODD (= the AvN -1; proof in comments below). Corollaries: no independent
#         transversal, alpha(X) = 5 < 6.
# Part C: state layer interface. sum_k P_k = 6 I (tight frame): S(psi) = 6 for EVERY
#         state; nonzero spec(G) = {6,6,6,6}; detector S > alpha = 5 for every state
#         (state-INDEPENDENT exit, same spectral detector as V49/V50); no global section
#         => strong contextuality, CF = 1 for all states.
# Part D: holonomy layer. The realized local system is PURE TORSION: all fundamental
#         Bargmann holonomies are +-1 (h in H^1(N;Z2) subset H^1(N;U(1))). The 18
#         eigenplane 4-cycles all carry holonomy -1; this is FORCED for every rank-1
#         realization of X in d=4 (two orthonormal bases of a 2-plane:
#         B(p1,q1,p2,q2) = -|alpha|^2 |beta|^2 < 0 for any U(2) change of basis).
#         Canonical class: z_obs = sum over the 9 observables of one eigenplane 4-cycle:
#         h(z_obs) = (-1)^9 = -1 independent of all choices. Cycle sign census.
# Part E: hexagon holonomy scan. Context hexagon R1 K1 R2 K2 R3 K3 (K_{3,3} 6-cycle);
#         for every hexagon-compatible transversal: diagonal (antipodal) pattern obeys
#         the parity law (odd number of MISSING diagonals). PINNED LAW (found here,
#         confirmed on all 12 labeled hexagons in Part G): the hexagon holonomy is
#         h6 = -1 exactly on CHORDLESS transversals (all three antipodal pairs
#         orthogonal) and +1 otherwise; every 4-cycle through a present antipodal
#         diagonal carries +1. The AvN sign -1 thus appears as the holonomy VALUE of
#         the maximally frustrated canonical 6-cycles.
# Part F: rigidity. Infinitesimal: nullity of the orthogonality-constraint Jacobian at
#         the PM realization vs gauge dimension (U(4) + 24 vertex phases - overlap = 39
#         over C; O(4) = 6 over R). Global evidence: re-realizations of X from noisy and
#         random starts; compare magnitudes and ALL fundamental holonomies (up to
#         complex conjugation = antiunitary). If nullity = gauge dim and all
#         re-realizations match, the graph PINS the whole local system: the torsion
#         class is forced = state-independence mechanism.
#
# PROOF OF THE TRANSVERSAL PARITY THEOREM (eigenvalue bookkeeping):
#   For a context C = {A,B,M} with joint eigenray r: a(r)b(r)m(r) = eps_C. Choose one
#   ray r_C per context and take the product of ALL eigenvalues over all 6 contexts:
#   prod_C eps_C = -1. Regroup by observable (each observable A sits in exactly two
#   contexts C, C'): -1 = prod_{A in 9 obs} a(r_C) a(r_{C'}). Now a(r_C)a(r_{C'}) = +1
#   iff the two rays lie in the SAME eigenplane of A, and (Part B block structure,
#   overlap 1/sqrt2 != 0) same eigenplane <=> NON-ORTHOGONAL, different eigenplane =>
#   orthogonal. Hence (-1)^{#orthogonal designated pairs} = -1: the parity is ODD. QED.
#   Well-definedness (gauge lemma): changing r_C within its context flips the three
#   incident indicators by a(r)a(r'), b(r)b(r'), m(r)m(r') whose product is
#   eps_C^2 = +1: an even number of flips, so the parity is transversal-independent --
#   it is a Z2 INVARIANT nu(L) of the support of the local system (block data are
#   graph data), and nu(L) = prod_C eps_C = -1 is the AvN class.
#
# Expected final line: 'CU UNIFY PROBE PASS'. Each part < 40 s (full run a few s;
# part F dominates). Usage: python3 cu_unify_probe.py [A B C D E F]
import sys, itertools, collections, time
import numpy as np

sys.path.append('/sessions/quirky-eloquent-babbage/mnt/contextuality-obstructions/verification')
try:
    from weyl import build
except Exception:
    def build(d, m):
        w = np.exp(2j*np.pi/d); tau = np.exp(1j*np.pi/d)
        X = np.roll(np.eye(d), 1, axis=0); Z = np.diag([w**k for k in range(d)])
        Xp = [np.linalg.matrix_power(X, a) for a in range(d)]
        Zp = [np.linalg.matrix_power(Z, b) for b in range(d)]
        def T1(a, b): return Xp[a % d] @ Zp[b % d]
        def W(v):
            M = T1(v[0], v[1])
            for i in range(1, m): M = np.kron(M, T1(v[2*i], v[2*i+1]))
            q = sum(int(v[2*i])*int(v[2*i+1]) for i in range(m))
            return tau**(-q)*M
        return (X, Z, w, tau, W, d**m)

OK = True
def check(name, cond):
    global OK
    print(("  PASS " if cond else "  FAIL ") + name)
    OK = OK and bool(cond)

rng = np.random.default_rng(5)

# PM contexts, V44 PMC lists (per-site layout (a1,b1,a2,b2)); rows R1..R3, cols K1..K3.
PMC = [[(1,0,0,0),(0,0,1,0),(1,0,1,0)],   # R1: X1, X2, X1X2
       [(0,0,0,1),(0,1,0,0),(0,1,0,1)],   # R2: Z2, Z1, Z1Z2
       [(1,0,0,1),(0,1,1,0),(1,1,1,1)],   # R3: X1Z2, Z1X2, Y1Y2
       [(1,0,0,0),(0,0,0,1),(1,0,0,1)],   # K1: X1, Z2, X1Z2
       [(0,0,1,0),(0,1,0,0),(0,1,1,0)],   # K2: X2, Z1, Z1X2
       [(1,0,1,0),(0,1,0,1),(1,1,1,1)]]   # K3: X1X2, Z1Z2, Y1Y2
CNAME = ["R1","R2","R3","K1","K2","K3"]
LABELS = sorted({t for C in PMC for t in C})
_,_,_,_,Wop,_ = build(2, 2)
OPS = {t: Wop(np.array(t)) for t in LABELS}
OBS2CTX = {t: [c for c in range(6) if t in PMC[c]] for t in LABELS}

# ---- 24 rays: joint eigenbases ----
RAYS = np.zeros((24, 4))
EIG = [dict() for _ in range(24)]
EPS = []
for c, C in enumerate(PMC):
    A, B, M = (OPS[t] for t in C)
    Ar, Br, Mr = A.real, B.real, M.real
    lam, U = np.linalg.eigh(Ar + 2*Br)
    for k in range(4):
        idx = 4*c + k; v = U[:, k]
        RAYS[idx] = v
        for t in C:
            e = float(v @ OPS[t].real @ v)
            assert abs(abs(e) - 1) < 1e-9
            EIG[idx][t] = int(round(e))
    EPS.append(int(round(np.trace(A @ B @ M).real/4)))
EPS = np.array(EPS)
GS = RAYS @ RAYS.T
TOL = 1e-9
def orth(i, j): return abs(GS[i, j]) < TOL
NADJ = (np.abs(GS) > TOL) & ~np.eye(24, dtype=bool)
NE = [(i, j) for i in range(24) for j in range(i+1, 24) if NADJ[i, j]]
XE = [(i, j) for i in range(24) for j in range(i+1, 24) if not NADJ[i, j]]

# ---- tree gauge / fundamental holonomies on N ----
def tree_data():
    par = {0: None}; dq = collections.deque([0])
    while dq:
        u = dq.popleft()
        for v in range(24):
            if NADJ[u, v] and v not in par:
                par[v] = u; dq.append(v)
    return par
PAR = tree_data()
TREE = {tuple(sorted((v, p))) for v, p in PAR.items() if p is not None}
COTREE = [e for e in NE if e not in TREE]
def path_to_root(v):
    p = [v]
    while PAR[p[-1]] is not None: p.append(PAR[p[-1]])
    return p
def fund_hols(IP):
    """Bargmann phase of each fundamental cycle (cotree edge closed through the tree)."""
    out = []
    for (u, v) in COTREE:
        pu, pv = path_to_root(u), path_to_root(v)
        su, sv = set(pu), set(pv)
        m = next(x for x in pu if x in sv)
        walk = pu[:pu.index(m)+1] + pv[:pv.index(m)][::-1] + [u]  # u..m..v,u closes
        pr = 1.0 + 0j
        for a, b in zip(walk, walk[1:]): pr *= IP[a, b]
        out.append(pr/abs(pr))
    return np.array(out)

def partA():
    print("== Part A: the 24-ray PM configuration (V44 contexts, weyl convention) ==")
    imx = max(np.max(np.abs(OPS[t].imag)) for t in LABELS)
    check(f"all 9 PM operators REAL symmetric (max imag {imx:.1e}) => rays real", imx < 1e-12)
    comm = max(np.max(np.abs(OPS[a] @ OPS[b] - OPS[b] @ OPS[a]))
               for C in PMC for a, b in itertools.combinations(C, 2))
    check(f"all 6 contexts commuting (max comm dev {comm:.1e})", comm < 1e-12)
    check(f"context signs eps = {EPS.tolist()} (R:+++ K:++-), prod = -1 = AvN class",
          EPS.tolist() == [1, 1, 1, 1, 1, -1] and EPS.prod() == -1)
    tri = max(abs(EIG[4*c+k][PMC[c][0]]*EIG[4*c+k][PMC[c][1]]*EIG[4*c+k][PMC[c][2]]
                  - EPS[c]) for c in range(6) for k in range(4))
    check("joint eigenvalue triples satisfy a*b*m = eps_C on every ray", tri == 0)
    dmin = min(1 - abs(GS[i, j]) for i in range(24) for j in range(i+1, 24))
    check(f"24 rays pairwise distinct as projectors (min 1-|<u,v>| = {dmin:.4f})",
          dmin > 0.29)
    cnt = collections.Counter()
    for v in RAYS:
        nz = np.sort(np.abs(v[np.abs(v) > 1e-9]))
        cnt[len(nz)] += 1
    check(f"Peres 24-ray census: {cnt[1]} x e_i, {cnt[2]} x (e_i+-e_j)/sqrt2, "
          f"{cnt[4]} x (+-1,..)/2", cnt[1] == 4 and cnt[2] == 12 and cnt[4] == 8)

def partB():
    print("== Part B: graphs, MUB blocks, TRANSVERSAL PARITY THEOREM, alpha = 5 ==")
    deg = (~NADJ).sum(axis=1) - 1
    check(f"exclusivity graph X is 9-regular, |E(X)| = {len(XE)} = 108; "
          f"|E(N)| = {len(NE)} = 168", np.all(deg == 9) and len(XE) == 108 and len(NE) == 168)
    okblk = True
    for ci, cj in itertools.combinations(range(6), 2):
        sh = set(PMC[ci]) & set(PMC[cj])
        sub = np.array([[abs(GS[4*ci+a, 4*cj+b]) for b in range(4)] for a in range(4)])
        if sh:
            t = next(iter(sh))
            for a in range(4):
                for b in range(4):
                    same = EIG[4*ci+a][t] == EIG[4*cj+b][t]
                    okblk &= (abs(sub[a, b] - 1/np.sqrt(2)) < 1e-9) if same \
                        else (sub[a, b] < TOL)
        else:
            okblk &= bool(np.all(np.abs(sub - 0.5) < 1e-9))
    check("sharing context pairs: 2+2 MUB blocks, |<p,q>| = 1/sqrt2, block = shared-"
          "observable eigenvalue; non-sharing pairs: fully unbiased (all 1/2)", okblk)
    # transversal parity theorem: every transversal has an ODD number of orthogonal
    # designated (shared-observable) pairs; parity via graph == parity via eigenvalues.
    pars = set(); agree = True; zero_cross = 0
    for T in itertools.product(range(4), repeat=6):
        k_orth = 0; k_eig = 0
        for t in LABELS:
            ci, cj = OBS2CTX[t]
            ri, rj = 4*ci + T[ci], 4*cj + T[cj]
            k_orth += orth(ri, rj)
            k_eig += EIG[ri][t] != EIG[rj][t]
        agree &= (k_orth == k_eig)
        pars.add(k_orth % 2); zero_cross += (k_orth == 0)
    check("PARITY THEOREM: all 4096 transversals have ODD orthogonal-pair count "
          f"(parities seen: {sorted(pars)}); graph parity == eigenvalue parity: {agree}",
          pars == {1} and agree)
    check("=> nu(L) := (-1)^parity = -1 = prod eps_C, well-defined (transversal-"
          "independent) invariant of supp(L); no crossing-free transversal "
          f"({zero_cross} found)", zero_cross == 0)
    # independence number: at most one ray per context (context = 4-clique in X)
    best = 0
    for T in itertools.product(range(5), repeat=6):
        sel = [4*c + T[c] for c in range(6) if T[c] < 4]
        if len(sel) <= best: continue
        if all(not orth(a, b) for a, b in itertools.combinations(sel, 2)):
            best = len(sel)
    check(f"alpha(X) = {best} = 5 < 6 = #contexts (no independent transversal: "
          "KS/strong-contextuality at ray level, graph-forced)", best == 5)

def partC():
    print("== Part C: state layer -- tight frame, spectral detector, CF = 1 ==")
    Sm = RAYS.T @ RAYS
    check(f"sum_k P_k = 6 I (tight frame; dev {np.max(np.abs(Sm - 6*np.eye(4))):.1e}) "
          "=> S(psi) = 6 for EVERY state", np.allclose(Sm, 6*np.eye(4), atol=1e-12))
    lam = np.linalg.eigvalsh(GS)
    check(f"nonzero spec(G) = {{6 x4}} (got {np.round(lam[-4:],10).tolist()}), rank 4: "
          "every state is a top eigenstate", np.allclose(lam[-4:], 6, atol=1e-9)
          and np.all(np.abs(lam[:-4]) < 1e-9))
    print("  => V49/V50 detector, same object: contextual states exist iff "
          "lam_max(G) > alpha;")
    print("     here lam_max(G) = 6 > 5 = alpha AND the top eigenspace is EVERYTHING:")
    print("     the exit from the NC polytope is state-independent -- the AvN layer.")
    print("  => no global section (Part B) = empty deterministic grid: CF(rho) = 1 for")
    print("     every rho (strong contextuality), the AvN signature in ABM form.")

def partD():
    print("== Part D: holonomy layer -- pure torsion; forced -1 eigenplane cycles ==")
    check(f"H^1(N;U(1)) rank = |E|-|V|+1 = {len(NE)}-24+1 = {len(NE)-23} "
          f"(cotree size {len(COTREE)}; N connected: {len(PAR)==24})",
          len(COTREE) == len(NE) - 23 and len(PAR) == 24)
    h = fund_hols(GS.astype(complex))
    re = np.max(np.abs(h.imag)); neg = int(np.sum(h.real < 0))
    check(f"realized class is PURE TORSION: all {len(h)} fundamental holonomies "
          f"in {{+1,-1}} (max |Im| {re:.1e}); {neg} of them = -1 (class nontrivial)",
          re < 1e-9 and neg > 0)
    D = np.diag(np.exp(1j*rng.uniform(0, 2*np.pi, 24)))
    hg = fund_hols(D @ GS.astype(complex) @ D.conj().T)
    check(f"gauge invariance of fundamental holonomies (dev {np.max(np.abs(hg-h)):.1e})",
          np.max(np.abs(hg - h)) < 1e-9)
    # 18 eigenplane 4-cycles
    vals = []
    for t in LABELS:
        ci, cj = OBS2CTX[t]
        for a in (1, -1):
            p = [4*ci+k for k in range(4) if EIG[4*ci+k][t] == a]
            q = [4*cj+k for k in range(4) if EIG[4*cj+k][t] == a]
            B = GS[p[0], q[0]]*GS[q[0], p[1]]*GS[p[1], q[1]]*GS[q[1], p[0]]
            vals.append(B)
    vals = np.array(vals)
    check(f"all 18 eigenplane 4-cycles carry holonomy -1 (values all = -1/4: "
          f"dev {np.max(np.abs(vals + 0.25)):.1e})", np.max(np.abs(vals + 0.25)) < 1e-9)
    # forced-ness lemma over C: any two ONBs of a 2-plane, any unitary U(2) transition
    dev = 0.0
    for _ in range(60):
        Aa = rng.normal(size=(2, 2)) + 1j*rng.normal(size=(2, 2))
        U, _ = np.linalg.qr(Aa)
        al, be = U[0, 0], U[1, 0]
        p1, p2 = np.eye(2)
        q1, q2 = U[:, 0], U[:, 1]
        B = (p1.conj() @ q1)*(q1.conj() @ p2)*(p2.conj() @ q2)*(q2.conj() @ p1)
        dev = max(dev, abs(B - (-(abs(al)*abs(be))**2)))
    check(f"LEMMA (forced torsion): B(p1,q1,p2,q2) = -|alpha|^2|beta|^2 < 0 for ANY "
          f"U(2) (60 draws, dev {dev:.1e}) -- holonomy -1 forced for every d=4 "
          "realization of X", dev < 1e-12)
    print("  => canonical class z_obs = sum over the 9 observables of one eigenplane")
    print("     4-cycle: h(z_obs) = (-1)^9 = -1, independent of the plane choices")
    print("     (both planes forced -1). A holonomy invariant pinned at -1.")
    # cycle sign census on N (triangles and 4-cycles)
    tri = collections.Counter()
    for i, j, k in itertools.combinations(range(24), 3):
        if NADJ[i, j] and NADJ[j, k] and NADJ[i, k]:
            tri[int(np.sign(GS[i, j]*GS[j, k]*GS[i, k]))] += 1
    q4 = collections.Counter()
    for i, j, k, l in itertools.combinations(range(24), 4):
        for (a, b, c, d) in ((i, j, k, l), (i, k, j, l), (i, j, l, k)):
            if NADJ[a, b] and NADJ[b, c] and NADJ[c, d] and NADJ[d, a]:
                q4[int(np.sign(GS[a, b]*GS[b, c]*GS[c, d]*GS[d, a]))] += 1
    print(f"  cycle sign census: triangles +1:{tri[1]} -1:{tri[-1]}; "
          f"4-cycles +1:{q4[1]} -1:{q4[-1]} (holonomy is genuinely mixed on N)")

def partE():
    print("== Part E: hexagon transversals -- diagonal parity law + holonomy table ==")
    hexc = [0, 3, 1, 4, 2, 5]                       # R1 K1 R2 K2 R3 K3
    hshare = [next(iter(set(PMC[hexc[i]]) & set(PMC[hexc[(i+1) % 6]])))
              for i in range(6)]
    dshare = [next(iter(set(PMC[hexc[i]]) & set(PMC[hexc[i+3]]))) for i in range(3)]
    check("context hexagon R1 K1 R2 K2 R3 K3: consecutive AND antipodal pairs each "
          "share one observable (6 + 3 = 9, all of K_{3,3})",
          all(s is not None for s in hshare + dshare)
          and len(set(hshare + dshare)) == 9)
    tab = collections.Counter(); c4tab = collections.Counter()
    nhex = 0; parbad = 0
    for T in itertools.product(range(4), repeat=6):
        r = [4*hexc[i] + T[i] for i in range(6)]
        if any(orth(r[i], r[(i+1) % 6]) for i in range(6)): continue
        nhex += 1
        pres = tuple(int(not orth(r[j], r[j+3])) for j in range(3))
        if (3 - sum(pres)) % 2 == 0: parbad += 1
        h6 = int(np.sign(np.prod([GS[r[i], r[(i+1) % 6]] for i in range(6)])))
        tab[(pres, h6)] += 1
        for j in range(3):
            if pres[j]:
                c = int(np.sign(GS[r[j], r[j+1]]*GS[r[j+1], r[j+2]]
                                * GS[r[j+2], r[j+3]]*GS[r[j+3], r[j]]))
                c4tab[c] += 1
    check(f"{nhex} hexagon transversals; DIAGONAL PARITY LAW: odd number of MISSING "
          f"antipodal diagonals in every one ({parbad} violations) -- the nu(L) = -1 "
          "class seen on the hexagon", nhex > 0 and parbad == 0)
    pats = sorted({p for (p, s) in tab})
    for p in pats:
        sp, sm = tab[(p, 1)], tab[(p, -1)]
        print(f"    diagonals present {p}: hexagon holonomy +1 x{sp}, -1 x{sm}")
    forced_hex = all((tab[(p, 1)] == 0) or (tab[(p, -1)] == 0) for p in pats)
    print(f"  hexagon holonomy forced by diagonal pattern: {forced_hex}")
    print(f"  4-cycles through a present diagonal: +1 x{c4tab[1]}, -1 x{c4tab[-1]}")
    if c4tab[-1] == 0:
        msg = "law: EVERY 4-cycle through a present antipodal diagonal has holonomy +1"
    elif c4tab[1] == 0:
        msg = "law: EVERY 4-cycle through a present antipodal diagonal has holonomy -1"
    else:
        msg = "no single-sign law for diagonal 4-cycles (mixed) -- recorded as data"
    check(msg + " => h6 = -1 exactly on CHORDLESS hexagon transversals: the AvN sign "
          "as a holonomy VALUE on canonical 6-cycles", c4tab[-1] == 0)

def partF():
    print("== Part F: rigidity -- is the whole local system graph-forced? ==")
    from scipy.optimize import least_squares
    def resid(x):
        V = (x[0::2] + 1j*x[1::2]).reshape(24, 4)
        ip = V.conj() @ V.T
        r = [np.real(np.sum(V.conj()*V, axis=1)) - 1]
        r.append(np.array([ip[i, j].real for (i, j) in XE]))
        r.append(np.array([ip[i, j].imag for (i, j) in XE]))
        return np.concatenate(r)
    x0 = np.zeros(192); x0[0::2] = RAYS.ravel()
    assert np.max(np.abs(resid(x0))) < 1e-12
    # infinitesimal rigidity: FD Jacobian, nullity vs gauge dimension
    J = np.zeros((24 + 2*len(XE), 192)); hstep = 1e-6
    for k in range(192):
        e = np.zeros(192); e[k] = hstep
        J[:, k] = (resid(x0 + e) - resid(x0 - e))/(2*hstep)
    sv = np.linalg.svd(J, compute_uv=False)
    nul = int(np.sum(sv < 1e-7*sv[0]))
    # gauge tangents: u(4) (16) + 24 vertex phases, overlap = global phase (dim 1)
    V0 = RAYS.astype(complex)
    tang = []
    Hb = []
    for a in range(4):
        for b in range(a, 4):
            H = np.zeros((4, 4), complex)
            if a == b: H[a, a] = 1
            else:
                H[a, b] = 1; H[b, a] = 1
                Hb.append(H.copy())
                H2 = np.zeros((4, 4), complex); H2[a, b] = 1j; H2[b, a] = -1j
                Hb.append(H2); continue
            Hb.append(H)
    for H in Hb:
        dV = V0 @ (1j*H).T
        t = np.zeros(192); t[0::2] = dV.real.ravel(); t[1::2] = dV.imag.ravel()
        tang.append(t)
    for i in range(24):
        dV = np.zeros((24, 4), complex); dV[i] = 1j*V0[i]
        t = np.zeros(192); t[0::2] = dV.real.ravel(); t[1::2] = dV.imag.ravel()
        tang.append(t)
    Tm = np.array(tang)
    gdim = int(np.linalg.matrix_rank(Tm, tol=1e-9))
    ingauge = np.max(np.abs(J @ Tm.T))
    check(f"gauge orbit dim = {gdim} = 39 (u(4)=16 + 24 phases - 1 global); gauge "
          f"tangents in ker(J) (dev {ingauge:.1e})", gdim == 39 and ingauge < 1e-4)
    check(f"INFINITESIMAL RIGIDITY over C: nullity(J) = {nul} = gauge dim (sv gap "
          f"{sv[-nul-1]/sv[0]:.1e} vs {sv[-nul]/sv[0]:.1e}) -- no flexes: the "
          "orthogonality graph pins the realization to first order", nul == gdim)
    # real-restricted: params = real parts only; residuals = norms + Re rows
    rows = list(range(24 + len(XE)))
    Jr = J[np.ix_(rows, list(range(0, 192, 2)))]
    svr = np.linalg.svd(Jr, compute_uv=False)
    nulr = int(np.sum(svr < 1e-7*svr[0]))
    check(f"real-restricted nullity = {nulr} = 6 = dim O(4): rigid over R too",
          nulr == 6)
    # global evidence: re-realizations from noisy + random starts
    href = fund_hols(GS.astype(complex))
    t0 = time.time(); stats = collections.Counter(); maxmag = 0.0; maxhol = 0.0
    starts = [x0 + rng.normal(scale=0.35, size=192) for _ in range(4)] \
        + [rng.normal(size=192) for _ in range(8)]
    for xs in starts:
        if time.time() - t0 > 30: stats['skipped'] += 1; continue
        sol = least_squares(resid, xs, method='trf', ftol=1e-15, xtol=1e-15,
                            gtol=1e-15, max_nfev=400)
        if np.max(np.abs(sol.fun)) > 1e-9: stats['noconv'] += 1; continue
        V = (sol.x[0::2] + 1j*sol.x[1::2]).reshape(24, 4)
        ip = V.conj() @ V.T
        if min(abs(ip[i, j]) for (i, j) in NE) < 1e-3:
            stats['unfaithful'] += 1; continue
        dmag = np.max(np.abs(np.abs(ip) - np.abs(GS)))
        hnew = fund_hols(ip)
        dh = min(np.max(np.abs(hnew - href)), np.max(np.abs(hnew.conj() - href)))
        maxmag = max(maxmag, dmag); maxhol = max(maxhol, dh)
        stats['match' if (dmag < 1e-6 and dh < 1e-6) else 'DIFFER'] += 1
    check(f"global re-realization: {stats['match']} faithful solutions, ALL with "
          f"identical magnitudes (max dev {maxmag:.1e}) and identical holonomy class "
          f"up to conjugation (max dev {maxhol:.1e}); DIFFER: {stats['DIFFER']}, "
          f"non-converged: {stats['noconv']}, unfaithful: {stats['unfaithful']}, "
          f"skipped: {stats['skipped']}",
          stats['match'] >= 1 and stats['DIFFER'] == 0)
    print("  => (rigidity, numeric) the exclusivity graph forces the ENTIRE local")
    print("     system (m, h) up to gauge and complex conjugation; h is pure torsion")
    print("     and every torsion invariant above (-1 eigenplane cycles, nu(L),")
    print("     h(z_obs)) is therefore realization-independent: state-independence.")

def partG():
    print("== Part G: law on ALL K33 hexagons; moduli contrast PM (rigid) vs C5 ==")
    # (i) every hexagon of K_{3,3}: chordless transversals -1, others +1
    nhexes = 0; bad = 0; chordless_total = 0
    for rperm in itertools.permutations([1, 2]):
        for cperm in itertools.permutations([3, 4, 5]):
            hexc = [0, cperm[0], rperm[0], cperm[1], rperm[1], cperm[2]]
            nhexes += 1
            for T in itertools.product(range(4), repeat=6):
                r = [4*hexc[i] + T[i] for i in range(6)]
                if any(orth(r[i], r[(i+1) % 6]) for i in range(6)): continue
                npres = sum(int(not orth(r[j], r[j+3])) for j in range(3))
                h6 = int(np.sign(np.prod([GS[r[i], r[(i+1) % 6]]
                                          for i in range(6)])))
                if npres == 0:
                    chordless_total += 1
                    if h6 != -1: bad += 1
                else:
                    if (3 - npres) % 2 == 0 or h6 != 1: bad += 1
    check(f"HEXAGON HOLONOMY LAW on all {nhexes} labeled hexagons of K33: h6 = -1 "
          f"exactly on chordless (all-crossing) transversals ({chordless_total} of "
          f"them), h6 = +1 otherwise, odd-missing parity throughout ({bad} violations)",
          bad == 0 and chordless_total >= 96)
    # (ii) same Jacobian machinery on the KCBS pentagon (d=3): flex dimension > 0
    c5 = np.cos(np.pi/5); ct2 = c5/(1+c5); st = np.sqrt(1-ct2)
    V5 = np.array([[st*np.cos(4*np.pi*k/5), st*np.sin(4*np.pi*k/5), np.sqrt(ct2)]
                   for k in range(5)])
    E5 = [(k, (k+1) % 5) for k in range(5)]
    def resid5(x):
        V = (x[0::2] + 1j*x[1::2]).reshape(5, 3)
        ip = V.conj() @ V.T
        return np.concatenate([np.real(np.sum(V.conj()*V, axis=1)) - 1,
                               [ip[i, j].real for (i, j) in E5],
                               [ip[i, j].imag for (i, j) in E5]])
    x5 = np.zeros(30); x5[0::2] = V5.ravel()
    assert np.max(np.abs(resid5(x5))) < 1e-12
    J5 = np.zeros((15, 30)); hh = 1e-6
    for k in range(30):
        e = np.zeros(30); e[k] = hh
        J5[:, k] = (resid5(x5 + e) - resid5(x5 - e))/(2*hh)
    sv5 = np.linalg.svd(J5, compute_uv=False)
    nul5 = 30 - int(np.sum(sv5 > 1e-7*sv5[0]))       # nullity = #params - rank
    g5 = 9 + 5 - 1                                   # u(3) + 5 phases - global
    check(f"KCBS C5 (d=3): nullity = {nul5}, gauge dim = {g5}, FLEXES = {nul5-g5} > 0 "
          "(the state-dependent moduli: magnitude/holonomy knob of V49) -- vs PM "
          "flexes = 0 (Part F): rigidity is the switch between the two layers",
          nul5 - g5 > 0)

PARTS = {"A": partA, "B": partB, "C": partC, "D": partD, "E": partE, "F": partF,
         "G": partG}
if __name__ == "__main__":
    sel = [s.upper() for s in sys.argv[1:]] or list("ABCDEFG")
    for s in sel: PARTS[s]()
    print("CU UNIFY PROBE " + ("PASS" if OK else "FAIL"))
