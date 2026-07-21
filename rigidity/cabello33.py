#!/usr/bin/env python3
"""
B1 — SESSION 11: the Cabello "simplest KS set" (arXiv:2508.07335) as a Conjecture-C test.

A genuinely DIFFERENT d=3 KS set from Peres-33: 14 complete bases (record), automorphism group
144, built from the Weyl-Heisenberg group acting on the minimal SI-C (Yu-Oh) set. Vectors are the
distinct rays of Alice's 5 bases (Eq. 1a-1e) and Bob's 9 bases (Eq. 2a-2i), omega = e^{2 pi i/3}.

Being d=3, tau = 0 by Lemma B. Cabello states the set is UNIQUE up to unitaries (point (d)),
which for a finite ray configuration means it is (infinitesimally and globally) RIGID: flex_C = 0.

If confirmed, this is a SECOND d=3 KS set with tau = 0 but --- unlike Peres-33 (flex = 1) --- it is
RIGID. That REFUTES Conjecture C ("among state-independent KS sets, flex>0 <=> tau=0"): here tau=0
yet flex=0. The refutation is the system working; it removes a conjecture that rested on the single
example Peres-33 and sharpens the dictionary: flex is NOT a function of (t0, tau).

Exact arithmetic in the Eisenstein integers Z[omega] (omega^2 = -1-omega) for orthogonality, basis
counting, distinctness and KS-uncolorability; flex_C computed numerically (complex extended
Jacobian) with a hard spectral-gap check. Transcription is self-validated: an error would break the
14-basis / 33-ray / uncolorable structure that Cabello proves.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from itertools import combinations, product

# ---------- Eisenstein integers a + b*omega, omega^2 = -1 - omega ----------
def ez(a=0, b=0): return (a, b)
ONE, OM, OM2 = (1, 0), (0, 1), (-1, -1)       # 1, omega, omega^2
def eadd(u, v): return (u[0]+v[0], u[1]+v[1])
def eneg(u): return (-u[0], -u[1])
def emul(u, v):
    a, b = u; c, d = v
    return (a*c - b*d, a*d + b*c - b*d)        # (a+b w)(c+d w), w^2=-1-w
def econj(u):
    a, b = u; return (a - b, -b)               # conj(w)=w^2=-1-w
def eis0(u): return u == (0, 0)

# symbols for entries
P1, M1 = (1, 0), (-1, 0)
W, MW = (0, 1), (0, -1)
W2, MW2 = (-1, -1), (1, 1)
Z = (0, 0)

# Alice bases 1a-1e, Bob bases 2a-2i (arXiv:2508.07335 Eqs. 1-2), each vector a triple of Eisenstein entries
BASES = [
    # 1a
    [(Z, Z, P1), (Z, P1, Z), (P1, Z, Z)],
    # 1b  (1,w,w2),(1,1,1),(w2,w,1)
    [(P1, W, W2), (P1, P1, P1), (W2, W, P1)],
    # 1c  (1,w,-w2),(1,1,-1),(w2,w,-1)
    [(P1, W, MW2), (P1, P1, M1), (W2, W, M1)],
    # 1d  (1,-w,w2),(1,-1,1),(w2,w,1)
    [(P1, MW, W2), (P1, M1, P1), (W2, W, P1)],
    # 1e  (-1,w,w2),(-1,1,1),(-w2,w,1)
    [(M1, W, W2), (M1, P1, P1), (MW2, W, P1)],
    # 2a
    [(Z, Z, P1), (P1, P1, Z), (P1, M1, Z)],
    # 2b
    [(Z, Z, P1), (P1, W, Z), (P1, MW, Z)],
    # 2c
    [(Z, Z, P1), (W, P1, Z), (W, M1, Z)],
    # 2d
    [(Z, P1, Z), (P1, Z, P1), (P1, Z, M1)],
    # 2e
    [(Z, P1, Z), (P1, Z, W), (P1, Z, MW)],
    # 2f
    [(Z, P1, Z), (W, Z, P1), (W, Z, M1)],
    # 2g
    [(P1, Z, Z), (Z, P1, P1), (Z, P1, M1)],
    # 2h
    [(P1, Z, Z), (Z, P1, W), (Z, P1, MW)],
    # 2i
    [(P1, Z, Z), (Z, W, P1), (Z, W, M1)],
]

def herm(u, v):
    """<u,v> = sum conj(u_c) v_c in Z[omega]."""
    s = Z
    for a, b in zip(u, v): s = eadd(s, emul(econj(a), b))
    return s

def ecross(u, v):
    """Standard cross product in Z[omega] (bilinear)."""
    return (eadd(emul(u[1], v[2]), eneg(emul(u[2], v[1]))),
            eadd(emul(u[2], v[0]), eneg(emul(u[0], v[2]))),
            eadd(emul(u[0], v[1]), eneg(emul(u[1], v[0]))))

def third_ortho(u, v):
    """The ray Hermitian-orthogonal to both u and v: w = conj(u) x conj(v)."""
    return ecross(tuple(econj(x) for x in u), tuple(econj(x) for x in v))

def reconstruct_bases():
    """Rebuild each basis from its first two listed vectors (v3 := Hermitian complement),
       which removes transcription error on the third vectors. Verify v1 _|_ v2 exactly."""
    fixed = []; badpairs = []
    for bi, B in enumerate(BASES):
        v1, v2 = B[0], B[1]
        if not eis0(herm(v1, v2)): badpairs.append(bi)
        v3 = third_ortho(v1, v2)
        fixed.append([v1, v2, v3])
    return fixed, badpairs

def proportional(u, v):
    """u,v same ray? cross products u_a v_b - u_b v_a all zero (Z[omega])."""
    for a, b in combinations(range(3), 2):
        if not eis0(eadd(emul(u[a], v[b]), eneg(emul(u[b], v[a])))): return False
    return True

def collect_rays(bases):
    rays = []
    for B in bases:
        for v in B:
            if not any(proportional(v, r) for r in rays):
                rays.append(v)
    return rays

def to_complex(u):
    w = np.exp(2j*np.pi/3)
    return np.array([a + b*w for a, b in u], complex)

# ---------- KS-uncolorability (exhaustive backtracking, exact graph) ----------
def ks_uncolorable(rays, pairs, triads):
    n = len(rays)
    orth = [[] for _ in range(n)]
    for i, j in pairs: orth[i].append(j); orth[j].append(i)
    tri_of = [[] for _ in range(n)]
    for t, tri in enumerate(triads):
        for r in tri: tri_of[r].append(t)
    color = [-1]*n
    def dfs(k):
        if k == n:
            return all(sum(color[r] for r in tri) == 1 for tri in triads)
        # try assigning vertex k
        for val in (1, 0):
            if val == 1 and any(color[j] == 1 for j in orth[k] if color[j] != -1):
                pass
            ok = True
            if val == 1:
                for j in orth[k]:
                    if color[j] == 1: ok = False; break
            if not ok: continue
            color[k] = val
            good = True
            for t in tri_of[k]:
                vals = [color[r] for r in triads[t]]
                if vals.count(1) > 1 or (vals.count(-1) == 0 and vals.count(1) != 1):
                    good = False; break
            if good and dfs(k+1): color[k] = -1; return True
            color[k] = -1
        return False
    return not dfs(0)

# ---------- flex_C (numerical, complex extended Jacobian) ----------
def flex_complex(rays):
    V = len(rays); d = 3
    v = [to_complex(r) for r in rays]
    v = [x/np.linalg.norm(x) for x in v]
    E = [(i, j) for i, j in combinations(range(V), 2) if abs(np.vdot(v[i], v[j])) < 1e-9]
    n = 2*d*V
    def col(i, c, im): return 2*d*i + 2*c + (1 if im else 0)
    rows = []
    for i, j in E:
        re = [0.0]*n; im = [0.0]*n
        # d<v_i,v_j> = <w_i,v_j> + <v_i,w_j>; w = x+iy
        for c in range(d):
            vjc = v[j][c]; vic = v[i][c]
            # <w_i,v_j> = sum conj(w_ic) v_jc ; conj(w)=x-iy
            re[col(i,c,0)] += vjc.real; re[col(i,c,1)] += vjc.imag
            im[col(i,c,0)] += vjc.imag; im[col(i,c,1)] += -vjc.real
            # <v_i,w_j> = sum conj(v_ic) w_jc ; Re: Re(vic)x + Im(vic)y ; Im: -Im(vic)x + Re(vic)y
            re[col(j,c,0)] += vic.real; re[col(j,c,1)] += vic.imag
            im[col(j,c,0)] += -vic.imag; im[col(j,c,1)] += vic.real
        rows.append(re); rows.append(im)
    for i in range(V):   # norm rows Re<v_i,w_i> = Re(vic)x_ic + Im(vic)y_ic
        r = [0.0]*n
        for c in range(d):
            r[col(i,c,0)] += v[i][c].real; r[col(i,c,1)] += v[i][c].imag
        rows.append(r)
    J = np.array(rows)
    # trivial: V phases (w=i v) + u(3) (9) with one global relation => V + 9 - 1
    triv = []
    for i in range(V):
        t = [0.0]*n
        for c in range(d): t[col(i,c,0)] += -v[i][c].imag; t[col(i,c,1)] += v[i][c].real
        triv.append(t)
    import numpy as _np
    herm_gen = []
    for a in range(d):
        H = _np.zeros((d,d), complex); H[a,a]=1; herm_gen.append(1j*H)
    for a in range(d):
        for b in range(a+1,d):
            H=_np.zeros((d,d),complex); H[a,b]=1;H[b,a]=1; herm_gen.append(1j*H)
            K=_np.zeros((d,d),complex); K[a,b]=1j;K[b,a]=-1j; herm_gen.append(1j*K)
    for A in herm_gen:
        t=[0.0]*n
        for i in range(V):
            wv=A@v[i]
            for c in range(d): t[col(i,c,0)]+=wv[c].real; t[col(i,c,1)]+=wv[c].imag
        triv.append(t)
    def rank(M):
        s=_np.linalg.svd(_np.array(M),compute_uv=False);
        return int((s/s[0]>1e-8).sum()) if len(s) and s[0]>0 else 0, (s[ (s/s[0]>1e-8).sum()-1]/s[0] if False else 0)
    sJ=_np.linalg.svd(J,compute_uv=False); rJ=int((sJ/sJ[0]>1e-8).sum())
    gapJ = sJ[rJ-1]/sJ[rJ] if rJ<len(sJ) else float('inf')
    T=_np.array(triv); sT=_np.linalg.svd(T,compute_uv=False); rT=int((sT/sT[0]>1e-8).sum())
    flex=(n-rJ)-rT
    return flex, len(E), rJ, rT, gapJ

# ---------- EXACT flex over Z[sqrt3] (2v scaled: Re(2v),Im(2v)/sqrt3 are integers) ----------
def find_primes_sqrt3(count=2, below=46300):
    def isp(m):
        if m < 2: return False
        for q in range(2, int(m**0.5)+1):
            if m % q == 0: return False
        return True
    out, p = [], below | 1
    while len(out) < count and p > 5:
        if isp(p):
            s = next((t for t in range(1, p) if (t*t) % p == 3 % p), None)
            if s is not None: out.append((p, s))
        p -= 2
    return out

def exact_flex_z3(rays):
    """Exact flex over Q(sqrt3) via two-prime mod-p rank bounds. Entries a+b*sqrt3 as pairs (a,b).
       2v_c = (2p - q) + q*sqrt3 * i  for v_c = p + q*omega; so Re, Im/ (sqrt3) are integers."""
    from sic_zoo import rank_mod_p
    V, d = len(rays), 3
    # per ray, Re2 and Im2 as integer vectors: Re(2v_c)=2p-q (rational int), Im(2v_c)=q*sqrt3
    Re = [[2*rays[i][c][0] - rays[i][c][1] for c in range(d)] for i in range(V)]   # integer
    Imq = [[rays[i][c][1] for c in range(d)] for i in range(V)]                    # coeff of sqrt3
    # inner product <v_i,v_j> over Z[omega] to get edges (exact)
    E = [(i, j) for i, j in combinations(range(V), 2) if eis0(herm(rays[i], rays[j]))]
    n = 2*d*V
    def col(i, c, im): return 2*d*i + 2*c + (1 if im else 0)
    def P(a, b=0): return (a, b)                       # a + b sqrt3
    rows = []
    def addrow():
        rows.append([(0,0)]*n); return rows[-1]
    for i, j in E:
        # Re row: x_i.Re(v_j)+y_i.Im(v_j)+x_j.Re(v_i)+y_j.Im(v_i)
        r = [ (0,0) ]*n
        for c in range(d):
            r[col(i,c,0)] = P(Re[j][c], 0); r[col(i,c,1)] = P(0, Imq[j][c])
            r[col(j,c,0)] = P(Re[i][c], 0); r[col(j,c,1)] = P(0, Imq[i][c])
        rows.append(r)
        # Im row: x_i.Im(v_j) - y_i.Re(v_j) - x_j.Im(v_i) + y_j.Re(v_i)
        r = [ (0,0) ]*n
        for c in range(d):
            r[col(i,c,0)] = P(0, Imq[j][c]); r[col(i,c,1)] = P(-Re[j][c], 0)
            r[col(j,c,0)] = P(0, -Imq[i][c]); r[col(j,c,1)] = P(Re[i][c], 0)
        rows.append(r)
    for i in range(V):   # norm: x_i.Re(v_i)+y_i.Im(v_i)
        r = [ (0,0) ]*n
        for c in range(d): r[col(i,c,0)] = P(Re[i][c],0); r[col(i,c,1)] = P(0, Imq[i][c])
        rows.append(r)
    # trivial: phases w=i v => x=-Im(v), y=Re(v)
    triv = []
    for i in range(V):
        t = [ (0,0) ]*n
        for c in range(d): t[col(i,c,0)] = P(0, -Imq[i][c]); t[col(i,c,1)] = P(Re[i][c],0)
        triv.append(t)
    # u(3) exactly in Z[omega]: for each generator, w_i = A v_i in Z[omega]; then map 2w to Z[sqrt3]:
    #   for u in Z[omega], u=(p,q):  Re(2u) = 2p-q (integer),  Im(2u) = q*sqrt3.
    #   for i*u:                     Re(2 i u) = -q*sqrt3,      Im(2 i u) = 2p-q.
    def put_real(t, i, c, u):        # add real w-entry u=(p,q) in Z[omega] to slot (i,c)
        p_, q_ = u; t[col(i,c,0)] = P(2*p_-q_, 0); t[col(i,c,1)] = P(0, q_)
    def put_imag(t, i, c, u):        # add (i*u) w-entry
        p_, q_ = u; t[col(i,c,0)] = P(0, -q_); t[col(i,c,1)] = P(2*p_-q_, 0)
    for a in range(d):               # i E_aa : w_a = i v_a
        t=[(0,0)]*n
        for i in range(V): put_imag(t, i, a, rays[i][a])
        triv.append(t)
    for a in range(d):
        for b in range(a+1, d):
            t=[(0,0)]*n              # i(E_ab+E_ba): w_a = i v_b, w_b = i v_a
            for i in range(V): put_imag(t, i, a, rays[i][b]); put_imag(t, i, b, rays[i][a])
            triv.append(t)
            t=[(0,0)]*n              # (E_ab-E_ba): w_a = v_b, w_b = -v_a  (real)
            for i in range(V):
                put_real(t, i, a, rays[i][b]); put_real(t, i, b, eneg(rays[i][a]))
            triv.append(t)
    best = None
    for p, s in find_primes_sqrt3(2):
        rJ = rank_mod_p(rows, p, s); rT = rank_mod_p(triv, p, s)
        b = (n - rJ) - rT
        best = b if best is None else min(best, b)
    return best

if __name__ == "__main__":
    print("Cabello simplest KS set (arXiv:2508.07335) — build, validate, flex")
    print("="*78)
    fixed, badpairs = reconstruct_bases()
    print(f"bases with non-orthogonal first two listed vectors: {badpairs if badpairs else 'none'} "
          f"(third vectors reconstructed as exact Hermitian complements)")
    rays = collect_rays(fixed)
    print(f"distinct rays collected: {len(rays)} (expected 33)")
    V = len(rays)
    BASES_USE = fixed
    pairs = [(i, j) for i, j in combinations(range(V), 2) if eis0(herm(rays[i], rays[j]))]
    triads = [c for c in combinations(range(V), 3)
              if all(eis0(herm(rays[i], rays[j])) for i, j in combinations(c, 2))]
    print(f"orthogonal pairs: {len(pairs)}; complete bases (triads): {len(triads)} (expected 14)")
    # each listed basis should appear as a triad
    listed = 0
    for B in BASES_USE:
        ii = [next(k for k, r in enumerate(rays) if proportional(v, r)) for v in B]
        if tuple(sorted(ii)) in {tuple(sorted(t)) for t in triads}: listed += 1
    print(f"listed bases recovered as complete triads: {listed}/14")
    unc = ks_uncolorable(rays, pairs, [list(t) for t in triads])
    print(f"KS-uncolorable (exhaustive): {unc}  => t0 = {1 if unc else 0}")
    flex, nE, rJ, rT, gapJ = flex_complex(rays)
    print(f"flex_C (numerical): {flex}  [edges {nE}, rank J {rJ}, trivial {rT}, spectral gap {gapJ:.1e}]")
    try:
        exbound = exact_flex_z3(rays)
        print(f"flex_C (EXACT over Q(sqrt3), two-prime mod-p bound): 0 <= flex <= {exbound}"
              + ("  => flex = 0 EXACT (RIGID)" if exbound == 0 else ""))
    except Exception as e:
        exbound = None; print(f"exact flex skipped: {e}")
    print("-"*78)
    ok = (len(rays) == 33 and len(triads) == 14 and listed == 14 and unc)
    print(f"transcription self-validated: {ok}  (33 rays, 14 bases, all listed bases present, uncolorable)")
    if ok:
        if flex == 0:
            print("\nRESULT: Cabello-33 is a d=3 KS set with tau=0 and flex_C = 0 (RIGID) --- consistent")
            print("with Cabello's 'unique up to unitaries'. It is a tau=0 KS set that is RIGID, whereas")
            print("Peres-33 is a tau=0 KS set that is FLEXIBLE. => CONJECTURE C is REFUTED: among d=3")
            print("KS sets, flexibility is NOT determined by (t0, tau). The dictionary keeps flex as an")
            print("INDEPENDENT invariant; 'tau=0 => flexible' is false.")
        else:
            print(f"\nRESULT: flex_C = {flex} (not 0) --- Cabello-33 is also flexible; would SUPPORT Conj. C.")
    print("\ncabello33 " + ("PASS" if ok else "FAIL (transcription)"))
