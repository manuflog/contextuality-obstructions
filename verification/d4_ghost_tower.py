# V33 - THE GHOST AT d=4: does the deleted context's ghost refract through the 2-adic tower?
# Family: cert4 (a d=4 PM square: 6 contexts, 9 observables, each in exactly 2 contexts,
# AvN value d/2=2). Delete a context C*; then over L = Z4-solutions of the remaining system:
#  (i)   solvability and |L|;
#  (ii)  the ghost value g(lam) = sum_{v in C*} lam_v mod 4: constant? value set vs quantum s*?
#        TOWER PREDICTION: offset lives at the top layer, g - s* in {2} or {1,3}-free coset of 2.
#  (iii) CF landscape of the deleted family: random states + C*-joint-eigenstates
#        (the d=4 'Bell' analogues) - do eigenstates saturate CF=1?
#  (iv)  top-layer law probe: apply the d=2 ghost formula to the ORDER-2 SHADOWS
#        W_v^2 of the deleted triple and compare to CF.
import numpy as np, itertools, json
from state_sector_probe import contextual_fraction
from weyl import build
X,Z,w,tau,W,N=build(4,2)   # dim 16, w=i
fam=[[tuple(v) for v in it["ctx"]] for it in json.load(open("cert4_min.json"))["items"]]
obs=sorted({t for C in fam for t in C}); oi={t:k for k,t in enumerate(obs)}
def s_of(C):
    M=np.eye(16,dtype=complex)
    for v in C: M=M@W(np.array(v))
    ph=np.angle(np.trace(M)/abs(np.trace(M)))
    return int(np.round(ph/(np.pi/2)))%4
S=[s_of(C) for C in fam]
# PINNED RESULTS (both deletions DROP=0 [all order-2 ghost] and DROP=1 [mixed 2/4]):
#  TOWER GHOST LAW: ghost offset gamma - s* = 2 = d/2 exactly - the class refracts as the
#    top 2-adic layer; unifies d=2, where the offset was likewise d/2 (opposite tetrahedra).
#  EIGENSTATE SATURATION: every C*-joint-eigenstate class gives CF = 1.0000 exactly.
#  POPULATED SECTOR: under correct spectral semantics Haar-random states have CF>0
#    (sampled up to ~0.21); the earlier CF=0 reading was the unrestricted-LP artifact.
#  NEGATIVE (pinned): the d=4 law is NOT a function of the Hermitian ghost shadows alone -
#    the Hermitianized d=2 formula predicts 0 exactly where random CF reaches 0.138+;
#    the operative facets involve order-4 moments; enumeration in moment space open.
#  SEMANTICS: L must be SPECTRALLY restricted (non-spectral assignments hit e=0 outcomes,
#    whose <=0 constraints the naive LP omits); unrestricted-L under-computes CF.
spec={}
for v in obs:
    ev=np.linalg.eigvals(W(np.array(v)))
    spec[v]=sorted(set(int(np.round(np.angle(x)/(np.pi/2)))%4 for x in ev))
def analyze(DROP):
    CTX=[C for i,C in enumerate(fam) if i!=DROP]; Cs=fam[DROP]; sstar=S[DROP]
    A=np.zeros((5,9),int)
    for r,C in enumerate(CTX):
        for v in C: A[r,oi[v]]+=1
    rhs=np.array([S[i] for i in range(6) if i!=DROP])
    grid=np.array(list(itertools.product(range(4),repeat=9)))
    ok=((grid@A.T)%4==rhs%4).all(axis=1)
    for v in obs: ok&=np.isin(grid[:,oi[v]],spec[v])
    L=grid[ok]
    g=sorted(set((L[:,[oi[v] for v in Cs]].sum(axis=1))%4))
    def jp(C,idx):
        Ws=[W(np.array(v)) for v in C]; P={}
        for j1,j2 in itertools.product(range(4),repeat=2):
            P1=sum((w**(-a*j1))*np.linalg.matrix_power(Ws[0],a) for a in range(4))/4
            P2=sum((w**(-a*j2))*np.linalg.matrix_power(Ws[1],a) for a in range(4))/4
            Pi=P1@P2
            if abs(np.trace(Pi))>1e-9: P[(j1,j2,(S[idx]-j1-j2)%4)]=Pi
        return P
    idxs=[i for i in range(6) if i!=DROP]
    PJ=[jp(C,idxs[k]) for k,C in enumerate(CTX)]; PJs=jp(Cs,DROP)
    import scipy.optimize as so
    def CF(psi):
        Aub=[]; bub=[]
        for ci,C in enumerate(CTX):
            ix=[oi[v] for v in C]
            for j,Pi in PJ[ci].items():
                Aub.append((L[:,ix]==np.array(j)).all(axis=1).astype(float))
                bub.append(float(np.real(psi.conj()@Pi@psi)))
        r=so.linprog(-np.ones(len(L)),A_ub=np.array(Aub),b_ub=np.array(bub),bounds=[(0,None)]*len(L),method="highs")
        return 1.0+r.fun
    ec=[]
    for j,Pi in PJs.items():
        _,vecs=np.linalg.eigh((Pi+Pi.conj().T)/2); ec.append(CF(vecs[:,-1]))
    rng=np.random.default_rng(21)
    rc=[CF((lambda x:x/np.linalg.norm(x))(rng.normal(size=16)+1j*rng.normal(size=16))) for _ in range(6)]
    orders=[4 if any(x%2 for x in v) else 2 for v in Cs]
    print(f"DROP={DROP}: orders {orders} | |L|={len(L)} | ghost offset {[int((x-sstar)%4) for x in g]} "
          f"| eig CF [{min(ec):.4f},{max(ec):.4f}] ({len(ec)} classes) | rand CF max {max(rc):.4f} (sector populated)")
    return set(int((x-sstar)%4) for x in g)=={2} and abs(min(ec)-1)<1e-7 and abs(max(ec)-1)<1e-7 and max(rc)>1e-3
print(f"context phase exponents S = {S}, sum mod 4 = {sum(S)%4} (the class value d/2)")
oks=[analyze(0),analyze(1)]
print("TOWER GHOST LAW + SATURATION: " + ("PASS on both deletions" if all(oks) else "FAIL"))
