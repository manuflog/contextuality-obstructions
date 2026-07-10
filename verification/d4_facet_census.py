# V43 - COMPLETE FACET CENSUS AND CLASSIFICATION THEOREM at d=4 (DROP=0).
# TWO INDEPENDENT DERIVATIONS, MUTUALLY CONFIRMING:
#   (1) Adjacency decomposition: BFS on the facet-ridge graph of conv(Y) modulo the
#       derived vertex-symmetry group (|G|=768), every step exactly certified (integer
#       nullspace hyperplane recovery through rank-32 tight sets; exact validity on all
#       64 integer vertices). Ridge enumeration per facet by exact engines; lrs WEDGES
#       reproducibly on several of these inputs (pinned), cdd(float) cracks most,
#       Normaliz 3.10.2 the rest. Closure: every neighbor of every class lies in the
#       class set.
#   (2) Normaliz direct enumeration of the full polytope: 23,256 exact integer support
#       hyperplanes.
# The two class sets coincide exactly: 61 classes, 23,256 facets. (An earlier in-walk
# count of "121 classes / 46,508" was a bookkeeping duplication - class dictionaries
# keyed by two different key types stored each orbit twice; caught ONLY by the
# independent cross-check. Pinned as a method lesson.)
# ON-HULL LEMMA (exact, this script): quantum behaviors satisfy normalization,
# shared-observable marginal consistency, and 24 spectral-support identities; the affine
# space these cut out has dimension EXACTLY 33 = the deterministic affine span (integer
# rank computation below). Hence quantum moment vectors lie in the hull's affine span as
# a THEOREM, not a numerical observation. With the LP pairing (V31):
#   THEOREM: CF(rho) > 0  <=>  mu(rho) violates one of the 23,256 facets (61 classes).
# This script verifies the artifact and the theorem's operational content.
import numpy as np, itertools, json, os, pickle
_GP=os.path.join(os.path.dirname(os.path.abspath(__file__)),'d4_group_seed.pkl')
if not os.path.exists(_GP): _GP='/tmp/v43_seed.pkl'
import scipy.optimize as so
from weyl import build
X,Z,w,tau,W,_=build(4,2)
fam=[[tuple(v) for v in it["ctx"]] for it in json.load(open("cert4_min.json"))["items"]]
obs=sorted({t for C in fam for t in C}); oi={t:k for k,t in enumerate(obs)}
def s_of(C):
    M=np.eye(16,dtype=complex)
    for v in C: M=M@W(np.array(v))
    return int(np.round(np.angle(np.trace(M)/abs(np.trace(M)))/(np.pi/2)))%4
S=[s_of(C) for C in fam]
CTX=[fam[i] for i in range(1,6)]
spec={v:sorted(set(int(np.round(np.angle(x)/(np.pi/2)))%4 for x in np.linalg.eigvals(W(np.array(v))))) for v in obs}
A=np.zeros((5,9),int)
for r,C in enumerate(CTX):
    for v in C: A[r,oi[v]]+=1
rhs=np.array([S[i] for i in range(6) if i!=0])
grid=np.array(list(itertools.product(range(4),repeat=9)))
ok=((grid@A.T)%4==rhs%4).all(axis=1)
for v in obs: ok&=np.isin(grid[:,oi[v]],spec[v])
L=grid[ok]
CH=[(a,b) for a in range(4) for b in range(4) if (a,b)!=(0,0)]
V150=[]
for row in L:
    vec=[]
    for C in CTX:
        j1,j2=int(row[oi[C[0]]]),int(row[oi[C[1]]])
        for (a,b) in CH:
            z=np.exp(-1j*np.pi/2*(a*j1+b*j2)); vec+=[z.real,z.imag]
    V150.append(vec)
V150=np.round(np.array(V150)).astype(np.int64)
d=np.load('d4_facet_classes.npz')
CEN=d['rows']; HN=d['full']; P=d['P']
Y=V150[:,P]
assert len(L)==64 and Y.shape==(64,33)
vals=HN[:,:33]@Y.T+HN[:,33][:,None]
assert (vals>=0).all() and (vals.min(axis=1)==0).all(), "exact validity"
tcs=(vals==0).sum(axis=1)
masks={np.packbits(vals[i]==0).tobytes() for i in range(len(HN))}
assert len(masks)==len(HN)==23256, "23,256 distinct facets"
cB=CEN[:,0]; cA=CEN[:,1:34]
cv=cB[:,None]+cA@Y.T
assert (cv>=0).all() and (cv.min(axis=1)==0).all() and len(CEN)==61
rks=[int(np.linalg.matrix_rank((Y[vals[i]==0]-Y[vals[i]==0][0]).astype(float))) for i in range(0,23256,911)]
assert set(rks)=={32}, "rank-32 spot checks"
assert int(CEN[:,36].sum())==23256, "orbit accounting"
gi=""
if os.path.exists(_GP):
    G=[np.array(g) for g in pickle.load(open(_GP,'rb'))['G']]
    keys=set()
    for i in range(len(CEN)):
        mb=cv[i]==0; best=None
        for g in G:
            nb=np.zeros(64,dtype=bool); nb[g]=mb
            k=np.packbits(nb).tobytes()
            if best is None or k<best: best=k
        keys.add(best)
    assert len(keys)==61, "61 distinct classes under G"
    gi=" | 61 distinct G-classes: True"
print(f"census artifact: 61 classes, 23,256 facets; validity/rank/accounting PASS{gi}")
def jp(C):
    Ws=[W(np.array(v)) for v in C]; Pj={}
    for j1,j2 in itertools.product(range(4),repeat=2):
        P1=sum((w**(-a*j1))*np.linalg.matrix_power(Ws[0],a) for a in range(4))/4
        P2=sum((w**(-a*j2))*np.linalg.matrix_power(Ws[1],a) for a in range(4))/4
        Pi=P1@P2
        if abs(np.trace(Pi))>1e-9: Pj[(j1,j2)]=Pi
    return Pj
PJ=[jp(C) for C in CTX]
def mu33(psi):
    out=[]
    for ci in range(5):
        e={j:float(np.real(psi.conj()@Pi@psi)) for j,Pi in PJ[ci].items()}
        for (a,b) in CH:
            z=sum(np.exp(-1j*np.pi/2*(a*j1+b*j2))*p for (j1,j2),p in e.items())
            out+=[z.real,z.imag]
    return np.array(out)[P]
def CF(psi):
    Aub=[]; bub=[]
    for ci,C in enumerate(CTX):
        eq={j:float(np.real(psi.conj()@Pi@psi)) for j,Pi in PJ[ci].items()}
        for j in PJ[ci]:
            Aub.append(((L[:,oi[C[0]]]==j[0])&(L[:,oi[C[1]]]==j[1])).astype(float)); bub.append(eq[j])
    r=so.linprog(-np.ones(len(L)),A_ub=np.array(Aub),b_ub=np.array(bub),bounds=[(0,None)]*len(L),method="highs")
    return 1.0+r.fun
rng=np.random.default_rng(314159); agree=0
for k in range(40):
    v=rng.normal(size=16)+1j*rng.normal(size=16); v/=np.linalg.norm(v)
    mu=mu33(v)
    viol=bool(((HN[:,:33]@mu+HN[:,33])<-1e-7).any())
    agree+=viol==(CF(v)>1e-4)
print(f"classifier (23,256 facets) vs CF LP on 40 fresh states: {agree}/40")
assert agree==40
# on-hull lemma: exact integer rank of the constraint system equals 80-33=47
SC=[s_of(C) for C in CTX]
def valof(c,pos,j1,j2):
    return j1 if pos==0 else (j2 if pos==1 else (SC[c]-j1-j2)%4)
nv=80; rows=[]
def vid(c,j1,j2): return c*16+j1*4+j2
for c in range(5):
    r=[0]*nv
    for j1 in range(4):
        for j2 in range(4): r[vid(c,j1,j2)]=1
    rows.append(r)
for v in obs:
    cs=[c for c in range(5) if v in CTX[c]]
    for a2,b2 in itertools.combinations(cs,2):
        pa=CTX[a2].index(v); pb=CTX[b2].index(v)
        for val in range(4):
            r=[0]*nv
            for j1 in range(4):
                for j2 in range(4):
                    if valof(a2,pa,j1,j2)==val: r[vid(a2,j1,j2)]+=1
                    if valof(b2,pb,j1,j2)==val: r[vid(b2,j1,j2)]-=1
            rows.append(r)
for c,C in enumerate(CTX):
    for j1 in range(4):
        for j2 in range(4):
            if any(valof(c,p,j1,j2) not in spec[C[p]] for p in range(3)):
                r=[0]*nv; r[vid(c,j1,j2)]=1; rows.append(r)
from fractions import Fraction
A2=[[Fraction(x) for x in row] for row in rows]
rk=0; m2=nv
for col in range(m2):
    pr=None
    for i2 in range(rk,len(A2)):
        if A2[i2][col]!=0: pr=i2; break
    if pr is None: continue
    A2[rk],A2[pr]=A2[pr],A2[rk]
    pv=A2[rk][col]; A2[rk]=[x/pv for x in A2[rk]]
    for i2 in range(len(A2)):
        if i2!=rk and A2[i2][col]!=0:
            f2=A2[i2][col]; A2[i2]=[A2[i2][k2]-f2*A2[rk][k2] for k2 in range(m2)]
    rk+=1
assert rk==47, f"constraint rank {rk}"
print(f"on-hull lemma: exact constraint rank 47 -> affine dim 33 == deterministic span: PASS")
print("V43 PASS")
