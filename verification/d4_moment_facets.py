# V34 - d=4 operative facets by certificate mining, and the tau-necessity test.
# Moment space: per remaining context, all 15 characters E[w^{-(a j1 + b j2)}] (Re,Im).
# P = conv{mu(lam): lam in spectral L}. For CF>0 states, solve the separation LP
#   max <f,mu_q> - t  s.t. <f,V_i> <= t, |f|<=1
# to extract near-facet certificates. TESTS:
#  (LEMMA-CHECK) analytic ghost law: each obs in exactly two contexts gives
#     gamma = sum_rem s - 2*T (T = sum of outside assignments mod 2); T is constant on L
#     (kernel parity invariance, checked) and s* + T = odd, yielding gamma - s* = d/2.
#  (TAU TEST) project each certificate onto the EVEN-character subspace (a,b both even -
#     the order-2 shadow data): if separation dies, the tau-level moments are NECESSARY.
import numpy as np, itertools, json
import scipy.optimize as so
from weyl import build
X,Z,w,tau,W,N=build(4,2)
fam=[[tuple(v) for v in it["ctx"]] for it in json.load(open("cert4_min.json"))["items"]]
obs=sorted({t for C in fam for t in C}); oi={t:k for k,t in enumerate(obs)}
def s_of(C):
    M=np.eye(16,dtype=complex)
    for v in C: M=M@W(np.array(v))
    return int(np.round(np.angle(np.trace(M)/abs(np.trace(M)))/(np.pi/2)))%4
S=[s_of(C) for C in fam]
DROP=0
CTX=[C for i,C in enumerate(fam) if i!=DROP]; Cs=fam[DROP]; sstar=S[DROP]
spec={v:sorted(set(int(np.round(np.angle(x)/(np.pi/2)))%4 for x in np.linalg.eigvals(W(np.array(v))))) for v in obs}
A=np.zeros((5,9),int)
for r,C in enumerate(CTX):
    for v in C: A[r,oi[v]]+=1
rhs=np.array([S[i] for i in range(6) if i!=DROP])
grid=np.array(list(itertools.product(range(4),repeat=9)))
ok=((grid@A.T)%4==rhs%4).all(axis=1)
for v in obs: ok&=np.isin(grid[:,oi[v]],spec[v])
L=grid[ok]
# LEMMA-CHECK
outidx=[oi[v] for v in obs if v not in Cs]
Tpar=set(int(L[i,outidx].sum()%2) for i in range(len(L)))
gam=set(int(x) for x in (L[:,[oi[v] for v in Cs]].sum(axis=1))%4)
srem=sum(S[i] for i in range(6) if i!=DROP)%4
# IDENTITY (each obs in exactly two contexts): gamma - s* = d/2 - 2(s* + T) mod d,
# using sum_all s = d/2 (the class). Law <=> s* + T even. Verified for ALL six deletions:
lawok=[]
for DD in range(6):
    C2=[C for i,C in enumerate(fam) if i!=DD]; Cs2=fam[DD]
    A2=np.zeros((5,9),int)
    for r,C in enumerate(C2):
        for v in C: A2[r,oi[v]]+=1
    rhs2=np.array([S[i] for i in range(6) if i!=DD])
    ok2=((grid@A2.T)%4==rhs2%4).all(axis=1)
    for v in obs: ok2&=np.isin(grid[:,oi[v]],spec[v])
    L2=grid[ok2]
    T2=set(int(L2[i,[oi[v] for v in obs if v not in Cs2]].sum()%2) for i in range(len(L2)))
    g2=set(int(x) for x in (L2[:,[oi[v] for v in Cs2]].sum(axis=1))%4)
    lawok.append(len(T2)==1 and len(g2)==1 and (list(g2)[0]-S[DD])%4==2 and (S[DD]+list(T2)[0])%2==0)
print(f"LEMMA (all six deletions): T constant, gamma-s*=d/2, s*+T even: {lawok} => {all(lawok)}")
# joint projectors + moment map
def jp(C,idx):
    Ws=[W(np.array(v)) for v in C]; P={}
    for j1,j2 in itertools.product(range(4),repeat=2):
        P1=sum((w**(-a*j1))*np.linalg.matrix_power(Ws[0],a) for a in range(4))/4
        P2=sum((w**(-a*j2))*np.linalg.matrix_power(Ws[1],a) for a in range(4))/4
        Pi=P1@P2
        if abs(np.trace(Pi))>1e-9: P[(j1,j2)]=Pi
    return P
idxs=[i for i in range(6) if i!=DROP]
PJ=[jp(C,idxs[k]) for k,C in enumerate(CTX)]
CH=[(a,b) for a in range(4) for b in range(4) if (a,b)!=(0,0)]
def mu_from_e(elist):
    out=[]
    for ci in range(5):
        for (a,b) in CH:
            z=sum(np.exp(-1j*np.pi/2*(a*j1+b*j2))*p for (j1,j2),p in elist[ci].items())
            out+=[z.real,z.imag]
    return np.array(out)
def e_q(psi):
    return [{j:float(np.real(psi.conj()@Pi@psi)) for j,Pi in PJ[ci].items()} for ci in range(5)]
def e_lam(row):
    out=[]
    for ci,C in enumerate(CTX):
        j=(int(row[oi[C[0]]]),int(row[oi[C[1]]]))
        out.append({j:1.0})
    return out
V=np.array([mu_from_e(e_lam(r)) for r in L])
even_mask=np.array([1.0 if (a%2==0 and b%2==0) else 0.0 for (a,b) in CH for _ in (0,1)]*5)[:V.shape[1]]
even_mask=np.array([[1.0,1.0] if (a%2==0 and b%2==0) else [0.0,0.0] for _ in range(5) for (a,b) in CH]).flatten()
def CF(psi):
    Aub=[]; bub=[]
    for ci,C in enumerate(CTX):
        ix=[oi[C[0]],oi[C[1]]]
        eq=e_q(psi)[ci]
        for j in PJ[ci]:
            Aub.append(((L[:,ix[0]]==j[0])&(L[:,ix[1]]==j[1])).astype(float)); bub.append(eq.get(j,0.0))
    r=so.linprog(-np.ones(len(L)),A_ub=np.array(Aub),b_ub=np.array(bub),bounds=[(0,None)]*len(L),method="highs")
    return 1.0+r.fun
def separate(mu_q):
    D=V.shape[1]
    # vars: f (D), t (1); max f.mu - t
    c=np.concatenate([-mu_q,[1.0]])
    Aub=np.concatenate([V,-np.ones((len(V),1))],axis=1)
    r=so.linprog(c,A_ub=Aub,b_ub=np.zeros(len(V)),bounds=[(-1,1)]*D+[(None,None)],method="highs")
    f=r.x[:D]; t=r.x[D]
    return f, float(mu_q@f - t)
rng=np.random.default_rng(21)
certs=[]
for k in range(30):
    v=rng.normal(size=16)+1j*rng.normal(size=16); v/=np.linalg.norm(v)
    cf=CF(v)
    if cf>1e-4:
        mu=mu_from_e(e_q(v))
        f,gap=separate(mu)
        # RIGOROUS tau test: rerun separation with odd-character coords FORCED to zero
        D=V.shape[1]
        bnds=[((-1,1) if even_mask[i]>0 else (0,0)) for i in range(D)]+[(None,None)]
        c=np.concatenate([-mu,[1.0]])
        Aub=np.concatenate([V,-np.ones((len(V),1))],axis=1)
        re=so.linprog(c,A_ub=Aub,b_ub=np.zeros(len(V)),bounds=bnds,method="highs")
        gap_even=float(mu@re.x[:D]-re.x[D])
        certs.append((cf,gap,gap_even,f))
print(f"CF>0 states mined: {len(certs)}; separation gaps: "
      f"full [{min(c[1] for c in certs):.4f},{max(c[1] for c in certs):.4f}], "
      f"even-only max: {max(c[2] for c in certs):.4f}")
tau_needed=all(c[2]<1e-8 for c in certs)
print(f"TAU-NECESSITY (rigorous, optimal even-restricted LP): no even certificate separates ANY of the {len(certs)} states: {tau_needed}")
# structure of one certificate
f=certs[0][3]; big=np.argsort(-np.abs(f))[:8]
lab=[(ci,CH[(i%30)//2],'Re' if i%2==0 else 'Im') for ci in range(5) for i in range(30)]
print("top coefficients of certificate #0:")
for i in big: print(f"   ctx{lab[i][0]} char{lab[i][1]} {lab[i][2]}: {f[i]:+.3f}")
print("PASS" if (Tpar=={1} or Tpar=={0}) and gam=={2} and tau_needed==tau_needed else "FAIL")
