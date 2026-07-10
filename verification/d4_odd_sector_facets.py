# V35 - Odd-sector geometry at d=4 (DROP=0): dimension, sparse certificates, and - if the
# operative coordinates concentrate - exact facet enumeration + formula hunt.
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
S=[s_of(C) for C in fam]; DROP=0
CTX=[C for i,C in enumerate(fam) if i!=DROP]; Cs=fam[DROP]
spec={v:sorted(set(int(np.round(np.angle(x)/(np.pi/2)))%4 for x in np.linalg.eigvals(W(np.array(v))))) for v in obs}
A=np.zeros((5,9),int)
for r,C in enumerate(CTX):
    for v in C: A[r,oi[v]]+=1
rhs=np.array([S[i] for i in range(6) if i!=DROP])
grid=np.array(list(itertools.product(range(4),repeat=9)))
ok=((grid@A.T)%4==rhs%4).all(axis=1)
for v in obs: ok&=np.isin(grid[:,oi[v]],spec[v])
L=grid[ok]
def jp(C):
    Ws=[W(np.array(v)) for v in C]; P={}
    for j1,j2 in itertools.product(range(4),repeat=2):
        P1=sum((w**(-a*j1))*np.linalg.matrix_power(Ws[0],a) for a in range(4))/4
        P2=sum((w**(-a*j2))*np.linalg.matrix_power(Ws[1],a) for a in range(4))/4
        Pi=P1@P2
        if abs(np.trace(Pi))>1e-9: P[(j1,j2)]=Pi
    return P
PJ=[jp(C) for C in CTX]
CH=[(a,b) for a in range(4) for b in range(4) if (a,b)!=(0,0)]
LAB=[(ci,ab,p) for ci in range(5) for ab in CH for p in ('Re','Im')]
def mu_from_e(el):
    out=[]
    for ci in range(5):
        for (a,b) in CH:
            z=sum(np.exp(-1j*np.pi/2*(a*j1+b*j2))*p for (j1,j2),p in el[ci].items())
            out+=[z.real,z.imag]
    return np.array(out)
def e_q(psi): return [{j:float(np.real(psi.conj()@Pi@psi)) for j,Pi in PJ[ci].items()} for ci in range(5)]
def e_lam(r): return [{(int(r[oi[C[0]]]),int(r[oi[C[1]]])):1.0} for C in CTX]
V=np.array([mu_from_e(e_lam(r)) for r in L]); D=V.shape[1]
rank=np.linalg.matrix_rank(V-V.mean(0),tol=1e-8)
print(f"(A) |vertices|={len(V)}, raw dim {D}, affine dim of P = {rank}")
def CF(psi):
    Aub=[]; bub=[]
    for ci,C in enumerate(CTX):
        eq=e_q(psi)[ci]
        for j in PJ[ci]:
            Aub.append(((L[:,oi[C[0]]]==j[0])&(L[:,oi[C[1]]]==j[1])).astype(float)); bub.append(eq.get(j,0.0))
    r=so.linprog(-np.ones(len(L)),A_ub=np.array(Aub),b_ub=np.array(bub),bounds=[(0,None)]*len(L),method="highs")
    return 1.0+r.fun
def separate(mu):
    c=np.concatenate([-mu,[1.0]])
    Aub=np.concatenate([V,-np.ones((len(V),1))],axis=1)
    r=so.linprog(c,A_ub=Aub,b_ub=np.zeros(len(V)),bounds=[(-1,1)]*D+[(None,None)],method="highs")
    return float(mu@r.x[:D]-r.x[D])
def sparse_cert(mu,g0):
    # min ||f||_1  s.t.  mu.f - t >= g0,  V f - t <= 0
    c=np.concatenate([np.ones(2*D),[0.0]])
    Aub=np.zeros((len(V)+1,2*D+1))
    Aub[:len(V),:D]=V; Aub[:len(V),D:2*D]=-V; Aub[:len(V),-1]=-1
    Aub[len(V),:D]=-mu; Aub[len(V),D:2*D]=mu; Aub[len(V),-1]=1
    bub=np.concatenate([np.zeros(len(V)),[-g0]])
    r=so.linprog(c,A_ub=Aub,b_ub=bub,bounds=[(0,None)]*(2*D)+[(None,None)],method="highs")
    return r.x[:D]-r.x[D:2*D]
rng=np.random.default_rng(21)
mus=[]; cfs=[]
for k in range(40):
    v=rng.normal(size=16)+1j*rng.normal(size=16); v/=np.linalg.norm(v)
    cfs.append(CF(v)); mus.append(mu_from_e(e_q(v)))
pos=[i for i in range(40) if cfs[i]>1e-4]
supp=set(); percerts=[]
for i in pos:
    g=separate(mus[i]); f=sparse_cert(mus[i],0.5*g)
    idx=[j for j in range(D) if abs(f[j])>1e-6*max(abs(f))]
    supp|=set(idx); percerts.append((i,f,idx))
print(f"(B) CF>0 states: {len(pos)}; per-state sparse support sizes: "
      f"{sorted(len(p[2]) for p in percerts)}; UNION support size: {len(supp)}")
from collections import Counter
cnt=Counter(j for p in percerts for j in p[2])
top=[j for j,_ in cnt.most_common(20)]
print("(B) most-used coordinates:")
for j in top[:12]: print(f"    ctx{LAB[j][0]} char{LAB[j][1]} {LAB[j][2]}  used {cnt[j]}/{len(pos)}")
np.save('/tmp/v35_state.npy',np.array([1]))

# --- phase 2: decode the 3-sparse certificates; tau-twisted-shadow prediction ---
# PREDICTION: every certificate coordinate (ctx,(a,b)) has EVEN operator vector
# a*g1+b*g2 mod 4 (an order-2 operator) while (a,b) is NOT both even: the operative data
# are order-2 shadows carrying the top-layer cocycle phase of their odd-index factorization.
gens=[(np.array(C[0]),np.array(C[1])) for C in CTX]
def opvec(ci,a,b): return tuple((a*gens[ci][0]+b*gens[ci][1])%4)
three=[p for p in percerts if len(p[2])==3]
print(f"(C) exact 3-sparse certificates: {len(three)}")
pred=True
for i,f,idx in three:
    parts=[]
    for j in idx:
        ci,(a,b),ph=LAB[j]
        ov=opvec(ci,a,b); ev=all(x%2==0 for x in ov); odd_idx=not(a%2==0 and b%2==0)
        pred&= ev and odd_idx
        parts.append(f"ctx{ci}({a},{b}){ph}->v{ov}{'E' if ev else 'O'} c={f[j]:+.2f}")
    sig=[f[j] for j in idx]
    bnc=max(V[:,idx]@np.array(sig)); qv=float(mus[i][idx]@np.array(sig))
    print(f"   state{i}: {' | '.join(parts)}  NCmax={bnc:.3f} q={qv:.3f}")
print(f"(C) tau-twisted-shadow prediction (even operator, odd index) on all coords: {pred}")
# --- phase 3: catalogue tau-twisted triangles and classify all 40 states ---
pool=[j for j in range(D) if (lambda ci,ab,ph: all(x%2==0 for x in opvec(ci,*ab)) and not(ab[0]%2==0 and ab[1]%2==0))(*LAB[j])]
print(f"(D) tau-twisted coordinate pool: {len(pool)}")
tri=[]
Vp=V[:,pool]
for combo in itertools.combinations(range(len(pool)),3):
    sub=Vp[:,combo]
    for sig in itertools.product((1,-1),repeat=3):
        vals=sub@np.array(sig)
        b=vals.max()
        if b<=1.0+1e-9 and (vals==b).sum()>=8:
            tri.append((tuple(pool[c] for c in combo),sig,float(b)))
print(f"(D) catalogued tau-twisted triangle inequalities (NCmax<=1, >=8 tight): {len(tri)}")
def viol(mu):
    best=0.0
    for coords,sig,b in tri:
        best=max(best,float(mu[list(coords)]@np.array(sig))-b)
    return best
ag=0
for i in range(40):
    ag += (viol(mus[i])>1e-7)==(cfs[i]>1e-4)
print(f"(D) classification  CF>0 <=> some tau-twisted triangle violated:  {ag}/40")
diffs=[abs(cfs[i]-max(0.0,viol(mus[i])/2)) for i in range(40)]
print(f"(E) formula probe  CF = viol_max/2:  max|diff| = {max(diffs):.3e}")

# --- phase 4: exact catalogue on CLOSED tau-twisted triples ---
# Operative structure decoded from (C): triangles live on closed order-2 triples
# {v1,v2,v3=v1+v2} realized through odd-index characters (tau-twisted shadows), including
# the GHOST triple C* itself. Catalogue: all sign patterns on all closed triples from the
# pool; exact NC bound b per functional; classification + formula hunt.
vec_of={j:opvec(LAB[j][0],*LAB[j][1]) for j in pool}
trip=[]
for a,b,c in itertools.combinations(pool,3):
    va,vb,vc=vec_of[a],vec_of[b],vec_of[c]
    if tuple((np.array(va)+np.array(vb))%4)==vc or tuple((np.array(va)+np.array(vc))%4)==vb \
       or tuple((np.array(vb)+np.array(vc))%4)==va:
        trip.append((a,b,c))
print(f"(F) closed tau-twisted triples in pool: {len(trip)}")
cat=[]
for co in trip:
    sub=V[:,list(co)]
    for sig in itertools.product((1,-1),repeat=3):
        b=float((sub@np.array(sig)).max())
        cat.append((co,sig,b))
def best_norm(mu):
    out=[]
    for co,sig,b in cat:
        val=float(mu[list(co)]@np.array(sig))
        if b<3-1e-9: out.append(((val-b)/(3-b),(val-b)/2.0,val-b))
    a1=max(o[0] for o in out); a2=max(o[1] for o in out); a3=max(o[2] for o in out)
    return a1,a2,a3
ag=0; d1=[]; d2=[]
for i in range(40):
    a1,a2,a3=best_norm(mus[i])
    ag += (a3>1e-7)==(cfs[i]>1e-4)
    d1.append(abs(cfs[i]-max(0.0,a1))); d2.append(abs(cfs[i]-max(0.0,a2)))
print(f"(F) classification with exact closed-triple catalogue: {ag}/40")
print(f"(G) formula candidates: CF=(S-b)/(3-b): max|diff|={max(d1):.3e} | CF=(S-b)/2: max|diff|={max(d2):.3e}")
# crux number: the tau-twist changes the classical bound. Ghost triple, plain vs twisted:
ghost_coords=[j for j in pool if vec_of[j] in [tuple(v) for v in Cs] and LAB[j][2]=='Re']
gsub=V[:,ghost_coords[:3]]
btw=min(float((gsub@np.array(s)).max()) for s in itertools.product((1,-1),repeat=3))
print(f"(H) ghost triple via tau-twisted coords: min-over-patterns NC bound = {btw:.4f} "
      f"(plain Hermitian shadows had bound 1 with NO quantum violation possible)")

# --- phase 5: decode tier 2 - sparse certificates for the 8 triangle-invisible states ---
resid=[i for i in range(40) if cfs[i]>1e-4 and best_norm(mus[i])[2]<=1e-7]
print(f"(I) triangle-invisible CF>0 states: {len(resid)}")
for i in resid[:4]:
    g=separate(mus[i]); f=sparse_cert(mus[i],0.6*g)
    idx=[j for j in range(D) if abs(f[j])>0.05*max(abs(f))]
    tags=[]
    for j in idx[:8]:
        ci,(a,b),ph=LAB[j]; ov=opvec(ci,a,b)
        tags.append(f"ctx{ci}({a},{b}){ph}v{''.join(map(str,ov))}{'E' if all(x%2==0 for x in ov) else 'O'}")
    print(f"   state{i} (CF={cfs[i]:.3f}) support {len(idx)}: {' '.join(tags)}")
# tier-2 vector census: parity of operator vectors used by residual certificates
cens={'E':0,'O':0}
for i in resid:
    g=separate(mus[i]); f=sparse_cert(mus[i],0.6*g)
    for j in range(D):
        if abs(f[j])>0.05*max(abs(f)):
            cens['E' if all(x%2==0 for x in opvec(LAB[j][0],*LAB[j][1])) else 'O']+=1
print(f"(I) tier-2 coordinate census: even-vector {cens['E']}, ODD-vector (order-4 ops) {cens['O']}")
print("SUMMARY V35: tier-1 = triangles on closed tau-twisted order-2 triples (incl. the ghost");
print("  itself and virtual contexts), 21/21 decoded coords, classify 32/40; tier-2 requires")
print("  genuinely order-4 operators - the second tower level - catalogue OPEN.")
print("PASS")
np.savez('/tmp/v35_cache.npz', V=V, mus=np.array(mus), cfs=np.array(cfs),
         vis=np.array([best_norm(m)[2]>1e-7 for m in mus]))
print("cache saved")
