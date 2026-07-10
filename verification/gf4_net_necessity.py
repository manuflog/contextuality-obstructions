# V30 - Necessity on the ACTUAL Gibbons-Hoffman-Wootters nets (GF(4) phase space).
# V29 proved fully Pauli-covariant nets don't exist (PM-type certificate); GHW nets
# survive by using GF(4)^2 with its 5 striations (rays), covariant under translations
# only. Freedom: one character choice per striation => 4^5 = 1024 nets.
# Questions: (a) are all CF>0 states (V27/E3 family) negative in EVERY GHW net?
# (b) sufficiency sanity: T x T best-case over nets (must stay negative);
# (c) census: how many of the 60 stabilizer states are rescuable by SOME net?
import numpy as np, itertools
from evend_frame_probe import CTX5, Pv, model, contextual_fraction, T
I2=np.eye(2); Xm=np.array([[0,1],[1,0]]); Ym=np.array([[0,-1j],[1j,0]]); Zm=np.diag([1,-1])
MUL=[[0,0,0,0],[0,1,2,3],[0,2,3,1],[0,3,1,2]]           # GF(4): 0,1,w,w^2
ADD=lambda a,b:a^b
bits=lambda g:(g&1,(g>>1)&1)
def pauli_of(point,Mz):
    (a,b)=point; (x0,x1)=bits(a); zb=bits(b); z0=(Mz[0][0]*zb[0]+Mz[0][1]*zb[1])%2; z1=(Mz[1][0]*zb[0]+Mz[1][1]*zb[1])%2
    def q(x,z): return {(0,0):I2,(1,0):Xm,(0,1):Zm,(1,1):Ym}[(x,z)]
    return np.kron(q(x0,z0),q(x1,z1))
DIRS=[(1,0),(1,1),(1,2),(1,3),(0,1)]                     # slopes 0,1,w,w^2,inf
RAYS=[[ (MUL[c][d[0]],MUL[c][d[1]]) for c in (1,2,3)] for d in DIRS]
Mz=None
for cand in [[[1,0],[0,1]],[[0,1],[1,0]],[[1,1],[0,1]],[[1,0],[1,1]],[[0,1],[1,1]],[[1,1],[1,0]]]:
    ok=True
    for R in RAYS:
        Ps=[pauli_of(p,cand) for p in R]
        ok &= all(np.allclose(Ps[i]@Ps[j],Ps[j]@Ps[i]) for i in range(3) for j in range(i+1,3))
    if ok: Mz=cand; break
assert Mz, "no Z-basis map makes rays commute"
POINTS=[(a,b) for a in range(4) for b in range(4)]
Tof={p:pauli_of(p,Mz) for p in POINTS}
def symp(u,v):
    A=Tof[u]@Tof[v]; B=Tof[v]@Tof[u]
    return 0 if np.allclose(A,B) else 1
# striation cosets: line id of point u in striation s
subgrp=[ [ (0,0) ]+R for R in RAYS ]
line_id=[]
for S in subgrp:
    seen={}; lid={}
    for u in POINTS:
        key=frozenset(((ADD(u[0],m[0]),ADD(u[1],m[1]))) for m in S)
        if key not in seen: seen[key]=len(seen)
        lid[u]=seen[key]
    line_id.append(lid)
# per striation: 4 valid base characters; projector table proj[s][line][choice]
def valid_chars(R):
    P1,P2,P3=[Tof[p] for p in R]
    eps=1 if np.allclose(P1@P2,P3) else -1
    return [c for c in itertools.product((1,-1),repeat=3) if c[0]*c[1]*c[2]==eps]
CH=[valid_chars(R) for R in RAYS]
reps=[]
for s,S in enumerate(subgrp):
    rep={}
    for u in POINTS:
        l=line_id[s][u]
        if l not in rep: rep[l]=u
    reps.append(rep)
def proj(s,l,ch):
    base=CH[s][ch]; v=reps[s][l]
    sg=[base[k]*(-1)**symp(v,RAYS[s][k]) for k in range(3)]
    P=np.eye(4)
    for k in (0,1): P=P@(np.eye(4)+sg[k]*Tof[RAYS[s][k]])/2
    return P
PT=[[[proj(s,l,ch) for ch in range(4)] for l in range(4)] for s in range(5)]
# sanity on one net: A_u traces & resolution
net0=[0]*5
A=lambda u,net: sum(PT[s][line_id[s][u]][net[s]] for s in range(5))-np.eye(4)
assert all(abs(np.trace(A(u,net0))-1)<1e-9 for u in POINTS)
assert np.allclose(sum(A(u,net0) for u in POINTS),4*np.eye(4))
rng=np.random.default_rng(5)
states={"|00>":np.array([1,0,0,0],complex),"TxT":np.kron(T,T)}
for k in range(20):
    v=rng.normal(size=4)+1j*rng.normal(size=4); states[f"rand{k}"]=v/np.linalg.norm(v)
NETS=list(itertools.product(range(4),repeat=5))
def sweep(ps):
    rho=np.outer(ps,ps.conj())
    t=np.array([[[np.real(np.trace(rho@PT[s][l][c])) for c in range(4)] for l in range(4)] for s in range(5)])
    best=-1e9; worst=1e9
    for net in NETS:
        W=[sum(t[s][line_id[s][u]][net[s]] for s in range(5))-1 for u in POINTS]
        m=min(W); best=max(best,m); worst=min(worst,m)
    return best,worst
rows=[]
for nm,ps in states.items():
    cf=contextual_fraction(CTX5,{o:(1,-1) for o in Pv},model(ps))
    b,wst=sweep(ps); rows.append((nm,cf,b))
pos=[r for r in rows if r[1]>1e-6]
nec=all(r[2]<-1e-9 for r in pos); resc=[r[0] for r in pos if r[2]>-1e-9]
tt=next(r for r in rows if r[0]=='TxT')
print(f"GHW nets: {len(NETS)}; Mz={Mz}")
print(f"(a) CF>0 states negative in EVERY GHW net: {nec} ({len(pos)} states; rescued: {resc if resc else 'none'})")
print(f"(b) TxT best-case over nets: {tt[2]:+.4f} (CF={tt[1]:.4f}) - sufficiency stays dead")
# (c) stabilizer census
P1={'I':I2,'X':Xm,'Y':Ym,'Z':Zm}
paulis={a+b:np.kron(P1[a],P1[b]) for a in P1 for b in P1 if a+b!='II'}
projs={}
nmsl=list(paulis)
for i,j in itertools.combinations(range(15),2):
    Pa,Qa=paulis[nmsl[i]],paulis[nmsl[j]]
    if not np.allclose(Pa@Qa,Qa@Pa): continue
    if np.allclose(Pa@Qa,np.eye(4)) or np.allclose(Pa@Qa,-np.eye(4)): continue
    for s1,s2 in itertools.product((1,-1),repeat=2):
        Pi=(np.eye(4)+s1*Pa)@(np.eye(4)+s2*Qa)/4
        if abs(np.trace(Pi)-1)>1e-9: continue
        projs[tuple(np.round(Pi,8).flatten())]=Pi
resc60=0
for Pi in projs.values():
    t=np.array([[[np.real(np.trace(Pi@PT[s][l][c])) for c in range(4)] for l in range(4)] for s in range(5)])
    best=max(min(sum(t[s][line_id[s][u]][net[s]] for s in range(5))-1 for u in POINTS) for net in NETS)
    resc60+= best>-1e-9
print(f"(c) stabilizer census: {resc60}/60 states nonnegative in SOME GHW net")
print("VERDICT:", "GHW-net negativity NECESSARY (never sufficient) - asymmetric lemma stands" if nec else "necessity fails on true GHW nets too - pairing carries both directions")
print("PASS" if (len(projs)==60 and tt[2]<-1e-9) else "FAIL")
