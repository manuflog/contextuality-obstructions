# V29 - Necessity on genuine GHW nets. V28 refuted sufficiency airtight and showed
# single-frame necessity fails under optimization over ALL covariant frames; but the
# rescuing frames may lie outside the net subset. Here: filter gamma in {+-1}^15 by the
# 15 Lagrangian sign conditions gamma(v1)gamma(v2)gamma(v1+v2)=eps (T_{v1}T_{v2}=eps T_{v3}),
# i.e. every line-sum is a stabilizer projector -> exactly the GHW quantum nets.
# Question: are all CF>0 states negative in EVERY net? (T x T stays negative automatically:
# subset of the V28 sweep.)
import numpy as np, itertools
from evend_frame_probe import CTX5, Pv, model, contextual_fraction, T
I2=np.eye(2); X=np.array([[0,1],[1,0]]); Y=np.array([[0,-1j],[1j,0]]); Z=np.diag([1,-1])
P1={'I':I2,'X':X,'Y':Y,'Z':Z}
names=[a+b for a in 'IXYZ' for b in 'IXYZ' if a+b!='II']
Tm={nm:np.kron(P1[nm[0]],P1[nm[1]]) for nm in names}
vec={'I':(0,0),'X':(1,0),'Y':(1,1),'Z':(0,1)}
V=np.array([vec[nm[0]]+vec[nm[1]] for nm in names]); idx={tuple(v):i for i,v in enumerate(V)}
sy2=lambda u,v:(u[0]*v[1]-u[1]*v[0]+u[2]*v[3]-u[3]*v[2])%2
lag=set()
for i,j in itertools.combinations(range(15),2):
    if sy2(V[i],V[j])==0:
        k=idx[tuple((V[i]+V[j])%2)]
        lag.add(frozenset((i,j,k)))
lag=[tuple(sorted(L)) for L in lag]
eps={}
for (i,j,k) in lag:
    Pr=Tm[names[i]]@Tm[names[j]]
    s=1 if np.allclose(Pr,Tm[names[k]]) else (-1 if np.allclose(Pr,-Tm[names[k]]) else None)
    assert s is not None
    eps[(i,j,k)]=s
# --- FINDING (was: sweep; the system is INCONSISTENT) ---
# Solve the F2 system: g_i+g_j+g_k = e(L) over the 15 Lagrangians. If inconsistent,
# NO frame covariant under ALL Pauli displacements has every line-sum a stabilizer
# projector: fully-covariant two-qubit quantum nets DO NOT EXIST. Extract the parity
# certificate (Lagrangian subset with dependent rows and odd e-sum) and print it as
# Pauli triples - the expected face of the order-2 class (PM-type obstruction).
A=np.zeros((len(lag),15),int); e=np.zeros(len(lag),int)
for r,(i,j,k) in enumerate(lag):
    A[r,[i,j,k]]=1; e[r]=(1-eps[(i,j,k)])//2
aug=np.concatenate([A.copy(),np.eye(len(lag),dtype=int)],axis=1)%2
rr=0
for c in range(15):
    p=next((x for x in range(rr,len(lag)) if aug[x,c]),None)
    if p is None: continue
    aug[[rr,p]]=aug[[p,rr]]
    for x in range(len(lag)):
        if x!=rr and aug[x,c]: aug[x]^=aug[rr]
    rr+=1
rank=rr
certs=[]
for x in range(len(lag)):
    if not aug[x,:15].any():
        y=aug[x,15:]
        if (y@e)%2==1: certs.append(y)
print(f"Lagrangians: {len(lag)} (expect 15) | rank over F2: {rank} | inconsistency certificates: {len(certs)}")
if certs:
    y=min(certs,key=lambda v:v.sum())
    triples=[[names[t] for t in lag[r]] for r in range(len(lag)) if y[r]]
    print(f"minimal certificate: {int(y.sum())} Lagrangians, eps-product = -1:")
    for t in triples: print("   ",t)
    from collections import Counter
    cnt=Counter(o for t in triples for o in t)
    is_pm_type = len(cnt)==9 and set(cnt.values())=={2} and len(triples)==6
    print(f"certificate is a Peres-Mermin-type magic configuration "
          f"(9 observables, each in exactly 2 of 6 contexts): {is_pm_type}")
print("VERDICT: fully Pauli-covariant 2-qubit quantum nets do not exist;")
print("the obstruction certificate is the order-2 class in yet another costume."
      if certs else "system consistent (unexpected)")
print("PASS" if (len(lag)==15 and certs) else "FAIL")
