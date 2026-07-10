# V27 - The even-d frame problem, probed at d=2 (the conjecture's hard half).
# (E1) single qubit, Wootters frame: 'qubit Hudson' HOLDS - all 6 stabilizer states
#      nonnegative, T-state negative (min (1-sqrt2)/4).
# (E2) two qubits, product Wootters frame: exhaustive over all 60 stabilizer pure
#      states - does Hudson FAIL (some stabilizer state negative)? This is the
#      structural reason the even-d state sector cannot be plain negativity and
#      must route through the 2-adic tower.
# (E3) state sector on a TRIVIAL-CLASS qubit Weyl family (PM minus its odd column):
#      CF via LP across stabilizer, magic (T x T), and random states - is the
#      qubit Weyl state sector empty (CF=0 always) or populated?
import numpy as np, itertools
from state_sector_probe import contextual_fraction
from criterion import carry_data, left_kernel
I2=np.eye(2); X=np.array([[0,1],[1,0]]); Y=np.array([[0,-1j],[1j,0]]); Z=np.diag([1,-1])
# E1
A1={(a,b): (I2+(-1)**a*Z+(-1)**b*X+(-1)**(a+b)*Y)/2 for a in range(2) for b in range(2)}
assert np.allclose(sum(A1.values()),2*I2)
wig1=lambda r: np.array([np.real(np.trace(r@A1[u]))/2 for u in A1])
stab1=[v for M in (X,Y,Z) for v in np.linalg.eigh(M)[1].T]
T=np.array([1,np.exp(1j*np.pi/4)])/np.sqrt(2)
m1=[wig1(np.outer(v,v.conj())).min() for v in stab1]
mT=wig1(np.outer(T,T.conj())).min()
print(f"(E1) qubit: stabilizer min {min(m1):+.4f} (>=0); T-state min {mT:+.4f} (=(1-sqrt2)/4={0.25*(1-np.sqrt(2)):+.4f})")
# E2
P1={'I':I2,'X':X,'Y':Y,'Z':Z}
paulis={a+b:np.kron(P1[a],P1[b]) for a in P1 for b in P1 if a+b!='II'}
names=list(paulis)
projs={}
for i,j in itertools.combinations(range(15),2):
    P,Q=paulis[names[i]],paulis[names[j]]
    if not np.allclose(P@Q,Q@P): continue
    if np.allclose(P@Q,np.eye(4)) or np.allclose(P@Q,-np.eye(4)): continue
    for s1,s2 in itertools.product((1,-1),repeat=2):
        Pi=(np.eye(4)+s1*P)@(np.eye(4)+s2*Q)/4
        if abs(np.trace(Pi)-1)>1e-9: continue
        projs[tuple(np.round(Pi,8).flatten())]=Pi
print(f"(E2) two-qubit stabilizer pure states enumerated: {len(projs)} (expect 60)")
A2={(u,v): np.kron(A1[u],A1[v]) for u in A1 for v in A1}
wig2=lambda r: np.array([np.real(np.trace(r@A2[k]))/4 for k in A2])
m2=[wig2(Pi).min() for Pi in projs.values()]
neg2=sum(1 for m in m2 if m<-1e-9)
print(f"(E2) product-net negativity: {neg2}/60 stabilizer states negative; global min {min(m2):+.4f}")
# E3
Pv={"XI":(1,0,0,0),"IX":(0,0,1,0),"XX":(1,0,1,0),"IZ":(0,0,0,1),"ZI":(0,1,0,0),
    "ZZ":(0,1,0,1),"XZ":(1,0,0,1),"ZX":(0,1,1,0),"YY":(1,1,1,1)}
CTX5=[["XI","IX","XX"],["IZ","ZI","ZZ"],["XZ","ZX","YY"],["XI","IZ","XZ"],["IX","ZI","ZX"]]
fam=[[Pv[o] for o in C] for C in CTX5]
K=left_kernel(carry_data(fam,2)[0])
print(f"(E3) PM-minus-odd-column: cycles in family = {len(K)} (trivial class)")
from weyl import build
_,_,_,_,Wop,_=build(2,2)
def model(psi):
    e={}
    for ci,C in enumerate(CTX5):
        Ms=[Wop(np.array(Pv[o])).real for o in C]
        for s in itertools.product((1,-1),repeat=3):
            Pi=np.eye(4)
            for M,si in zip(Ms,s): Pi=Pi@(np.eye(4)+si*M)/2
            e[(ci,s)]=float(np.real(psi.conj()@Pi@psi))
    return e
rng=np.random.default_rng(5)
tests={"<00|":np.array([1,0,0,0],complex),"TxT":np.kron(T,T)}
for k in range(20):
    v=rng.normal(size=4)+1j*rng.normal(size=4); tests[f"rand{k}"]=v/np.linalg.norm(v)
cfs={nm:contextual_fraction(CTX5,{o:(1,-1) for o in Pv},model(ps)) for nm,ps in tests.items()}
mx=max(cfs.values())
print(f"(E3) CF over 22 states (stabilizer, TxT, 20 random): max = {mx:.6f}")
verdict = "EMPTY - qubit Weyl state sector vanishes on trivial-class family" if mx<1e-6 else f"POPULATED - state sector exists at d=2 (max CF {mx:.4f})"
print(f"(E3) verdict: {verdict}")
print("PASS" if (min(m1)>-1e-9 and mT<-0.05 and len(projs)==60) else "FAIL")

# (E4) classifier test on the trivial-class family: negativity NECESSARY, NOT SUFFICIENT.
#   - all CF>0 states have negative product-net Wigner (10/10)
#   - T x T is negative (-0.0625) yet CF=0: one net's negativity cannot classify the sector
#   => the even-d classifier must pair the STATE against the FAMILY's tower data
#      (scenario-dependent frame), not against a global net. Refines Conjecture 1 (even half).
rows=[]
for nm,ps in tests.items():
    wm=wig2(np.outer(ps,ps.conj())).min()
    rows.append((nm,cfs[nm],wm))
nec = all(r[2]<-1e-9 for r in rows if r[1]>1e-6)
tt = next(r for r in rows if r[0]=="TxT")
print(f"(E4) negativity necessary for CF>0: {nec} ({sum(1 for r in rows if r[1]>1e-6)}/22 positive)")
print(f"(E4) sufficiency counterexample: TxT wig_min={tt[2]:+.4f} but CF={tt[1]:.4f}")
print("PASS(E4)" if (nec and tt[2]<-0.05 and tt[1]<1e-6) else "FAIL(E4)")
