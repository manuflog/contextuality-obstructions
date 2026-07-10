# V26 - The state-dependent sector, probed. Three machine-checked anchors for the
# open problem (Paper C, OP): (1) PM has contextual fraction CF=1 (AvN => maximal, any state);
# (2) KCBS has CF>0 while its class shadow is 0 (the sector the class misses, quantified);
# (3) at d=3 the discrete Wigner function (pinned Weyl convention) is >=0 on ALL 12 stabilizer
#     pure states and <0 on the strange state - the discrete-Hudson anchor for the conjecture
#     that the state sector is classified by pairing the state against the order-d class
#     (Howard-Wallman-Veitch-Emerson at odd prime d; 2-adic tower conjectured at even d).
import numpy as np, itertools
from scipy.optimize import linprog
from weyl import build

def contextual_fraction(contexts, outcomes, e):
    """CF = 1 - max total weight of a deterministic-global-assignment submodel <= e."""
    obs = sorted({o for C in contexts for o in C})
    G = list(itertools.product(*[outcomes[o] for o in obs]))
    A_ub=[]; b_ub=[]
    for ci,C in enumerate(contexts):
        for oc in itertools.product(*[outcomes[o] for o in C]):
            row=[1.0 if all(g[obs.index(o)]==oc[j] for j,o in enumerate(C)) else 0.0 for g in G]
            A_ub.append(row); b_ub.append(e[(ci,oc)])
    r=linprog(-np.ones(len(G)), A_ub=A_ub, b_ub=b_ub, bounds=[(0,None)]*len(G), method="highs")
    assert r.status==0
    return 1.0+r.fun

# (1) PM, state |00>
X,Z,w,tau,W,N = build(2,2)
P={"XI":(1,0,0,0),"IX":(0,0,1,0),"XX":(1,0,1,0),"IZ":(0,0,0,1),"ZI":(0,1,0,0),
   "ZZ":(0,1,0,1),"XZ":(1,0,0,1),"ZX":(0,1,1,0),"YY":(1,1,1,1)}
CTX=[["XI","IX","XX"],["IZ","ZI","ZZ"],["XZ","ZX","YY"],
     ["XI","IZ","XZ"],["IX","ZI","ZX"],["XX","ZZ","YY"]]
psi=np.zeros(4); psi[0]=1
e={}
for ci,C in enumerate(CTX):
    Ms=[W(np.array(P[o])).real for o in C]
    for s in itertools.product((1,-1),repeat=3):
        Pi=np.eye(4)
        for M,si in zip(Ms,s): Pi=Pi@(np.eye(4)+si*M)/2
        e[(ci,s)]=float(np.real(psi@Pi@psi))
cf_pm=contextual_fraction(CTX,{o:(1,-1) for o in P},e)

# (2) KCBS, optimal state
c=np.cos(np.pi/5); ct2=c/(1+c); st=np.sqrt(1-ct2)
V=[np.array([st*np.cos(4*np.pi*k/5),st*np.sin(4*np.pi*k/5),np.sqrt(ct2)]) for k in range(5)]
ctxK=[[k,(k+1)%5] for k in range(5)]; ket=np.array([0,0,1.0])
eK={}
for ci,(i,j) in enumerate(ctxK):
    Pi_i=np.outer(V[i],V[i]); Pi_j=np.outer(V[j],V[j])
    for a,b in itertools.product((1,0),repeat=2):
        Op=(Pi_i if a else np.eye(3)-Pi_i)@(Pi_j if b else np.eye(3)-Pi_j)
        eK[(ci,(a,b))]=float(np.real(ket@Op@ket))
cf_k=contextual_fraction(ctxK,{o:(1,0) for o in range(5)},eK)

# (3) discrete Wigner at d=3 via parity phase-point operators A_u = D_u P D_u^dag
# (convention-free: operator phases cancel under conjugation). NOTE: the repo's
# tau-convention W(v) does NOT generate this frame at odd d (A0 != parity; stabilizer
# states acquire -1/18 pseudo-negativity under the naive kernel) - tau is the even-d
# tower object; odd-d Wigner needs Gross/parity. Checks: sum_u A_u = dI, normalization,
# EXHAUSTIVE nonnegativity on all 12 stabilizer pure states, strange-state min = -1/3.
w3_=np.exp(2j*np.pi/3)
X3_=np.roll(np.eye(3),1,axis=0); Z3_=np.diag([w3_**k for k in range(3)])
PAR=np.array([[1,0,0],[0,0,1],[0,1,0]])
def Au(a,b):
    m=np.linalg.matrix_power(X3_,a)@np.linalg.matrix_power(Z3_,b)
    return m@PAR@m.conj().T
assert np.allclose(sum(Au(a,b) for a in range(3) for b in range(3)),3*np.eye(3))
def wig(rho):
    return np.array([[np.real(np.trace(rho@Au(a,b)))/3 for b in range(3)] for a in range(3)])
stab=[]
for a,b in [(1,0),(0,1),(1,1),(1,2)]:
    _,vecs=np.linalg.eig(np.linalg.matrix_power(X3_,a)@np.linalg.matrix_power(Z3_,b))
    for k in range(3): stab.append(vecs[:,k]/np.linalg.norm(vecs[:,k]))
mins=[wig(np.outer(v,v.conj())).min() for v in stab]
assert all(abs(wig(np.outer(v,v.conj())).sum()-1)<1e-9 for v in stab)
strange=np.array([0,1,-1])/np.sqrt(2)
mn_str=wig(np.outer(strange,strange.conj())).min()
print(f"(1) PM contextual fraction (state |00>): {cf_pm:.4f}   [AvN => expect 1]")
print(f"(2) KCBS contextual fraction (optimal state): {cf_k:.4f}   [class shadow 0, CF>0]")
print(f"(3) Wigner d=3: min over 12 stabilizer states = {min(mins):+.4f} (>=0); strange state min = {mn_str:+.4f} (<0)")
ok = abs(cf_pm-1)<1e-6 and cf_k>0.01 and min(mins)>-1e-9 and mn_str<-0.3
print("PASS: state sector anchored (CF hierarchy + discrete Hudson)" if ok else "FAIL")
