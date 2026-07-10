# V28 - Net-robust negativity: is 'negative in EVERY covariant qubit frame' the even-d
# state-sector classifier? Frame family: A_u = T_u A_0 T_u^dag, A_0=(1/2^n) sum gamma(v)T_v,
# gamma: nonzero Paulis -> {+-1} (2^15 frames at n=2; superset of all GHW nets - every frame
# is Hermitian, unit-trace, orthogonal, covariant by construction).
# Pinned sanity: n=1 T-state is frame-ROBUSTLY negative, value (1-sqrt2)/4 in all 8 frames.
# Verdict (computed, not assumed): per state, best-case min over all 32768 frames, vs CF
# on the trivial-class family of V27/E3.
import numpy as np, itertools
from evend_frame_probe import tests as _mk_unused  # noqa: reuse state defs below instead
from evend_frame_probe import CTX5, Pv, model, contextual_fraction, T
I2=np.eye(2); X=np.array([[0,1],[1,0]]); Y=np.array([[0,-1j],[1j,0]]); Z=np.diag([1,-1])
P1={'I':I2,'X':X,'Y':Y,'Z':Z}
# n=1 sanity
c1=lambda v: np.array([np.real(v.conj()@P1[p]@v) for p in ('X','Y','Z')])
signs1=lambda u: np.array([[(-1)**b,(-1)**(a+b),(-1)**a] for a,b in itertools.product(range(2),repeat=2)])[u]
cT=c1(T)
S1=np.array([[(-1)**b,(-1)**(a+b),(-1)**a] for a,b in itertools.product(range(2),repeat=2)])
best1=max(min((1+ (S1*g)@cT)/4) for g in itertools.product((1,-1),repeat=3))
print(f"(sanity n=1) T-state best-case min over all 8 frames: {best1:+.4f} (=(1-sqrt2)/4)")
assert abs(best1-0.25*(1-np.sqrt(2)))<1e-9
# n=2
names=[a+b for a in 'IXYZ' for b in 'IXYZ' if a+b!='II']
Tm={nm:np.kron(P1[nm[0]],P1[nm[1]]) for nm in names}
vec={'I':(0,0),'X':(1,0),'Y':(1,1),'Z':(0,1)}  # (x,z) per qubit
V=np.array([vec[nm[0]]+vec[nm[1]] for nm in names])          # 15 x 4
U=np.array(list(itertools.product(range(2),repeat=4)))        # 16 x 4 (x1,z1,x2,z2)
symp=(U[:,0,None]*V[None:,None][0][:,1] ) # placeholder
sy=np.zeros((16,15),dtype=int)
for i,u in enumerate(U):
    for j,v in enumerate(V):
        sy[i,j]=(u[0]*v[1]-u[1]*v[0]+u[2]*v[3]-u[3]*v[2])%2
M=(-1.0)**sy                                                  # 16 x 15
G=np.array(list(itertools.product((1,-1),repeat=15)))         # 32768 x 15
rng=np.random.default_rng(5)
states={"|00>":np.array([1,0,0,0],complex),"TxT":np.kron(T,T)}
for k in range(20):
    v=rng.normal(size=4)+1j*rng.normal(size=4); states[f"rand{k}"]=v/np.linalg.norm(v)
rows=[]
for nm,ps in states.items():
    c=np.array([np.real(ps.conj()@Tm[n]@ps) for n in names])   # 15 char-fn values
    W=(1.0+ G@(M*c).T)/16.0                                    # 32768 x 16
    best=W.min(axis=1).max()
    cf=contextual_fraction(CTX5,{o:(1,-1) for o in Pv},model(ps))
    rows.append((nm,cf,best))
pos=[r for r in rows if r[1]>1e-6]; zer=[r for r in rows if r[1]<=1e-6]
robust_neg_pos=all(r[2]<-1e-9 for r in pos)
rescued_zer=sum(1 for r in zer if r[2]>-1e-9)
print(f"(n=2) CF>0 states robustly negative in ALL 32768 frames: {robust_neg_pos} ({len(pos)} states)")
print(f"(n=2) CF=0 states rescued to nonnegativity by SOME frame: {rescued_zer}/{len(zer)}")
tt=next(r for r in rows if r[0]=='TxT'); z00=next(r for r in rows if r[0]=='|00>')
print(f"      TxT: CF={tt[1]:.4f}, best-case min={tt[2]:+.4f} | |00>: CF={z00[1]:.4f}, best={z00[2]:+.4f}")
miss=[r[0] for r in zer if r[2]<-1e-9]
print(f"      CF=0 yet robustly negative (classifier counterexamples): {miss if miss else 'NONE'}")
verdict = "SUPPORTED: robust negativity <=> CF>0 on this family (0 counterexamples)" if (robust_neg_pos and not miss) else f"REFUTED on this family: {len(miss)} CF=0 robustly-negative states"
print(f"VERDICT: net-robust-negativity classifier {verdict}")
# Pinned expectations (now known): n=1 sanity (asserted above); |00> rescued to 0;
# TxT robustly negative across ALL frames with CF=0. Caveat: the 2^15 family is a
# SUPERSET of GHW nets, so 'rescued' necessity-failures are inconclusive pending a
# net-restricted sweep; the SUFFICIENCY refutation (TxT) is airtight (nets subset family).
print("PASS" if (abs(z00[2])<1e-9 and tt[2]<-0.05 and tt[1]<1e-6) else "FAIL")
