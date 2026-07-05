"""Independent verification of the d=16 S=8 certificate.
Path: raw numpy matrices (256x256), no algebraic phase formula, no Howell solver."""
import numpy as np, json

d,m=16,2
w=np.exp(2j*np.pi/d); tau=np.exp(1j*np.pi/d)
X=np.roll(np.eye(d),1,axis=0); Zm=np.diag([w**k for k in range(d)])
def T1(a,bb): return np.linalg.matrix_power(X,a)@np.linalg.matrix_power(Zm,bb)
def W(v):
    M=np.kron(T1(v[0],v[1]),T1(v[2],v[3]))
    return tau**(-(v[0]*v[1]+v[2]*v[3]))*M

ctxs=[[tuple(v) for v in C] for C in json.load(open("fiber16_ctxs.json"))]
lam=np.load("cert16_lambda.npy")
sup=[i for i in range(len(ctxs)) if lam[i]%16]
print("certificate support:",len(sup),"contexts")

I=np.eye(d**m)
obs_mult={}
S=0
for i in sup:
    C=ctxs[i]
    P=I
    for v in C:
        P=P@W(v)
    z=P[0,0]
    assert np.allclose(P,z*I), ("not scalar",i)
    sval=round(np.angle(z)/(2*np.pi/d))%d
    assert np.allclose(z,w**sval), ("not in mu_d",i,z)
    S=(S+int(lam[i])*sval)%d
    for v in C: obs_mult[v]=obs_mult.get(v,0)+int(lam[i])
bad=[v for v,c in obs_mult.items() if c%d]
print("lambda^T A == 0 mod 16:",len(bad)==0)
print("certificate value S =",S,"(claim: 8)")
print("VERDICT:", "PASS - independent matrix path confirms S=8 at d=16" if (not bad and S==8) else "FAIL")
