# V12 - Wigner-friend loop T --H--> Tx --S--> Ty --close--> T (friend qubit; U1=I(x)H, U2=I(x)S).
# FINDING (sharpens the CMP repair-plan claim 'holonomy -i'):
#   (a) ray-level (Bargmann/Pancharatnam) holonomy of the loop is +1: strands are e^{+i pi/4}
#       and e^{-i pi/4} (opposite Bloch octants, Omega/2 law), product 1; gauge-checked.
#   (b) det/metaplectic-section holonomy is EXACTLY -i for every SU(2) closer (500/500).
#       -i = tau^{-1} at d=2 (tau=e^{i pi/d}) => the invariant lives at the tau / 2-adic-tower
#       level (Paper B), not at the projective level. (-i)^2 = -1 is section-independent.
# => the CMP draft must pin the section for its -i claim; interferometry targets (a) or (b)^2.
import numpy as np
H=np.array([[1,1],[1,-1]])/np.sqrt(2); S=np.diag([1,1j]); Vz=np.eye(2); Vx=H; Vy=S@H
def barg(ch):
    p=1+0j
    for a in range(len(ch)): p*=np.vdot(ch[a],ch[(a+1)%len(ch)])
    return p/abs(p)
rng=np.random.default_rng(0); ray_ok=True
for _ in range(200):
    ph=lambda:np.exp(1j*rng.uniform(0,2*np.pi))
    Bz,Bx,By=(V@np.diag([ph(),ph()]) for V in (Vz,Vx,Vy))
    s0=barg([Bz[:,0],Bx[:,0],By[:,0]]); s1=barg([Bz[:,1],Bx[:,1],By[:,1]])
    ray_ok &= abs(s0*s1-1)<1e-9
s0=barg([Vz[:,0],Vx[:,0],Vy[:,0]]); s1=barg([Vz[:,1],Vx[:,1],Vy[:,1]])
vals=set()
for _ in range(500):
    th=rng.uniform(0,2*np.pi); D=np.diag([np.exp(1j*th),np.exp(-1j*th)])
    P=np.eye(2) if rng.integers(2)==0 else np.array([[0,1],[-1,0]])
    Wc=D@P@Vy.conj().T; Wc/=np.linalg.det(Wc)**0.5
    L=Wc@S@H
    assert np.allclose(L[np.abs(L)<1e-9],0)
    vals.add(complex(np.round(np.linalg.det(L),6)))
print(f"ray-level: strands {np.angle(s0)/np.pi:+.2f}pi, {np.angle(s1)/np.pi:+.2f}pi; "
      f"loop product = 1, gauge-invariant 200/200: {ray_ok}")
print(f"det-section holonomy over 500 SU(2) closers: {vals}  (square = -1)")
print("PASS" if (ray_ok and vals=={-1j}) else "FAIL")
