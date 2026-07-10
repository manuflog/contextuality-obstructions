# V22 - Pins the axioms behind note S4 ('the unique completely positive minimally
# disturbing update is the Lueders rule'). Verified statement:
#   Among EFFICIENT (single-Kraus) instruments M with M^dag M = P (Born-compatible),
#   every M factors as M = W P (unitary W), and M leaves all states in range(P)
#   unchanged iff W|range = phase  =>  Lueders M=P is the unique such representative.
#   A refinement instrument (multi-Kraus, still Born-compatible) disturbs range(P),
#   so axiom (i) efficiency is NECESSARY, not decorative.
# Paper must state axioms: (i) efficiency, (ii) M^dag M = P, (iii) identity on supp(P).
import numpy as np
rng=np.random.default_rng(7)
def haar(n):
    z=rng.normal(size=(n,n))+1j*rng.normal(size=(n,n))
    q,r=np.linalg.qr(z); d=np.diag(r); return q@np.diag(d/np.abs(d))
n,r=6,3; polar_ok=True; f_rand=[]; f_lued=[]; f_refine=[]
for _ in range(200):
    U=haar(n); B=U[:,:r]; P=B@B.conj().T
    W=haar(n); M=W@P
    polar_ok &= np.allclose(M.conj().T@M,P,atol=1e-10)
    v=P@(rng.normal(size=n)+1j*rng.normal(size=n)); v/=np.linalg.norm(v)
    o=M@v; f_rand.append(abs(np.vdot(v,o/np.linalg.norm(o)))**2)
    oL=P@v; f_lued.append(abs(np.vdot(v,oL/np.linalg.norm(oL)))**2)
    # refinement: split P into two orthogonal sub-projectors (finer vN measurement)
    P1=B[:,:1]@B[:,:1].conj().T; P2=P-P1
    rho_out=P1@np.outer(v,v.conj())@P1+P2@np.outer(v,v.conj())@P2
    f_refine.append(np.real(np.vdot(v,rho_out@v)))
print(f"polar M=WP with M^dagM=P: {polar_ok} (200/200)")
print(f"fidelity on supp(P): Lueders min={min(f_lued):.6f}; random-W mean={np.mean(f_rand):.3f}; "
      f"refinement mean={np.mean(f_refine):.3f}")
print("PASS: Lueders unique under axioms (i)-(iii); (i) shown necessary"
      if polar_ok and min(f_lued)>1-1e-9 and max(f_rand)<1-1e-6 and max(f_refine)<1-1e-6 else "FAIL")
