# V39 - THE FRAUCHIGER-RENNER OBSERVER CYCLE, LOCALIZED.
# Model: each lab one qubit (memory identified with the measured system). The FR chain's
# probabilistic core is Hardy's structure on psi = (|h,dn> + |t,dn> + |t,up>)/sqrt3 with
# the four observer contexts {Zb Z},{Zb X},{Xb X},{Xb Z} - a closed 4-cycle Weyl family.
# CLAIMS TESTED:
#  (1) CLASS: the family is Z2-solvable (all context phases +1) - trivial class; and the
#      switch loop around the observer cycle has hol = +1 EXACTLY (2-qubit Weyl dets are
#      all +1, so hol is fully gauge-invariant here). => the Paper D conjecture
#      "FR paradoxes measure holonomy of observer cycles" is REFUTED by our own machinery.
#  (2) STATE SECTOR: CF(F_FR, psi_FR) > 0; by the codeword-polytope theorem (V31) the
#      FR contradiction <=> moment vector outside conv L. The separating facet is
#      extracted and identified; ABM: CF = (S - 2)/2 for the CHSH-normalized facet.
import numpy as np, itertools
import scipy.optimize as so
I2=np.eye(2); X=np.array([[0.,1],[1,0]]); Z=np.diag([1.,-1]); H=(X+Z)/np.sqrt(2)
def kron(a,b): return np.kron(a,b)
OBS={'Zb':kron(Z,I2),'Xb':kron(X,I2),'Z':kron(I2,Z),'X':kron(I2,X),
     'ZbZ':kron(Z,Z),'ZbX':kron(Z,X),'XbX':kron(X,X),'XbZ':kron(X,Z)}
names=list(OBS); ni={n:k for k,n in enumerate(names)}
CTX=[('Zb','Z','ZbZ'),('Zb','X','ZbX'),('Xb','X','XbX'),('Xb','Z','XbZ')]
# (1a) context phases and solvability over Z2
svals=[]
for a,b,c in CTX:
    M=OBS[a]@OBS[b]@OBS[c]
    svals.append(int(np.round(np.real(np.trace(M))/4)))   # +1 or -1
A=np.zeros((4,8),int)
for r,(a,b,c) in enumerate(CTX):
    for n in (a,b,c): A[r,ni[n]]=1
rhs=np.array([0 if s==1 else 1 for s in svals])
lam=np.array(list(itertools.product((0,1),repeat=8)))
L=lam[((lam@A.T)%2==rhs%2).all(axis=1)]
print(f"(1a) context phases {svals} (all +1 => trivial class), |L| = {len(L)} (solvable: {len(L)>0})")
# (1b) switch-loop holonomy around the observer cycle
sw=[kron(I2,H),kron(H,I2),kron(I2,H),kron(H,I2)]
acc=np.eye(4,dtype=complex)
for U in sw: acc=U@acc
hol=np.linalg.det(acc)   # loop closes exactly: acc should be I
dets=[np.round(np.linalg.det(kron(P,Q)),9) for P in (I2,X,Z,X@Z) for Q in (I2,X,Z,X@Z)]
print(f"(1b) switch loop closes: {np.allclose(acc,np.eye(4))}, hol = {np.round(hol,9)}; "
      f"all 16 two-qubit Weyl dets = +1: {all(abs(d-1)<1e-9 for d in dets)} (hol fully invariant)")
# (2) state sector
psi=np.zeros(4); psi[0]=1; psi[2]=1; psi[3]=1; psi/=np.linalg.norm(psi)   # |00>+|10>+|11>
def joint_e(a,b,ctxops):
    P={}
    for sa,sb in itertools.product((1,-1),repeat=2):
        Pa=(np.eye(4)+sa*ctxops[0])/2; Pb=(np.eye(4)+sb*ctxops[1])/2
        P[(sa,sb)]=float(np.real(psi@Pa@Pb@psi))
    return P
E={n:float(np.real(psi@OBS[n]@psi)) for n in names}
# CF LP: variables over L; constraints per context outcome (sa,sb) [third determined]
Aub=[]; bub=[]
for r,(a,b,c) in enumerate(CTX):
    P=joint_e(a,b,(OBS[a],OBS[b]))
    for (sa,sb),p in P.items():
        la=0 if sa==1 else 1; lb=0 if sb==1 else 1
        Aub.append(((L[:,ni[a]]==la)&(L[:,ni[b]]==lb)).astype(float)); bub.append(p)
r=so.linprog(-np.ones(len(L)),A_ub=np.array(Aub),b_ub=np.array(bub),bounds=[(0,None)]*len(L),method="highs")
CF=1.0+r.fun
print(f"(2a) CF(FR family, psi_FR) = {CF:.6f}  (>0: state-sector contextual)")
# codeword-polytope instance + separating facet
V=(1-2*L).astype(float)              # chi = (-1)^lambda
c=np.array([E[n] for n in names])
D=8
cc=np.concatenate([-c,[1.0]])
Asep=np.concatenate([V,-np.ones((len(V),1))],axis=1)
rs=so.linprog(cc,A_ub=Asep,b_ub=np.zeros(len(V)),bounds=[(-1,1)]*D+[(None,None)],method="highs")
f=rs.x[:D]; gap=float(c@f-rs.x[D])
fs=np.round(f*2)/2
b_nc=float((V@fs).max()); qv=float(c@fs)
print(f"(2b) codeword-polytope: outside (gap {gap:.4f}); snapped facet coefficients:")
print("     "+", ".join(f"{n}:{fs[ni[n]]:+.1f}" for n in names if abs(fs[ni[n]])>1e-9))
print(f"     NC bound {b_nc:.1f}, quantum value {qv:.4f}")
chsh=abs(fs[ni['ZbZ']])+abs(fs[ni['ZbX']])+abs(fs[ni['XbX']])+abs(fs[ni['XbZ']])
is_chsh= chsh==4 and all(abs(fs[ni[n]])<1e-9 for n in ('Zb','Xb','Z','X'))
print(f"(2c) facet is product-only CHSH-type: {is_chsh}; ABM formula CF=(S-b)/2 = {max(0,(qv-b_nc)/2):.6f} vs LP CF {CF:.6f}")
ok=(all(s==1 for s in svals) and len(L)>0 and abs(hol-1)<1e-9 and CF>1e-6
    and abs(max(0,(qv-b_nc)/2)-CF)<1e-6)
print("PASS" if ok else "FAIL")
