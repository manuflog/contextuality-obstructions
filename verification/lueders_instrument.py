# V37 - INSTRUMENT-LEVEL LUEDERS THEOREM (machine-certified instances).
# Observable A = sum_j a_j P_j on C^d with degenerate eigenspaces (d_j = rank P_j).
# An instrument {I_j} (CP maps, sum trace-preserving) measuring A satisfies:
#  (i)   statistics:    tr I_j(rho) = tr(P_j rho)
#  (ii)  repeatability:  I_j(rho) = P_j I_j(rho) P_j   (output in the j-th sector)
#  (iii) covariance under the COMMUTANT unitaries U = sum_j U_j (block unitaries):
#            I_j(U rho U*) = U I_j(rho) U*
# THEOREM (verified exactly here at (d,degs)=(4,[2,2]) and (6,[3,2,1])):
#   the solution set is exactly  I_j(rho) = p_j P_j rho P_j + (1-p_j) tr(P_j rho) P_j/d_j,
#   p_j in [0,1]  (CP forces the interval);  EFFICIENCY (pure Kraus) <=> p_j = 1: LUEDERS.
# SHARPNESS: with covariance only under the diagonal MASA the solution space is strictly
# larger (dims reported): the commutant-covariance hypothesis is what pins Lueders.
# Analytic core: (i)+(ii) reduce I_j to a map into the j-block; block-covariance under
# the full unitary group of the block makes it a depolarizing-family member (Schur).
import numpy as np, itertools
rng=np.random.default_rng(5)
def haar(n):
    z=rng.normal(size=(n,n))+1j*rng.normal(size=(n,n))
    q,r=np.linalg.qr(z); return q*np.exp(1j*np.angle(np.diag(r)))
def solve_instance(d,degs,commutant=True,nU=6):
    js=len(degs); P=[]; blk=[]; s0=0
    for dj in degs:
        M=np.zeros((d,d)); M[s0:s0+dj,s0:s0+dj]=np.eye(dj); P.append(M)
        blk.append(list(range(s0,s0+dj))); s0+=dj
    # unknowns AFTER repeatability reduction: B_j[k,l][r,c] with r,c in block j
    index={}; cnt=0
    for j in range(js):
        for k in range(d):
            for l in range(d):
                for r in blk[j]:
                    for c in blk[j]:
                        index[(j,k,l,r,c)]=cnt; cnt+=1
    rowsA=[]; rowsb=[]
    def addrow(co,rhs):
        v=np.zeros(cnt,complex)
        for key,x in co.items():
            if key in index: v[index[key]]+=x
        rowsA.append(v); rowsb.append(rhs)
    for j in range(js):
        for k in range(d):
            for l in range(d):
                addrow({(j,k,l,r,r):1.0 for r in blk[j]}, P[j][l,k])
    Us=[]
    for _ in range(nU):
        if commutant:
            U=np.zeros((d,d),complex)
            for j,dj in enumerate(degs):
                U[np.ix_(blk[j],blk[j])]=haar(dj)
        else:
            U=np.diag(np.exp(2j*np.pi*rng.random(d)))
        Us.append(U)
    for U in Us:
        Uc=U.conj()
        for j in range(js):
            for k0,l0 in itertools.product(range(d),repeat=2):
                for r in blk[j]:
                    for c in blk[j]:
                        co={}
                        for k in range(d):
                            for l in range(d):
                                x=U[k,k0]*Uc[l,l0]
                                if abs(x)>1e-14: co[(j,k,l,r,c)]=co.get((j,k,l,r,c),0)+x
                        for rr in blk[j]:
                            for cc in blk[j]:
                                x=-U[r,rr]*Uc[c,cc]
                                if abs(x)>1e-14: co[(j,k0,l0,rr,cc)]=co.get((j,k0,l0,rr,cc),0)+x
                        addrow(co,0.0)
    A=np.array(rowsA); b=np.array(rowsb)
    sol,res,rk,sv=np.linalg.lstsq(A,b,rcond=None)
    resid=float(np.linalg.norm(A@sol-b)); ns=A.shape[1]-rk
    def instr_from(pvec):
        vv=np.zeros(cnt,complex)
        for j,pj in enumerate(pvec):
            dj=len(blk[j])
            for k in range(d):
                for l in range(d):
                    E=np.zeros((d,d)); E[k,l]=1
                    lud=P[j]@E@P[j]; dep=P[j][l,k]*P[j]/dj
                    Bkl=pj*lud+(1-pj)*dep
                    for r in blk[j]:
                        for c in blk[j]:
                            vv[index[(j,k,l,r,c)]]=Bkl[r,c]
        return vv
    tests=[[1]*js,[0]*js,[0.3]*js,[1,0]+[0.5]*(js-2)]
    inside=all(np.linalg.norm(A@instr_from(p)-b)<1e-8 for p in tests)
    return ns,resid,inside
if __name__=='__main__':
 RES=[]
 for d,degs in [(4,[2,2]),(6,[3,2,1])]:
     ns,resid,inside=solve_instance(d,degs,commutant=True)
     nsM,_,_=solve_instance(d,degs,commutant=False)
     nfree=sum(1 for dj in degs if dj>1)
     ok=(ns==nfree) and resid<1e-8 and inside and nsM>ns
     RES.append(ok)
     print(f"(d={d},degs={degs}) commutant-covariant: nullity {ns} (expected {nfree}), "
           f"resid {resid:.1e}, family inside: {inside} | MASA-only nullity {nsM} (sharp: {nsM>ns}) -> {ok}")
 print("PASS" if all(RES) else "FAIL")
