# V40 - THE S4 TREATMENT'S THREE CERTIFICATES.
# (A) SHARP-CORE THEOREM (POVM case). For an instrument with (i) statistics
#     tr I_j(rho)=tr(E_j rho) and (ii) repeatability tr(E_j I_j(rho))=tr I_j(rho):
#     outputs of I_j are supported in K_j = ker(I - E_j), the eigenvalue-1 space of E_j
#     (proof: tr((I-E_j)sigma)=0 with I-E_j>=0 forces support in its kernel), and the
#     K_j are mutually orthogonal. Corollary: any effect with NO 1-eigenvector admits
#     no repeatable instrument. NOTE "repeatable => projective" is FALSE: E=diag(1,c)
#     admits a fully axiom-satisfying instrument. Machine: (a) infeasibility certificate
#     for strictly-unsharp effects; (b) explicit instrument for the sharp-core example;
#     (c) with (iii) commutant covariance the solution family on the core is exactly the
#     V37 Lueders/uniformize interpolation - nullity counted.
# (B) CLASSICAL TWIN (Bayes). On a finite sample space with partition {A_j}, classical
#     instruments T_j with (i) mass = p(A_j), (ii) support in A_j, (iii) covariance under
#     partition-preserving permutations form exactly the one-parameter family
#     t * (Bayes conditioning) + (1-t) * (measure-and-uniformize) per cell |A_j|>1;
#     the deterministic/efficient member is Bayes. Lueders is to quantum what Bayes is
#     to classical, by the same three axioms.
import numpy as np, itertools
rng=np.random.default_rng(7)
# ---- (A-a) strictly unsharp => infeasible ----
c=0.6; E1=np.diag([c,1-0*0+0.3-0.3+c*0+0.7-0.7+0.4]) # build explicitly below instead
E1=np.diag([0.6,0.4]); E2=np.eye(2)-E1     # both effects strictly inside (0,1): no 1-eigenvectors
# (ii) forces output support in ker(I-E_j) = {0} => I_j = 0 => (i) fails on any rho.
infeasible = np.linalg.matrix_rank(np.eye(2)-E1)==2 and np.linalg.matrix_rank(np.eye(2)-E2)==2
print(f"(A-a) strictly unsharp two-outcome qubit POVM: ker(I-E_j)=0 for both => repeatable instrument impossible: {infeasible}")
# ---- (A-b) sharp-core example: E1=diag(1,c) ----
c=0.35; E1=np.diag([1.0,c]); E2=np.diag([0.0,1-c])
I1=lambda rho: np.trace(E1@rho)*np.diag([1.0,0.0])
I2=lambda rho: np.trace(E2@rho)*np.diag([0.0,1.0])
ok=True
for _ in range(200):
    v=rng.normal(size=2)+1j*rng.normal(size=2); v/=np.linalg.norm(v); rho=np.outer(v,v.conj())
    ok &= abs(np.trace(I1(rho))-np.trace(E1@rho))<1e-12
    ok &= abs(np.trace(E1@I1(rho))-np.trace(I1(rho)))<1e-12
    ok &= abs(np.trace(I1(rho)+I2(rho))-1)<1e-12
    U=np.diag(np.exp(1j*rng.uniform(0,2*np.pi,2)))   # commutant of {E_j}
    ok &= np.allclose(I1(U@rho@U.conj().T),U@I1(rho)@U.conj().T)
print(f"(A-b) non-projective POVM diag(1,{c}) WITH a repeatable covariant instrument (sharp core e1): {ok}"
      f"  => 'repeatable => projective' is FALSE; the sharp-core theorem is the correct statement")
# ---- (A-c) core classification with (iii): reuse V37 machinery on projective core ----
from lueders_instrument import solve_instance
ns,resid,inside=solve_instance(4,[2,2],commutant=True)
print(f"(A-c) on the sharp core the (i)-(iii) family is the V37 interpolation: nullity {ns} (=2), resid {resid:.1e}, family inside: {inside}")
# ---- (B) classical twin ----
def classical(nA=[2,2]):
    n=sum(nA); cells=[]; s=0
    for k in nA: cells.append(list(range(s,s+k))); s+=k
    # unknowns T_j (|A_j| x n); constraints from (i),(ii implicit by shape),(iii) permutations
    idx={}; cnt=0
    for j,A in enumerate(cells):
        for r in A:
            for col in range(n): idx[(j,r,col)]=cnt; cnt+=1
    rows=[]; rhs=[]
    def add(co,b):
        v=np.zeros(cnt)
        for k,x in co.items(): v[idx[k]]+=x
        rows.append(v); rhs.append(b)
    for j,A in enumerate(cells):
        for col in range(n):
            add({(j,r,col):1.0 for r in A}, 1.0 if col in A else 0.0)   # (i) per basis vector
    perms=[]
    for j,A in enumerate(cells):
        if len(A)>1:
            for a,b in itertools.combinations(A,2):
                p=list(range(n)); p[a],p[b]=p[b],p[a]; perms.append(p)
    for p in perms:
        for j,A in enumerate(cells):
            for r in A:
                for col in range(n):
                    co={}
                    for key,x in (((j,p[r],p[col]),1.0),((j,r,col),-1.0)):
                        co[key]=co.get(key,0.0)+x
                    if any(abs(x)>0 for x in co.values()): add(co,0.0)
    A_=np.array(rows); b_=np.array(rhs)
    sol,res,rk,sv=np.linalg.lstsq(A_,b_,rcond=None)
    ns=A_.shape[1]-rk
    def member(ts):
        v=np.zeros(cnt)
        for j,A in enumerate(cells):
            t=ts[j]; m=len(A)
            for r in A:
                for col in range(n):
                    bay=1.0 if col==r else 0.0
                    uni=(1.0/m) if col in A else 0.0
                    v[idx[(j,r,col)]]=t*bay+(1-t)*uni if col in A else 0.0
        return v
    inside=all(np.linalg.norm(A_@member(t)-b_)<1e-9 for t in ([1,1],[0,0],[0.4,0.9]))
    return ns,float(np.linalg.norm(A_@sol-b_)),inside
ns,resid,inside=classical([2,2])
print(f"(B) classical (i)-(iii) on |A|=[2,2]: nullity {ns} (=2: one t per cell), resid {resid:.1e}, "
      f"Bayes<->uniformize family inside: {inside}; deterministic member = Bayes conditioning")
nz,_,_=classical([3,1])
print(f"(B) [3,1] cells: nullity {nz} (=1: singleton cell pinned) - exact parallel of V37's nondegenerate case")
print("PASS" if infeasible and ok and ns==2 and inside and nz==1 else "FAIL")
