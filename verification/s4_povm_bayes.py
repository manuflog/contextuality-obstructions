# V40 - THE S4 TREATMENT'S THREE CERTIFICATES.  [CORRECTED 2026-07-11 - see INDEX.md ledger]
# (A) SHARP-CORE THEOREM (POVM case). For an instrument with (i) statistics
#     tr I_j(rho)=tr(E_j rho) and (ii) repeatability tr(E_j I_j(rho))=tr I_j(rho)
#     FOR EVERY outcome j: outputs of I_j are supported in K_j = ker(I - E_j), the
#     eigenvalue-1 space of E_j (proof: tr((I-E_j)sigma)=0 with I-E_j>=0 forces support
#     in its kernel), and the K_j are mutually orthogonal. Corollary: any effect with NO
#     1-eigenvector admits no repeatable instrument for that outcome.
#     "repeatable => projective" is FALSE, but a valid witness must be >=3-DIMENSIONAL:
#     every nonzero outcome needs an eigenvalue-1 sharp core, so a non-projective repeatable
#     POVM needs each effect to have a 1-eigenvector while at least one effect stays unsharp.
#   CORRECTION (2026-07-11): the earlier qubit example E1=diag(1,c), E2=diag(0,1-c) was
#     INVALID - E2 has no eigenvalue 1, so outcome 2 is NOT repeatable (the old script checked
#     repeatability only for outcome 1). It is replaced by the qutrit example below and kept
#     as a pinned NEGATIVE regression.
# (B) CLASSICAL TWIN. On a finite sample space with partition {A_j}, classical instruments
#     T_j with (i) mass=p(A_j), (ii) support in A_j, (iii) covariance under partition-preserving
#     permutations form exactly the one-parameter AFFINE family per cell of size m=|A_j|>1:
#         T_t = t*I + (1-t)*J/m,     -1/(m-1) <= t <= 1   (nonnegativity forces the interval).
#     Deterministic covariant members: identity (t=1, "Bayes conditioning") for every m, AND
#     for m=2 also the swap (t=-1). So Bayes is the UNIQUE deterministic covariant member only
#     for m>=3; at m=2 identity and swap are both deterministic. Lueders is to quantum what this
#     affine family is to classical, by the same three axioms.
import numpy as np, itertools
rng=np.random.default_rng(7)

def repeatable(E, I, dim, tol=1e-12):
    """tr(E I(rho)) == tr(I(rho)) for random rho (repeatability of this outcome)."""
    for _ in range(50):
        v=rng.normal(size=dim)+1j*rng.normal(size=dim); v/=np.linalg.norm(v)
        rho=np.outer(v,v.conj())
        if abs(np.trace(E@I(rho))-np.trace(I(rho)))>tol: return False
    return True

# ---- (A-a) strictly unsharp => infeasible ----
E1=np.diag([0.6,0.4]); E2=np.eye(2)-E1     # both effects strictly inside (0,1): no 1-eigenvectors
infeasible = np.linalg.matrix_rank(np.eye(2)-E1)==2 and np.linalg.matrix_rank(np.eye(2)-E2)==2
print(f"(A-a) strictly unsharp two-outcome qubit POVM: ker(I-E_j)=0 for both => repeatable instrument impossible: {infeasible}")

# ---- (A-b NEGATIVE regression): the OLD qubit example fails outcome-2 repeatability ----
c=0.35; oE1=np.diag([1.0,c]); oE2=np.diag([0.0,1-c])
oI1=lambda rho: np.trace(oE1@rho)*np.diag([1.0,0.0])
oI2=lambda rho: np.trace(oE2@rho)*np.diag([0.0,1.0])
old_out1=repeatable(oE1,oI1,2); old_out2=repeatable(oE2,oI2,2)
print(f"(A-b regression) OLD diag(1,{c}) qubit example: outcome1 repeatable={old_out1}, "
      f"outcome2 repeatable={old_out2}  => INVALID as a non-projective repeatable instrument "
      f"(E2 has no eigenvalue 1: {not np.isclose(np.linalg.eigvalsh(oE2).max(),1)})")

# ---- (A-b) VALID qutrit sharp-core example: E1=diag(1,0,c), E2=diag(0,1,1-c) ----
c=0.5; E1=np.diag([1.0,0.0,c]); E2=np.diag([0.0,1.0,1-c])
I1=lambda rho: np.trace(E1@rho)*np.diag([1.0,0.0,0.0])   # sharp core |0>
I2=lambda rho: np.trace(E2@rho)*np.diag([0.0,1.0,0.0])   # sharp core |1>
ok=True
for _ in range(200):
    v=rng.normal(size=3)+1j*rng.normal(size=3); v/=np.linalg.norm(v); rho=np.outer(v,v.conj())
    ok &= abs(np.trace(I1(rho))-np.trace(E1@rho))<1e-12     # (i) statistics, outcome 1
    ok &= abs(np.trace(I2(rho))-np.trace(E2@rho))<1e-12     # (i) statistics, outcome 2
    ok &= abs(np.trace(E1@I1(rho))-np.trace(I1(rho)))<1e-12 # (ii) repeatability, outcome 1
    ok &= abs(np.trace(E2@I2(rho))-np.trace(I2(rho)))<1e-12 # (ii) repeatability, outcome 2  <-- now checked
    ok &= abs(np.trace(I1(rho)+I2(rho))-1)<1e-12            # trace preservation
    U=np.diag(np.exp(1j*rng.uniform(0,2*np.pi,3)))          # commutant of {E_j}
    ok &= np.allclose(I1(U@rho@U.conj().T),U@I1(rho)@U.conj().T)
    ok &= np.allclose(I2(U@rho@U.conj().T),U@I2(rho)@U.conj().T)
unsharp = not np.allclose(E1@E1,E1)
print(f"(A-b) VALID qutrit POVM diag(1,0,{c}) with a repeatable covariant instrument, EVERY outcome "
      f"checked: {ok}; E1 genuinely unsharp: {unsharp}  => 'repeatable => projective' is FALSE, "
      f"sharp-core theorem is the correct statement")

# ---- (A-c) core classification with (iii): reuse V37 machinery on the projective core ----
from lueders_instrument import solve_instance
ns,resid,inside=solve_instance(4,[2,2],commutant=True)
print(f"(A-c) on the sharp core the (i)-(iii) linear solution set is the V37 affine depolarizing "
      f"line: nullity {ns} (=2), resid {resid:.1e}, family inside: {inside} "
      f"(CP interval p in [-1/(r^2-1),1] handled analytically / in lueders_cp_interval.py)")

# ---- (B) classical twin: affine family, exact interval, m=2 swap ----
def T_of(t,m):
    return t*np.eye(m)+(1-t)*np.ones((m,m))/m
def stochastic(T):
    return (T>-1e-12).all() and np.allclose(T.sum(0),1)
# interval endpoints and the m=2 swap
cl_ok=True
for m in [2,3,4]:
    lo=-1/(m-1)
    cl_ok &= stochastic(T_of(1,m)) and stochastic(T_of(lo,m)) and not stochastic(T_of(lo-1e-6,m)) and not stochastic(T_of(1+1e-6,m))
swap=T_of(-1,2)
swap_ok=np.allclose(swap,[[0,1],[1,0]])                      # m=2 lower endpoint is the deterministic swap
det3=T_of(-1/2,3)                                            # m=3 lower endpoint is NOT deterministic
m3_lo_not_det = not np.allclose(det3,det3.round())
print(f"(B) classical affine family T_t=t I+(1-t)J/m: interval [-1/(m-1),1] holds for m=2,3,4: {cl_ok}; "
      f"m=2 t=-1 is the swap (deterministic): {swap_ok}; m=3 lower endpoint non-deterministic: {m3_lo_not_det}")
print(f"    deterministic covariant members: identity(t=1,'Bayes') for all m; swap(t=-1) only for m=2 "
      f"=> Bayes unique deterministic member iff m>=3")

# nullity structure of the linear (i)-(iii) system (one free parameter per cell of size>1)
def classical_nullity(nA):
    n=sum(nA); cells=[]; s=0
    for k in nA: cells.append(list(range(s,s+k))); s+=k
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
            add({(j,r,col):1.0 for r in A}, 1.0 if col in A else 0.0)      # (i)
    for j,A in enumerate(cells):
        if len(A)>1:
            for a,b in itertools.combinations(A,2):
                p=list(range(n)); p[a],p[b]=p[b],p[a]
                for r in A:
                    for col in range(n):
                        co={}
                        for key,x in (((j,p[r],p[col]),1.0),((j,r,col),-1.0)): co[key]=co.get(key,0.0)+x
                        if any(abs(x)>0 for x in co.values()): add(co,0.0)   # (iii)
    A_=np.array(rows); b_=np.array(rhs)
    sol,res,rk,sv=np.linalg.lstsq(A_,b_,rcond=None)
    return A_.shape[1]-rk, float(np.linalg.norm(A_@sol-b_))
ns2,r2=classical_nullity([2,2]); nz,rz=classical_nullity([3,1])
print(f"(B) linear (i)-(iii) nullity: [2,2] -> {ns2} (one t per size-2 cell), resid {r2:.1e}; "
      f"[3,1] -> {nz} (singleton cell pinned), resid {rz:.1e}")

PASS = (infeasible and (old_out2 is False) and ok and unsharp and cl_ok and swap_ok
        and ns==2 and inside and ns2==2 and nz==1)
print("PASS" if PASS else "FAIL")
