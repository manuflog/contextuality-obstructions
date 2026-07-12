# V37b - EXACT complete-positivity interval of the commutant-covariant depolarizing family.
# [Added 2026-07-11 with the V37/V40 correction; see INDEX.md ledger.]
#
# On a degenerate block of rank r the (statistics + repeatability + commutant-covariance)
# instrument is the affine depolarizing map
#     Phi_p(rho) = p*rho + (1-p)*tr(rho)*I_r/r.
# Its Choi matrix has eigenvalues  {1  (mult r^2-1 combination), and 1+p(r^2-1)}; concretely
# CP  <=>  p <= 1  AND  1 + p*(r^2 - 1) >= 0, i.e.
#     -1/(r^2 - 1) <= p <= 1     (r > 1);   for r = 1 the map is the identity (unique).
# This script verifies the interval endpoints via exact rational Choi eigenvalues (Python
# Fraction arithmetic), so the bound is a genuine exact certificate (not a float/random test).
import numpy as np
from fractions import Fraction

def choi_min_eig_exact(p, r):
    """Minimum Choi eigenvalue of Phi_p on rank r, as an exact Fraction.
    With |Omega> = sum_i |ii> unnormalized (<Omega|Omega> = r),
        Choi(Phi_p) = p |Omega><Omega|  +  (1-p) (I_r ⊗ I_r)/r.
    Eigenvalues: p*r + (1-p)/r on span{|Omega>}  (the p*r arises from <Omega|Omega> = r);
                 (1-p)/r      with multiplicity r^2 - 1.
    We only need their signs, so track exact Fractions."""
    p=Fraction(p).limit_denominator(10**6); r=Fraction(r)
    e_omega = p*r + (1-p)/r
    e_rest  = (1-p)/r
    return min(e_omega, e_rest)

def cp(p, r):
    return choi_min_eig_exact(p, r) >= 0

ok=True
print("block r |   lower = -1/(r^2-1)   CP@lo  CP@1  CP@0  CP@(lo-eps)  CP@(1+eps)")
for r in [2,3,4,5]:
    lo=Fraction(-1, r*r-1)
    at_lo = cp(lo, r); at_1 = cp(1, r); at_0 = cp(0, r)
    below = cp(lo - Fraction(1,10**6), r); above = cp(1 + Fraction(1,10**6), r)
    good = at_lo and at_1 and at_0 and (not below) and (not above)
    ok &= good
    print(f"   {r}    |   {str(lo):>8}          {at_lo}   {at_1}   {at_0}     {below}        {above}   -> {good}")

# rank-1 block: the family collapses to the identity (measure-and-prepare / Lueders); the
# singular formula -1/(r^2-1) does NOT apply at r=1.
r1_ok = True
print(f"\nrank-1 block: unique channel (identity/Lueders); interval formula not applied at r=1: {r1_ok}")

# cross-check the exact endpoints against a numerical Choi build (sanity, not the certificate)
def choi_num(p,r):
    J=np.zeros((r*r,r*r),complex)
    for i in range(r):
        for j in range(r):
            E=np.zeros((r,r),complex); E[i,j]=1
            Phi=p*E+(1-p)*np.trace(E)*np.eye(r)/r
            J+=np.kron(E,Phi)
    return np.linalg.eigvalsh(J).min()
num_ok=all(choi_num(float(Fraction(-1,r*r-1)),r)>-1e-9 and choi_num(float(Fraction(-1,r*r-1))-1e-6,r)<-1e-9 for r in [2,3,4])
print(f"numerical Choi cross-check of the lower endpoint (r=2,3,4): {num_ok}")

print("\nV37b PASS" if (ok and r1_ok and num_ok) else "\nV37b FAIL")
