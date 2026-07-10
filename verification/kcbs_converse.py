# V25 - Converse gap certificate (operational reading of note S3):
# state-DEPENDENT contextuality exists at d=3 while the state-independent class shadow
# is {0} (V13 / Thm J). KCBS pentagon: quantum value sqrt(5)=2.2360 vs classical bound 2
# (exhaustive over all 32 exclusivity-respecting assignments; independence number of C5).
# => 'holonomy encodes contextuality' FAILS in the <= direction on the operational reading;
#    the correct repaired claim is the Thm Q criterion (AvN part only).
import numpy as np, itertools
c=np.cos(np.pi/5); ct2=c/(1+c); st=np.sqrt(1-ct2)
V=[np.array([st*np.cos(4*np.pi*k/5), st*np.sin(4*np.pi*k/5), np.sqrt(ct2)]) for k in range(5)]
orth=max(abs(V[k]@V[(k+1)%5]) for k in range(5))
psi=np.array([0,0,1.0]); q=sum((psi@v)**2 for v in V)
cl=max(sum(a) for a in itertools.product([0,1],repeat=5)
       if all(a[k]*a[(k+1)%5]==0 for k in range(5)))
print(f"KCBS orthogonality residual: {orth:.2e}; quantum value: {q:.5f}; classical bound: {cl}")
print("PASS: contextual (2.23607 > 2) with zero d=3 class shadow => converse gap certified"
      if orth<1e-12 and q>2.236 and cl==2 else "FAIL")
