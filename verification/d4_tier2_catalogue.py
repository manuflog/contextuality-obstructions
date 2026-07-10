# V36 - TIER-2 EXACT FACETS at d=4 (DROP=0): exact arithmetic pinned; completeness RETRACTED.
# STATUS (corrected after fresh-sample audit): the 8 extracted facets classify the
# construction sample 40/40, but FRESH samples leak (32/40 raw; ~37-40/40 after closing
# under the PROVEN symmetries: kernel translations x global conjugation, ~1152 facets).
# Per-context outcome relabelings are NOT symmetries (each observable lives in two
# contexts; pinned counterexample: bound violation 25.9 on vertices). Every facet ever
# extracted shares the exact arithmetic: tight-rank 32, |coeff| = 3/(2*sqrt2), integer
# bound in {6,7} (unit-normalized 4*sqrt2, 14*sqrt2/3). The complete tier-2 family =
# one orbit question for the family's true automorphism group - OPEN.
# Method: for each triangle-invisible CF>0 state, extract a supporting inequality via the
# separation LP, keep those with tight-vertex rank exactly 32 (true facets of the 33-dim
# hull), and SNAP the normal onto the (1/(2*sqrt2))Z grid. PINNED FACTS:
#   * snapping is EXACT (error 0): tier-2 facet normals live on the Z[1/sqrt2] grid with a
#     SINGLE coefficient magnitude 3/(2*sqrt2) across all extracted facets;
#   * bounds are integers {6,7}; in unit-coefficient normalization: {4*sqrt2, 14*sqrt2/3} -
#     Tsirelson's sqrt2 sits in the bound;
#   * five (bound, tight-count) classes; the b=7 class is uniform (48 tight vertices);
#   * tier-1 (tau-twisted ghost triangles) + tier-2 exact facets classify 40/40 states.
import numpy as np, itertools, json, os
import scipy.optimize as so
if not os.path.exists('/tmp/v35_cache.npz'):
    import d4_odd_sector_facets  # regenerates the cache
d=np.load('/tmp/v35_cache.npz'); V=d['V']; mus=d['mus']; cfs=d['cfs']; vis=d['vis']
D=V.shape[1]; s2=np.sqrt(2)
resid=[i for i in range(40) if cfs[i]>1e-4 and not vis[i]]
rng=np.random.default_rng(3)
def sep(mu):
    c=np.concatenate([-mu,[1.0]])
    A=np.concatenate([V,-np.ones((len(V),1))],axis=1)
    r=so.linprog(c,A_ub=A,b_ub=np.zeros(len(V)),bounds=[(-1,1)]*D+[(None,None)],method="highs")
    return r.x[:D]
snap=lambda f:(lambda g1,g2: np.where(np.abs(f-g1)<=np.abs(f-g2),g1,g2))(np.round(f*2)/2,np.round(f*2*s2)/(2*s2))
facets=[]
for i in resid:
    for trial in range(6):
        mu=mus[i]+(0 if trial==0 else 1e-3*rng.normal(size=D))
        f=sep(mu); vals=V@f
        T=V[vals>vals.max()-1e-7]
        if len(T)<2 or np.linalg.matrix_rank(T-T.mean(0),tol=1e-7)!=32: continue
        fs=snap(f); b=(V@fs).max()
        assert np.max(np.abs(f-fs))<1e-9 and abs(b-round(b))<1e-9, "snap must be exact"
        facets.append((i,fs,int(round(b))))
        break
mags=sorted(set(int(round(x)) for fc in facets for x in np.abs(fc[1][np.abs(fc[1])>1e-9])*2*s2))
bset=sorted(set(b for _,_,b in facets))
ag=sum((bool(vis[j]) or any(float(mus[j]@fs-b)>1e-7 for _,fs,b in facets))==(cfs[j]>1e-4) for j in range(40))
print(f"tier-2 exact facets: {len(facets)} | coeff |c|*2sqrt2 values: {mags} | bounds: {bset}")
print(f"unit-normalized bounds: {[round(b*2*s2/3,4) for b in bset]}  (= 4*sqrt2 = {4*s2:.4f}, 14*sqrt2/3 = {14*s2/3:.4f})")
print(f"COMBINED CLASSIFICATION (tier1 + tier2 exact): {ag}/40")
print("NOTE: 40/40 above is the CONSTRUCTION sample (partially circular); fresh-sample")
print("completeness is open - see header. Exact facet arithmetic is the pinned result.")
print("PASS" if (mags==[3] and bset==[6,7] and ag==40) else "FAIL")
