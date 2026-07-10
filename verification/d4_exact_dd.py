# V42 (PREPARED, PENDING COMPUTE) - EXACT COMPLETE FACET ENUMERATION of the d=4 deleted-
# family vertex polytope: 64 integer points (entries in {-1,0,1}) in affine dimension 33
# (faithful integer coordinates = 33 pivot columns; projection is an isomorphism of the
# polytope, so its H-representation IS the complete facet catalogue - settling tier-2
# completeness as a THEOREM). Double description did not finish within the interactive
# session budget; run this file to completion (Colab/overnight; try float first, then
# gmp for exact rational certificates). Requires: pip install pycddlib (needs libcdd-dev
# libgmp-dev on Linux).
import numpy as np, itertools, json, time, pickle
def build_Y():
    from weyl import build
    X,Z,w,tau,W,_=build(4,2)
    fam=[[tuple(v) for v in it["ctx"]] for it in json.load(open("cert4_min.json"))["items"]]
    obs=sorted({t for C in fam for t in C}); oi={t:k for k,t in enumerate(obs)}
    def s_of(C):
        M=np.eye(16,dtype=complex)
        for v in C: M=M@W(np.array(v))
        return int(np.round(np.angle(np.trace(M)/abs(np.trace(M)))/(np.pi/2)))%4
    S=[s_of(C) for C in fam]
    CTX=[fam[i] for i in range(1,6)]
    spec={v:sorted(set(int(np.round(np.angle(x)/(np.pi/2)))%4 for x in np.linalg.eigvals(W(np.array(v))))) for v in obs}
    A=np.zeros((5,9),int)
    for r,C in enumerate(CTX):
        for v in C: A[r,oi[v]]+=1
    rhs=np.array([S[i] for i in range(6) if i!=0])
    grid=np.array(list(itertools.product(range(4),repeat=9)))
    ok=((grid@A.T)%4==rhs%4).all(axis=1)
    for v in obs: ok&=np.isin(grid[:,oi[v]],spec[v])
    L=grid[ok]
    CH=[(a,b) for a in range(4) for b in range(4) if (a,b)!=(0,0)]
    V=[]
    for row in L:
        vec=[]
        for C in CTX:
            j1,j2=int(row[oi[C[0]]]),int(row[oi[C[1]]])
            for (a,b) in CH:
                z=np.exp(-1j*np.pi/2*(a*j1+b*j2)); vec+=[z.real,z.imag]
        V.append(vec)
    V=np.round(np.array(V)).astype(int)
    D=V-V[0]; rank=int(np.linalg.matrix_rank(D.astype(float)))
    P=[]; cur=np.zeros((V.shape[0],0))
    for c in range(150):
        t=np.concatenate([cur,D[:,c:c+1].astype(float)],axis=1)
        if np.linalg.matrix_rank(t)>cur.shape[1]: P.append(c); cur=t
        if len(P)==rank: break
    return V[:,P], np.array(P)
if __name__=='__main__':
    Y,P=build_Y()
    print(f"polytope: {Y.shape} integer points, entries {sorted(set(Y.flatten().tolist()))}")
    np.savez('d4_exact_dd_input.npz',Y=Y,P=P)
    import cdd
    rows=[[1.0]+[float(x) for x in r] for r in Y]
    t0=time.time()
    mat=cdd.matrix_from_array(rows,rep_type=cdd.RepType.GENERATOR)
    poly=cdd.polyhedron_from_matrix(mat)
    H=cdd.copy_inequalities(poly)
    arr=np.array(H.array,float); lin=sorted(H.lin_set)
    print(f"FLOAT DD: {arr.shape[0]} rows, {len(lin)} equalities, {arr.shape[0]-len(lin)} facets, t={time.time()-t0:.0f}s")
    pickle.dump({'arr':arr,'lin':lin},open('d4_exact_dd_float.pkl','wb'))
# STATUS NOTES (session-measured):
# * plain lrs streams ~1 facet/sec single-core on this polytope (1392 exact integer
#   facets in ~1400s, saved in d4_exact_dd_partial.npz, all verified b+a.y>=0 exactly);
#   with tier-2 alone >=21504 members, expect hours: run offline to completion, or use
#   lrs's maxcobases/restart relay, or mplrs with MPI for parallel speedup.
# * QUANTUM STATES ARE ON-HULL (residual ~1e-15 over samples): the classifier needs the
#   facet inequalities only - no equality part - so the finished H-representation IS the
#   complete classification theorem: CF>0 <=> some listed facet violated.
# * in-session alternative for full completeness without full enumeration: RIDGE-WALK
#   MODULO THE GROUP - pivot across ridges from known facet classes, quotient by the
#   derived 1536-group, BFS to closure; facet-graph connectivity of polytopes makes
#   closure a completeness proof with only (#classes x ridges) pivots. Designed, not yet
#   implemented.
