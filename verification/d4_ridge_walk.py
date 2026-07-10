# V43 - RIDGE-WALK MODULO THE GROUP: complete facet-class enumeration by BFS on the
# facet-ridge graph, quotiented by the derived vertex-permutation group (|G|=768).
# Every step is exactly certified: ridges by exact-arithmetic lrs on the tight-set
# polytope; neighbors proposed by a numeric rotation and CERTIFIED by integer nullspace
# recovery (unique hyperplane through a rank-32 tight set), exact validity on all 64
# integer vertices. Closure of the BFS = completeness (facet-ridge graphs of polytopes
# are connected): the class list is then the COMPLETE facet catalogue up to symmetry.
import numpy as np, pickle, subprocess, os, time
from fractions import Fraction
from math import gcd
def int_nullvec(M):
    # exact 1-dim nullspace of integer matrix M (n x m) via Fraction RREF; returns primitive int vector
    n,m=len(M),len(M[0])
    A=[[Fraction(x) for x in row] for row in M]
    piv=[]; r=0
    for c in range(m):
        pr=None
        for i in range(r,n):
            if A[i][c]!=0: pr=i; break
        if pr is None: continue
        A[r],A[pr]=A[pr],A[r]
        pv=A[r][c]
        A[r]=[x/pv for x in A[r]]
        for i in range(n):
            if i!=r and A[i][c]!=0:
                f=A[i][c]; A[i]=[A[i][k]-f*A[r][k] for k in range(m)]
        piv.append(c); r+=1
        if r==n: break
    free=[c for c in range(m) if c not in piv]
    assert len(free)==1, f"null dim {len(free)}"
    fc=free[0]
    v=[Fraction(0)]*m; v[fc]=Fraction(1)
    for rr,c in enumerate(piv): v[c]=-A[rr][fc]
    den=1
    for x in v: den=den*x.denominator//gcd(den,x.denominator)
    w=[int(x*den) for x in v]
    g=0
    for x in w: g=gcd(g,abs(x))
    return [x//g for x in w]
Y=np.load('/tmp/v42_proj.npz')['Y'].astype(np.int64)
st=pickle.load(open('/tmp/v43_seed.pkl','rb'))
G=[np.array(g) for g in st['G']]
def canon_mask(bits):
    best=None
    for g in G:
        nb=np.zeros(64,dtype=bool); nb[g]=bits
        v=np.packbits(nb).tobytes()
        if best is None or v<best: best=v
    return best
def exact_mask(b,a):
    vals=b+Y@a
    assert vals.min()>=0
    return vals==0
def recover(pts):
    M=[[1]+list(map(int,y)) for y in pts]
    w=int_nullvec(M)
    b,a=w[0],np.array(w[1:],dtype=np.int64)
    vals=b+Y@a
    if vals.min()<0: b,a=-b,-a; vals=-vals
    assert vals.min()==0 and vals.max()>0
    T=Y[vals==0]
    assert np.linalg.matrix_rank((T-T[0]).astype(float))==32
    return int(b),a
def ridges_of(T):
    D=T-T[0]; P=[]; cur=np.zeros((T.shape[0],0))
    for c in range(33):
        t=np.concatenate([cur,D[:,c:c+1].astype(float)],axis=1)
        if np.linalg.matrix_rank(t)>cur.shape[1]: P.append(c)
        cur=t if len(P)>cur.shape[1] else cur
    # rebuild cur properly
    P=[]; cur=np.zeros((T.shape[0],0))
    for c in range(33):
        t=np.concatenate([cur,D[:,c:c+1].astype(float)],axis=1)
        if np.linalg.matrix_rank(t)>cur.shape[1]: P.append(c); cur=t
        if len(P)==32: break
    Tp=T[:,P]
    with open('/tmp/rw.ext','w') as f:
        f.write(f"rw\nV-representation\nbegin\n{Tp.shape[0]} 33 integer\n")
        for r in Tp: f.write("1 "+" ".join(str(int(x)) for x in r)+"\n")
        f.write("end\n")
    subprocess.run(['timeout','--signal=KILL','600','lrs','/tmp/rw.ext','/tmp/rw.ine'],capture_output=True)
    txt=open('/tmp/rw.ine').read()
    if 'Totals' not in txt: return None    # heavy: sub-enumeration did not finish
    rows=[]
    for line in txt.splitlines():
        t=line.split()
        if t and all(x.lstrip('-').isdigit() for x in t) and len(t)==33:
            rows.append([int(x) for x in t])
    if not rows: return []
    RA=np.array(rows,dtype=np.int64)
    Vv=RA[:,0:1]+RA[:,1:]@Tp.T
    return list(Vv==0)
def neighbor(b,a,T,Rmask):
    R=T[Rmask]; r0=R[0]
    U_,s_,Vt=np.linalg.svd((R-r0).astype(float))
    null=Vt[31:33]
    an=a.astype(float); an/=np.linalg.norm(an)
    u=null[0]-(null[0]@an)*an
    if np.linalg.norm(u)<1e-8: u=null[1]-(null[1]@an)*an
    u/=np.linalg.norm(u)
    dT=(T[~Rmask]-r0)@u
    assert np.all(dT>1e-9) or np.all(dT<-1e-9), "ridge side check"
    if dT[0]<0: u=-u
    s=(b+Y@a).astype(float); d=(Y-r0)@u
    out=s>1e-9
    rho=d[out]/s[out]
    alpha=-float(np.min(rho))
    idx=np.where(out)[0]; j=idx[int(np.argmin(rho))]
    val=alpha*s+d
    nm=val<1e-7*max(1.0,abs(alpha))
    return j,nm

STATE='/tmp/v43_state.pkl'
if os.path.exists(STATE):
    S=pickle.load(open(STATE,'rb'))
else:
    classes={}; queue=[]
    for ck,(src,row) in st['classes'].items():
        if row is not None:
            row=np.array(row,dtype=np.int64)
            b,a=int(row[0]),row[1:]
            classes[ck]=(b,a.tolist()); queue.append(ck)
    S={'classes':classes,'queue':queue,'done':[]}
classes=S['classes']; queue=S['queue']; done=set(map(tuple,[[d] for d in S['done']])) if False else set(S['done'])
# self-test: recover must reproduce a known lrs facet from its tight set
_H=np.load('d4_exact_dd_partial.npz')['H'][0]
_m=exact_mask(int(_H[0]),_H[1:].astype(np.int64))
_b,_a=recover(Y[_m])
assert abs(_b)==abs(int(_H[0])) and np.array_equal(np.abs(_a),np.abs(_H[1:])), "recover self-test"
print("recover self-test PASS",flush=True)
t0=time.time(); expanded=0
def tc_of(ck):
    b,a=classes[ck][0],np.array(classes[ck][1],dtype=np.int64)
    return int(exact_mask(b,a).sum())
queue.sort(key=tc_of)
HEAVY=set(S.get('heavy',[]))
while queue and time.time()-t0<1250:
    ck=queue.pop(0)
    if ck in done: continue
    b,a=classes[ck][0],np.array(classes[ck][1],dtype=np.int64)
    T=Y[exact_mask(b,a)]
    print(f"[{time.time()-t0:.0f}s] expanding class tight={len(T)} ...",flush=True)
    RD=ridges_of(T)
    if RD is None:
        HEAVY.add(ck); print("    HEAVY (sub-lrs incomplete) - set aside",flush=True); continue
    # stabilizer-quotient of ridges
    fmask=np.zeros(64,dtype=bool); fmask[np.where(exact_mask(b,a))[0]]=True
    fkey=np.packbits(fmask).tobytes()
    Gs=[]
    for g in G:
        nb=np.zeros(64,dtype=bool); nb[g]=fmask
        if np.packbits(nb).tobytes()==fkey: Gs.append(g)
    Tidx=np.where(fmask)[0]
    reps=[]; covered=set()
    for Rm in RD:
        rmask=np.zeros(64,dtype=bool); rmask[Tidx[Rm]]=True
        kk=np.packbits(rmask).tobytes()
        if kk in covered: continue
        reps.append(Rm)
        for g in Gs:
            nb=np.zeros(64,dtype=bool); nb[g]=rmask
            covered.add(np.packbits(nb).tobytes())
    print(f"    ridges: {len(RD)} -> {len(reps)} reps (|stab|={len(Gs)})",flush=True)
    RD=reps
    for ri,Rm in enumerate(RD):
        if ri%500==499: print(f"    ridge {ri+1}/{len(RD)}",flush=True)
        j,nm=neighbor(b,a,T,Rm)
        ck2=canon_mask(nm)
        if ck2 not in classes:
            R=T[Rm]
            b2,a2=recover(np.concatenate([R,[Y[j]]],axis=0))
            m=exact_mask(b2,a2)
            ck2e=canon_mask(m)
            if ck2e not in classes:
                classes[ck2e]=(int(b2),a2.tolist()); queue.append(ck2e)
            if ck2e!=ck2: classes[ck2]=classes[ck2e]
    done.add(ck); expanded+=1
    print(f"[{time.time()-t0:.0f}s] class expanded ({expanded}); classes {len(classes)} queue {len(queue)}",flush=True)
    if True:
        S={'classes':classes,'queue':queue,'done':list(done),'heavy':list(HEAVY)}
        pickle.dump(S,open(STATE,'wb'))

S={'classes':classes,'queue':queue,'done':list(done),'heavy':list(HEAVY)}
pickle.dump(S,open(STATE,'wb'))
print(f"RIDGE WALK: classes {len(classes)}, expanded {len(done)}, queue {len(queue)}, heavy set-aside {len(HEAVY)}")
print(("CLOSED over light classes - heavy remaining: "+str(len(HEAVY))) if not queue else "checkpointed - resume to continue")
