# V43b - RECURSIVE ADJACENCY DECOMPOSITION for the six heavy classes.
# class_walk(pts_idx, Gsub, level): complete facet-class reps of conv(Y[pts_idx]) modulo
# Gsub, each as (global tight mask, local integer inequality). Direct lrs when it
# finishes under budget; otherwise seed from a capped harvest and close by pivot-walk,
# recursing for ridge enumeration. All certifications exact-integer.
import numpy as np, pickle, subprocess, time, os
from fractions import Fraction
from math import gcd
Y=np.load('/tmp/v42_proj.npz')['Y'].astype(np.int64)
SEED=pickle.load(open('/tmp/v43_seed.pkl','rb'))
G=[np.array(g) for g in SEED['G']]
def int_nullvec(M):
    n,m=len(M),len(M[0])
    A=[[Fraction(x) for x in row] for row in M]
    piv=[]; r=0
    for c in range(m):
        pr=None
        for i in range(r,n):
            if A[i][c]!=0: pr=i; break
        if pr is None: continue
        A[r],A[pr]=A[pr],A[r]; pv=A[r][c]; A[r]=[x/pv for x in A[r]]
        for i in range(n):
            if i!=r and A[i][c]!=0:
                f=A[i][c]; A[i]=[A[i][k]-f*A[r][k] for k in range(m)]
        piv.append(c); r+=1
    free=[c for c in range(m) if c not in piv]
    assert len(free)==1,f"nulldim {len(free)}"
    fc=free[0]; v=[Fraction(0)]*m; v[fc]=Fraction(1)
    for rr,c in enumerate(piv): v[c]=-A[rr][fc]
    den=1
    for x in v: den=den*x.denominator//gcd(den,x.denominator)
    w=[int(x*den) for x in v]; g=0
    for x in w: g=gcd(g,abs(x))
    return [x//g for x in w]
def pivots_of(pts):
    D=pts-pts[0]; P=[]; cur=np.zeros((pts.shape[0],0))
    for c in range(pts.shape[1]):
        t=np.concatenate([cur,D[:,c:c+1].astype(float)],axis=1)
        if np.linalg.matrix_rank(t)>cur.shape[1]: P.append(c); cur=t
    return P
def run_lrs(Pl,cap=None,tsec=240):
    with open('/tmp/hw.ext','w') as f:
        f.write(f"hw\nV-representation\nbegin\n{Pl.shape[0]} {Pl.shape[1]+1} integer\n")
        for r in Pl: f.write("1 "+" ".join(str(int(x)) for x in r)+"\n")
        f.write("end\n")
        if cap: f.write(f"maxoutput {cap}\n")
    subprocess.run(['timeout','--signal=KILL',str(tsec),'lrs','/tmp/hw.ext','/tmp/hw.ine'],capture_output=True)
    txt=open('/tmp/hw.ine').read()
    rows=[]
    d=Pl.shape[1]
    for line in txt.splitlines():
        t=line.split()
        if t and all(x.lstrip('-').isdigit() for x in t) and len(t)==d+1:
            rows.append([int(x) for x in t])
    complete=('Totals' in txt)
    return rows,complete
def stab_of(mask_bytes,mask_bits,Gsub):
    out=[]
    for g in Gsub:
        nb=np.zeros(64,dtype=bool); nb[g]=mask_bits
        if np.packbits(nb).tobytes()==mask_bytes: out.append(g)
    return out
def canon(mask_bits,Gsub):
    best=None
    for g in Gsub:
        nb=np.zeros(64,dtype=bool); nb[g]=mask_bits
        k=np.packbits(nb).tobytes()
        if best is None or k<best: best=k
    return best
def class_walk(idx,Gsub,level,budget):
    t0=time.time()
    pts=Y[idx]; P=pivots_of(pts); Pl=pts[:,P]; k=len(P)
    rows,complete=run_lrs(Pl,cap=None if level==0 else 3000,tsec=min(240,budget))
    def mask_of_local(row):
        rv=row[0]+Pl@np.array(row[1:],dtype=np.int64)
        mb=np.zeros(64,dtype=bool); mb[idx[rv==0]]=True
        return mb,rv==0
    if complete:
        seen={}; out=[]
        for row in rows:
            mb,loc=mask_of_local(row)
            ck=canon(mb,Gsub)
            if ck not in seen:
                seen[ck]=1; out.append((mb,np.array(row,dtype=np.int64),loc))
        return out,True
    # incomplete: pivot-walk with seeds
    print("  "*level+f"[L{level}] lrs incomplete ({len(rows)} seed rows) -> pivot walk, dim {k}, pts {len(idx)}",flush=True)
    classes={}; queue=[]
    for row in rows:
        mb,loc=mask_of_local(row)
        ck=canon(mb,Gsub)
        if ck not in classes:
            classes[ck]=(mb,np.array(row,dtype=np.int64),loc); queue.append(ck)
    done=set()
    while queue:
        if time.time()-t0>budget:
            raise TimeoutError(f"L{level} budget")
        ck=queue.pop(0)
        if ck in done: continue
        mb,rowv,loc=classes[ck]
        b1,a1=int(rowv[0]),rowv[1:]
        T1=Pl[loc]; T1g=idx[loc]
        st=stab_of(np.packbits(mb).tobytes(),mb,Gsub)
        sub,_=class_walk(T1g,st,level+1,budget-(time.time()-t0))
        for (rmb,rrow,rloc) in sub:
            # neighbor pivot within current hull (local coords)
            Rl=T1[rloc]; r0=Rl[0]
            U_,s_,Vt=np.linalg.svd((Rl-r0).astype(float))
            null=Vt[k-2:k]
            an=a1.astype(float); an/=np.linalg.norm(an)
            u=null[0]-(null[0]@an)*an
            if np.linalg.norm(u)<1e-8: u=null[1]-(null[1]@an)*an
            u/=np.linalg.norm(u)
            dT=(T1[~rloc]-r0)@u
            assert np.all(dT>1e-9) or np.all(dT<-1e-9)
            if dT[0]<0: u=-u
            sv=(b1+Pl@a1).astype(float); dv=(Pl-r0)@u
            out_=sv>1e-9
            rho=dv[out_]/sv[out_]
            alpha=-float(np.min(rho)); jl=np.where(out_)[0][int(np.argmin(rho))]
            Rg=Y[T1g[rloc]]
            w=int_nullvec([[1]+list(map(int,y)) for y in np.concatenate([Rl[ :],[Pl[jl]]],axis=0)])
            b2,a2=w[0],np.array(w[1:],dtype=np.int64)
            vals=b2+Pl@a2
            if vals.min()<0: b2,a2=-b2,-a2; vals=-vals
            assert vals.min()==0
            mb2=np.zeros(64,dtype=bool); mb2[idx[vals==0]]=True
            ck2=canon(mb2,Gsub)
            if ck2 not in classes:
                classes[ck2]=(mb2,np.array([b2]+list(a2),dtype=np.int64),vals==0); queue.append(ck2)
        done.add(ck)
    return [classes[c] for c in classes],True
if __name__=='__main__':
    S=pickle.load(open('/tmp/v43_state.pkl','rb'))
    classes=S['classes']; done=set(S['done'])
    rem=[ck for ck in S['queue'] if ck not in done]
    print(f"heavies to finish: {len(rem)}; known classes {len(classes)}",flush=True)
    def canon_top(mb): return canon(mb,G)
    t00=time.time(); new=0
    for hk in rem:
        b,a=classes[hk][0],np.array(classes[hk][1],dtype=np.int64)
        m=(b+Y@a)==0; T=Y[m]; idx=np.where(m)[0]
        st=stab_of(np.packbits(m).tobytes(),m,G)
        print(f"[{time.time()-t00:.0f}s] HEAVY tight={int(m.sum())} |stab|={len(st)}",flush=True)
        ridge_reps,_=class_walk(idx,st,1,1000)
        print(f"   ridge classes: {len(ridge_reps)}",flush=True)
        k=33
        for (rmb,rrow,rloc) in ridge_reps:
            Rg=Y[np.where(rmb)[0]]
            r0=Rg[0]
            U_,s_,Vt=np.linalg.svd((Rg-r0).astype(float))
            null=Vt[31:33]
            an=a.astype(float); an/=np.linalg.norm(an)
            u=null[0]-(null[0]@an)*an
            if np.linalg.norm(u)<1e-8: u=null[1]-(null[1]@an)*an
            u/=np.linalg.norm(u)
            Trest=Y[m&~rmb]
            dT=(Trest-r0)@u
            assert np.all(dT>1e-9) or np.all(dT<-1e-9)
            if dT[0]<0: u=-u
            sv=(b+Y@a).astype(float); dv=(Y-r0)@u
            out_=sv>1e-9; rho=dv[out_]/sv[out_]
            j=np.where(out_)[0][int(np.argmin(rho))]
            w=int_nullvec([[1]+list(map(int,y)) for y in np.concatenate([Rg,[Y[j]]],axis=0)])
            b2,a2=w[0],np.array(w[1:],dtype=np.int64)
            vals=b2+Y@a2
            if vals.min()<0: b2,a2=-b2,-a2; vals=-vals
            assert vals.min()==0
            T2=Y[vals==0]
            assert np.linalg.matrix_rank((T2-T2[0]).astype(float))==32
            mb2=vals==0
            ck2=canon_top(mb2)
            if ck2 not in classes:
                classes[ck2]=(int(b2),a2.tolist()); new+=1
                print(f"   !!! NEW top-level class discovered from heavy ridge",flush=True)
        done.add(hk)
        S={'classes':classes,'queue':[c for c in rem if c not in done],'done':list(done),'heavy':[]}
        pickle.dump(S,open('/tmp/v43_state.pkl','wb'))
    print(f"HEAVY PHASE DONE: total classes {len(classes)}, new from heavies: {new}",flush=True)
    print("RIDGE WALK CLOSED - COMPLETENESS ACHIEVED" if not [c for c in rem if c not in done] else "incomplete")
