import numpy as np, json
from itertools import product
from kernelpk import kernel_modpk
from spectrum_test2 import make_alg2
def sympZ(u,v): return u[1]*v[0]-u[0]*v[1]+u[3]*v[2]-u[2]*v[3]
rng=np.random.default_rng(41)

def fiber_all(C, dd):
    letters=sorted(set(C)); nv=len(letters); li={v:k for k,v in enumerate(letters)}
    eqs=[];rhs=[]
    for c in range(4):
        t=sum(v[c] for v in C)
        row=np.zeros(4*nv,dtype=np.int8)
        for v in C: row[4*li[v]+c]^=1
        eqs.append(row);rhs.append((t//dd)%2)
    for a in range(len(C)):
        for b2 in range(a+1,len(C)):
            u,v=C[a],C[b2]; S=sympZ(u,v)
            row=np.zeros(4*nv,dtype=np.int8)
            for i in range(2):
                row[4*li[u]+2*i+1]^=v[2*i]%2; row[4*li[u]+2*i]^=v[2*i+1]%2
                row[4*li[v]+2*i+1]^=u[2*i]%2; row[4*li[v]+2*i]^=u[2*i+1]%2
            eqs.append(row);rhs.append((S//dd)%2)
    M=np.array(eqs);y=np.array(rhs,dtype=np.int8)
    Maug=np.hstack([M,y[:,None]]).copy();r=0;piv=[]
    R,Cn=Maug.shape
    for c in range(Cn-1):
        p=None
        for i in range(r,R):
            if Maug[i,c]:p=i;break
        if p is None:continue
        Maug[[r,p]]=Maug[[p,r]]
        for i in np.nonzero(Maug[:,c])[0]:
            if i!=r:Maug[i]^=Maug[r]
        piv.append(c);r+=1
    if any((Maug[i,:-1]==0).all() and Maug[i,-1] for i in range(R)): return []
    x0=np.zeros(Cn-1,dtype=np.int8)
    for i in range(r-1,-1,-1):
        c=piv[i]; x0[c]=(Maug[i,-1]^(int((Maug[i,c+1:-1]@x0[c+1:])%2)))%2
    free=[c for c in range(Cn-1) if c not in piv]
    basis=[]
    for f in free:
        vec=np.zeros(Cn-1,dtype=np.int8); vec[f]=1
        for i in range(r-1,-1,-1):
            c=piv[i]; vec[c]=int((Maug[i,c+1:-1]@vec[c+1:])%2)
        basis.append(vec)
    out=set()
    for cb in product([0,1],repeat=len(basis)):
        x=x0.copy()
        for bit,vec in zip(cb,basis):
            if bit: x^=vec
        out.add(tuple(sorted(tuple((v[c]+dd*int(x[4*li[v]+c]))%(2*dd) for c in range(4)) for v in C)))
    return sorted(out)

def rand_ctx(dd):
    while True:
        u=tuple(int(a) for a in rng.integers(0,dd,4)); v=tuple(int(a) for a in rng.integers(0,dd,4))
        if sympZ(u,v)%dd: continue
        w=tuple((-(u[k]+v[k]))%dd for k in range(4))
        if any(u) and any(v) and any(w): return [u,v,w]

def test_instance(base, dd, label, ngens=None):
    d2=2*dd
    fibC=[]; fibO=[]; seen={}
    for j,C in enumerate(base):
        for Cc in fiber_all(C,dd):
            if Cc not in seen:
                seen[Cc]=1; fibC.append(Cc); fibO.append(j)
    obs=sorted({v for Cc in fibC for v in Cc}); oi={v:k for k,v in enumerate(obs)}
    A=np.zeros((len(fibC),len(obs)),dtype=np.int64)
    q,_,_,ph=make_alg2(d2,2)
    b2v=np.zeros(len(fibC),dtype=np.int64); s=np.zeros(len(fibC),dtype=np.int64)
    for r,Cc in enumerate(fibC):
        for v in Cc: A[r,oi[v]]+=1
        kk=0
        for a in range(len(Cc)):
            for b2 in range(a+1,len(Cc)): kk+=sympZ(Cc[a],Cc[b2])//d2
        b2v[r]=kk%2
        e=ph(list(Cc)); s[r]=(e//2)%d2
    bd=[]
    for C in base:
        kk=0
        for a in range(len(C)):
            for b2 in range(a+1,len(C)): kk+=sympZ(C[a],C[b2])//dd
        bd.append(kk%2)
    bd=np.array(bd)
    K=kernel_modpk(A,2,{4:2,8:3,16:4}[d2])
    if ngens: K=K[:ngens]
    bad=0; tested=0; nzP=0
    gens=[np.asarray(t,dtype=np.int64) for t in K]
    cands=list(gens)
    for _ in range(min(200,4*len(gens))):
        t=gens[rng.integers(len(gens))].copy()
        for _ in range(int(rng.integers(1,4))):
            t=(t+gens[rng.integers(len(gens))])%d2
        cands.append(t)
    for t in cands:
        lamb=t%2
        P=int((lamb@b2v)%2)
        pi=np.zeros(len(base),dtype=np.int64)
        for idx in np.nonzero(t)[0]:
            pi[fibO[idx]]+= t[idx]
        rhs=int(((pi%2)@bd)%2)
        vbit=(int((t@s)%d2))//dd
        assert vbit==P or int((t@s)%d2)%dd==0, "J2 sanity"
        if P!=rhs: bad+=1
        nzP+=P; tested+=1
    print(f"{label}: fiber {len(fibC)} ctx; kernel gens {len(gens)}; identity violations {bad}/{tested} (P=1 cases: {nzP})")

if __name__=="__main__":
    XI=(1,0,0,0); IX=(0,0,1,0); XX=(1,0,1,0)
    IY=(0,0,1,1); YI=(1,1,0,0); YY=(1,1,1,1)
    XY=(1,0,1,1); YX=(1,1,1,0); ZZ=(0,1,0,1)
    PM=[[XI,IX,XX],[IY,YI,YY],[XY,YX,ZZ],[XI,IY,XY],[IX,YI,YX],[XX,YY,ZZ]]
    test_instance(PM,2,"PM 2->4 (attaining base)")
    c4=json.load(open("cert4_min.json"))
    test_instance([[tuple(v) for v in it["ctx"]] for it in c4["items"]],4,"cert4 4->8 (attaining base)")
    for tr in range(2):
        test_instance([rand_ctx(2) for _ in range(6)],2,f"random d=2 family #{tr}")
    for tr in range(2):
        test_instance([rand_ctx(4) for _ in range(6)],4,f"random d=4 family #{tr}")
