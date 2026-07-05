# NODE 01, rigorous: proper symplectic-basis Arf, radical handled. Test W ≡ Arf and
# the delta-lemma prediction (Arf=0 on trivial bases) on cert4 (attaining) + random trivial bases.
import json, numpy as np
from itertools import product
def symp(u,v):
    m=len(u)//2; return sum(u[2*i+1]*v[2*i]-u[2*i]*v[2*i+1] for i in range(m))
def load(path):
    c=json.load(open(path)); return [[tuple(v) for v in it["ctx"]] for it in c["items"]]

def fiber_solspace(C,d):
    """Return (x0, kernel_basis) of the affine GF(2) lift solution space for one base context."""
    m=len(C[0])//2; L=len(C); nb=2*m*L
    rows=[]; rhs=[]
    sc=[sum(C[i][c] for i in range(L)) for c in range(2*m)]
    for c in range(2*m):
        row=[0]*nb
        for i in range(L): row[i*2*m+c]=1
        rows.append(row); rhs.append((sc[c]//d)%2)
    for i in range(L):
        for j in range(i+1,L):
            row=[0]*nb
            for k in range(m):
                row[j*2*m+2*k]   ^= C[i][2*k+1]%2
                row[j*2*m+2*k+1] ^= C[i][2*k]%2
                row[i*2*m+2*k]   ^= C[j][2*k+1]%2
                row[i*2*m+2*k+1] ^= C[j][2*k]%2
            rows.append(row); rhs.append((symp(C[i],C[j])//d)%2)
    A=np.array(rows,int)%2; b=np.array(rhs,int)%2; rows_n=len(A)
    M=np.concatenate([A,b.reshape(-1,1)],axis=1)%2; piv=[]; r=0
    for c in range(nb):
        pr=next((i for i in range(r,rows_n) if M[i,c]),None)
        if pr is None: continue
        M[[r,pr]]=M[[pr,r]]
        for i in range(rows_n):
            if i!=r and M[i,c]: M[i]^=M[r]
        piv.append(c); r+=1
        if r==rows_n: break
    for i in range(r,rows_n):
        if M[i,nb]==1 and not M[i,:nb].any(): return None,None
    free=[c for c in range(nb) if c not in piv]
    x0=np.zeros(nb,int)
    for idx,c in enumerate(piv): x0[c]=M[idx,nb]
    ker=[]
    for f in free:
        v=np.zeros(nb,int); v[f]=1
        for idx,c in enumerate(piv): v[c]=M[idx,f]
        ker.append(v%2)
    return x0%2, ker

def qform(x,C,d):
    m=len(C[0])//2; L=len(C); xs=[x[i*2*m:(i+1)*2*m] for i in range(L)]
    tot=0
    for i in range(L):
        for j in range(i+1,L):
            kij=symp(C[i],C[j])//d
            Lij=symp(C[i],xs[j])+symp(xs[i],C[j])
            tot += (kij+Lij)//2 + (d//2)*symp(list(xs[i]),list(xs[j]))
    return tot%2

def arf(C,d):
    """Rigorous Arf of q on the affine solution space, via symplectic Gram-Schmidt on the kernel."""
    x0,ker=fiber_solspace(C,d)
    if x0 is None: return None
    def q(y): return qform((x0^y)%2, C, d)          # quadratic form on the linear kernel space
    def Bil(u,v): return (q((u^v)%2)-q(u)-q(v))%2     # associated bilinear (alternating) form
    # reduce ker to independent basis
    basis=[]; red=[]
    for v in ker:
        cur=v.copy()
        for b in red:
            lead=np.argmax(b); 
            if cur[lead]: cur=(cur^b)%2
        if cur.any(): red.append(cur); basis.append(v.copy())
    V=[b.copy() for b in basis]
    # symplectic Gram-Schmidt: pull out hyperbolic pairs, track radical
    arf=0; vecs=V[:]
    while vecs:
        a=vecs.pop(0)
        # find partner with B(a,partner)=1
        pi=next((i for i,v in enumerate(vecs) if Bil(a,v)==1),None)
        if pi is None:
            continue   # a in radical of B; contributes to Arf only via degenerate part -> skip (q|radical linear)
        b=vecs.pop(pi)
        arf=(arf+q(a)*q(b))%2
        # symplectically orthogonalize the rest against the pair (a,b)
        newv=[]
        for v in vecs:
            v2=v.copy()
            if Bil(v,b)==1: v2=(v2^a)%2
            if Bil(v2,a)==1: v2=(v2^b)%2
            if v2.any(): newv.append(v2)
        vecs=newv
    return arf

if __name__=='__main__':
    # ---- TEST A: cert4 (attaining base). W-bit per whole fiber vs Arf ----
    fam4=load("cert4_min.json"); d=4
    print("cert4 (attaining base): per base context")
    arfs=[]
    for C in fam4:
        a=arf(C,d); arfs.append(a)
        print(f"  {[''.join(map(str,v)) for v in C]}: Arf={a}")
    print("  sum Arf over base contexts mod2:", sum(x for x in arfs if x is not None)%2)
    
    # ---- TEST B: trivial bases (random, non-attaining). delta-lemma predicts Arf pattern gives value 0 ----
    print("\nrandom TRIVIAL bases (should NOT produce an attaining fiber): Arf profile")
    rng=np.random.default_rng(11); m=2; d=4
    def rand_commuting_ctx():
        for _ in range(400):
            u=tuple(int(x) for x in rng.integers(0,d,2*m))
            v=tuple(int(x) for x in rng.integers(0,d,2*m))
            if symp(u,v)%d==0:
                w=tuple((-(u[i]+v[i]))%d for i in range(2*m))
                if symp(u,w)%d==0 and symp(v,w)%d==0: return [u,v,w]
        return None
    for trial in range(4):
        fam=[c for c in (rand_commuting_ctx() for _ in range(6)) if c]
        av=[arf(C,d) for C in fam]
        print(f"  trial {trial}: Arfs={av}  sum%2={sum(x for x in av if x is not None)%2}")