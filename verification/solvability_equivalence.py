# V44 - THE DETECTION EQUIVALENCE, ASSEMBLED AND CERTIFIED.
# Theorem (equivalence; note v3 Thm 3'): for integral closed-triple Weyl families at
# even d (all context phases omega-powers), TFAE:
#   (a) no noncontextual value assignment over Z_d exists (A.gamma = s mod d unsolvable);
#   (b) some left-kernel vector u (uA = 0 mod d) has u.s = d/2 (mod d);
#   (c) some certificate carries odd commutator carry (Paper B Thm 4).
# Proof assembly: (b)=>(a) trivial pairing; (a)=>(b): constructive double-annihilator
# over Z_d via Smith normal form (u = (d/g_i) U_i for the failing invariant factor),
# then u.s in {0, d/2} by Paper B Thm 1 (Obstruction Spectrum: 2S = 0 mod d for EVERY
# certificate, any incidence structure/multiplicity - Remark 3), so u.s = d/2;
# (b)<=>(c) is Paper B Thm 4. The previously "machine-verified converse" is thus a
# corollary of proven theorems; this script certifies the assembly empirically:
# spectrum confinement, the double-annihilator equivalence, and witness existence,
# on random integral families at d=2,4 plus the pinned PM and cert4 positives.
import numpy as np, itertools, json
from weyl import build
def kernel_left(A,d,cap=8):
    nC=A.shape[0]
    if nC>cap: return None
    K=[]
    for u in itertools.product(range(d),repeat=nC):
        u=np.array(u)
        if ((u@A)%d==0).all(): K.append(u)
    return K
def solvable_bf(A,s,d):
    nobs=A.shape[1]
    G=np.array(list(itertools.product(range(d),repeat=nobs)),dtype=np.int64)
    return bool((((G@A.T)%d==s%d).all(axis=1)).any())
def family_from_labels(labels,d,m,W,rng,max_ctx=8,must=None):
    def sym(a,b): return int(a[:m]@b[m:]-a[m:]@b[:m])%d
    idx={tuple(v):i for i,v in enumerate(labels)}
    ctxs=[]
    for i,j in itertools.combinations(range(len(labels)),2):
        v3=tuple((-(labels[i]+labels[j]))%d)
        if v3 in idx:
            k=idx[v3]
            if k in (i,j): continue
            trip=[labels[i],labels[j],labels[k]]
            if any(sym(a,b)!=0 for a,b in itertools.combinations(trip,2)): continue
            key=tuple(sorted((i,j,k)))
            if key not in [tuple(sorted(x)) for x in ctxs]: ctxs.append((i,j,k))
    if len(ctxs)>max_ctx:
        keep=[c for c in ctxs if must and tuple(sorted(c)) in must]
        rest=[c for c in ctxs if not (must and tuple(sorted(c)) in must)]
        sel=rng.choice(len(rest),size=max(0,max_ctx-len(keep)),replace=False)
        ctxs=keep+[rest[int(i)] for i in sel]
    return ctxs
def check(labels,d,m,W,rng,stats,must=None):
    ctxs=family_from_labels(labels,d,m,W,rng,must=must)
    if len(ctxs)<3: return
    A=np.zeros((len(ctxs),len(labels)),int); st=[]
    for r,(i,j,k) in enumerate(ctxs):
        A[r,i]+=1; A[r,j]+=1; A[r,k]+=1
        M=W(labels[i])@W(labels[j])@W(labels[k])
        tr=np.trace(M); ph=np.angle(tr/abs(tr)); sc=ph/(np.pi/d)
        assert abs(sc-round(sc))<1e-6, "context phase off the tau grid"
        st.append(int(round(sc))%(2*d))
    st=np.array(st)
    # GAUGE-INTEGRALITY: solve A a = st (mod 2) over GF(2); always solvable (Paper B
    # Thm 1 consequence) - asserted; kernel values are gauge-invariant.
    A2=(A%2).astype(np.uint8); b2=(st%2).astype(np.uint8)
    M2=np.concatenate([A2,b2[:,None]],axis=1).astype(np.uint8)
    rr=0
    for c in range(A2.shape[1]):
        pr=None
        for i2 in range(rr,M2.shape[0]):
            if M2[i2,c]: pr=i2; break
        if pr is None: continue
        M2[[rr,pr]]=M2[[pr,rr]]
        for i2 in range(M2.shape[0]):
            if i2!=rr and M2[i2,c]: M2[i2]^=M2[rr]
        rr+=1
    assert not any(M2[i2,:-1].sum()==0 and M2[i2,-1] for i2 in range(M2.shape[0])), "GAUGE LEMMA VIOLATED"
    # extract one solution a
    a=np.zeros(A2.shape[1],dtype=int)
    M2b=np.concatenate([A2,b2[:,None]],axis=1).astype(np.uint8)
    piv=[]; rr=0
    for c in range(A2.shape[1]):
        pr=None
        for i2 in range(rr,M2b.shape[0]):
            if M2b[i2,c]: pr=i2; break
        if pr is None: continue
        M2b[[rr,pr]]=M2b[[pr,rr]]
        for i2 in range(M2b.shape[0]):
            if i2!=rr and M2b[i2,c]: M2b[i2]^=M2b[rr]
        piv.append(c); rr+=1
    for r2,c in enumerate(piv): a[c]=int(M2b[r2,-1])
    assert ((A@a-st)%2==0).all()
    s=((st-A@a)//2)%d
    if A.shape[1]>(14 if d==2 else 9): return
    K=kernel_left(A,d)
    if K is None: return
    solv=solvable_bf(A,s,d)
    vals=set(int((u@s)%d) for u in K)
    stats['checked']+=1
    assert vals.issubset({0,d//2}), f"SPECTRUM VIOLATION {sorted(vals)}"
    assert solv==(vals<={0}), f"DOUBLE-ANNIHILATOR MISMATCH solv={solv} vals={sorted(vals)}"
    if not solv:
        stats['uns']+=1
        assert (d//2) in vals, "unsolvable without d/2 witness"
if __name__=='__main__':
    import json as _json
    fam4=[[tuple(v) for v in it["ctx"]] for it in _json.load(open("cert4_min.json"))["items"]]
    pin4=[np.array(v) for v in sorted({t for C in fam4 for t in C})]
    # PM labels at d=2 (two-qubit Paulis, 3x3 grid): X1,X2,X1X2 / Z2,Z1,Z1Z2 / X1Z2,Z1X2,Y1Y2
    pin2=[np.array(v) for v in [(1,0,0,0),(0,1,0,0),(1,1,0,0),(0,0,0,1),(0,0,1,0),(0,0,1,1),(1,0,0,1),(0,1,1,0),(1,1,1,1)]]
    for d,m,trials,seed,pin in [(2,2,500,11,pin2),(4,2,400,12,pin4)]:
        X,Z,w,tau,W,_=build(d,m)
        rng=np.random.default_rng(seed)
        stats={'checked':0,'uns':0,'nonintegral':0}
        allv=[np.array(v) for v in itertools.product(range(d),repeat=2*m) if any(v)]
        for t in range(trials):
            npin=int(rng.integers(4,10)); nrand=int(rng.integers(0,5))
            psel=rng.choice(len(pin),size=min(npin,len(pin)),replace=False)
            labels=[pin[int(i)] for i in psel]
            rsel=rng.choice(len(allv),size=nrand,replace=False)
            for i in rsel:
                v=allv[int(i)]
                if not any((v==x).all() for x in labels): labels.append(v)
            for _ in range(6):
                i,j=rng.integers(0,len(labels),size=2)
                if int(i)==int(j): continue
                v3=(-(labels[int(i)]+labels[int(j)]))%d
                if v3.any() and not any((v3==x).all() for x in labels) and len(labels)<13: labels.append(v3)
            must=None
            if rng.random()<0.6:
                labels=[x for x in pin]+[x for x in labels if not any((x==p).all() for p in pin)]
                li={tuple(v):i for i,v in enumerate(labels)}
                if d==2:
                    PMC=[[(1,0,0,0),(0,1,0,0),(1,1,0,0)],[(0,0,0,1),(0,0,1,0),(0,0,1,1)],
                         [(1,0,0,1),(0,1,1,0),(1,1,1,1)],[(1,0,0,0),(0,0,0,1),(1,0,0,1)],
                         [(0,1,0,0),(0,0,1,0),(0,1,1,0)],[(1,1,0,0),(0,0,1,1),(1,1,1,1)]]
                    must={tuple(sorted(li[t] for t in C)) for C in PMC}
                else:
                    must={tuple(sorted(li[t] for t in C)) for C in fam4 if all(t in li for t in C)}
            check(labels,d,m,W,rng,stats,must=must)
        print(f"d={d}: integral families checked {stats['checked']} (nonintegral skipped {stats['nonintegral']}), unsolvable {stats['uns']}, spectrum+DA+witness all PASS")
    # pinned positives
    X,Z,w,tau,W,_=build(2,2)
    P={'X':np.array([1,0,0,0]),'Z':np.array([0,0,1,0])}
    lab=lambda a,b:(np.array([a[0],b[0],a[1],b[1]]))
    XX=lab([1,0],[1,0]) # placeholder structure; PM via cert file if available
    fam4=[[tuple(v) for v in it["ctx"]] for it in json.load(open("cert4_min.json"))["items"]]
    X4,Z4,w4,tau4,W4,_=build(4,2)
    labs=sorted({t for C in fam4 for t in C}); oi={t:k for k,t in enumerate(labs)}
    A=np.zeros((6,len(labs)),int); s=[]
    for r,C in enumerate(fam4):
        for v in C: A[r,oi[v]]+=1
        M=np.eye(16,dtype=complex)
        for v in C: M=M@W4(np.array(v))
        ph=np.angle(np.trace(M)/abs(np.trace(M))); s.append(int(round(ph/(np.pi/2)))%4)
    s=np.array(s)
    K=kernel_left(A,4)
    vals=sorted(set(int((u@s)%4) for u in K))
    assert vals==[0,2] and not solvable_bf(A,s,4)
    print(f"cert4 pinned: unsolvable, kernel values {vals} (witness d/2=2 present): PASS")
    print("V44 PASS")
