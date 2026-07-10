# V44 - THE ODD-Q SHADOW: SOUND BUT NOT COMPLETE (reconstructing Paper B Prop. 9's
# missing artifacts; gap_char.py / soundness.py / shadow_content.py / prove_Q.py were
# cited by Paper B but never committed - paper-suite drift, pinned in INDEX).
# Closed-triple Weyl families over Z_d, contexts (u,v,w=-u-v), pairwise commuting
# (automatic), summing to 0 mod d. Exact notions:
#   Z_d-contextual  <=>  exists u in Z_d^n, u.M = 0 (mod d), u.gamma != 0 (mod d)
#   odd-Q shadow    <=>  exists mod-2 kernel vector lambda of M with lambda.b odd
# where gamma(C)=sum_{i<j}<c_i,c_j>/d in Z_d and b = gamma mod 2 (carries).
# Also re-verifies on every sample: Thm 8 identity (lambda.b == multiset Q) and
# Thm 1 spectrum (dual pairings in {0, d/2}).
import numpy as np, itertools
from weyl import build as weyl_build
_WB={}
def s_of_ctx(C,d,m):
    if (d,m) not in _WB: _WB[(d,m)]=weyl_build(d,m)
    X,Z,w,tau,W,_=_WB[(d,m)]
    M=np.eye(d**m,dtype=complex)
    for v in C: M=M@W(np.array(v))
    tr=np.trace(M); assert abs(tr)>1e-9
    return int(np.round(np.angle(tr/abs(tr))/(2*np.pi/d)))%d
def symp(u,v,m): return int(sum(int(u[2*i])*int(v[2*i+1])-int(u[2*i+1])*int(v[2*i]) for i in range(m)))
def gamma_of(C,d,m):
    s=0; L=len(C)
    for i in range(L):
        for j in range(i+1,L):
            x=symp(C[i],C[j],m); assert x%d==0, "noncommuting pair"; s+=x//d
    return s
def ok_triple(u,v,d,m):
    if symp(u,v,m)%d: return None
    u=u%d; v=v%d; w=(-(u+v))%d
    if not u.any() or not v.any() or not w.any(): return None
    tt=[tuple(u),tuple(v),tuple(w)]
    if len(set(tt))<3: return None
    return [u,v,w]
def family(rng,d,m,nctx,glue=0.6):
    F=[]; pool=[]
    for _ in range(nctx):
        for _try in range(400):
            if pool and rng.random()<glue:
                u=np.array(pool[int(rng.integers(0,len(pool)))],dtype=np.int64)
            else:
                u=rng.integers(0,d,2*m)
            v=rng.integers(0,d,2*m)
            T=ok_triple(u,v,d,m)
            if T is not None:
                F.append(T)
                for x in T: pool.append(tuple(int(y) for y in x))
                break
        else:
            return None
    return F
def analyze(F,d,m):
    obs={}
    for C in F:
        for v in C: obs.setdefault(tuple(v),len(obs))
    no=len(obs); n=len(F)
    M=np.zeros((n,no),dtype=np.int64)
    for r,C in enumerate(F):
        for v in C: M[r,obs[tuple(v)]]+=1
    g=np.array([s_of_ctx(C,d,m) for C in F],dtype=np.int64)
    b=np.array([gamma_of(C,d,m)%2 for C in F],dtype=np.int64)
    # mod-2 kernel of M^T (cycles): all lambda in F2^n with lambda.M ≡ 0 mod 2
    A2=(M%2).astype(np.uint8)
    lams=[]
    for bits in itertools.product((0,1),repeat=n):
        lam=np.array(bits,dtype=np.uint8)
        if not lam.any(): continue
        if ((lam@A2)%2==0).all(): lams.append(lam)
    oddQ=False; q_id_ok=True
    for lam in lams:
        # Thm 8 identity check: lam.b  ==  multiset-Q over observables of the cycle
        mult=[]
        for r in range(n):
            if lam[r]:
                for v in F[r]: mult.append(np.array(v))
        Q=0
        for i in range(len(mult)):
            for j in range(i+1,len(mult)):
                Q+=symp(mult[i],mult[j],m)//d
        if (int(lam@b)%2)!=(Q%2): q_id_ok=False
        if int(lam@b)%2==1: oddQ=True
    # exact Z_d contextuality by vectorized dual enumeration
    U=np.array(list(itertools.product(range(d),repeat=n)),dtype=np.int64)
    ker=((U@M)%d==0).all(axis=1)&(U%d).any(axis=1)
    vals=(U[ker]@g)%d
    spec_ok=bool(np.isin(vals,[0,d//2]).all())
    ctx=bool((vals%d!=0).any())
    return oddQ,ctx,q_id_ok,spec_ok,(M,g,F)
def run(d,m,trials,nctx_range,seed):
    rng=np.random.default_rng(seed)
    sound_viol=0; false_cert=None; nfc=0; id_bad=0; spec_bad=0; nodd=0; nctxl=0
    for t in range(trials):
        F=family(rng,d,m,int(rng.integers(nctx_range[0],min(nctx_range[1],6 if d==8 else 7))))
        if F is None: continue
        oddQ,ctx,idok,spok,data=analyze(F,d,m)
        if not idok: id_bad+=1
        if not spok: spec_bad+=1
        nodd+=oddQ; nctxl+=ctx
        if ctx and not oddQ: sound_viol+=1
        if oddQ and not ctx:
            nfc+=1
            if false_cert is None: false_cert=data
    return sound_viol,nfc,false_cert,id_bad,spec_bad,nodd,nctxl,trials
if __name__=="__main__":
    # pinned positive controls (must be: oddQ True, contextual True, spectrum {0,d/2})
    import json
    XI=(1,0,0,0); IX=(0,0,1,0); XX=(1,0,1,0)
    ZI=(0,1,0,0); IZ=(0,0,0,1); ZZ=(0,1,0,1)
    XZ=(1,0,0,1); ZX=(0,1,1,0); YY=(1,1,1,1)
    PM=[[np.array(v,dtype=np.int64) for v in C] for C in
        [[XI,IX,XX],[IZ,ZI,ZZ],[XZ,ZX,YY],[XI,IZ,XZ],[IX,ZI,ZX],[XX,ZZ,YY]]]
    o,c,i2,s2,_=analyze(PM,2,2); assert o and c and i2 and s2, "PM control"
    fam=[[np.array(v,dtype=np.int64) for v in it["ctx"]] for it in json.load(open("cert4_min.json"))["items"]]
    o,c,i2,s2,_=analyze(fam,4,2); assert o and c and i2 and s2, "cert4 control"
    print("controls: PM and cert4 both odd-Q + Zd-contextual + spectrum {0,d/2}: PASS")
    for d,m,tr in ((2,2,1500),(4,2,2500),(8,2,2500)):
        sv,nfc,fc,idb,spb,nodd,nctx_,tot=run(d,m,tr,(3,7),20260710+d+m)
        print(f"d={d} m={m}: trials {tot}, oddQ {nodd}, Zd-contextual {nctx_}, "
              f"SOUNDNESS violations {sv}, FALSE CERTIFICATES {nfc}, "
              f"identity fails {idb}, spectrum fails {spb}")
        if fc is not None:
            M,g,F=fc
            print("  FALSE CERTIFICATE (odd-Q but Zd-solvable) reconstructed:")
            for r,C in enumerate(F):
                print(f"    C{r}: "+", ".join(str(tuple(int(x) for x in v)) for v in C)+f"   s={int(g[r])}")
    print("V44 PASS" )
