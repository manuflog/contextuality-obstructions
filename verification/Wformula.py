import numpy as np, json
from itertools import product
from kernelpk import kernel_modpk
from spectrum_test2 import make_alg2
from projection import fiber_all, rand_ctx
def sympZ(u,v): return u[1]*v[0]-u[0]*v[1]+u[3]*v[2]-u[2]*v[3]
rng=np.random.default_rng(53)

def test_formula(base, dd, label):
    d2=2*dd
    fibC=[]; seen=set()
    for C in base:
        for Cc in fiber_all(C,dd):
            if Cc not in seen: seen.add(Cc); fibC.append(Cc)
    obs=sorted({v for Cc in fibC for v in Cc}); oi={v:k for k,v in enumerate(obs)}
    A=np.zeros((len(fibC),len(obs)),dtype=np.int64)
    q2,_,_,ph2=make_alg2(d2,2)
    s2=np.zeros(len(fibC),dtype=np.int64)
    red=[]; hib=[]
    for r,Cc in enumerate(fibC):
        for v in Cc: A[r,oi[v]]+=1
        e=ph2(list(Cc)); s2[r]=(e//2)%d2
        us=[tuple(a%dd for a in v) for v in Cc]
        xs=[tuple(a//dd for a in v) for v in Cc]
        red.append(us); hib.append(xs)
    K=[np.asarray(t,dtype=np.int64) for t in kernel_modpk(A,2,{4:2,8:3,16:4}[d2])]
    cands=list(K)[:400]
    for _ in range(150):
        t=K[rng.integers(len(K))].copy()
        for _ in range(int(rng.integers(1,4))): t=(t+K[rng.integers(len(K))])%d2
        cands.append(t)
    bad=0; tested=0; p1=0
    for t in cands:
        lamb=(t%2)
        supp=np.nonzero(lamb)[0]
        S=int((t@s2)%d2); P = S//dd; assert S%dd==0
        from collections import Counter
        groups=Counter()
        for i in supp: groups[tuple(sorted(red[i]))]+=1
        TA=0
        for i in supp:
            us=red[i]
            K_j=sum(sympZ(us[a],us[b])//dd for a in range(len(us)) for b in range(a+1,len(us)))
            TA+=K_j
        TB=0
        for i in supp:
            us=red[i]; xs=hib[i]
            X=[sum(x[c] for x in xs) for c in range(4)]
            sig=[sum(u[c] for u in us) for c in range(4)]
            TB+= (X[1]*sig[0]-X[0]*sig[1]+X[3]*sig[2]-X[2]*sig[3])
            TB-= sum(sympZ(xs[a],us[a]) for a in range(len(us)))
            pref=[0,0,0,0]
            for a in range(len(us)):
                TB-= 2*sympZ(xs[a],tuple(pref))
                for c in range(4): pref[c]+=us[a][c]
        bracket=TA+TB
        assert bracket%2==0,(label,"bracket odd")
        F = (bracket//2) % 2
        TQ=0
        for i in supp:
            xs=hib[i]
            TQ+= sum(sympZ(xs[a],xs[b]) for a in range(len(xs)) for b in range(a+1,len(xs)))
        F = (F + (dd//2)*TQ) % 2
        if F!=P: bad+=1
        p1+=P; tested+=1
    print(f"{label}: derived-W formula violations {bad}/{tested} (P=1 cases {p1})")

XI=(1,0,0,0); IX=(0,0,1,0); XX=(1,0,1,0)
IY=(0,0,1,1); YI=(1,1,0,0); YY=(1,1,1,1)
XY=(1,0,1,1); YX=(1,1,1,0); ZZ=(0,1,0,1)
PM=[[XI,IX,XX],[IY,YI,YY],[XY,YX,ZZ],[XI,IY,XY],[IX,YI,YX],[XX,YY,ZZ]]
test_formula(PM,2,"PM 2->4")
c4=json.load(open("cert4_min.json"))
test_formula([[tuple(v) for v in it["ctx"]] for it in c4["items"]],4,"cert4 4->8")
test_formula([rand_ctx(2) for _ in range(6)],2,"random d=2")
test_formula([rand_ctx(4) for _ in range(6)],4,"random d=4")
