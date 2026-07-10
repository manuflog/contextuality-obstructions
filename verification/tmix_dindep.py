import numpy as np
from arf_global import symp, fiber_all
lbl={'XI':(1,0,0,0),'IX':(0,0,1,0),'XX':(1,0,1,0),'IY':(0,0,1,1),'YI':(1,1,0,0),
     'YY':(1,1,1,1),'XY':(1,0,1,1),'YX':(1,1,1,0),'ZZ':(0,1,0,1)}
R=[['XI','IX','XX'],['IY','YI','YY'],['XY','YX','ZZ']]
Co=[['XI','IY','XY'],['IX','YI','YX'],['XX','YY','ZZ']]
P={k:np.array(v) for k,v in lbl.items()}
def build(d):
    h=d//2; base=[[tuple((h*P[l])%d) for l in C] for C in R+Co]; pool=[]
    for C in base:
        for lc in fiber_all(C,d): pool.append([tuple(v) for v in lc])
    seen=set(); uniq=[]
    for C in pool:
        k=tuple(sorted(C))
        if k not in seen: seen.add(k); uniq.append(sorted(C))
    pool=uniq
    # canonicalize each lifted context by its (base-pauli-index, lift-bits) signature so ordering is d-independent
    def sig(C):
        s=[]
        for v in C:
            u=tuple(np.array(v)%d); x=tuple(((np.array(v)-(np.array(v)%d))//d)%2)
            up=tuple((np.array(u)//h)%2)  # recover pauli symbol (u = h*p mod d, h=d/2)
            s.append((up,x))
        return tuple(sorted(s))
    rows=[]
    for C in pool:
        sg=sig(C)
        # T value
        us=[np.array(v)%d for v in C]; xs=[((np.array(v)-(np.array(v)%d))//d)%2 for v in C]
        t=0; n=len(C)
        for a in range(n):
            for b in range(a+1,n):
                qu=symp(us[a],us[b])//d; Mab=symp(us[a],xs[b])+symp(xs[a],us[b])
                t^=((qu&1)*(Mab&1))&1
        rows.append((sg,t%2))
    rows.sort()
    T=np.array([t for _,t in rows])%2
    sigs=[sg for sg,_ in rows]
    return sigs,T
# compare across d = 2 mod 4
data={d:build(d) for d in [6,10,14,18,22,26]}
ref_sig,ref_T=data[6]
print("d≡2 mod4: same context signature-set as d=6?  and same T vector?")
for d,(sg,T) in data.items():
    same_ctx = (sg==ref_sig)
    same_T = bool(np.array_equal(T,ref_T)) if same_ctx else "n/a"
    print(f"  d={d:>3}: same_contexts={same_ctx}  same_T={same_T}  (|T|={T.sum()})")
# also test whether T depends only on d mod 8 within 2 mod4
print("\nGroup by d mod 8:", {d:(d%8) for d in data})
