# PROVE T2 = sum_{a<b in word} symp(x_a,x_b) = 0 mod2 over closed fiber cycles.
# Decompose: within-context + cross-context.
# WITHIN a context C: observables commute at level 2d: symp(v_a,v_b) = 0 mod 2d.
#   symp(v_a,v_b)=symp(u_a,u_b)+d M_ab + d^2 symp(x_a,x_b) = 0 mod 2d.
#   mod 2: but we want symp(x_a,x_b) mod 2. From =0 mod 2d, divide: symp(u_a,u_b)+dM_ab+d^2 sxx ≡0 mod 2d.
#   Hmm, this constrains but doesn't directly give sxx mod2. Let me test the CLEANEST claim instead:
# CLAIM: sum_{a<b in word} symp(x_a,x_b) mod2 = Q(x-cycle) where the x-bits form their OWN Z_2 cycle,
#   and Q of a Z_2 cycle in a symplectic space is a well-defined invariant that = 0 because the
#   x-lift-bits do NOT themselves close as a nontrivial cycle (they're free lift choices).
# Actually the RIGHT structural fact: build the map that sends the fiber cycle to its x-bit pattern;
# test if sum symp(x_a,x_b) mod2 is always 0 by checking it equals symp-self-pairing of a boundary.
import numpy as np, json
from arf_global import symp, build_fiber_pool
def data(fam,dd):
    obs=sorted({tuple(v) for CC in fam for v in CC}); oi={v:k for k,v in enumerate(obs)}
    A=np.zeros((len(fam),len(obs)),int); b=np.zeros(len(fam),int)
    for r,CC in enumerate(fam):
        b[r]=sum(symp(CC[i],CC[j])//dd for i in range(len(CC)) for j in range(i+1,len(CC)))%2
        for v in CC: A[r,oi[tuple(v)]]^=1
    return A%2,b%2
def lk(A):
    rows,ncols=A.shape; aug=np.concatenate([A,np.eye(rows,dtype=int)],axis=1)%2; r=0
    for c in range(ncols):
        pr=next((i for i in range(r,rows) if aug[i,c]),None)
        if pr is None: continue
        aug[[r,pr]]=aug[[pr,r]]
        for i in range(rows):
            if i!=r and aug[i,c]: aug[i]^=aug[r]
        r+=1
        if r==rows: break
    return [aug[i,ncols:].copy() for i in range(rows) if not aug[i,:ncols].any()]
# Test the two structural facts that would PROVE T2=0:
# (P1) WITHIN each context, sum_{a<b} symp(x_a,x_b) = 0 mod 2.
# (P2) CROSS: sum_{C<C'} symp(SX_C, SX_C') = 0 mod 2, where SX_C = sum of x-bits in context C.
#      Since SX_C = rc = -(sum u_C)/d mod 2 (per-context closure), cross = sum_{C<C'} symp(rc,rc').
#      And sum_C rc over the cycle relates to base cycle closure.
for tag,(d,seed) in {'d4':(4,'cert4'),'d6':(6,'PM6')}.items():
    if seed=='cert4':
        cert=json.load(open('cert4_min.json')); base=[[tuple(v) for v in it['ctx']] for it in cert['items']]
    else:
        lbl={'XI':(1,0,0,0),'IX':(0,0,1,0),'XX':(1,0,1,0),'IY':(0,0,1,1),'YI':(1,1,0,0),
             'YY':(1,1,1,1),'XY':(1,0,1,1),'YX':(1,1,1,0),'ZZ':(0,1,0,1)}
        R=[['XI','IX','XX'],['IY','YI','YY'],['XY','YX','ZZ']];Co=[['XI','IY','XY'],['IX','YI','YX'],['XX','YY','ZZ']]
        base=[[tuple((3*np.array(lbl[l]))%6) for l in C] for C in R+Co]
    pool=build_fiber_pool(base,d)
    A,b=data(pool,2*d); cyc=lk(A)
    p1_all=p2_all=True
    for lam in cyc[:60]:
        sel=[i for i in range(len(pool)) if lam[i]%2]
        ctxs=[[np.array(v) for v in pool[i]] for i in sel]
        xs=[[(v-v%d)//d for v in C] for C in ctxs]
        # P1 within
        for xc in xs:
            w=sum(symp(xc[a],xc[b]) for a in range(len(xc)) for b in range(a+1,len(xc)))%2
            if w!=0: p1_all=False
        # P2 cross
        SX=[sum(xc)%2 for xc in xs]
        cross=sum(symp(SX[i],SX[j]) for i in range(len(SX)) for j in range(i+1,len(SX)))%2
        if cross!=0: p2_all=False
    print(f"{tag}: (P1) within-context sum symp(x,x)=0: {p1_all};  (P2) cross-context =0: {p2_all}")
print()
print("If P1 and P2 both hold universally, T2=within+cross=0 is PROVED structurally per fiber cycle.")
print("P1 should follow from within-context commutation; P2 from per-context lift-closure + base cycle.")
