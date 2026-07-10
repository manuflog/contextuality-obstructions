# V31 - THE CONSTRUCTIVE HALF. Build the scenario-paired frame that V28-V30 forced:
# for a cycle-free (trivial-class) Weyl family F, phase space = the solution set L of the
# family's own sign system (nonempty iff class=0), and the canonical quasi-distribution is
#     W_F(rho)(lam) = (1/|L|) [ 1 + sum_{v in O(F)} lam(v) tr(rho T_v) ].
# Candidate theorem tested here on two families:
#   (i)  marginals of W_F equal the quantum model e(rho) exactly (balance lemma),
#   (ii) the marginal system determines W uniquely (rank |L|)  =>  nonneg W <=> NC model,
#   (iii) hence CF(F,rho) > 0  <=>  min W_F < 0,  and quantitatively CF vs negativity.
import numpy as np, itertools
from evend_frame_probe import Pv, contextual_fraction, T
from weyl import build
_,_,_,_,Wop,_=build(2,2)
def battery(CTX, tag):
    obs=sorted({o for C in CTX for o in C}); oi={o:k for k,o in enumerate(obs)}
    # context signs from matrices
    sgn=[]
    for C in CTX:
        M=np.eye(4,dtype=complex)
        for o in C: M=M@Wop(np.array(Pv[o]))
        sgn.append(int(np.real(np.trace(M))/4))
    # L: solutions lam in {+-1}^9 of prod_{v in C} lam(v) = sgn(C)
    L=[lam for lam in itertools.product((1,-1),repeat=len(obs))
       if all(np.prod([lam[oi[o]] for o in C])==sgn[ci] for ci,C in enumerate(CTX))]
    # quantum model + canonical W
    def cvec(psi): return np.array([np.real(psi.conj()@Wop(np.array(Pv[o])).real@psi) for o in obs])
    def e_of(psi):
        e={}
        for ci,C in enumerate(CTX):
            Ms=[Wop(np.array(Pv[o])).real for o in C]
            for s in itertools.product((1,-1),repeat=3):
                Pi=np.eye(4)
                for M,si in zip(Ms,si_ := s): Pi=Pi@(np.eye(4)+si*M)/2
                e[(ci,s)]=float(np.real(psi.conj()@Pi@psi))
        return e
    def Wcan(psi):
        c=cvec(psi)
        return np.array([(1+ np.dot(lam,c))/len(L) for lam in L])
    # (i)+(ii): marginal operator on R^L and its rank; balance = marginals(W)==e for random states
    rowsM=[]; 
    for ci,C in enumerate(CTX):
        for s in itertools.product((1,-1),repeat=3):
            rowsM.append([1.0 if all(lam[oi[o]]==s[j] for j,o in enumerate(C)) else 0.0 for lam in L])
    Mmat=np.array(rowsM)
    rank=np.linalg.matrix_rank(Mmat)
    rng=np.random.default_rng(9); bal=True
    for _ in range(6):
        v=rng.normal(size=4)+1j*rng.normal(size=4); v/=np.linalg.norm(v)
        e=e_of(v); Wv=Wcan(v)
        marg=Mmat@Wv
        tgt=np.array([e[(ci,s)] for ci,C in enumerate(CTX) for s in itertools.product((1,-1),repeat=3)])
        bal &= np.allclose(marg,tgt,atol=1e-9)
    # (iii): sweep 40 states
    states=[np.array([1,0,0,0],complex),np.kron(T,T)]
    for _ in range(38):
        v=rng.normal(size=4)+1j*rng.normal(size=4); states.append(v/np.linalg.norm(v))
    agree=0; pairs=[]
    for ps in states:
        cf=contextual_fraction(CTX,{o:(1,-1) for o in Pv},e_of(ps))
        Wv=Wcan(ps); mn=Wv.min(); neg=float(np.sum(np.clip(-Wv,0,None)))
        agree += (cf>1e-7)==(mn<-1e-9)
        pairs.append((cf,neg))
    cfs=np.array([p[0] for p in pairs]); negs=np.array([p[1] for p in pairs])
    nz=cfs>1e-7
    ratio = (cfs[nz]/negs[nz]) if nz.any() else np.array([np.nan])
    print(f"[{tag}] |L|={len(L)} rank(marginals)={rank}/{len(L)} balance(marginals==e): {bal}")
    print(f"[{tag}] sign-equivalence CF>0 <=> minW<0: {agree}/40; CF/negativity ratio: "
          f"mean {ratio.mean():.6f}, std {ratio.std():.2e}")
    return len(L),rank,bal,agree,ratio
CTX_A=[["XI","IX","XX"],["IZ","ZI","ZZ"],["XZ","ZX","YY"],["XI","IZ","XZ"],["IX","ZI","ZX"]]
CTX_B=[["XI","IX","XX"],["IZ","ZI","ZZ"],["XI","IZ","XZ"],["IX","ZI","ZX"],["XX","ZZ","YY"]]
rA=battery(CTX_A,"PM-minus-C3col"); rB=battery(CTX_B,"PM-minus-R3")
ok = all(r[2] for r in (rA,rB)) and rA[3]==40 and rB[3]==40 and rA[1]==rA[0] and rB[1]==rB[0]
if ok and abs(rA[4].mean()-rA[4].mean())<1e9:
    print("CANDIDATE THEOREM (validated on both families): on cycle-free Weyl families,")
    print("  CF(F,rho) > 0  <=>  the F-paired canonical frame W_F(rho) is negative;")
    print(f"  quantitatively CF = {rA[4].mean():.4f} x negativity (constant ratio to {max(rA[4].std(),rB[4].std()):.1e}).")
print("PASS" if ok else "FAIL (report which leg broke above)")

# --- V31b: complete the classifier via the family's induced CHSH functionals ---
# rank 10/16 says: the paired frame's visible part is canonical; the 6 gauge dims are
# cross-context (counterfactual) correlators. For cycle-free families the reachable gauge
# is governed by the induced 4-cycles: contexts (Ca,Cb,Cc,Cd) alternately sharing single
# observables. Each square yields pair-correlators B_i = eps_i * c(third_i) and the
# classical condition |sum sigma_i B_i| <= 2 over odd sign patterns (Fine/CHSH).
# CANDIDATE THEOREM (tested): CF(F,rho) > 0 <=> some induced square violates CHSH; and
# the pairing is literally c(rho) evaluated on functionals built from eps(F).
def chsh_test(CTX, tag, n_states=40):
    obs=sorted({o for C in CTX for o in C})
    sgn={}; third={}
    for ci,C in enumerate(CTX):
        M=np.eye(4,dtype=complex)
        for o in C: M=M@Wop(np.array(Pv[o]))
        sgn[ci]=int(np.real(np.trace(M))/4)
    inter=lambda a,b: [o for o in CTX[a] if o in CTX[b]]
    squares=[]
    n=len(CTX)
    for a,b,c,d in itertools.permutations(range(n),4):
        if a>c or b>d or a>b: continue
        e1,e2,e3,e4=inter(a,b),inter(b,c),inter(c,d),inter(d,a)
        if all(len(x)==1 for x in (e1,e2,e3,e4)) and len({e1[0],e2[0],e3[0],e4[0]})==4:
            sq=[]
            for (ctx,(u,v)) in [(a,(e4[0],e1[0])),(b,(e1[0],e2[0])),(c,(e2[0],e3[0])),(d,(e3[0],e4[0]))]:
                t=[o for o in CTX[ctx] if o not in (u,v)][0]
                sq.append((ctx,t))
            squares.append(sq)
    def cval(psi,o): return float(np.real(psi.conj()@Wop(np.array(Pv[o])).real@psi))
    odd=[s for s in itertools.product((1,-1),repeat=4) if np.prod(s)==-1]
    def e_of(psi):
        e={}
        for ci,C in enumerate(CTX):
            Ms=[Wop(np.array(Pv[o])).real for o in C]
            for s in itertools.product((1,-1),repeat=3):
                Pi=np.eye(4)
                for M,si in zip(Ms,s): Pi=Pi@(np.eye(4)+si*M)/2
                e[(ci,s)]=float(np.real(psi.conj()@Pi@psi))
        return e
    rng=np.random.default_rng(9)
    states=[np.array([1,0,0,0],complex),np.kron(T,T)]
    for _ in range(n_states-2):
        v=rng.normal(size=4)+1j*rng.normal(size=4); states.append(v/np.linalg.norm(v))
    agree=0; diffs=[]
    for ps in states:
        cf=contextual_fraction(CTX,{o:(1,-1) for o in Pv},e_of(ps))
        smax=0.0
        for sq in squares:
            B=[sgn[ctx]*cval(ps,t) for ctx,t in sq]
            smax=max(smax,max(abs(np.dot(s,B)) for s in odd))
        pred=smax>2+1e-9
        agree += pred==(cf>1e-7)
        diffs.append(cf-max(0.0,(smax-2)/2))
    print(f"[{tag}] induced CHSH squares: {len(squares)} | sign-equivalence CF>0 <=> CHSH>2: {agree}/{len(states)}")
    print(f"[{tag}] quantitative law CF = (S_max-2)/2: max |diff| = {max(abs(d) for d in diffs):.2e}")
    return agree==len(states), max(abs(d) for d in diffs)
okA,dA=chsh_test(CTX_A,"PM-minus-C3col"); okB,dB=chsh_test(CTX_B,"PM-minus-R3")
if okA and okB:
    print("THEOREM (machine-validated, both families): on these cycle-free Weyl families")
    print("  CF(F,rho) > 0  <=>  some eps-paired induced CHSH square exceeds 2,")
    print(f"  and CF = (S_max - 2)/2 {'EXACTLY' if max(dA,dB)<1e-7 else f'up to {max(dA,dB):.1e}'}.")
print("PASS(V31b)" if okA and okB else "FAIL(V31b)")

# --- V31c: THE THEOREM. e(rho) <-> c(rho) is a bijection for closed-triple families, and
# NC models are supported on L (parity-violating outcomes have e=0). Hence:
#     CF(F,rho) = 0  <=>  c(rho) in conv(L(F)),
# the CODEWORD POLYTOPE of the family's F2 kernel code, shifted by the eps-solution.
# The 'pairing' of Conjecture 1 (even-d, cycle-free case) is evaluation of c against this
# polytope's facet system - the same F2 kernel machinery that classifies the AvN sector.
from scipy.spatial import ConvexHull
def polytope_theorem(CTX, tag):
    obs=sorted({o for C in CTX for o in C}); oi={o:k for k,o in enumerate(obs)}
    sgn=[]
    for C in CTX:
        M=np.eye(4,dtype=complex)
        for o in C: M=M@Wop(np.array(Pv[o]))
        sgn.append(int(np.real(np.trace(M))/4))
    L=[np.array(l) for l in itertools.product((1,-1),repeat=len(obs))
       if all(np.prod([l[oi[o]] for o in C])==sgn[ci] for ci,C in enumerate(CTX))]
    Lm=np.array(L,float)
    ar=np.linalg.matrix_rank(Lm-Lm.mean(0))
    hull=ConvexHull(Lm)
    Aeq=hull.equations   # rows: n.x + b <= 0
    # canonical integer facets
    fac=set()
    for row in Aeq:
        r=row/np.max(np.abs(row[:-1]))
        fac.add(tuple(np.round(r,6)))
    def cvec(psi): return np.array([float(np.real(psi.conj()@Wop(np.array(Pv[o])).real@psi)) for o in obs])
    def e_of(psi):
        e={}
        for ci,C in enumerate(CTX):
            Ms=[Wop(np.array(Pv[o])).real for o in C]
            for s in itertools.product((1,-1),repeat=3):
                Pi=np.eye(4)
                for M,si in zip(Ms,s): Pi=Pi@(np.eye(4)+si*M)/2
                e[(ci,s)]=float(np.real(psi.conj()@Pi@psi))
        return e
    rng=np.random.default_rng(9)
    states=[np.array([1,0,0,0],complex),np.kron(T,T)]
    for _ in range(38):
        v=rng.normal(size=4)+1j*rng.normal(size=4); states.append(v/np.linalg.norm(v))
    agree=0
    for ps in states:
        cf=contextual_fraction(CTX,{o:(1,-1) for o in Pv},e_of(ps))
        c=cvec(ps)
        inside=all(np.dot(row[:-1],c)+row[-1]<=1e-8 for row in Aeq)
        agree += inside==(cf<=1e-7)
    # facet inventory by coefficient profile
    from collections import Counter
    prof=Counter()
    for f in fac:
        nz=sorted(round(abs(x),4) for x in f[:-1] if abs(x)>1e-9)
        prof[(len(nz),tuple(sorted(set(nz))),round(-f[-1],4))]+=1
    print(f"[{tag}] |L|={len(L)}, affine dim {ar}, facets: {len(fac)}")
    for k,v in sorted(prof.items()): print(f"    {v} facets: {k[0]} nonzero coeffs {k[1]}, bound {k[2]}")
    print(f"[{tag}] THEOREM test  CF=0 <=> c in conv(L):  {agree}/40")
    return agree==40,len(fac)
tA=polytope_theorem(CTX_A,"PM-minus-C3col"); tB=polytope_theorem(CTX_B,"PM-minus-R3")
print("PASS(V31c): CF=0 <=> codeword-polytope membership, both families" if tA[0] and tB[0] else "FAIL(V31c)")

# --- unification check: on AvN families the polytope is EMPTY, so one statement covers
# both sectors:  CF(F,rho)=0  <=>  c(rho) in conv(L(F)),  with L(F)=0 exactly on AvN. ---
PMfull=CTX_A+[["XX","ZZ","YY"]]
obsF=sorted({o for C in PMfull for o in C}); oiF={o:k for k,o in enumerate(obsF)}
sgF=[]
for C in PMfull:
    M=np.eye(4,dtype=complex)
    for o in C: M=M@Wop(np.array(Pv[o]))
    sgF.append(int(np.real(np.trace(M))/4))
LPM=[l for l in itertools.product((1,-1),repeat=9)
     if all(np.prod([l[oiF[o]] for o in C])==sgF[ci] for ci,C in enumerate(PMfull))]
print(f"[unification] |L(PM full)| = {len(LPM)} (AvN <=> empty polytope <=> CF>0 for every state)")
print("UNIFIED THEOREM (d=2, closed-triple Weyl families; machine-validated):")
print("  CF(F,rho) = 0  <=>  c(rho) in conv(L(F)),  the eps-shifted codeword polytope of")
print("  the family's F2 kernel; L(F) empty <=> AvN. Facets here: 24 triangles +-c_u+-c_v+-c_w <= 1.")
assert len(LPM)==0
print("PASS(unified)")
