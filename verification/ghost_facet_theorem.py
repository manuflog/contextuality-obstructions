# V32 - GHOST FACET THEOREM + CLOSED FORMULA (the written-proof core).
# For F = PM minus one context C*, the dual code rowspace(A) has EXACTLY six weight-3
# words: the five remaining contexts and the GHOST = sum of all five rows = C* itself.
# Facets of conv L(F): 4 sign patterns per word (Pi sigma = -gamma(word)) = 24 triangles.
#   - the 20 in-context facets are quantum-trivial (they are e_C(s) >= 0);
#   - the 4 GHOST facets carry the whole state sector, with operative parity
#     Pi sigma = eps(C*)  (since gamma_ghost = Pi eps_remaining = -1/eps* by the class).
# CLOSED FORMULA:  CF(F,rho) = max(0, (S* - 1)/2),
#   S* = max over the 4 operative sigma of  sum_i sigma_i c_{v_i}(rho),  v_i in C*.
# Sharp consequence: joint eigenstates of C* with an operative pattern (e.g. Bell states
# for the deleted odd column) attain S*=3 and hence CF=1 - maximal contextuality survives
# the deletion of the context that carried the class.
import numpy as np, itertools
from scipy.spatial import ConvexHull
from state_sector_probe import contextual_fraction
from evend_frame_probe import Pv, T
from weyl import build
_,_,_,_,Wop,_=build(2,2)
PM=[["XI","IX","XX"],["IZ","ZI","ZZ"],["XZ","ZX","YY"],["XI","IZ","XZ"],["IX","ZI","ZX"],["XX","ZZ","YY"]]
def eps_of(C):
    M=np.eye(4,dtype=complex)
    for o in C: M=M@Wop(np.array(Pv[o]))
    return int(np.real(np.trace(M))/4)
def analyze(drop):
    CTX=[C for i,C in enumerate(PM) if i!=drop]; Cs=PM[drop]; es=eps_of(Cs)
    obs=sorted({o for C in CTX for o in C}); oi={o:k for k,o in enumerate(obs)}
    L=[np.array(l) for l in itertools.product((1,-1),repeat=9)
       if all(np.prod([l[oi[o]] for o in C])==eps_of(C) for C in CTX)]
    hull=ConvexHull(np.array(L,float))
    fac=set(tuple(np.round(r/np.max(np.abs(r[:-1])),6)) for r in hull.equations)
    ctx_sets=[frozenset(C) for C in CTX]; ghost=frozenset(Cs)
    n_in=n_gh=0; op_par=set()
    for f0 in fac:
        f=np.array(f0)
        if f[-1]>0: f=-f            # normalize hull row so inequality reads sigma.c <= -b > 0
        sup=frozenset(obs[k] for k in range(9) if abs(f[k])>1e-9)
        if sup in ctx_sets: n_in+=1
        elif sup==ghost:
            n_gh+=1; op_par.add(int(np.prod([np.sign(f[k]) for k in range(9) if abs(f[k])>1e-9])))
        else: n_in=-999
    # TWO-TETRAHEDRA LEMMA: on the deleted triple, L has constant parity -eps* while
    # quantum joint eigenstates have parity eps*: classical and quantum occupy OPPOSITE
    # tetrahedra. Hence eps*-parity functionals are capped at 1 on L (odd-disagreement
    # argument: Pi(sigma)Pi(lambda)= -1 => sum <= 1) and reach 3 quantumly: the 4 facets.
    gidx=[oi[o] for o in Cs]
    Lpar=set(int(np.prod([l[g] for g in gidx])) for l in L)
    ncmax_op=max(max(sum(s*l[g] for s,g in zip(sig,gidx)) for l in L)
                 for sig in itertools.product((1,-1),repeat=3) if np.prod(sig)==es)
    assert Lpar=={-es} and ncmax_op==1
    rng=np.random.default_rng(13)
    states={"Phi+":np.array([1,0,0,1],complex)/np.sqrt(2),"TxT":np.kron(T,T),"|00>":np.array([1,0,0,0],complex)}
    for k in range(57):
        v=rng.normal(size=4)+1j*rng.normal(size=4); states[f"r{k}"]=v/np.linalg.norm(v)
    def e_of(psi):
        e={}
        for ci,C in enumerate(CTX):
            Ms=[Wop(np.array(Pv[o])).real for o in C]
            for s in itertools.product((1,-1),repeat=3):
                Pi=np.eye(4)
                for M,si in zip(Ms,s): Pi=Pi@(np.eye(4)+si*M)/2
                e[(ci,s)]=float(np.real(psi.conj()@Pi@psi))
        return e
    opsig=[s for s in itertools.product((1,-1),repeat=3) if np.prod(s)==es]
    md=0.0; phi_cf=None
    for nm,ps in states.items():
        cf=contextual_fraction(CTX,{o:(1,-1) for o in Pv},e_of(ps))
        cv=[float(np.real(ps.conj()@Wop(np.array(Pv[o])).real@ps)) for o in Cs]
        S=max(np.dot(s,cv) for s in opsig)
        md=max(md,abs(cf-max(0.0,(S-1)/2)))
        if nm=="Phi+": phi_cf=cf
    print(f"[drop {Cs} eps*={es:+d}] facets: {len(fac)} = {n_in} in-context + {n_gh} ghost; "
          f"ghost operative parity {sorted(op_par)} (expect [{es}])")
    print(f"[drop {Cs}] FORMULA CF = max(0,(S*-1)/2): max|diff| over 60 states = {md:.2e}"
          + (f" | CF(Phi+)={phi_cf:.4f}" if drop==5 else ""))
    return len(fac)==24 and n_in==20 and n_gh==4 and op_par=={es} and md<1e-7 and (drop!=5 or abs(phi_cf-1)<1e-7)
oks=[analyze(5),analyze(2),analyze(0)]
print("GHOST FACET THEOREM + FORMULA: " + ("PASS on all three deletions" if all(oks) else "FAIL"))

# --- COMPLETION LEMMA (closes the written proof of the formula) ---
# For each operative ghost pattern sigma (Pi sigma = eps*), the point
#     y_sigma = (0,...,0 ; sigma)   (all non-ghost coordinates zero)
# is a valid parity-supported no-signalling model attaining sigma . c = 3: every remaining
# context contains at most one ghost observable, so its probabilities are (1 +- 1)/4 or 1/4,
# all nonnegative. With NS-attainability = 3 and the 20 in-context facets quantum-inert,
# ABM duality gives CF = max_sigma (S_sigma - 1)/(3 - 1) exactly: the formula is PROVED.
def completion_lemma(drop):
    CTX=[C for i,C in enumerate(PM) if i!=drop]; Cs=PM[drop]; es=eps_of(Cs)
    obs=sorted({o for C in CTX for o in C})
    ok=True
    for sig in itertools.product((1,-1),repeat=3):
        if np.prod(sig)!=es: continue
        y={o:0.0 for o in obs}
        for o,s in zip(Cs,sig): y[o]=float(s)
        for C in CTX:
            eC=eps_of(C)
            for s in itertools.product((1,-1),repeat=3):
                if np.prod(s)!=eC: continue
                p=(1+sum(si*y[o] for si,o in zip(s,C)))/4
                ok &= p>=-1e-12
    return ok
cl=all(completion_lemma(d) for d in (5,2,0))
print(f"COMPLETION LEMMA: y_sigma in NS for all operative sigma, all three deletions: {cl}")
print("PASS(completion)" if cl else "FAIL(completion)")
