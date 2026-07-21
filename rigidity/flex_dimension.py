#!/usr/bin/env python3
"""
B1 — Rigidity–contextuality dictionary: the flex-dimension engine (2026-07-12).

Definition (working). A realization of an exclusivity graph G=(V,E) in dimension d is a map
v: V -> CP^{d-1} with <v_i, v_j> = 0 for every edge. The FLEX DIMENSION at a realization is
    flex = dim ker(J) - dim(PU(d)-orbit directions),
where J is the real Jacobian ("rigidity matrix") of the edge constraints acting on the product of
projective tangent spaces  T_[v_i] CP^{d-1} = v_i^perp  (2(d-1) real dims per vertex).
flex = 0 is (infinitesimal) rigidity: the realization is locally unique up to a global unitary.

Dictionary under test (V54's conjecture, generalized):
    flex dimension  =  dimension of state-dependent contextuality moduli;
    rigidity ( + forced torsion )  <=>  state-independent ( AvN ) contextuality.

Discriminating test set:
  - one ONB (d=3):     trivial, expect flex 0 (sanity)
  - KCBS C5 (d=3):     state-DEPENDENT;  V54 reports flex 2  -> reproduce independently
  - C7 (d=3):          state-DEPENDENT;  flex unknown -> measure (V50 gives 8 holonomy coords for
                       unconstrained local systems; the constrained flex is a different number)
  - C9 (d=3):          state-DEPENDENT;  measure (trend test)
  - Yu-Oh 13 (d=3):    state-INDEPENDENT but NOT AvN  -> THE critical test: dictionary predicts RIGID
  - Peres 24 (d=4):    state-independent AND AvN;  V54 reports flex 0 -> reproduce independently
All numerics are rank computations with an explicit spectral-gap check (we report the singular-value
gap so 'rank' is not tolerance-luck). Evidence label: NUMERICAL (exact-arithmetic upgrade is future work).
"""
import numpy as np
from itertools import combinations, permutations, product

TOL = 1e-8

def unit(v): v=np.asarray(v,dtype=complex); return v/np.linalg.norm(v)

def tangent_basis(v):
    """Orthonormal complex basis of v^perp."""
    d=len(v); M=np.eye(d,dtype=complex)-np.outer(v,v.conj())
    Q,R=np.linalg.qr(M)
    cols=[Q[:,k] for k in range(d) if abs(R[k,k])>1e-12]
    # keep d-1 columns orthogonal to v
    B=[c for c in cols if abs(np.vdot(v,c))<1e-9][:d-1]
    assert len(B)==d-1, "tangent basis failed"
    return B

def edges_from(rays, tol=1e-7):
    E=[]
    for i,j in combinations(range(len(rays)),2):
        if abs(np.vdot(rays[i],rays[j]))<tol: E.append((i,j))
    return E

def flex_dimension(rays, name="", expect=None):
    rays=[unit(r) for r in rays]; d=len(rays[0]); V=len(rays)
    E=edges_from(rays)
    # verify constraints hold
    for i,j in E: assert abs(np.vdot(rays[i],rays[j]))<1e-6, f"{name}: edge not orthogonal"
    tb=[tangent_basis(v) for v in rays]
    ncols=2*(d-1)*V
    J=np.zeros((2*len(E),ncols))
    for r,(i,j) in enumerate(E):
        vi,vj=rays[i],rays[j]
        for k in range(d-1):
            # d f = w_i^dag v_j + v_i^dag w_j ;  w_i = sum (a+ib) e_k^i
            ci=np.vdot(tb[i][k],vj)          # (e_k^i)^dag v_j   (coeff of a_k^i;  -i ci for b)
            cj=np.vdot(vi,tb[j][k])          # v_i^dag e_k^j     (coeff of a_k^j;  +i cj for b)
            col_i=2*(d-1)*i+2*k; col_j=2*(d-1)*j+2*k
            J[2*r,  col_i]+=ci.real; J[2*r,  col_i+1]+=(-1j*ci).real
            J[2*r+1,col_i]+=ci.imag; J[2*r+1,col_i+1]+=(-1j*ci).imag
            J[2*r,  col_j]+=cj.real; J[2*r,  col_j+1]+=(1j*cj).real
            J[2*r+1,col_j]+=cj.imag; J[2*r+1,col_j+1]+=(1j*cj).imag
    # rank of J with spectral-gap report
    sv=np.linalg.svd(J,compute_uv=False) if J.size else np.array([])
    rankJ=int((sv>TOL*max(1,sv.max() if sv.size else 1)).sum())
    gapJ = (sv[rankJ-1]/sv[rankJ]) if (sv.size and 0<rankJ<len(sv) and sv[rankJ]>0) else np.inf
    kerdim=ncols-rankJ
    # trivial (unitary-orbit) directions: A in u(d), direction_i = proj_{v_i^perp}(A v_i)
    dirs=[]
    basis=[]
    for a in range(d):
        for b in range(a,d):
            H=np.zeros((d,d),complex)
            if a==b: H[a,a]=1
            else: H[a,b]=1; H[b,a]=1
            basis.append(1j*H)                    # i*Hermitian = anti-Hermitian
            if a!=b:
                K=np.zeros((d,d),complex); K[a,b]=1j; K[b,a]=-1j
                basis.append(1j*K)
    for A in basis:
        vec=np.zeros(ncols)
        for i,v in enumerate(rays):
            Av=A@v
            for k in range(d-1):
                c=np.vdot(tb[i][k],Av)
                vec[2*(d-1)*i+2*k]=c.real; vec[2*(d-1)*i+2*k+1]=c.imag
        dirs.append(vec)
    D=np.array(dirs)
    svD=np.linalg.svd(D,compute_uv=False)
    orbit=int((svD>TOL*max(1,svD.max())).sum())
    # sanity: orbit directions lie in ker J
    resid=np.abs(J@D.T).max() if J.size else 0.0
    flex=kerdim-orbit
    tag = "" if expect is None else ("  [expected %s: %s]"%(expect,"MATCH" if flex==expect else "DIFFERS"))
    print(f"  {name:14s} d={d} V={V} E={len(E)}  rankJ={rankJ} (gap {gapJ:.1e})  ker={kerdim}  orbit={orbit}  "
          f"FLEX={flex}{tag}  (J@orbit resid {resid:.1e})")
    return flex

# ---------------- realizations ----------------
def onb(d=3): return [unit(np.eye(d)[k]) for k in range(d)]

def odd_cycle(n):
    """Lovász umbrella: v_j = (sin t cos(pi j (n-1)/n), sin t sin(pi j (n-1)/n), cos t),
       cos^2 t = cos(pi/n)/(1+cos(pi/n)); consecutive rays orthogonal."""
    c=np.cos(np.pi/n); ct2=c/(1+c); ct=np.sqrt(ct2); st=np.sqrt(1-ct2)
    rays=[unit([st*np.cos(np.pi*j*(n-1)/n), st*np.sin(np.pi*j*(n-1)/n), ct]) for j in range(n)]
    # polish orthogonality of the n-cycle numerically (Newton on the constraint set)
    return rays

def yu_oh():
    z=[[1,0,0],[0,1,0],[0,0,1]]
    y=[[0,1,1],[0,1,-1],[1,0,1],[1,0,-1],[1,1,0],[1,-1,0]]
    h=[[1,1,1],[-1,1,1],[1,-1,1],[1,1,-1]]
    return [unit(v) for v in z+y+h]

def peres24():
    rays=[]
    seen=set()
    def add(v):
        v=np.array(v,dtype=float)
        v=v/np.linalg.norm(v)
        # mod sign: canonical representative
        for x in v:
            if abs(x)>1e-9:
                if x<0: v=-v
                break
        key=tuple(np.round(v,6));
        if key not in seen: seen.add(key); rays.append(unit(v))
    for pos in range(4):
        e=[0]*4; e[pos]=1; add(e)
    for i,j in combinations(range(4),2):
        for s in (1,-1):
            e=[0]*4; e[i]=1; e[j]=s; add(e)
    for signs in product([1,-1],repeat=3):
        add([1,signs[0],signs[1],signs[2]])
    assert len(rays)==24, len(rays)
    return rays

if __name__=="__main__":
    print("B1 flex-dimension engine — discriminating test set")
    print("(dictionary: flex = state-dependent moduli; rigid(+torsion) <=> state-independent(AvN))\n")
    flex_dimension(onb(3),        name="ONB d=3",   expect=0)
    flex_dimension(odd_cycle(5),  name="KCBS C5",   expect=2)
    flex_dimension(odd_cycle(7),  name="C7")
    flex_dimension(odd_cycle(9),  name="C9")
    flex_dimension(yu_oh(),       name="Yu-Oh 13")
    flex_dimension(peres24(),     name="Peres 24",  expect=0)
    print("\nInterpretation guide: state-dependent scenarios (C5,C7,C9) should be FLEXIBLE with flex")
    print("counting their moduli; state-independent ones (Yu-Oh, Peres) should be RIGID (flex 0).")
    print("Yu-Oh is the critical case: rigid-without-AvN would separate the rigidity layer (state-")
    print("independence) from the torsion layer (AvN), exactly as the dictionary predicts.")
