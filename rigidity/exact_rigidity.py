#!/usr/bin/env python3
"""
B1 — EXACT rigidity certificates for Yu-Oh 13 (d=3) and Peres 24 (d=4).

Upgrades the numerical flex-0 findings to exact rational arithmetic. Formulation avoids
square roots entirely by using UNNORMALIZED INTEGER rays and an extended Jacobian:

  variables:   w_i in C^d per vertex (real dim 2dV)   [ambient tangent, no phase gauge]
  constraints: per edge (i,j):   w_i^dag v_j + v_i^dag w_j = 0     (2 real rows, integer coeffs)
               per vertex i:     Re(v_i^dag w_i) = 0               (norm preservation; 1 real row)
  trivial:     per-vertex phases  w_i = i t v_i                    (V directions)
               unitary orbit      w = (A v_i)_i,  A in u(d)        (d^2 directions)

  flex = dim ker(J_ext)  -  rank(span(phases ∪ orbit)).

All entries are integers (rays integer, i-coefficients folded into real/imag parts), so sympy
computes ranks EXACTLY over Q. flex = 0 is then an exact certificate of infinitesimal rigidity.
"""
from sympy import Matrix, Rational, I as sI
from itertools import combinations, product

def integer_rays_yuoh():
    z=[[1,0,0],[0,1,0],[0,0,1]]
    y=[[0,1,1],[0,1,-1],[1,0,1],[1,0,-1],[1,1,0],[1,-1,0]]
    h=[[1,1,1],[-1,1,1],[1,-1,1],[1,1,-1]]
    return [tuple(v) for v in z+y+h]

def integer_rays_peres24():
    rays=[]; seen=set()
    def add(v):
        v=list(v)
        for x in v:
            if x!=0:
                if x<0: v=[-t for t in v]
                break
        t=tuple(v)
        if t not in seen: seen.add(t); rays.append(t)
    for p in range(4):
        e=[0]*4; e[p]=1; add(e)
    for i,j in combinations(range(4),2):
        for s in (1,-1):
            e=[0]*4; e[i]=1; e[j]=s; add(e)
    for signs in product([1,-1],repeat=3):
        add([1,signs[0],signs[1],signs[2]])
    assert len(rays)==24
    return rays

def ip(u,v):  # integer <u,v> for real integer vectors
    return sum(a*b for a,b in zip(u,v))

def exact_flex(rays, name):
    d=len(rays[0]); V=len(rays)
    E=[(i,j) for i,j in combinations(range(V),2) if ip(rays[i],rays[j])==0]
    n=2*d*V   # real coords: w_i = x_i + i y_i, x,y in R^d
    rows=[]
    def coord(i,comp,real): return 2*d*i + 2*comp + (0 if real else 1)
    # edge rows: f = w_i^dag v_j + v_i^dag w_j  (v real integer). Re f, Im f.
    #   w_i^dag v_j = sum_c (x_ic - i y_ic) v_jc ;  v_i^dag w_j = sum_c v_ic (x_jc + i y_jc)
    for (i,j) in E:
        re=[0]*n; im=[0]*n
        for c in range(d):
            re[coord(i,c,True)] += rays[j][c]; im[coord(i,c,False)] -= rays[j][c]
            re[coord(j,c,True)] += rays[i][c]; im[coord(j,c,False)] += rays[i][c]
        rows.append(re); rows.append(im)
    # norm rows: Re(v_i^dag w_i) = sum_c v_ic x_ic
    for i in range(V):
        r=[0]*n
        for c in range(d): r[coord(i,c,True)] = rays[i][c]
        rows.append(r)
    J=Matrix(rows)
    rankJ=J.rank()
    ker=n-rankJ
    # trivial directions
    triv=[]
    for i in range(V):        # phase at vertex i: w_i = i v_i  -> x=0, y=v_i
        t=[0]*n
        for c in range(d): t[coord(i,c,False)] = rays[i][c]
        triv.append(t)
    # u(d) basis: A = i E_aa ; (E_ab+E_ba) with a<b -> w=Av real-imag decomposition ; i(E_ab-E_ba)? enumerate:
    # anti-Hermitian basis over R: iH for H Hermitian: H=E_aa; H=E_ab+E_ba; H=i(E_ab-E_ba)
    for a in range(d):
        t=[0]*n   # A=iE_aa: w_i = i v_ia e_a
        for i in range(V): t[coord(i,a,False)] = rays[i][a]
        triv.append(t)
    for a in range(d):
        for b in range(a+1,d):
            t=[0]*n  # A=i(E_ab+E_ba): w_i = i(v_ib e_a + v_ia e_b)
            for i in range(V):
                t[coord(i,a,False)] += rays[i][b]; t[coord(i,b,False)] += rays[i][a]
            triv.append(t)
            t=[0]*n  # A=(E_ab-E_ba): w_i = v_ib e_a - v_ia e_b   (real)
            for i in range(V):
                t[coord(i,a,True)] += rays[i][b]; t[coord(i,b,True)] -= rays[i][a]
            triv.append(t)
    T=Matrix(triv)
    rankT=T.rank()
    # sanity: trivial directions satisfy the constraints exactly
    resid = (J*T.T)
    ok0 = all(x==0 for x in resid)
    flex = ker - rankT
    print(f"  {name:10s} d={d} V={V} E={len(E)} | rank J = {rankJ} (exact), ker = {ker}, "
          f"trivial rank = {rankT}, J*triv==0: {ok0}  ->  FLEX = {flex} (EXACT over Q)")
    return flex

if __name__=="__main__":
    print("EXACT rational-arithmetic rigidity certificates (no numerics, no tolerances):")
    f1=exact_flex(integer_rays_yuoh(),  "Yu-Oh 13")
    f2=exact_flex(integer_rays_peres24(),"Peres 24")
    verdict = (f1==0 and f2==0)
    print("\nTHEOREM-GRADE CERTIFICATE:" if verdict else "MISMATCH:",
          "Yu-Oh 13 and Peres 24 are infinitesimally rigid (flex 0), exactly."
          if verdict else "check formulation")
    print("exact_rigidity PASS" if verdict else "exact_rigidity FAIL")
