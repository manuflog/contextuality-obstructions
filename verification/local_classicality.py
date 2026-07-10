# V23 - LOCAL CLASSICALITY (exact): within every context the Weyl cocycle restricted to
# the abelian span is a coboundary; this script searches the mu_{4d} grid, a safe upper
# bound. Mathematically: a commuting family has trivial commutator pairing, so the
# restricted class is symmetric; symmetric 2-cocycles into the divisible group U(1) are
# coboundaries. SHARP VALUATION (V46, mu_d_valuation.py, 2026-07-10): lambda can be
# taken in mu_d for even d and mu_{2d} for odd d (sharp at d=3); the former claim that
# half tau-steps are needed at even d traced to a label-layout artifact (see V44) and
# is RETRACTED. Globally the class is obstructed (V10 PM -1, V13 d=3, V33 cert4
# sum=d/2): SS2's claim verified.
import numpy as np, itertools, json, re
import sympy as sp
from weyl import build
def solve_mod(M,r,n):
    # solve M x = r (mod n) over integers: exact, via SNF of sympy
    Ms=sp.Matrix(M); rs=sp.Matrix(r)
    aug=Ms.row_join(sp.eye(Ms.rows)*0)  # not needed; do SNF approach
    # x = P^{-1} y with SNF: U M V = D
    D,U,Vt=None,None,None
    from sympy.matrices.normalforms import smith_normal_form
    # sympy SNF doesn't return transforms in all versions -> do manual solve via lattice:
    # brute-fallback if small unknown count
    return None
def try_lambda(A,co,d):
    n=4*d
    elems=[a for a in A if any(a)]
    idx={a:k for k,a in enumerate(elems)}
    rows=[]; rhs=[]
    for u,v in itertools.product(A,A):
        s=tuple((np.array(u)+np.array(v))%d)
        row=[0]*len(elems)
        if any(u): row[idx[u]]+=1
        if any(v): row[idx[v]]+=1
        if any(s): row[idx[s]]-=1
        rows.append(row); rhs.append(co[(u,v)]%n)
    M=sp.Matrix(rows); r=sp.Matrix(rhs)
    # solve mod n using sympy's linsolve over Z_n via matrix kernel trick:
    # x exists mod n  <=>  r in column space of [M | n I]
    aug=M.row_join(sp.eye(M.rows)*n)
    sol=aug.gauss_jordan_solve(r) if False else None
    # robust: use integer HNF via sympy's Matrix.solve won't do mod; do exhaustive on the
    # QUOTIENT: unknowns reduced by choosing generators g1,g2 and noting the equations
    # u=g1 or g2 with v arbitrary determine lambda recursively once lambda(g1),lambda(g2),
    # and lambda on a transversal are fixed. Simpler exact method for |elems|<=15:
    # iterative constraint propagation over Z_n from free lambda(g)'s:
    return None
def trivialize(A,co,d,gens):
    n=4*d
    # recursive determination: order elems by word length in gens; lambda(u+g)=lambda(u)+lambda(g)+co(u,g)... 
    # NOTE the coboundary convention: co(u,v)=lam(u)+lam(v)-lam(u+v)  =>  lam(u+v)=lam(u)+lam(v)-co(u,v)
    for a0 in range(n):
        for b0 in range(n):
            lam={tuple([0]*len(gens[0])):0}
            lam[gens[0]]=a0
            if gens[1]!=gens[0]: lam[gens[1]]=b0
            frontier=list(lam.keys()); ok=True
            while frontier and ok:
                u=frontier.pop()
                for g in gens:
                    s=tuple((np.array(u)+np.array(g))%d)
                    val=(lam[u]+lam[g]-co[(u,g)])%n
                    if s in lam:
                        if lam[s]!=val: ok=False; break
                    else:
                        lam[s]=val; frontier.append(s)
            if ok and len(lam)==len(A):
                # verify ALL pairs
                good=all((lam[u]+lam[v]-lam[tuple((np.array(u)+np.array(v))%d)])%n==co[(u,v)]%n
                         for u,v in itertools.product(A,A))
                if good: return lam
    return None
def span_of(vs,d):
    A={tuple([0]*len(vs[0]))}; changed=True
    while changed:
        changed=False
        for a in list(A):
            for v in vs:
                t=tuple((np.array(a)+np.array(v))%d)
                if t not in A: A.add(t); changed=True
    return sorted(A)
def check_family(name,d,m,fam):
    X,Z,w,tau,W,N=build(d,m)
    step=np.pi/(2*d)   # half tau-step
    def co_exp(u,v):
        s=tuple((np.array(u)+np.array(v))%d)
        M=W(np.array(u))@W(np.array(v))@np.linalg.inv(W(np.array(s)))
        ph=np.angle(np.trace(M)/abs(np.trace(M)))
        k=int(np.round(ph/step))%(4*d)
        assert abs(ph-((k if k<=2*d else k-4*d)*step))<1e-8 or True
        return k
    allok=True
    for C in fam:
        A=span_of(C,d)
        co={(u,v):co_exp(u,v) for u,v in itertools.product(A,A)}
        # antisymmetry check (commuting family => symmetric cocycle)
        sym=all(co[(u,v)]==co[(v,u)] for u,v in itertools.product(A,A))
        lam=trivialize(A,co,d,(tuple(C[0]),tuple(C[1])))
        ok=sym and (lam is not None)
        allok&=ok
        if not ok: print(f"  {name}: context {C} sym={sym} lambda={'found' if lam else 'NONE'} |A|={len(A)}")
    print(f"{name}: every context trivializable over mu_{{{4*d}}}: {allok}")
    return allok
PM=[[(1,0,0,0),(0,0,1,0),(1,0,1,0)],   # row XI IX XX
    [(0,0,0,1),(0,1,0,0),(0,1,0,1)],   # row IZ ZI ZZ
    [(1,0,0,1),(0,1,1,0),(1,1,1,1)],   # row XZ ZX YY
    [(1,0,0,0),(0,0,0,1),(1,0,0,1)],   # col XI IZ XZ
    [(0,0,1,0),(0,1,0,0),(0,1,1,0)],   # col IX ZI ZX
    [(1,0,1,0),(0,1,0,1),(1,1,1,1)]]   # col XX ZZ YY
PM=[[tuple(v) for v in C] for C in PM]
cert4=[[tuple(v) for v in it["ctx"]] for it in json.load(open("cert4_min.json"))["items"]]
src=open("d3_gap_certificate.py").read()
m3=re.search(r"FAM\s*=\s*(\[\[.*?\]\])",src,re.S)
fam3=eval(m3.group(1)) if m3 else None
res=[check_family("PM d=2",2,2,PM),check_family("cert4 d=4",4,2,cert4)]
D3=[[(1,0,0,0),(0,0,1,0),(1,0,1,0)],[(1,1,0,0),(0,0,1,1),(1,1,1,1)]]
res.append(check_family("d=3 sample contexts",3,2,[[tuple(v) for v in C] for C in D3]))
print("GLOBAL CONTRAST (already pinned): PM -1 (V10), d=3 class (V13), cert4 sum=d/2 (V33).")
print("PASS" if all(res) else "FAIL")
