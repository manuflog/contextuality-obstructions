import numpy as np
from weyl import build
from phase import make_alg

rng = np.random.default_rng(42)

def make_alg2(d,m,odd_gross=False):
    """even d: tau-exponent mod 2d (exact). odd d + gross: phases in Z_d with inv2."""
    def q(v): return sum(int(v[2*i])*int(v[2*i+1]) for i in range(m))
    def beta(u,v): return sum(int(u[2*i+1])*int(v[2*i]) for i in range(m))
    def symp(u,v): return (sum(int(u[2*i+1])*int(v[2*i])-int(u[2*i])*int(v[2*i+1]) for i in range(m)))%d
    if not odd_gross:
        def phexp(ctx):  # exact tau exponent mod 2d
            cur=tuple([0]*(2*m)); ph=0
            for v in ctx:
                nxt=tuple((cur[i]+v[i])%d for i in range(2*m))
                ph=(ph-q(cur)-q(v)+2*beta(cur,v)+q(nxt))%(2*d)
                cur=nxt
            assert all(x==0 for x in cur)
            return ph
        return q,beta,symp,phexp
    else:
        i2=(d+1)//2
        def sg(ctx):  # W'(v)=w^{-i2 q(v)}T(v); scalar = w^{sg}
            cur=tuple([0]*(2*m)); s=0
            for v in ctx:
                nxt=tuple((cur[i]+v[i])%d for i in range(2*m))
                s=(s-i2*q(cur)-i2*q(v)+beta(cur,v)+i2*q(nxt))%d
                cur=nxt
            assert all(x==0 for x in cur)
            return s
        return q,beta,symp,sg

def rand_context(d,m,length,symp):
    for _ in range(4000):
        vs=[]
        ok=True
        for i in range(length-1):
            for _ in range(600):
                v=tuple(int(x) for x in rng.integers(0,d,2*m))
                if all(symp(v,u)==0 for u in vs): vs.append(v); break
            else: ok=False; break
        if not ok: continue
        last=tuple((-sum(v[i] for v in vs))%d for i in range(2*m))
        if all(symp(last,u)==0 for u in vs): return vs+[last]
    return None

if __name__=="__main__":
    print("A) exactness of the tau/gross phase formulas vs matrices (m=2)")
    for d in (4,6,9,16):
        m=2; _,_,w,tau,W,_=build(d,m)
        odd = (d%2==1)
        q,beta,symp,ph=make_alg2(d,m,odd_gross=odd)
        I=np.eye(d**m); N=25; okc=0
        for t in range(N):
            C=rand_context(d,m,int(rng.integers(2,6)),symp)
            if odd:
                X=np.roll(np.eye(d),1,axis=0); Z=np.diag([w**k for k in range(d)])
                def T1(a,b): return np.linalg.matrix_power(X,a)@np.linalg.matrix_power(Z,b)
                i2=(d+1)//2
                def Wv(v):
                    M=T1(v[0],v[1])
                    for i in range(1,m): M=np.kron(M,T1(v[2*i],v[2*i+1]))
                    return (w**((-i2*q(v))%d))*M
                P=I
                for v in C: P=P@Wv(v)
                z=P[0,0]; assert np.allclose(P,z*I)
                s=ph(C); assert np.allclose(z,w**s),(d,C,s)
            else:
                P=I
                for v in C: P=P@W(v)
                z=P[0,0]; assert np.allclose(P,z*I)
                e=ph(C); assert np.allclose(z,tau**e),(d,C,e)
            okc+=1
        print(f"  d={d} ({'gross' if odd else 'tau'}): {okc}/{N} exact")

    print()
    print("B) EVEN d: tau-exponent identity  phexp(C) + 2*sum q(u) ?= 0 mod d")
    for d in (2,4,6,8,12,16):
        q,beta,symp,ph=make_alg2(d,2)
        bad=0; oddexp=0; N=3000
        for t in range(N):
            C=rand_context(d,2,int(rng.integers(2,7)),symp)
            if C is None: continue
            e=ph(C)
            if e%2: oddexp+=1
            if (e+2*sum(q(v) for v in C))%d: bad+=1
        print(f"  d={d}: identity violations {bad}/{N}, odd tau-exponents {oddexp}")

    print()
    print("B-strong) EVEN d: full mod-2d identity  phexp(C) ?= -2*sum q(u) mod 2d")
    for d in (4,8,16):
        q,beta,symp,ph=make_alg2(d,2)
        bad=0; N=2000
        for t in range(N):
            C=rand_context(d,2,int(rng.integers(2,7)),symp)
            e=ph(C)
            if (e+2*sum(q(v) for v in C))%(2*d): bad+=1
        print(f"  d={d}: mod-2d violations {bad}/{N}")

    print()
    print("C) ODD d (gross): potential identity  s(C) + sum q(u) ?= 0 mod d")
    for d in (3,5,9):
        q,beta,symp,sg=make_alg2(d,2,odd_gross=True)
        for form in [1]:
            bad=0; N=2000; vals=set()
            for t in range(N):
                C=rand_context(d,2,int(rng.integers(2,7)),symp)
                s=sg(C); vals.add((s+sum(q(v) for v in C))%d)
            print(f"  d={d}: distinct values of s+Sigma q mod d over {N} contexts: {sorted(vals)}")
