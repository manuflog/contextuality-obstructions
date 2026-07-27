"""Thm J support (Obstruction Spectrum) -- see INDEX.md.

VERIFIED CLAIMS (all asserted below; a failure raises and exits nonzero):
  [A] the closed-form tau-exponent / Gross-phase formulas reproduce the actual matrix
      product scalar exactly, for every sampled context (even d via tau, odd d via the
      Gross phase).
  [B] EVEN d: the mod-d tau-exponent identity  phexp(C) + 2*sum q(u) == 0 (mod d)
      holds with ZERO violations, and every tau-exponent is EVEN (=> context products
      are omega-integral; the "tau-odd" gotcha is an artifact -- see INDEX.md V46).
  [C] ODD d (Gross): the value set of  s(C) + sum q(u) mod d  is exactly {0}
      -- the odd-d half of Thm J (achievable value set {0} for odd d).

EXPECTED-NEGATIVE (section B-strong): the *stronger* mod-2d refinement of [B] is an
EXPLORATORY probe that is KNOWN AND EXPECTED TO FAIL. Its large violation counts are a
recorded negative result, NOT a suite failure, and are deliberately NOT asserted.
The theorem needs only the mod-d identity [B]; the mod-2d version is false.
"""
import sys
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
    checks=[]   # (label, ok) -- every load-bearing quantity gets an entry
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
        checks.append((f"A:d={d} phase formula exact on all {N} sampled contexts", okc==N))

    print()
    print("B) EVEN d: tau-exponent identity  phexp(C) + 2*sum q(u) ?= 0 mod d")
    for d in (2,4,6,8,12,16):
        q,beta,symp,ph=make_alg2(d,2)
        bad=0; oddexp=0; N=3000; tested=0
        for t in range(N):
            C=rand_context(d,2,int(rng.integers(2,7)),symp)
            if C is None: continue
            tested+=1
            e=ph(C)
            if e%2: oddexp+=1
            if (e+2*sum(q(v) for v in C))%d: bad+=1
        print(f"  d={d}: identity violations {bad}/{N}, odd tau-exponents {oddexp}")
        # load-bearing: zero violations, zero odd exponents, and a non-vacuous sample
        checks.append((f"B:d={d} mod-d identity violations == 0 (got {bad})", bad==0))
        checks.append((f"B:d={d} odd tau-exponents == 0 (got {oddexp})", oddexp==0))
        checks.append((f"B:d={d} sample non-vacuous ({tested}/{N} contexts built)", tested==N))

    print()
    print("B-strong) [EXPLORATORY -- EXPECTED NEGATIVE, NOT ASSERTED]")
    print("          EVEN d: full mod-2d identity  phexp(C) ?= -2*sum q(u) mod 2d")
    print("          This stronger refinement is KNOWN FALSE; the large violation counts")
    print("          below are the recorded negative result, not a suite failure.")
    print("          Thm J needs only the mod-d identity verified in section B.")
    strong_bad={}
    for d in (4,8,16):
        q,beta,symp,ph=make_alg2(d,2)
        bad=0; N=2000
        for t in range(N):
            C=rand_context(d,2,int(rng.integers(2,7)),symp)
            e=ph(C)
            if (e+2*sum(q(v) for v in C))%(2*d): bad+=1
        strong_bad[d]=(bad,N)
        print(f"  d={d}: mod-2d violations {bad}/{N}   [expected-negative: nonzero is CORRECT]")

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
            # load-bearing (Thm J, odd-d half): the achievable value set is exactly {0}
            checks.append((f"C:d={d} value set == {{0}} (got {sorted(vals)})", sorted(vals)==[0]))

    print()
    print("--- verdict ---")
    for label,ok in checks:
        if not ok: print(f"  FAILED CHECK: {label}")
    bad_checks=[l for l,ok in checks if not ok]
    for d,(bad,N) in strong_bad.items():
        print(f"  (expected-negative, unasserted) B-strong d={d}: {bad}/{N} mod-2d violations")
    assert not bad_checks, "spectrum_test2 load-bearing checks FAILED: "+"; ".join(bad_checks)
    print(f"spectrum_test2: {len(checks)}/{len(checks)} load-bearing checks passed")
    print("spectrum_test2 PASS")
    sys.exit(0)
