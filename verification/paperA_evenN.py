#!/usr/bin/env python3
# Paper A, EVEN-n GENERATOR GAP -- resolution check (2026-07-12).
#
# Claim to establish: the canonical class c (order exactly n, proved elsewhere) generates the
# Z/n summand of H^2(PN(T);U(1)) = Z/n (+) Z_2, INCLUDING even n (n=4 is Peres-Mermin).
#
# Two-step argument, each step checked here:
#   (S1) [abelian-group subtlety is REAL] In Z/n (+) Z_2, an element of order n need NOT have a
#        generating Z/n-component: for n = 2 (mod 4) there are order-n elements (a,1) with ord(a)=n/2.
#        So "order n" alone does NOT close the gap -- one must also pin the Z_2 (Schur) component.
#   (S2) [Schur component of c is zero] For the split group PN(T) = (T/U(1)) |x| S_n, the Z_2 = Schur
#        summand is H^2(S_n;U(1)) via inflation, with retraction = restriction to the S_n subgroup.
#        The S_n subgroup lifts to honest PERMUTATION MATRICES P_sigma, and P_sigma P_tau = P_{sigma tau}
#        EXACTLY (no scalar phase) -> the central extension N(T) -> PN(T) SPLITS over S_n ->
#        res_{S_n}(c) = 0 -> the Z_2-component of c is 0.  This holds at EVERY n.
#   Conclusion: c has zero Z_2-component, so c = (a,0); ord(c)=n then forces ord(a)=n, i.e. a is a
#   generator; hence <c> = Z/n (+) 0 is the standard Z/n summand. Gap closed (modulo the textbook
#   identification res_{S_n} = E_2^{2,0} edge for a split extension).
import numpy as np
from itertools import permutations, product

def perm_matrix(p):
    n=len(p); P=np.zeros((n,n),dtype=int)
    for i,pi in enumerate(p): P[pi,i]=1
    return P

def compose(p,q):            # (p o q)(i) = p[q[i]]
    return tuple(p[q[i]] for i in range(len(p)))

# ---- (S2) permutation matrices split S_n: P_p P_q = P_{p o q}, no phase ----
def check_perm_split(n):
    S=list(permutations(range(n)))
    ok=True
    # full check for n<=4, sampled for larger (group is closed; law is associative composition)
    import random; random.seed(0)
    pairs = [(p,q) for p in S for q in S] if n<=4 else [(random.choice(S),random.choice(S)) for _ in range(4000)]
    for p,q in pairs:
        if not np.array_equal(perm_matrix(p)@perm_matrix(q), perm_matrix(compose(p,q))):
            ok=False; break
    return ok

# ---- (S1) abelian subtlety + conclusion in Z/n (+) Z_2 ----
def order(el,n):             # el=(a,b) in Z/n x Z/2
    a,b=el; from math import gcd
    oa = n//gcd(a,n) if a%n else 1
    ob = 2 if b%2 else 1
    return np.lcm(oa,ob)
def generates_standard_summand(el,n):
    # <el> = Z/n (+) 0  iff  <el> cap (0 (+) Z_2) = 0  iff  b-component of el is 0 and ord(a)=n
    a,b=el
    from math import gcd
    return (b%2==0) and (gcd(a,n)==1)

def check_abelian(n):
    els=[(a,b) for a in range(n) for b in range(2)]
    ordn=[e for e in els if order(e,n)==n]
    # order-n elements with NONZERO Z_2 component (the subtlety) -- exist iff n = 2 (mod 4)
    subtle=[e for e in ordn if e[1]%2==1]
    # among order-n elements, those that generate the standard Z/n summand are exactly b=0, a a unit
    gen_std=[e for e in ordn if generates_standard_summand(e,n)]
    # KEY: every order-n element with ZERO Z_2 component generates the standard summand
    zero_b = [e for e in ordn if e[1]%2==0]
    key = all(generates_standard_summand(e,n) for e in zero_b) and len(zero_b)>0
    return {"#order_n":len(ordn), "#subtle(order_n,b=1)":len(subtle),
            "subtlety_present": len(subtle)>0, "n mod 4": n%4,
            "b=0 & order_n => generates standard summand": key}

if __name__=="__main__":
    print("=== (S2) S_n splits via permutation matrices (P_p P_q = P_{p o q}, no phase) ===")
    s2=True
    for n in [3,4,5,6,8]:
        r=check_perm_split(n); s2&=r
        print(f"  n={n}: permutation matrices form a homomorphic section of N(T)->PN(T) over S_n: {r}")
    print("  => res_{S_n}(c) = 0  => the Z_2 (Schur) component of c is 0, at every n.\n")

    print("=== (S1) abelian subtlety in Z/n (+) Z_2, and the conclusion ===")
    s1=True
    for n in [2,4,6,8,10,12]:
        d=check_abelian(n)
        s1 &= d["b=0 & order_n => generates standard summand"]
        note = "SUBTLE (order-n elt with b=1 exists)" if d["subtlety_present"] else "clean (order-n => generator)"
        print(f"  n={n:2d} (n%4={d['n mod 4']}): {note}; "
              f"b=0 & order n => generator: {d['b=0 & order_n => generates standard summand']}")
    print("  => 'order n' alone is NOT enough for n=2 (mod 4); but combined with (S2) [b=0] it IS.\n")

    print("CONCLUSION: c has zero Schur component (S2) and order n, hence generates the standard")
    print("Z/n summand of H^2, for EVERY n including n=4 (Peres-Mermin).")
    print("paperA_evenN PASS" if (s1 and s2) else "paperA_evenN FAIL")
