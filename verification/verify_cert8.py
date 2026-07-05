"""Independent verification of the explicit S=4 certificate at (d,m)=(8,2).
Uses ONLY explicit 64x64 matrices (no algebraic phase formula)."""
import numpy as np, json
from collections import Counter

d, m = 8, 2
w = np.exp(2j*np.pi/d); tau = np.exp(1j*np.pi/d)
X = np.roll(np.eye(d),1,axis=0); Z = np.diag([w**k for k in range(d)])
def T1(a,b): return np.linalg.matrix_power(X,a)@np.linalg.matrix_power(Z,b)
def W(v):
    a1,b1,a2,b2 = [x%d for x in v]
    return tau**(-(a1*b1+a2*b2)) * np.kron(T1(a1,b1),T1(a2,b2))
I = np.eye(d**m)

cert = json.load(open("cert8_min.json"))
items = cert["items"]
print(f"certificate: {len(items)} contexts")

mult = Counter(); total = 0
for it in items:
    ctx = [tuple(v) for v in it["ctx"]]; lam = it["lam"]
    Ws = [W(v) for v in ctx]
    for A in Ws:
        assert np.allclose(np.linalg.matrix_power(A,d), I), "order violation"
    for i in range(len(Ws)):
        for j in range(i+1,len(Ws)):
            assert np.allclose(Ws[i]@Ws[j], Ws[j]@Ws[i]), ("noncommuting", ctx)
    P = I.copy()
    for A in Ws: P = P@A
    ph = P[0,0]
    assert np.allclose(P, ph*I), ("not scalar", ctx)
    s = round(np.angle(ph)/(2*np.pi/d)) % d
    assert np.allclose(ph, w**s)
    assert s == it["s"] % d, ("s mismatch", ctx, s, it["s"])
    for v in ctx: mult[v] += lam
    total = (total + lam*s) % d

assert all(c % d == 0 for c in mult.values()), "multiplicity condition FAILS"
print("every observable's total lambda-weighted multiplicity == 0 (mod 8):", True)
print("certificate value  sum_j lambda_j * s_j  mod 8 =", total)
assert total == 4

rng = np.random.default_rng(9)
obs = sorted(mult.keys())
for trial in range(50):
    c = {v:int(rng.integers(0,d)) for v in obs}
    tot=0
    for it in items:
        s2 = (it["s"] + sum(c[tuple(v)] for v in it["ctx"])) % d
        tot = (tot + it["lam"]*s2) % d
    assert tot == 4
print("value invariant under 50 random mu_8 relifts of all letters:", True)

import sys; sys.path.insert(0,".")
from howell import kernel_mod
rows = []; svals=[]
obsix = {v:i for i,v in enumerate(obs)}
for it in items:
    r = np.zeros(len(obs),dtype=np.int64)
    for v in it["ctx"]: r[obsix[tuple(v)]] += 1
    rows.append(r%8); svals.append(it["s"]%8)
A = np.array(rows); svals=np.array(svals)
KT = kernel_mod(A,8)
bad = [int((np.array(t)@svals)%8) for t in KT if (np.array(t)@svals)%8]
print(f"left-kernel of the 89-row system: {len(KT)} generators; nonzero pairings with s: {sorted(set(bad))}")
assert bad, "system unexpectedly feasible"
print("=> the 89-context system admits NO value assignment gamma: obs -> Z_8. AvN certified, S = 4.")
print("ALL CHECKS PASS")
