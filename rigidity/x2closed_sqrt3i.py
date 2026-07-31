# Independent verification: x = sqrt(3) + i. Ring Z[sqrt3, i]: elements (a,b,c,d) = a + b*sqrt3 + c*i + d*i*sqrt3.
from itertools import combinations, product

def add(u, v): return tuple(x + y for x, y in zip(u, v))
def neg(u): return tuple(-x for x in u)
def sub(u, v): return tuple(x - y for x, y in zip(u, v))
def mul(u, v):
    a1, b1, c1, d1 = u; a2, b2, c2, d2 = v
    # (a + b r + c i + d i r)(a' + b' r + c' i + d' i r), r^2 = 3, i^2 = -1
    a = a1*a2 + 3*b1*b2 - c1*c2 - 3*d1*d2
    b = a1*b2 + b1*a2 - c1*d2 - d1*c2
    c = a1*c2 + c1*a2 + 3*(b1*d2 + d1*b2)
    d = a1*d2 + d1*a2 + b1*c2 + c1*b2
    return (a, b, c, d)
def conj(u): return (u[0], u[1], -u[2], -u[3])   # i -> -i

ZERO = (0,0,0,0); ONE = (1,0,0,0)
X  = (0,1,1,0)          # sqrt3 + i
XS = conj(X)            # sqrt3 - i
N  = mul(X, XS)
print("N =", N, "(must be (4,0,0,0))")
w = add(add(mul(X,X), mul(XS,XS)), neg(N))
print("x^2 + x*^2 - N =", w, "(must be zero) => witness identity on the arg-30-degree line")

# locus check: M1 x in {+-2,+-1/2}? no. M2 N in {2,1/2}? N=4 no. M3 Re x = +-1/2? Re = sqrt3 no.
# M4 x real? no. M5 2 Re x = +-N? 2 sqrt3 = +-4? no.  => avoids ALL five published loci.
alpha = [ZERO, ONE, neg(ONE), X, neg(X), XS, neg(XS)]
def hdot(u, v):
    s = ZERO
    for a, b in zip(u, v): s = add(s, mul(conj(a), b))
    return s
def cross(u, v):
    return (sub(mul(u[1],v[2]), mul(u[2],v[1])),
            sub(mul(u[2],v[0]), mul(u[0],v[2])),
            sub(mul(u[0],v[1]), mul(u[1],v[0])))
def isnull(t): return all(e == ZERO for e in t)
def same(u, v): return isnull(cross(u, v))

rays = []
for v in product(alpha, repeat=3):
    if all(e == ZERO for e in v): continue
    if not any(same(v, r) for r in rays): rays.append(v)
n = len(rays)
E = [(i, j) for i in range(n) for j in range(i+1, n) if hdot(rays[i], rays[j]) == ZERO]
Es = set(E)
T = [t for t in combinations(range(n), 3)
     if all((min(a,b), max(a,b)) in Es for a, b in combinations(t, 2))]
print("closed pool at x = sqrt3+i: %d rays / %d pairs / %d triads" % (n, len(E), len(T)), flush=True)

from pysat.solvers import Glucose3
s = Glucose3()
for i, j in E: s.add_clause([-(i+1), -(j+1)])
for (i, j, k) in T: s.add_clause([i+1, j+1, k+1])
res = s.solve()
print("closed pool KS-colorable (SAT):", res, "=>", "COLORABLE" if res else "UNCOLORABLE")

# UNCLOSED pool (entries only {0,+-1,+-x}) for the convention question:
alpha_u = [ZERO, ONE, neg(ONE), X, neg(X)]
rays_u = []
for v in product(alpha_u, repeat=3):
    if all(e == ZERO for e in v): continue
    if not any(same(v, r) for r in rays_u): rays_u.append(v)
nu = len(rays_u)
Eu = [(i, j) for i in range(nu) for j in range(i+1, nu) if hdot(rays_u[i], rays_u[j]) == ZERO]
Eus = set(Eu)
Tu = [t for t in combinations(range(nu), 3)
      if all((min(a,b), max(a,b)) in Eus for a, b in combinations(t, 2))]
s2 = Glucose3()
for i, j in Eu: s2.add_clause([-(i+1), -(j+1)])
for (i, j, k) in Tu: s2.add_clause([i+1, j+1, k+1])
r2 = s2.solve()
print("UNCLOSED pool: %d rays / %d pairs / %d triads -> %s" % (nu, len(Eu), len(Tu), "COLORABLE" if r2 else "UNCOLORABLE"))
