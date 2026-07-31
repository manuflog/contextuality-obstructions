# Independent verification: x = 2 + omega = (3 + i sqrt3)/2, Hermitian-closed pool in C^3.
# Work in the ring Z[omega] (Eisenstein), omega^2 = -1 - omega. Elements (a, b) = a + b*omega.
# x = 2 + omega, x* = conj = 2 + conj(omega) = 2 + omega^2 = 2 + (-1-omega) = 1 - omega.
from itertools import combinations, product

def add(u, v): return (u[0] + v[0], u[1] + v[1])
def neg(u): return (-u[0], -u[1])
def sub(u, v): return (u[0] - v[0], u[1] - v[1])
def mul(u, v):
    a1, b1 = u; a2, b2 = v
    # (a1 + b1 w)(a2 + b2 w) = a1a2 + (a1b2 + b1a2) w + b1b2 w^2 ; w^2 = -1 - w
    return (a1 * a2 - b1 * b2, a1 * b2 + b1 * a2 - b1 * b2)
def conj(u):
    # conj(a + b w) = a + b conj(w) = a + b(-1-w) = (a - b) - b w
    return (u[0] - u[1], -u[1])

ZERO = (0, 0); ONE = (1, 0)
X = (2, 1)            # 2 + omega
XS = conj(X)          # 1 - omega
assert mul(X, XS) == (3, 0)   # N = 3

# sanity: the witness identity  x^2 + x*^2 - N = 0
lhs = add(add(mul(X, X), mul(XS, XS)), neg((3, 0)))
print("x^2 + x*^2 - N =", lhs, "(must be (0,0))")

alpha = [ZERO, ONE, neg(ONE), X, neg(X), XS, neg(XS)]
def hdot(u, v):
    s = ZERO
    for a, b in zip(u, v): s = add(s, mul(conj(a), b))
    return s
def cross(u, v):
    return (sub(mul(u[1], v[2]), mul(u[2], v[1])),
            sub(mul(u[2], v[0]), mul(u[0], v[2])),
            sub(mul(u[0], v[1]), mul(u[1], v[0])))
def isnull(w): return all(e == ZERO for e in w)
def same(u, v): return isnull(cross(u, v))

rays = []
for v in product(alpha, repeat=3):
    if all(e == ZERO for e in v): continue
    if not any(same(v, w) for w in rays): rays.append(v)
n = len(rays)
E = [(i, j) for i in range(n) for j in range(i + 1, n) if hdot(rays[i], rays[j]) == ZERO]
Es = set(E)
T = [t for t in combinations(range(n), 3)
     if all((min(a, b), max(a, b)) in Es for a, b in combinations(t, 2))]
print("pool at x = 2+omega: %d rays / %d pairs / %d triads" % (n, len(E), len(T)), flush=True)

# the witness edge exists in the pool
u = (XS, X, X); v = (X, XS, neg(X))
print("witness <(x*,x,x),(x,x*,-x)> =", hdot(u, v), "(must be (0,0))")

from pysat.solvers import Glucose3
s = Glucose3()
for i, j in E: s.add_clause([-(i + 1), -(j + 1)])
for (i, j, k) in T: s.add_clause([i + 1, j + 1, k + 1])
res = s.solve()
print("KS-colorable (SAT):", res, "=>", "COLORABLE" if res else "UNCOLORABLE")

# raw mechanism check: x = 2+omega on neither (A) nor (B):
# N = 3 not in {2, 1/2}; x not in {2, -2, 1/2, -1/2}; x not a root of unity (N != 1).
print("N =", mul(X, XS), " -> (A) requires N in {2,1/2} or x in {+-2,+-1/2}: NO ; (B) requires N=1: NO")
