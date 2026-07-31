# line30_ring.py -- shared EXACT arithmetic for the arg-30-degree line  x = t * zeta,
# zeta = e^{i pi/6} (primitive 12th root of unity), t a FORMAL positive variable.
#
# Ring Z[zeta] as 4-tuples (a,b,c,d) = a + b*z + c*z^2 + d*z^3 w.r.t. the Z-basis
# 1, z, z^2, z^3 with the cyclotomic relation Phi_12: z^4 = z^2 - 1  (so z^6 = -1).
# conj(z) = z^{-1} = z^{11} = z - z^3.
#
# Polynomials in t with Z[zeta] coefficients: 3-tuples (deg-0, deg-1, deg-2 coefficient)
# -- every quantity we ever form (Hermitian dots, cross-product entries) has t-degree <= 2
# because every alphabet entry is c * t^e with e in {0,1}.
#
# Everything is python-int exact.  No floats anywhere.
from itertools import combinations, product
from fractions import Fraction

Z0 = (0, 0, 0, 0)
Z1 = (1, 0, 0, 0)
ZETA = (0, 1, 0, 0)


def zadd(u, v): return (u[0]+v[0], u[1]+v[1], u[2]+v[2], u[3]+v[3])
def zsub(u, v): return (u[0]-v[0], u[1]-v[1], u[2]-v[2], u[3]-v[3])
def zneg(u):    return (-u[0], -u[1], -u[2], -u[3])
def zscale(k, u): return (k*u[0], k*u[1], k*u[2], k*u[3])


def zmul(u, v):
    a0, a1, a2, a3 = u
    b0, b1, b2, b3 = v
    # convolution, degree <= 6
    c0 = a0*b0
    c1 = a0*b1 + a1*b0
    c2 = a0*b2 + a1*b1 + a2*b0
    c3 = a0*b3 + a1*b2 + a2*b1 + a3*b0
    c4 = a1*b3 + a2*b2 + a3*b1
    c5 = a2*b3 + a3*b2
    c6 = a3*b3
    # reduce: z^4 = z^2 - 1, z^5 = z^3 - z, z^6 = -1
    return (c0 - c4 - c6, c1 - c5, c2 + c4, c3 + c5)


def zconj(u):
    # conj: z -> z^{11} = z - z^3, z^2 -> z^{10} = 1 - z^2, z^3 -> z^9 = -z^3
    a, b, c, d = u
    return (a + c, b, -c, -b - d)


ZETA_INV = zconj(ZETA)          # (0, 1, 0, -1)
assert zmul(ZETA, ZETA_INV) == Z1
assert zmul(ZETA, zmul(ZETA, zmul(ZETA, ZETA))) == zsub((0, 0, 1, 0), Z1)  # z^4 = z^2-1

# --- the witness identity x^2 + x*^2 = N = t^2, i.e. zeta^2 + zeta^{-2} = 1 ------------
assert zadd(zmul(ZETA, ZETA), zmul(ZETA_INV, ZETA_INV)) == Z1

# ======================================================================================
# real/imag decomposition over Q(sqrt3):  Re/Im of (a,b,c,d) are (u + v*sqrt3)/2.
#   1 = (1, 0);  z = (sqrt3 + i)/2;  z^2 = (1 + i sqrt3)/2;  z^3 = i.
# Returns ((2*Re as (u,v) int pair meaning (u+v*sqrt3))/... ) -- we return EXACT doubled
# parts: re2 = 2*Re = (2a + c) + b*sqrt3, im2 = 2*Im = (b + 2d) + c*sqrt3.
# ======================================================================================
def z_reim2(u):
    a, b, c, d = u
    return ((2*a + c, b), (b + 2*d, c))    # each pair (p, q) means p + q*sqrt3


# ======================================================================================
# t-polynomials: 3-tuples of Z[zeta] elements (coefficients of t^0, t^1, t^2).
# ======================================================================================
P0 = (Z0, Z0, Z0)


def padd(P, Q): return (zadd(P[0], Q[0]), zadd(P[1], Q[1]), zadd(P[2], Q[2]))
def psub(P, Q): return (zsub(P[0], Q[0]), zsub(P[1], Q[1]), zsub(P[2], Q[2]))
def pzero(P):   return P[0] == Z0 and P[1] == Z0 and P[2] == Z0


def peval_scaled(P, p, q):
    """P(p/q) * q^2 as an EXACT Z[zeta] element (zero iff P(p/q) = 0, q > 0)."""
    return zadd(zadd(zscale(q*q, P[0]), zscale(p*q, P[1])), zscale(p*p, P[2]))


# ======================================================================================
# The alphabet {0, +-1, +-x, +-x*} with x = t*zeta, x* = t*zeta^{-1}.
# Symbol -> (Z[zeta] coefficient, t-exponent).
# ======================================================================================
SYMS = ["0", "1", "-1", "X", "-X", "Y", "-Y"]      # Y := x* (conjugate symbol)
SYMVAL = {
    "0":  (Z0, 0),
    "1":  (Z1, 0),
    "-1": (zneg(Z1), 0),
    "X":  (ZETA, 1),
    "-X": (zneg(ZETA), 1),
    "Y":  (ZETA_INV, 1),
    "-Y": (zneg(ZETA_INV), 1),
}


def sym_prod(su, sv, conj_left=False):
    """(coeff_u [conjugated], coeff_v) product as a t-polynomial (degree e_u + e_v)."""
    cu, eu = SYMVAL[su]
    cv, ev = SYMVAL[sv]
    if conj_left:
        cu = zconj(cu)
    w = zmul(cu, cv)
    out = [Z0, Z0, Z0]
    out[eu + ev] = w
    return tuple(out)


def hdot_pol(u, v):
    """Hermitian dot <u, v> = sum conj(u_c) v_c as a t-polynomial."""
    P = P0
    for su, sv in zip(u, v):
        P = padd(P, sym_prod(su, sv, conj_left=True))
    return P


def cross_pol(u, v):
    """Cross product entries (t-polynomials); all three zero identically <=> proportional
    over the function field Q(zeta)(t)."""
    def m(si, sj):
        return sym_prod(si, sj, conj_left=False)
    return (psub(m(u[1], v[2]), m(u[2], v[1])),
            psub(m(u[2], v[0]), m(u[0], v[2])),
            psub(m(u[0], v[1]), m(u[1], v[0])))


def same_formal(u, v):
    return all(pzero(P) for P in cross_pol(u, v))


def build_stable_pool():
    """Rays over the 7-symbol alphabet, identified iff proportional IDENTICALLY in t.
    Returns (rays, edges, triads): edges = Hermitian dot vanishes identically in t;
    triads = triangles of the edge set (in C^3 a triangle is a full basis)."""
    rays = []
    for vec in product(SYMS, repeat=3):
        if all(s == "0" for s in vec):
            continue
        if not any(same_formal(vec, r) for r in rays):
            rays.append(vec)
    n = len(rays)
    E = [(i, j) for i in range(n) for j in range(i+1, n)
         if pzero(hdot_pol(rays[i], rays[j]))]
    Es = set(E)
    T = [t for t in combinations(range(n), 3)
         if all((a, b) in Es for a, b in combinations(t, 2))]
    return rays, E, T


# ======================================================================================
# K = Q(sqrt3) exact field arithmetic: elements (p, q) = p + q*sqrt3, p,q Fractions.
# ======================================================================================
KZERO = (Fraction(0), Fraction(0))
KONE = (Fraction(1), Fraction(0))


def kadd(x, y): return (x[0]+y[0], x[1]+y[1])
def ksub(x, y): return (x[0]-y[0], x[1]-y[1])
def kneg(x):    return (-x[0], -x[1])
def kmul(x, y): return (x[0]*y[0] + 3*x[1]*y[1], x[0]*y[1] + x[1]*y[0])
def kscale(f, x): return (f*x[0], f*x[1])
def kzero(x):   return x[0] == 0 and x[1] == 0


def kinv(x):
    d = x[0]*x[0] - 3*x[1]*x[1]     # nonzero for x != 0 (sqrt3 irrational)
    return (x[0]/d, -x[1]/d)


def ksign(x):
    """Exact sign of p + q*sqrt3."""
    p, q = x
    if p == 0 and q == 0:
        return 0
    if p >= 0 and q >= 0:
        return 1
    if p <= 0 and q <= 0:
        return -1
    # opposite signs: sign(p + q sqrt3) = sign(p) * sign(p^2 - 3 q^2)
    s = p*p - 3*q*q
    return (1 if p > 0 else -1) * (1 if s > 0 else -1)


def krank(rows):
    """Exact rank over K = Q(sqrt3) by Gaussian elimination (rows: lists of K elements)."""
    rows = [list(r) for r in rows if any(not kzero(x) for x in r)]
    if not rows:
        return 0
    ncols = len(rows[0])
    rank = 0
    for col in range(ncols):
        piv = None
        for r in range(rank, len(rows)):
            if not kzero(rows[r][col]):
                piv = r
                break
        if piv is None:
            continue
        rows[rank], rows[piv] = rows[piv], rows[rank]
        prow = rows[rank]
        pinv = kinv(prow[col])
        for r in range(rank + 1, len(rows)):
            f = rows[r][col]
            if not kzero(f):
                m = kmul(f, pinv)
                rows[r] = [ksub(a, kmul(m, b)) for a, b in zip(rows[r], prow)]
        rank += 1
        if rank == len(rows):
            break
    return rank
