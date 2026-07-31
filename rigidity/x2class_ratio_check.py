# Independent check of the RATIO LEMMA.
# Claim: for the Hermitian-closed pool over {0,+-1,+-x,+-x*} in C^3, the degree-one subpool
# D(x) (rays with all entries in {0,+-x,+-x*}) rescaled projectively by 1/x* is isomorphic
# as an orthogonality graph to the pool over {0,+-1,+-r}, r = x/x*.  |r| = 1 always.
# Consequence: D(x) is KS-uncolorable iff r is a primitive 3rd or 6th root of unity.
#
# Checked here symbolically (exact, sympy) for arbitrary x, then numerically-exactly at
# several named points including the two counterexamples.
import sympy as sp
from itertools import combinations, product

p, q = sp.symbols('p q', real=True)
x = p + sp.I*q
xs = sp.conjugate(x)
N = sp.expand(x*xs)

# --- Leg 1: the rescaling multiplies every Hermitian inner product by N (so preserves zeros).
# A degree-one ray has entries a_c * x or a_c * x*, a_c in {0,+-1}. Rescale by 1/x*:
#   x  -> x/x*  = r ;   x* -> 1.
# For two degree-one rays u, v the inner product is sum conj(u_c) v_c.
# After rescaling u -> u/x*, v -> v/x*, the inner product becomes
#   sum conj(u_c/x*) (v_c/x*) = (1/(x* conj(x*))) sum conj(u_c) v_c = (1/N) <u,v>.
print("Leg 1 (scaling factor): conj(x*)*x* =", sp.simplify(sp.conjugate(xs)*xs), " = N =", sp.simplify(N))
print("  => <u/x*, v/x*> = <u,v> / N, and N != 0 for x != 0, so ZEROS ARE PRESERVED. Exact.\n")

# --- Leg 2: r = x/x* is unimodular, and identify when it is a primitive 3rd/6th root.
r = sp.simplify(x/xs)
print("Leg 2: |r|^2 =", sp.simplify(sp.expand(r*sp.conjugate(r))), "(must be 1)\n")

def ratio_of(xv):
    xv = sp.nsimplify(xv)
    return sp.simplify(xv/sp.conjugate(xv))

named = {
    'sqrt3 + i        (L30, our counterexample)': sp.sqrt(3) + sp.I,
    '2 + omega3       (our 2nd counterexample)': sp.Rational(3,2) + sp.I*sp.sqrt(3)/2,
    '2*(sqrt3 + i)    (L30, other modulus)':      2*(sp.sqrt(3) + sp.I),
    '1 + i*sqrt3      (L60)':                     1 + sp.I*sp.sqrt(3),
    '1 + i            (arg 45, sterile)':         1 + sp.I,
    '3 + i*sqrt7      (H+ generic)':              sp.Rational(3,2) + sp.I*sp.sqrt(7)/2,
    '1 + 2i           (generic avoiding)':        1 + 2*sp.I,
    '(3+4i)/5         (unimodular)':              sp.Rational(3,5) + sp.Rational(4,5)*sp.I,
}
print("Leg 3: the ratio r = x/x* and its order as a root of unity")
for name, xv in named.items():
    rv = ratio_of(xv)
    order = None
    for k in range(1, 25):
        if sp.simplify(rv**k - 1) == 0:
            order = k; break
    print(f"  {name:42s} r = {sp.nsimplify(sp.simplify(rv))!s:28s} ord = {order}")
print("\n  Predicted uncolorable  <=>  ord(r) in {3, 6}.")
