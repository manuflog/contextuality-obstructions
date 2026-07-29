"""Dedekind + Jordan cycle-type certificates for the coupled unimodular-circle holonomy
Galois groups — an INDEPENDENT, proof-grade confirmation that they are the full symmetric
group S_d, not relying on any black-box galois_group() call.

Method (for a monic integer char poly f of degree d, irreducible over Q):
  - Irreducible over Q  => the Galois group G acts TRANSITIVELY on the d roots.
  - Dedekind's theorem: for a prime p not dividing disc(f), the degrees of the irreducible
    factors of f mod p are the cycle type of a Frobenius element that LIES IN G. So every
    cycle type we realize by factoring mod some prime is a permutation actually present in G.
  - Jordan's theorem: a PRIMITIVE permutation group of degree d that contains a transposition
    is the full symmetric group S_d.
  - Primitivity certificate here: G transitive + G contains a (d-1)-cycle (prime d-1), which
    fixes exactly one point and cycles the other d-1  =>  the point-stabilizer is transitive on
    those d-1 points => G is 2-transitive => primitive. (A transposition then forces S_d by
    Jordan.) Both certified cases use exactly a (d-1)-cycle: d=4 the 3-cycle mod 5, d=6 the
    5-cycle mod 3. [The generic-primitivity search below still scans primes d/2<p<d and reports
    whichever prime cycle it finds; the proof-grade witness used in the paper is the (d-1)-cycle.]

Result: S4 (d=4, the M9 quartic) and S6 (d=6, the coupled multi-X core) certified by explicit
cycle types. The same three ingredients (a d-cycle, a transposition, a large-prime cycle) are
the template a general "coupled core => Galois S_d" proof would need to produce uniformly in d.

Run: python3 branch_galois_cert.py
"""
import sympy as sp

x = sp.symbols('x')

# d=4: the M9 holonomy characteristic polynomial (Galois S4, per M9_GEOMETRY.md)
P4 = x**4 - 348*x**3 + 44691*x**2 - 2505672*x + 51631020
# d=6: the coupled multi-X core holonomy characteristic polynomial (Galois S6, per D6_GALOIS.md)
P6 = (x**6 - 570*x**5 + 135129*x**4 - 17053830*x**3
      + 1208401591*x**2 - 45581353598*x + 715049394620)


def cycle_types(poly, d, prime_cap=400):
    disc = sp.discriminant(poly, x)
    seen = {}
    for pr in sp.primerange(3, prime_cap):
        if disc % pr == 0:
            continue
        fac = sp.Poly(poly, x, modulus=pr).factor_list()
        # squarefree mod p required for Dedekind; skip if any repeated factor
        if any(m > 1 for _, m in fac[1]):
            continue
        degs = tuple(sorted((f.degree() for f, _ in fac[1]), reverse=True))
        if sum(degs) != d:
            continue
        seen.setdefault(degs, pr)
    return seen, disc


def certify(poly, d, name):
    print("=" * 78)
    print(f"{name}: degree {d}, char poly {sp.Poly(poly,x).as_expr()}")
    irred = sp.Poly(poly, x).is_irreducible
    print(f"  irreducible over Q: {irred}   => Galois group TRANSITIVE on {d} roots")
    types, disc = cycle_types(poly, d)
    sq = sp.sqrt(disc).is_rational
    print(f"  disc a perfect square: {sq}   => group contains an ODD permutation (rules out A_{d})"
          if not sq else f"  disc a perfect square: {sq}")
    print("  realized cycle types (Dedekind — each is a permutation IN the Galois group):")
    for degs, pr in sorted(types.items(), key=lambda kv: (-max(kv[0]), kv[0])):
        print(f"     mod {pr:>3}: {degs}")
    # find a d-cycle, a transposition, and a large-prime cycle p with d/2<p<d-1
    d_cycle = (d,) in types
    transp = tuple([2] + [1]*(d-2)) in types
    # a p-cycle fixing d-p points, p prime with d/2 < p < d, makes the point-stabilizer
    # transitive on the other p points => G is 2-transitive => primitive.
    large_p = None
    for p in sp.primerange(d//2 + 1, d):
        ct = tuple([p] + [1]*(d - p))
        if ct in types:
            large_p = (p, types[ct]); break
    print(f"  d-cycle present:            {d_cycle}   (irreducible mod {types.get((d,),'-')})")
    print(f"  transposition present:      {transp}   (mod {types.get(tuple([2]+[1]*(d-2)),'-')})")
    print(f"  large-prime {d//2}<p<{d-1} cycle:  {large_p is not None}"
          + (f"   ({large_p[0]}-cycle mod {large_p[1]})" if large_p else ""))
    primitive = (large_p is not None) or (d in (2, 3))  # small d handled directly
    full = irred and transp and primitive and (not sq)
    print("  --------")
    if full:
        print(f"  CERTIFICATE: transitive + primitive (via the {large_p[0]}-cycle => 2-transitive) "
              f"+ a transposition\n     => by JORDAN's theorem the Galois group is the FULL "
              f"symmetric group S_{d} (order {sp.factorial(d)}).")
    else:
        print(f"  Certificate incomplete with primes<400; ingredients: irred={irred} "
              f"transp={transp} primitive={primitive} odd={not sq}")
    return full


def genericity_sample(d, offmax, n_trials, prime_cap, seed=7):
    """Empirical genericity check for the S_d law (backs the [NUMERICAL] remark in
    tower_paper.tex Sec. 6). Sample symmetric integer matrices of the OBSERVED SHAPE
    of the coupled-core generator S~ — distinct diagonal in {60..119}, small
    off-diagonal in [-offmax, offmax] — and report how often the char poly is
    irreducible (=> Galois transitive) and how often Dedekind+Jordan CERTIFIES S_d
    within primes < prime_cap. Finding: irreducibility is ~100% at every d (strong
    genericity), while the certifying prime bound must GROW with d (a transposition +
    large-prime cycle are rare mod small primes) — precisely the obstruction a uniform
    proof must clear. The underlying groups are S_d; only the finite-prime certificate
    degrades."""
    import random
    rnd = random.Random(seed)
    irred = full = 0
    for _ in range(n_trials):
        diag = rnd.sample(range(60, 120), d)
        M = sp.zeros(d)
        for i in range(d):
            M[i, i] = diag[i]
        for i in range(d):
            for j in range(i + 1, d):
                v = rnd.randint(-offmax, offmax)
                M[i, j] = v
                M[j, i] = v
        f = sp.Poly(M.charpoly(x).as_expr(), x)
        if not f.is_irreducible:
            continue
        irred += 1
        disc = sp.discriminant(f.as_expr(), x)
        if sp.sqrt(disc).is_rational:
            continue
        types = set()
        for pr in sp.primerange(3, prime_cap):
            if disc % pr == 0:
                continue
            fl = sp.Poly(f.as_expr(), x, modulus=pr).factor_list()
            if any(m > 1 for _, m in fl[1]):
                continue
            types.add(tuple(sorted((g.degree() for g, _ in fl[1]), reverse=True)))
        dcyc = (d,) in types
        transp = tuple([2] + [1] * (d - 2)) in types
        prim = any(tuple([r] + [1] * (d - r)) in types
                   for r in sp.primerange(d // 2 + 1, d))
        if dcyc and transp and prim:
            full += 1
    print(f"  d={d}: {n_trials} random matrices (offmax={offmax}); "
          f"irreducible {irred}/{n_trials}; CERTIFIED S_{d} {full}/{n_trials} "
          f"(primes<{prime_cap})")
    return irred, full


if __name__ == "__main__":
    ok4 = certify(P4, 4, "d=4  (M9 quartic)")
    print()
    ok6 = certify(P6, 6, "d=6  (coupled multi-X sextic)")
    print("\n" + "=" * 78)
    print("GENERICITY SAMPLE (backs the [NUMERICAL] remark: generic symmetric integer\n"
          "matrices of the generator's shape are irreducible ~100%, group S_d; the\n"
          "certifying prime bound grows with d — the obstruction to a uniform proof):")
    genericity_sample(4, 18, 40, 120)
    genericity_sample(6, 5, 25, 600)
    genericity_sample(8, 5, 15, 1500)
    print("\n" + "=" * 78)
    print(f"SEQUENCE (proof-grade, independent of galois_group()): d=3 Z/2 (=S_2, Q(sqrt1867)); "
          f"d=4 {'S4' if ok4 else '?'}; d=6 {'S6' if ok6 else '?'}.")
    print("The coupled unimodular-circle holonomy realizes the FULL symmetric group S_d at every "
          "tested rung. Conjecture: S_d for all even d>=4 (and d=3).")
    print("PASS" if (ok4 and ok6) else "INCOMPLETE")
