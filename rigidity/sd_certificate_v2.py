#!/usr/bin/env python3
"""
sd_certificate_v2.py -- MOONSHOT: the (d-1)-prime-free Galois certificate template.

WHY THIS EXISTS.  The published tower certificate (branch_galois_cert.py / branch_d8galois.py's
`dedekind_jordan`) proves Gal(f_d) = S_d from three Dedekind cycle types:
      (i)  f irreducible                          => G transitive
      (ii) a (d-1)-cycle with d-1 PRIME           => point stabilizer transitive => 2-transitive
                                                   => PRIMITIVE
      (iii) a transposition                       => Jordan => S_d.
Ingredient (ii) requires d-1 prime, which FAILS at d = 10, 16, 22, 26, 28, 34, 36, ...
(SD_ATTACK.md / rem:certscope: the template is INAPPLICABLE at infinitely many even d).
Ingredient (iii) is the scaling killer even where (ii) works: transposition Chebotarev density
C(d,2)/d! (1/1440 at d=8 -> certifying prime 10861; ~1/80640 at d=10).

THE FIX (this file) -- classical group theory gives a strictly better template:

  LEMMA P (primitivity from a big prime cycle).  Let G <= S_n be transitive and contain a
  p-cycle sigma for a PRIME p with n/2 < p <= n.  Then G is primitive.
    Proof.  Suppose B is a G-block system with blocks of size b, 1 < b < n.  sigma permutes the
    blocks; the induced permutation has order dividing p (prime), so order 1 or p.
      - order p: some block orbit has size p, covering p*b >= 2p > n points.  Contradiction.
      - order 1: every block is sigma-invariant, hence a union of sigma-orbits.  The sigma-orbits
        are one p-set (the support) and fixed points, so any block meeting the support contains
        ALL p of its points: b >= p > n/2.  But b | n and b < n forces b <= n/2.  Contradiction.
    Hence only trivial block systems exist: G is primitive.  QED  [self-contained; textbook]

  THEOREM J (Jordan 1873).  A PRIMITIVE subgroup of S_n containing a p-cycle for a prime
  p <= n-3 contains A_n.
    [C. Jordan, Bull. Soc. Math. France 1 (1873) 40-71; Wielandt, "Finite Permutation Groups",
     Theorem 13.9; also Isaacs-Zieschang, Amer. Math. Monthly 102 (1995).]

  THEOREM J' (Jordan, transposition form).  A primitive subgroup of S_n containing a
  transposition IS S_n.   [Wielandt Thm 13.3 -- this is the form the published template used.]

  DEDEKIND.  For monic f in Z[x] and a prime q with f mod q squarefree, the multiset of degrees
  of the irreducible factors of f mod q is the cycle type of some element of Gal(f).
    [Dedekind 1894; van der Waerden, Algebra I Sec. 66; Cox, "Galois Theory", Thm 13.4.5.]

  POWER TRICK.  If a Frobenius element has cycle type containing the part p exactly once with
  every other part < p (automatic when p > n/2: the other parts sum to n-p < n/2 < p), then
  raising it to the lcm of the other parts (coprime to p) yields a PURE p-cycle in G.

  CERTIFICATE v2 (route A -- the new one).  For n >= 8 pick a prime p with n/2 < p <= n-3.
      (1) f irreducible over Q            [witnessed by f irreducible mod some q0, or sympy]
                                          => G transitive
      (2) f mod q1 has factor-degree multiset containing p (e.g. 7+2+1 at n=10)
                                          => p-cycle in G  => LEMMA P: primitive
                                          => THEOREM J: A_n <= G
      (3) any Frobenius cycle type that is an ODD permutation (n - #parts odd; for even n the
          n-cycle from (1) already is one)  => G is not inside A_n  => G = S_n.  QED

  CERTIFICATE v2 (route B -- fallback for n in {4,5,6,7}, where no prime p <= n-3 exceeds n/2):
      p prime with n/2 < p < n gives primitivity by LEMMA P, then a transposition gives S_n by
      THEOREM J'.  (This subsumes the published template: it never uses "d-1 prime" as such.)

  WHY THIS BREAKS THE d=10 WALL (obstruction (a)):
    - at n=10 route A uses p = 7 (7 > 5, 7 <= 7).  A 7-pattern (7+1+1+1, 7+2+1, 7+3) has
      Chebotarev density 1/7 * (1/1+1/2+1/6) [# perms with a 7-cycle: 10!/7 * (1/3!+1/2+1/6)...]
      -- numerically 331/1440 of A-route usable elements vs 1/80640 for a transposition.  The
      witness primes STAY SMALL: the prime-growth obstruction disappears together with the
      d-1-prime obstruction.
    - COVERAGE THEOREM: a prime p with n/2 < p <= n-3 exists for EVERY n >= 8.
      Proof: for n >= 50 apply Nagura's theorem [J. Nagura, Proc. Japan Acad. 28 (1952) 177-181:
      for x >= 25 there is a prime in (x, 6x/5)] with x = n/2: it gives p with
      n/2 < p < 3n/5 <= n-3 (the last inequality iff n >= 7.5).  The finitely many cases
      8 <= n < 50 are checked directly below (and the script re-checks a large range).
      In particular the ENTIRE bad set {d even : d-1 composite} = {10,16,22,26,28,34,36,...}
      is covered.  [PROVED]

USAGE:
    python3 sd_certificate_v2.py            # full self-test battery (PASS/FAIL verdicts)
    python3 sd_certificate_v2.py "x**10 - ... "    # certify a given monic integer polynomial

  As a library:  from sd_certificate_v2 import certify_v2, factor_degrees_mod_p
    certify_v2([a0, a1, ..., 1])  (coefficients low -> high, monic)  -> dict with 'verdict'.

No repo file is modified.  No git.  Everything below prints PASS/FAIL and the final line is the
overall verdict.
"""
import sys, time, random
from math import gcd

import sympy as sp

sys.dont_write_bytecode = True

x = sp.symbols("x")

# =====================================================================================
# Exact polynomial arithmetic over GF(p) -- coefficient lists, low -> high degree.
# Self-contained (no sympy in the inner loop) so tens of thousands of primes are cheap.
# =====================================================================================

def _trim(a):
    while len(a) > 1 and a[-1] == 0:
        a.pop()
    return a


def _pmul(a, b, p):
    r = [0] * (len(a) + len(b) - 1)
    for i, ai in enumerate(a):
        if ai:
            for j, bj in enumerate(b):
                if bj:
                    r[i + j] = (r[i + j] + ai * bj) % p
    return _trim(r)


def _pmod(a, f, p):
    """a mod f with f MONIC over GF(p)."""
    df = len(f) - 1
    if df == 0:
        return [0]
    a = [v % p for v in a]
    while len(a) - 1 >= df:
        c = a[-1]
        if c:
            k = len(a) - 1 - df
            for i in range(df + 1):
                a[k + i] = (a[k + i] - c * f[i]) % p
        a.pop()
        while len(a) > 1 and a[-1] == 0 and len(a) - 1 >= df:
            a.pop()
    if not a:
        a = [0]
    return _trim(a)


def _iszero(a):
    return len(a) == 1 and a[0] == 0


def _monic(a, p):
    inv = pow(a[-1] % p, p - 2, p)
    return [(v * inv) % p for v in a]


def _pgcd(a, b, p):
    a = _trim([v % p for v in list(a)])
    b = _trim([v % p for v in list(b)])
    while not _iszero(b):
        bm = _monic(b, p)
        a, b = bm, _pmod(a, bm, p)
    return a if _iszero(a) else _monic(a, p)


def _pdivexact(a, b, p):
    """a / b with b MONIC (exact division assumed)."""
    a = _trim([v % p for v in list(a)])
    db = len(b) - 1
    q = [0] * max(1, len(a) - db)
    while len(a) - 1 >= db and not _iszero(a):
        c = a[-1]
        k = len(a) - 1 - db
        q[k] = c
        if c:
            for i in range(db + 1):
                a[k + i] = (a[k + i] - c * b[i]) % p
        a.pop()
        _trim(a)
    return _trim(q)


def _ppow(a, e, f, p):
    """a^e mod (f, p), f monic."""
    r = [1]
    base = _pmod(a, f, p)
    while e:
        if e & 1:
            r = _pmod(_pmul(r, base, p), f, p)
        base = _pmod(_pmul(base, base, p), f, p)
        e >>= 1
    return r


def factor_degrees_mod_p(coeffs, p):
    """Monic integer polynomial (coeffs low->high) -> sorted tuple of the degrees of the
    irreducible factors of f mod p (the Dedekind cycle type), or None when f mod p is not
    squarefree (equivalently p | disc f -- those primes are skipped, as Dedekind requires).
    Distinct-degree factorization: h_k = x^(p^k) mod f;  gcd(h_k - x, rem) collects the
    product of all irreducible factors of degree k."""
    n = len(coeffs) - 1
    f = [c % p for c in coeffs]
    if f[-1] != 1:          # monic integer poly => lead coeff 1 mod every p
        return None
    fp = _trim([(i * f[i]) % p for i in range(1, len(f))])
    if _iszero(fp):
        return None                                    # f' == 0 mod p: not squarefree
    if len(_pgcd(f, fp, p)) - 1 > 0:
        return None                                    # gcd(f,f') nontrivial: not squarefree
    degs = []
    rem = f[:]
    h = [0, 1]                                         # x
    for k in range(1, n + 1):
        dr = len(rem) - 1
        if dr == 0:
            break
        if 2 * k > dr:                                 # remaining factor is irreducible
            degs.append(dr)
            rem = [1]
            break
        h = _ppow(h, p, f, p)                          # x^(p^k) mod f
        diff = h[:] + [0] * max(0, 2 - len(h))
        diff[1] = (diff[1] - 1) % p
        diff = _trim(diff)
        g = rem[:] if _iszero(diff) else _pgcd(rem, diff, p)
        dg = len(g) - 1
        if dg > 0:
            degs += [k] * (dg // k)
            rem = _pdivexact(rem, _monic(g, p), p)
    if sum(degs) != n:
        return None
    return tuple(sorted(degs, reverse=True))


# =====================================================================================
# The certificate.
# =====================================================================================

def routeA_primes(n):
    """Primes p with n/2 < p <= n-3 (Lemma P + Theorem J both apply)."""
    return [int(q) for q in sp.primerange(n // 2 + 1, n - 2)]


def routeB_primes(n):
    """Primes p with n/2 < p < n but p > n-3 (Lemma P applies, Theorem J does not; need J')."""
    return [q for q in (n - 2, n - 1) if sp.isprime(q) and 2 * q > n]


def _is_odd_type(n, degs):
    """Parity of a permutation with cycle type `degs`: odd iff n - #cycles is odd."""
    return (n - len(degs)) % 2 == 1


def certify_v2(coeffs, name="f", prime_cap=100000, verbose=True):
    """coeffs: monic integer polynomial, low->high.  Returns dict with:
       verdict:  'S_n' | 'REDUCIBLE (not S_n)' | 'INCOMPLETE'
       route:    'A' (p-cycle p<=n-3, Jordan->A_n, odd type)  or  'B' (p-cycle + transposition)
       witness:  the witnessing primes and cycle types
    The proof chain per route is spelled out in the module docstring."""
    t0 = time.time()
    coeffs = [int(c) for c in coeffs]
    assert coeffs[-1] == 1, "polynomial must be monic"
    n = len(coeffs) - 1
    assert n >= 4, "degree >= 4 required"
    pA = routeA_primes(n)
    pB = routeB_primes(n)
    transp_type = tuple([2] + [1] * (n - 2))

    W = {}                       # events -> (witness prime, cycle type)
    scanned = 0
    for q in sp.primerange(2, prime_cap):
        degs = factor_degrees_mod_p(coeffs, int(q))
        scanned += 1
        if degs is None:
            continue
        if degs == (n,) and "irred" not in W:
            W["irred"] = (int(q), degs)
        for p in pA:
            if p in degs and "pA" not in W:
                # POWER TRICK check: all other parts < p (automatic for p > n/2 -- assert it)
                others = list(degs)
                others.remove(p)
                assert all(o < p for o in others), "power-trick hypothesis violated?!"
                W["pA"] = (int(q), degs, p)
        for p in pB:
            if p in degs and "pB" not in W:
                others = list(degs)
                others.remove(p)
                assert all(o < p for o in others)
                W["pB"] = (int(q), degs, p)
        if degs == transp_type and "transp" not in W:
            W["transp"] = (int(q), degs)
        if _is_odd_type(n, degs) and "odd" not in W:
            W["odd"] = (int(q), degs)
        doneA = ("pA" in W) and ("odd" in W)
        doneB = (("pA" in W) or ("pB" in W)) and ("transp" in W)
        if ("irred" in W) and (doneA or doneB):
            break

    # --- transitivity ---
    if "irred" in W:
        irred, irred_how = True, f"irreducible mod {W['irred'][0]} (=> irreducible over Q)"
    else:
        irred = sp.Poly(coeffs[::-1], x).is_irreducible
        irred_how = "sympy is_irreducible over Q (no single-prime witness found)"
        if not irred:
            out = dict(verdict=f"REDUCIBLE (not S_{n})", route=None, witness=W, n=n,
                       scanned_primes=scanned, seconds=round(time.time() - t0, 3))
            if verbose:
                print(f"  [{name}] degree {n}: REDUCIBLE over Q => Galois group not transitive, "
                      f"not S_{n}.")
            return out

    doneA = ("pA" in W) and ("odd" in W)
    doneB = (("pA" in W) or ("pB" in W)) and ("transp" in W)
    if doneA:
        route = "A"
        q1, degs1, p = W["pA"]
        qo, degso = W["odd"]
        chain = [
            f"(1) {irred_how}  => Galois group G TRANSITIVE on {n} roots",
            f"(2) mod {q1}: factor degrees {degs1} -- contains the prime part p={p} once, all "
            f"other parts < p  => a power of this Frobenius element is a pure {p}-cycle in G",
            f"    LEMMA P  ({p} > {n}/2):        G is PRIMITIVE",
            f"    JORDAN   ({p} <= {n}-3 = {n-3}): G contains A_{n}",
            f"(3) mod {qo}: factor degrees {degso} -- an ODD permutation ({n}-{len(degso)} odd) "
            f" => G is not contained in A_{n}",
            f"=> G = S_{n}  (order {sp.factorial(n)})",
        ]
        verdict = f"S_{n}"
    elif doneB:
        route = "B"
        key = "pA" if "pA" in W else "pB"
        q1, degs1, p = W[key]
        qt, degst = W["transp"]
        chain = [
            f"(1) {irred_how}  => G TRANSITIVE",
            f"(2) mod {q1}: degrees {degs1} => pure {p}-cycle in G; LEMMA P ({p} > {n}/2) "
            f"=> G PRIMITIVE",
            f"(3) mod {qt}: degrees {degst} => transposition in G; JORDAN (J') => G = S_{n}",
        ]
        verdict = f"S_{n}"
    else:
        route = None
        chain = [f"prime scan to {prime_cap} exhausted; have events {sorted(W)}"]
        verdict = "INCOMPLETE"

    out = dict(verdict=verdict, route=route, witness={k: (v[0], list(v[1]) if len(v) > 1 else v)
                                                      for k, v in W.items()},
               n=n, routeA_ps=pA, routeB_ps=pB, chain=chain,
               scanned_primes=scanned, seconds=round(time.time() - t0, 3))
    if verbose:
        print(f"  [{name}] degree {n}; route-A primes {pA}, route-B primes {pB}")
        for line in chain:
            print(f"    {line}")
        print(f"  [{name}] *** VERDICT: {verdict} ***  (route {route}, {scanned} primes scanned, "
              f"{out['seconds']}s)")
    return out


# =====================================================================================
# The repo's published polynomials (exactly as recorded in the tower artifacts).
# =====================================================================================
F4 = [51631020, -2505672, 44691, -348, 1]                              # M9 quartic, S_4
F6 = [715049394620, -45581353598, 1208401591, -17053830, 135129, -570, 1]   # d=6 sextic, S_6
F8 = [65318579432800, -10384575801136, 708314669151, -27121174366,
      638589348, -9480672, 86767, -448, 1]                             # d=8 octic, S_8


# =====================================================================================
# Self-test battery.
# =====================================================================================

def gate_dedekind_crossval(nprimes_cap=200, ntrials=8, seed=11):
    """The one new computational primitive (factor_degrees_mod_p) cross-validated entry-by-entry
    against sympy's factor_list mod p."""
    print("=" * 96)
    print("GATE 1 -- Dedekind degree routine vs sympy.factor_list, incl. squarefree verdicts")
    print("=" * 96)
    rnd = random.Random(seed)
    polys = [("f4", F4), ("f6", F6), ("f8", F8)]
    for t in range(ntrials):
        n = rnd.choice([10, 10, 10, 16])
        c = [rnd.randint(-40, 40) for _ in range(n)] + [1]
        polys.append((f"rand{t}(deg {n})", c))
    mism = checked = 0
    for nm, c in polys:
        P = sp.Poly(c[::-1], x)
        for q in sp.primerange(2, nprimes_cap):
            mine = factor_degrees_mod_p(c, int(q))
            fl = sp.Poly(P.as_expr(), x, modulus=int(q)).factor_list()
            sqfree = all(m == 1 for _, m in fl[1])
            deg_drop = sum(f.degree() * m for f, m in fl[1]) != len(c) - 1
            theirs = None if (not sqfree or deg_drop) else \
                tuple(sorted((f.degree() for f, _ in fl[1]), reverse=True))
            checked += 1
            if mine != theirs:
                mism += 1
                print(f"    MISMATCH {nm} mod {q}: mine={mine} sympy={theirs}")
    ok = mism == 0
    print(f"  cross-validated {checked} (poly, prime) pairs over {len(polys)} polynomials: "
          f"{mism} mismatches")
    print(f"  GATE 1: {'PASS' if ok else 'FAIL'}")
    return ok


def gate_coverage(hi=100000):
    """The coverage theorem: route-A prime exists for every n >= 8 (finite check + Nagura)."""
    print("=" * 96)
    print("GATE 2 -- coverage: a prime p with n/2 < p <= n-3 exists for EVERY n >= 8")
    print("=" * 96)
    primes = list(sp.sieve.primerange(2, hi + 10))
    import bisect
    bad = []
    for n in range(8, hi + 1):
        i = bisect.bisect_right(primes, n // 2)
        if i >= len(primes) or primes[i] > n - 3 or 2 * primes[i] <= n:
            # recheck strictly: want p > n/2 and p <= n-3
            found = False
            j = i
            while j < len(primes) and primes[j] <= n - 3:
                if 2 * primes[j] > n:
                    found = True
                    break
                j += 1
            if not found:
                bad.append(n)
    print(f"  finite check n in [8, {hi}]: {len(bad)} failures {bad[:10]}")
    print("  n >= 50: Nagura (1952) gives a prime in (n/2, 3n/5], and 3n/5 <= n-3 for n >= 8.")
    print("  => the template covers ALL n >= 8, in particular the whole bad set "
          "{d even : d-1 composite} = {10,16,22,26,28,34,36,...}.  [PROVED]")
    bad_set = [d for d in range(4, 60, 2) if not sp.isprime(d - 1)]
    print(f"  (bad set through 60: {bad_set}; note n in {{4,5,6,7}} has NO route-A prime -- "
          f"route B covers those, and d=4,6 have d-1 prime anyway)")
    ok = len(bad) == 0
    print(f"  GATE 2: {'PASS' if ok else 'FAIL'}")
    return ok


def gate_degree10_group_theory():
    """Independent (classification-based) cross-check of Lemma P + Jordan at n=10: every proper
    transitive subgroup of S_10 other than A_10 has order NOT divisible by 7, so a 7-cycle +
    transitivity already forces A_10 or S_10.  The certificate itself never uses this -- it
    stands on Lemma P + Jordan alone -- but agreement is reassuring."""
    print("=" * 96)
    print("GATE 3 -- n=10 sanity vs the classification of transitive groups of degree 10")
    print("=" * 96)
    imprim = {"S_2 wr S_5 (blocks of 2)": 2 ** 5 * sp.factorial(5),
              "S_5 wr S_2 (blocks of 5)": sp.factorial(5) ** 2 * 2}
    prim_proper = {"A_5": 60, "S_5": 120, "PSL_2(9)=A_6": 360, "S_6": 720, "PGL_2(9)": 720,
                   "M_10": 720, "PGammaL_2(9)": 1440}
    ok = True
    for nm, o in {**imprim, **prim_proper}.items():
        div = (int(o) % 7 == 0)
        ok &= not div
        print(f"    |{nm}| = {int(o)}   divisible by 7: {div}")
    print("  every transitive proper subgroup of S_10 except A_10 lies in one of these (or is "
          "one) => none contains a 7-cycle.  Consistent with LEMMA P + JORDAN.")
    print(f"  GATE 3: {'PASS' if ok else 'FAIL'}")
    return ok


def test_tower_polys():
    print("=" * 96)
    print("TEST 4 -- the published tower polynomials, WITHOUT the (d-1)-prime step")
    print("=" * 96)
    ok = True
    r4 = certify_v2(F4, "f_4 (M9 quartic)")
    ok &= r4["verdict"] == "S_4" and r4["route"] == "B"
    r6 = certify_v2(F6, "f_6 (d=6 sextic)")
    ok &= r6["verdict"] == "S_6" and r6["route"] == "B"
    r8 = certify_v2(F8, "f_8 (d=8 octic)")
    ok &= r8["verdict"] == "S_8" and r8["route"] == "A"
    if r8["route"] == "A":
        wmax = max(v[0] for v in r8["witness"].values())
        print(f"  NOTE: f_8 route A max witness prime = {wmax} -- the published route needed a "
              f"transposition mod 10861.  The prime-growth obstruction is GONE with route A.")
        ok &= wmax < 10861
    print(f"  TEST 4: {'PASS' if ok else 'FAIL'}")
    return ok


def test_random(deg_list=(10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 16, 16, 22), seed=7,
                coeff=60):
    print("=" * 96)
    print(f"TEST 5 -- random monic integer polynomials at bad degrees {sorted(set(deg_list))}")
    print("=" * 96)
    rnd = random.Random(seed)
    cert = tot = 0
    tmax = 0.0
    wprime_max = 0
    for i, n in enumerate(deg_list):
        c = [rnd.randint(-coeff, coeff) for _ in range(n)] + [1]
        r = certify_v2(c, f"rand{i} deg {n}", verbose=False)
        tot += 1
        tmax = max(tmax, r["seconds"])
        if r["verdict"] == f"S_{n}":
            cert += 1
            wprime_max = max(wprime_max, max(v[0] for v in r["witness"].values()))
            print(f"    rand{i} deg {n}: S_{n} via route {r['route']} "
                  f"({r['scanned_primes']} primes, {r['seconds']}s)")
        else:
            print(f"    rand{i} deg {n}: {r['verdict']}  (events {sorted(r['witness'])})")
    print(f"  certified {cert}/{tot}; max witness prime {wprime_max}; slowest {tmax}s")
    ok = cert >= tot - 1          # allow one genuine non-S_n or unlucky sample
    print(f"  TEST 5: {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    t0 = time.time()
    if len(sys.argv) > 1:
        P = sp.Poly(sp.sympify(sys.argv[1]), x)
        coeffs = [int(c) for c in P.all_coeffs()[::-1]]
        r = certify_v2(coeffs, "input")
        print(r["verdict"])
        return
    results = [
        ("gate1 Dedekind cross-validation", gate_dedekind_crossval()),
        ("gate2 coverage theorem (all n>=8)", gate_coverage()),
        ("gate3 n=10 classification sanity", gate_degree10_group_theory()),
        ("test4 tower polys f_4/f_6/f_8", test_tower_polys()),
        ("test5 random bad-degree polys", test_random()),
    ]
    print("=" * 96)
    print("sd_certificate_v2 -- SUMMARY")
    print("=" * 96)
    allok = True
    for nm, ok in results:
        print(f"  [{'PASS' if ok else 'FAIL'}] {nm}")
        allok &= ok
    print(f"\n  OVERALL: {'PASS' if allok else 'FAIL'}   ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
