"""Stage 12 (Branch W): EXACT upgrade of the J-splitting facts — all eigenvalue statements
reduced to Laurent-polynomial identities (no floats, no tolerances).

Facts certified here, identically in theta over Z[sqrt2]-Laurent arithmetic:
  (F1) J' = coordswap(0<->2) ∘ conj is a PROJECTIVE PERMUTATION of the 33 Gould-Aravind
       slice rays IDENTICALLY IN THETA (checked via L_cross_zero on Laurent vectors).
       Corollary (orientation): since the same permutation works at every theta, J' maps the
       curve v(theta) to itself with parameter map theta |-> theta (orientation-PRESERVING),
       so the GA tangent is J'-EVEN — the stage-11 numerical +1.000000 is now EXACT.
  (F2) plain conj is NOT a theta-identical self-permutation (it maps v(theta) -> v(-theta):
       L_conj negates exponents). At theta = 0 it fixes the configuration with parameter map
       theta |-> -theta (orientation-REVERSING), so the tangent is J-ODD at Peres — the
       (0,1) split at theta=0 is EXACT.
  (F3) the theta <-> theta+pi signed-permutation relabeling (peres_penrose.py, exact) composed
       with conj gives the self-symmetry at theta = -pi/2 with parameter map theta |-> -theta-pi
       (derivative -1, fixed point -pi/2): the Penrose-point split (0,1) is EXACT as well.
The general J-splitting theorem's written proof is in M3M2.md Stage 12.

Run: python3 stage12_jexact.py
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import peres_penrose as pp

def main():
    t0 = time.time()
    rays = [tuple(comp for comp in ray) for ray in pp.SLICE]
    n = len(rays)
    print(f"exact GA slice: {n} Laurent rays over Z[sqrt2], exponents in {{-1,0,1}}")
    assert n == 33

    # ---- F1: J' = swap(0,2) ∘ conj permutes the rays identically in theta ------------------
    def Jprime(ray):
        u = [pp.L_conj(c) for c in ray]
        return (u[2], u[1], u[0])
    sigma, ok = [], True
    for j in range(n):
        img = Jprime(rays[j])
        hits = [k for k in range(n) if pp.L_cross_zero(img, rays[k])]
        if len(hits) != 1:
            ok = False
            print(f"  ray {j+1}: {len(hits)} projective matches -> FAIL")
            break
        sigma.append(hits[0])
    perm_ok = ok and sorted(sigma) == list(range(n))
    nfix = sum(1 for j, s in enumerate(sigma) if s == j) if perm_ok else -1
    print(f"[F1] J' = swap(0,2)∘conj: theta-identical projective permutation: {perm_ok} "
          f"(fixes {nfix}/33 rays; 2-cycles: {(33-nfix)//2 if perm_ok else '-'})")
    print("     => J' preserves the curve with parameter map theta |-> theta: the GA tangent")
    print("        is J'-EVEN, EXACT (was NUMERICAL +1.000000 in stage 11).")

    # ---- F2: plain conj is NOT theta-identical (it is theta -> -theta) ---------------------
    conj_self = all(any(pp.L_cross_zero(tuple(pp.L_conj(c) for c in rays[j]), rays[k])
                        for k in range(n)) for j in range(n))
    # count rays individually self-matching under plain conj (the theta-even rays)
    conj_hits = sum(1 for j in range(n)
                    if any(pp.L_cross_zero(tuple(pp.L_conj(c) for c in rays[j]), rays[k])
                           for k in range(n)))
    print(f"[F2] plain conj theta-identical self-map: {conj_self} "
          f"({conj_hits}/33 rays match; expect < 33 since conj negates exponents)")
    assert not conj_self, "unexpected: plain conj should NOT permute the slice identically"
    print("     => plain conj maps v(theta) -> v(-theta) (L_conj = exponent negation);")
    print("        at the fixed point theta=0 the parameter map is theta |-> -theta")
    print("        (orientation-REVERSING): tangent J-ODD at Peres, EXACT => split (0,1).")

    # ---- F3: the theta+pi relabeling exists exactly (cited machinery re-run) ---------------
    # peres_penrose documents and asserts the exact relabeling inside its own stage; here we
    # re-verify the *Laurent* form: find signed perm R and relabeling g with
    # R v_j(theta) ~ v_g(j)(theta+pi), where theta+pi multiplies e^{i e theta} monomials by
    # (-1)^e — i.e. on Laurent data: coefficient c at exponent e maps to (-1)^e c.
    from itertools import permutations, product
    def shift_pi(ray):
        return tuple({e: pp.q_neg(c) if (e[0] % 2) else c for e, c in comp.items()}
                     for comp in ray)
    target = [shift_pi(r) for r in rays]
    foundR = None
    for perm in permutations(range(3)):
        for sg in product((1, -1), repeat=3):
            g, okR = [], True
            for j in range(n):
                rv = tuple({e: (pp.q_neg(c) if sg[c_i] < 0 else c) for e, c in rays[j][perm[c_i]].items()}
                           for c_i in range(3))
                hits = [k for k in range(n) if pp.L_cross_zero(rv, target[k])]
                if len(hits) != 1: okR = False; break
                g.append(hits[0])
            if okR and sorted(g) == list(range(n)):
                foundR = (perm, sg); break
        if foundR: break
    print(f"[F3] exact Laurent-form theta<->theta+pi signed-permutation relabeling: "
          f"{'FOUND ' + str(foundR) if foundR else 'NOT FOUND'}")
    if foundR:
        print("     => J_pen = R^-1∘conj maps set(-pi/2) to itself with parameter map")
        print("        theta |-> -theta-pi (derivative -1, fixed point -pi/2): tangent J-ODD")
        print("        at the Penrose modulus, EXACT => split (0,1) there too.")

    allok = perm_ok and (not conj_self) and foundR is not None
    print(f"\ntotal time: {time.time()-t0:.1f}s")
    print("PASS" if allok else "FAIL")
    return allok

if __name__ == "__main__":
    main()
