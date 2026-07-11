# V53 - ARITHMETIC PROFILE OF THE 61 d=4 FACET CLASSES (open problem (b), profile
# part). 18 arithmetic species; ALL facet functionals RATIONAL - the pinned tier-2
# unit bounds 4*sqrt2, 14*sqrt2/3 were a normalization artifact (exact {0,+-1} lifts
# with INTEGER bounds {6,7} exist); quadratic fields enter only via Euclidean norms.
# Run: python3 d4_arithmetic_profile.py 1|2|3|4. See INDEX.md.
# ap_profile.py - ARITHMETIC PROFILE of the 61 facet classes of the certified d=4 census
# (note v3, open problem (b), profile half). All load-bearing arithmetic is EXACT
# (Python Fraction / integer Bareiss); numpy is used only for exact int64 linear algebra
# and for non-load-bearing rank pre-selection.
#
# Geometry and conventions (from verification/d4_facet_census.py, V43):
#   Y150 : 64 vertices of the moment polytope in the raw 150-dim character coordinates
#          (5 contexts x 15 characters x Re/Im), entries in {-1,0,1};
#   P    : 33 pivot columns; Y = Y150[:,P] faithful integer coordinates;
#   census class c: exact integer facet  cB_c + cA_c . y >= 0  on Y (min 0), i.e.
#          (-cA_c) . y <= cB_c ("bound" = cB_c in the integer-primitive normalization).
# Canonical metric object: the unique MINIMAL-NORM lift lambda_c in the 150-dim
# character metric of the facet's linear part along the affine hull:
#          lambda = D^T G^{-1} D a~,  G = D D^T,  D = 33 independent vertex differences,
#          a~ = cA placed on the pivot columns. lambda is RATIONAL; the facet reads
#          (-lambda) . y <= b'  on the hull with  b' = cB + a~.y0 - lambda.y0  (exact,
#          lift-independent). Unit-normalized bound = b'/||lambda|| - THE arithmetic
#          invariant of the class (independent of every normalization choice).
# Usage:  python3 ap_profile.py [1|2|3|4]   (default: all parts in order)
# Parts:  1 = species table -> ap_species_table.json ; 2 = tier-2 orbit match + pinned
#         surd verification ; 3 = separating invariants ; 4 = general-d feasibility scan.
import sys, os, json, pickle, itertools, time
from fractions import Fraction
from math import gcd, isqrt
import numpy as np

T0 = time.time()
OUT = os.path.dirname(os.path.abspath(__file__))
for _cand in ('/sessions/quirky-eloquent-babbage/mnt/contextuality-obstructions/verification',
              os.path.expanduser('~/Developer/contextuality-obstructions/verification')):
    if os.path.isdir(_cand):
        VER = _cand; break
else:
    raise SystemExit('verification dir not found')
os.chdir(VER); sys.path.insert(0, VER)
JS = os.path.join(OUT, 'ap_species_table.json')

# ---------- shared rebuild (exactly the census loading code, V43) ----------
from weyl import build
X, Z, w, tau, W, _ = build(4, 2)
fam = [[tuple(v) for v in it["ctx"]] for it in json.load(open("cert4_min.json"))["items"]]
obs = sorted({t for C in fam for t in C}); oi = {t: k for k, t in enumerate(obs)}
def s_of(C):
    M = np.eye(16, dtype=complex)
    for v in C: M = M @ W(np.array(v))
    return int(np.round(np.angle(np.trace(M) / abs(np.trace(M))) / (np.pi / 2))) % 4
S = [s_of(C) for C in fam]
CTX = [fam[i] for i in range(1, 6)]; Cs = fam[0]           # DROP = 0 (deleted context)
spec = {v: sorted(set(int(np.round(np.angle(x) / (np.pi / 2))) % 4
                      for x in np.linalg.eigvals(W(np.array(v))))) for v in obs}
A = np.zeros((5, 9), int)
for r, C in enumerate(CTX):
    for v in C: A[r, oi[v]] += 1
rhs = np.array([S[i] for i in range(6) if i != 0])
grid = np.array(list(itertools.product(range(4), repeat=9)))
ok = ((grid @ A.T) % 4 == rhs % 4).all(axis=1)
for v in obs: ok &= np.isin(grid[:, oi[v]], spec[v])
L = grid[ok]
CH = [(a, b) for a in range(4) for b in range(4) if (a, b) != (0, 0)]
V150 = []
for row in L:
    vec = []
    for C in CTX:
        j1, j2 = int(row[oi[C[0]]]), int(row[oi[C[1]]])
        for (a, b) in CH:
            z = np.exp(-1j * np.pi / 2 * (a * j1 + b * j2)); vec += [z.real, z.imag]
    V150.append(vec)
V150 = np.round(np.array(V150)).astype(np.int64)
d = np.load('d4_facet_classes.npz')
CEN = d['rows']; HN = d['full']; P = d['P']
Y = V150[:, P]
assert len(L) == 64 and Y.shape == (64, 33) and len(CEN) == 61 and len(HN) == 23256
cB = CEN[:, 0].astype(np.int64); cA = CEN[:, 1:34].astype(np.int64)
TC = CEN[:, 34].astype(int); STAB = CEN[:, 35].astype(int); ORB = CEN[:, 36].astype(int)
vals = cB[:, None] + cA @ Y.T                                # 61 x 64 exact
assert (vals >= 0).all() and (vals.min(axis=1) == 0).all()
assert all(STAB[i] * ORB[i] == 768 for i in range(61)), "col35 = stabilizer order"

# ---------- exact helpers ----------
def bareiss_rank(M):
    """Exact rank of an integer matrix (list of lists of ints), fraction-free."""
    M = [row[:] for row in M]; n = len(M); m = len(M[0]) if n else 0
    piv = 1; r = 0
    for c in range(m):
        pr = next((i for i in range(r, n) if M[i][c] != 0), None)
        if pr is None: continue
        M[r], M[pr] = M[pr], M[r]
        for i in range(r + 1, n):
            M[i] = [(M[r][c] * M[i][j] - M[i][c] * M[r][j]) // piv for j in range(m)]
        piv = M[r][c]; r += 1
        if r == n: break
    return r

def solve_gauss_frac(G, RHS):
    """Exact solve G X = RHS, G int (n x n) nonsingular, RHS int (n x k) -> Fractions."""
    n = len(G); k = len(RHS[0])
    Aug = [[Fraction(G[i][j]) for j in range(n)] + [Fraction(RHS[i][j]) for j in range(k)]
           for i in range(n)]
    for col in range(n):
        pr = next(i for i in range(col, n) if Aug[i][col] != 0)
        Aug[col], Aug[pr] = Aug[pr], Aug[col]
        pv = Aug[col][col]
        Aug[col] = [x / pv for x in Aug[col]]
        for i in range(n):
            if i != col and Aug[i][col] != 0:
                f = Aug[i][col]
                Aug[i] = [Aug[i][j] - f * Aug[col][j] for j in range(n + k)]
    return [[Aug[i][n + j] for j in range(k)] for i in range(n)]

def squarefree_part(n):
    n = abs(n); m = 1; p = 2
    while p * p <= n:
        e = 0
        while n % p == 0: n //= p; e += 1
        if e % 2: m *= p
        p += 1 if p == 2 else 2
    return m * n

def surd_form(q):
    """q = r^2 * m exactly, q Fraction >= 0, m squarefree int, r Fraction > 0."""
    if q == 0: return Fraction(0), 1
    m = squarefree_part(q.numerator * q.denominator)
    r2 = q / m
    rn, rd = isqrt(r2.numerator), isqrt(r2.denominator)
    assert Fraction(rn * rn, rd * rd) == r2, "not of the form r^2*m"
    return Fraction(rn, rd), m

def fstr(f): return f"{f.numerator}" if f.denominator == 1 else f"{f.numerator}/{f.denominator}"

def mask_key(mb):  # ONE key encoding only (pinned lesson): packbits bytes
    return np.packbits(mb).tobytes()

def class_orbit_keys(G_perms, mb):
    ks = set()
    for g in G_perms:
        nb = np.zeros(64, dtype=bool); nb[g] = mb
        ks.add(mask_key(nb))
    return ks

# ---------- PART 1: species table ----------
def part1():
    # exact affine-hull basis: 33 independent vertex differences (int)
    diffs = (V150 - V150[0]).astype(np.int64)
    sel = []; cur = np.zeros((0, 150))
    for i in range(1, 64):
        t = np.vstack([cur, diffs[i]])
        if np.linalg.matrix_rank(t) > len(cur): sel.append(i); cur = t
        if len(sel) == 33: break
    D = diffs[sel]                                            # 33 x 150 int
    assert bareiss_rank(D.tolist()) == 33, "hull basis exact rank 33"
    Gm = (D @ D.T).tolist()                                   # 33x33 int Gram
    RHS = (D[:, P] @ cA.T).tolist()                           # 33x61 int  ( = D a~ )
    Xs = solve_gauss_frac(Gm, RHS)                            # 33x61 Fractions
    Dy0 = (D @ V150[0]).tolist()                              # 33 int
    DYT = (D @ V150.T).tolist()                               # 33x64 int (for exact check)
    Dlist = D.tolist()
    # ghost / even factor-through data (exact, lift-independent):
    gidx = [oi[v] for v in Cs]                                # deleted-context observables
    fib_ghost = [tuple(int(L[i, g]) for g in gidx) for i in range(64)]
    fib_even  = [tuple(int(x) % 2 for x in L[i]) for i in range(64)]
    fib_geven = [tuple(int(L[i, g]) % 2 for g in gidx) for i in range(64)]
    def factors_through(fib, va):
        d_ = {}
        for i in range(64):
            if fib[i] in d_ and d_[fib[i]] != va[i]: return False
            d_[fib[i]] = va[i]
        return True
    # character typing per context: single-body lines vs two-body characters
    single = {k for k, (a, b) in enumerate(CH) if b == 0 or a == 0 or a == b}
    rows_out = []
    for c in range(61):
        x = [Xs[j][c] for j in range(33)]
        norm2 = sum(Fraction(RHS[j][c]) * x[j] for j in range(33))      # ||lambda||^2
        lam_y0 = sum(x[j] * Dy0[j] for j in range(33))
        a_y0 = int(cA[c] @ Y[0])
        bprime = Fraction(int(cB[c])) + a_y0 - lam_y0                    # exact bound b'
        # exact consistency: lambda.(Y_i - Y_0) == vals[c,i] - vals[c,0] for ALL 64 vertices
        chk = [sum(x[j] * (DYT[j][i] - Dy0[j]) for j in range(33)) for i in range(64)]
        assert all(chk[i] == int(vals[c, i]) - int(vals[c, 0])
                   for i in range(64)), f"class {c}: lift mismatch"
        # lambda entries (150, rational)
        lam = [sum(Fraction(Dlist[j][k]) * x[j] for j in range(33)) for k in range(150)]
        supp = [k for k in range(150) if lam[k] != 0]
        from math import lcm
        den = 1
        for k in supp: den = lcm(den, lam[k].denominator)
        nums = [int(lam[k] * den) for k in supp]
        cont = 0
        for v_ in nums: cont = gcd(cont, abs(v_))
        prim = [v_ // cont for v_ in nums] if cont else []
        mags = sorted(set(abs(v_) for v_ in prim))
        # unit-normalized data
        q = bprime * bprime / norm2
        r, m = surd_form(q)                                              # b'/||l|| = r*sqrt(m)
        rn, mn = surd_form(norm2)                                        # ||l|| = rn*sqrt(mn)
        assert m == mn or bprime == 0, "field of bound = field of norm"
        # unit-normal grid: u = lam/||lam||; check u in (1/4)Z resp. (1/(2 sqrt2))Z
        if mn == 1:
            ugrid = all((4 * lam[k] / rn).denominator == 1 for k in supp)      # (1/4)Z
        elif mn == 2:
            ugrid = all((2 * lam[k] / rn).denominator == 1 for k in supp)      # (1/(2s2))Z
        else:
            ugrid = False
        va = vals[c]
        tight = [i for i in range(64) if va[i] == 0]
        rk = bareiss_rank((Y[tight] - Y[tight[0]]).tolist())
        ctxs = sorted(set(k // 30 for k in supp))
        twob = sum(1 for k in supp if ((k % 30) // 2) not in single)
        g33 = 0
        for v_ in cA[c]: g33 = gcd(g33, abs(int(v_)))
        n33 = int((cA[c].astype(np.int64) ** 2).sum())
        r33, m33 = surd_form(Fraction(int(cB[c]) ** 2, n33))
        rows_out.append(dict(
            id=c, bound_int=int(cB[c]), tight_count=int(TC[c]), tight_rank=rk,
            orbit=int(ORB[c]), stab=int(STAB[c]), gcd33=g33,
            supp33=int((cA[c] != 0).sum()), mod2_weight=int((cA[c] % 2 != 0).sum()),
            norm2=fstr(norm2), bprime=fstr(bprime),
            norm33_sq=n33,
            unit33={'r': fstr(r33), 'm': m33,
                    'str': (fstr(r33) if m33 == 1 else (fstr(r33) + '*sqrt(%d)' % m33))},
            species33=('rational' if m33 == 1 else 'sqrt%d' % m33),
            unit_bound={'r': fstr(r), 'm': m,
                        'str': (fstr(r) if m == 1 else (fstr(r) + '*sqrt(%d)' % m))},
            unit_bound_sq=fstr(q), species=('quarter' if m == 1 else
                                            ('sqrt2' if m == 2 else 'sqrt%d' % m)),
            unit_normal_on_grid=bool(ugrid),
            lam_supp150=len(supp), lam_den=den, lam_content=fstr(Fraction(cont, den)),
            lam_mags=mags[:6], ctxs_touched=len(ctxs), twobody_chars=twob,
            ghost_factor=factors_through(fib_ghost, va.tolist()),
            even_factor=factors_through(fib_even, va.tolist()),
            ghost_even_factor=factors_through(fib_geven, va.tolist()),
            tier2_raw_bound=None))
        assert rk == 32, f"class {c}: tight rank {rk}"
        assert va[tight].max() == 0 and len(tight) == TC[c]
    json.dump(rows_out, open(JS, 'w'), indent=1)
    hdr = f"{'id':>2} {'b':>2} {'tc':>2} {'rk':>2} {'orb':>4} {'st':>3} {'sup33':>5} " \
          f"{'w2':>2} {'|A|^2':>5} {'|l|^2':>8} {'unit bound (150)':>16} {'unit33':>14} " \
          f"{'gh':>2} {'s150':>4}"
    print(hdr)
    for r_ in rows_out:
        print(f"{r_['id']:>2} {r_['bound_int']:>2} {r_['tight_count']:>2} {r_['tight_rank']:>2} "
              f"{r_['orbit']:>4} {r_['stab']:>3} {r_['supp33']:>5} {r_['mod2_weight']:>2} "
              f"{r_['norm33_sq']:>5} {r_['norm2']:>8} {r_['unit_bound']['str']:>16} "
              f"{r_['unit33']['str']:>14} {int(r_['ghost_factor']):>2} {r_['lam_supp150']:>4}")
    from collections import Counter
    cs = Counter((r_['species'], r_['unit_bound']['str']) for r_ in rows_out)
    co = Counter(r_['species'] for r_ in rows_out)
    orbsum = Counter()
    for r_ in rows_out: orbsum[r_['species']] += r_['orbit']
    print(f"[part1] species: {dict(co)} | facets per species: {dict(orbsum)} "
          f"| distinct unit bounds: {sorted(set(r_['unit_bound']['str'] for r_ in rows_out))} "
          f"| t={time.time()-T0:.1f}s")
    print(f"[part1] table -> {JS}")

# ---------- PART 2: tier-2 orbit match + pinned surd verification ----------
def part2():
    rows_out = json.load(open(JS))
    Gp = [np.array(g) for g in pickle.load(open('d4_group_seed.pkl', 'rb'))['G']]
    # class label for every census facet: generate the 61 mask orbits under G (exact)
    key2class = {}
    for c in range(61):
        mb = vals[c] == 0
        ks = class_orbit_keys(Gp, mb)
        assert len(ks) == ORB[c], f"class {c}: orbit {len(ks)} != {ORB[c]}"
        for k in ks:
            assert k not in key2class, "orbit overlap"
            key2class[k] = c
    hvals = HN[:, :33] @ Y.T + HN[:, 33][:, None]
    hkeys = {mask_key(hvals[i] == 0) for i in range(23256)}
    assert len(hkeys) == 23256 and hkeys == set(key2class), \
        "61 orbits tile the 23,256 census facets exactly"
    # key -> (class, permutation) so each stored facet can be checked EXACTLY
    key2cg = {}
    for c in range(61):
        mb = vals[c] == 0
        for g in Gp:
            nb = np.zeros(64, dtype=bool); nb[g] = mb
            key2cg.setdefault(mask_key(nb), (c, g))
    t = np.load('d4_tier2_orbit_data.npz')
    enc = t['enc'].astype(np.int64); B2 = t['B'].astype(int)
    Wv = enc @ V150.T                                          # 21504 x 64 exact int
    mx = Wv.max(axis=1)
    # CORRECTED EXACT ARITHMETIC of the tier-2 family. The stored lift enc/4 has all
    # coefficients in {0,+-1} and satisfies max_V (enc/4).v = B EXACTLY (mx == 4B), with
    # slack vector exactly proportional to the integer census slacks (factor 1 for B=6,
    # 2 for B=7 classes). Hence the true single coefficient magnitude is 1 and the true
    # unit-COEFFICIENT bounds are the INTEGERS {6,7}. The pinned values 4*sqrt2 = 6/(3/(2s2))
    # and 14*sqrt2/3 = 7/(3/(2s2)) came from dividing by a nominal grid unit 3/(2*sqrt2):
    # in V36's magnitude bookkeeping |x|*2*sqrt2 rounded to int, |+-1| -> round(2.828)=3
    # collides with |3/(2 sqrt2)| -> 3 exactly; the sqrt2 was injected by that convention.
    assert (mx == 4 * B2).all(), "exact integer bounds {6,7} for the +-1 lifts"
    assert set(np.abs(enc).flatten().tolist()) == {0, 4}
    # no constant-magnitude 3/(2 sqrt2) vector on integer vertices can have integer bound:
    c0 = 3 / (2 * np.sqrt(2))
    assert min(abs(k * c0 - round(k * c0)) for k in range(1, 201)) > 1e-3, \
        "irrationality margin: k*3/(2 sqrt2) never within 1e-3 of Z for k<=200"
    hits = {}
    okslack = True
    for i in range(len(enc)):
        k = mask_key(Wv[i] == mx[i])
        c, g = key2cg[k]                                       # KeyError would mean non-facet
        vp = np.zeros(64, dtype=np.int64); vp[g] = vals[c]
        sl = mx[i] - Wv[i]                                     # 4 * slack of the +-1 lift
        fac = {6: 4, 7: 8}[int(B2[i])]                         # observed exact proportionality
        okslack &= (sl == fac * vp).all()
        hits.setdefault(c, [0, set()])
        hits[c][0] += 1; hits[c][1].add(int(B2[i]))
    tot = sum(v[0] for v in hits.values())
    print(f"[part2] tier-2 orbit file: {len(enc)} stored lifts -> {len(hits)} census classes "
          f"({tot} matched); EXACT slack proportionality to census integers: {okslack}")
    for c in sorted(hits):
        r_ = rows_out[c]
        bset = sorted(hits[c][1]); assert len(bset) == 1
        r_['tier2_raw_bound'] = bset[0]
        print(f"  class {c:>2}: +-1-lift bound {bset[0]} (census b={int(cB[c])}), stored reps "
              f"{hits[c][0]:>5}, orbit {ORB[c]:>4}, canonical unit bound "
              f"{r_['unit_bound']['str']:>15}, unit33 {r_['unit33']['str']}")
    print(f"[part2] tier-2 family: {sorted(hits)} = {sum(ORB[c] for c in hits)} facets of 23256.")
    print("[part2] CORRECTION (pinned): tier-2 unit-COEFFICIENT bounds are the integers {6,7};")
    print("        '4*sqrt2, 14*sqrt2/3' = {6,7}/(3/(2 sqrt2)) is a bookkeeping artifact of the")
    print("        V36 2*sqrt2-scale magnitude rounding (|+-1| and |3/(2 sqrt2)| both -> '3').")
    print("        All tier-2 facet functionals are RATIONAL: they admit exact {0,+-1}-coefficient")
    print("        lifts in the 150 character coordinates with integer bounds {6,7}.")
    print(f"[part2] t={time.time()-T0:.1f}s")
    json.dump(rows_out, open(JS, 'w'), indent=1)

# ---------- PART 3: separating invariants ----------
def part3():
    rows_out = json.load(open(JS))
    from collections import Counter, defaultdict
    def xtab(name, f):
        d_ = defaultdict(Counter)
        for r_ in rows_out: d_[r_['species']][f(r_)] += 1
        print(f"  {name:<28} " + " | ".join(
            f"{sp}: {dict(sorted(d_[sp].items()))}" for sp in sorted(d_)))
    print("[part3] species cross-tabulations (61 classes):")
    xtab('bound_int', lambda r_: r_['bound_int'])
    xtab('tight_count', lambda r_: r_['tight_count'])
    xtab('orbit size', lambda r_: r_['orbit'])
    xtab('stabilizer', lambda r_: r_['stab'])
    xtab('ghost_factor', lambda r_: r_['ghost_factor'])
    xtab('even_factor', lambda r_: r_['even_factor'])
    xtab('ghost_even_factor', lambda r_: r_['ghost_even_factor'])
    xtab('mod2 weight of cA', lambda r_: r_['mod2_weight'])
    xtab('supp33', lambda r_: r_['supp33'])
    xtab('lam supp150', lambda r_: r_['lam_supp150'])
    xtab('ctxs touched', lambda r_: r_['ctxs_touched'])
    xtab('two-body chars', lambda r_: r_['twobody_chars'])
    xtab('||lam||^2', lambda r_: r_['norm2'])
    xtab('unit bound', lambda r_: r_['unit_bound']['str'])
    xtab('species33 (33-coord metric)', lambda r_: r_['species33'])
    # does (bound, ||lambda||^2) determine the unit bound? (it must); is it injective on classes?
    key = lambda r_: (r_['bound_int'], r_['norm2'])
    kmap = defaultdict(set)
    for r_ in rows_out: kmap[key(r_)].add(r_['unit_bound']['str'])
    print(f"  (b, ||lam||^2) -> unit bound well-defined: {all(len(v) == 1 for v in kmap.values())}; "
          f"{len(kmap)} distinct arithmetic types over 61 classes")
    # tier-2 membership vs arithmetic type: twins (same full invariant tuple, different membership)
    sig = lambda r_: (r_['bound_int'], r_['tight_count'], r_['orbit'], r_['stab'],
                      r_['norm2'], r_['supp33'], r_['mod2_weight'], r_['lam_supp150'])
    t2 = {r_['id'] for r_ in rows_out if r_['tier2_raw_bound'] is not None}
    bysig = defaultdict(list)
    for r_ in rows_out: bysig[sig(r_)].append(r_['id'])
    mixed = {str(k): v for k, v in bysig.items()
             if any(i in t2 for i in v) and any(i not in t2 for i in v)}
    print(f"  tier-2 classes (stored orbit): {sorted(t2)}")
    print(f"  arithmetic twins straddling the tier-2 orbit boundary: {list(mixed.values())}")
    print("  -> the stored tier-2 orbit is NOT arithmetically closed: its invariant types also")
    print("     occur outside it, consistent with the V36/V41 'orbit completion OPEN' caveat.")
    # ghost / tau summary
    gh = [r_['id'] for r_ in rows_out if r_['ghost_factor']]
    ev = [r_['id'] for r_ in rows_out if r_['even_factor']]
    print(f"  ghost classes (factor through deleted-context outcomes): {gh} "
          f"(facets: {sum(r_['orbit'] for r_ in rows_out if r_['ghost_factor'])})")
    print(f"  order-2-shadow classes (factor through lambda mod 2): {ev} "
          f"-> tau-level (order-4) data necessary for ALL {61 - len(ev)} remaining classes")
    print(f"[part3] t={time.time()-T0:.1f}s")

# ---------- PART 4: general-d feasibility scan (report only) ----------
def part4():
    # exact d=4 anchors (recomputed live)
    n_obs, n_ctx = 9, 5
    print("[part4] d=4 anchors: |L|=64 vertices, raw dim 150, affine dim 33, "
          "23,256 facets / 61 classes; vertex group |G|=768 (x2 conj = 1536 on tier-2); "
          "engines: Normaliz 3.10.2 + exact adjacency decomposition (lrs WEDGES, pinned).")
    # constraint-rank bookkeeping (V43 on-hull lemma): rank 47 of 80 = 5d^2 prob coords:
    #   5 normalizations + 9(d-1) shared-marginal + spectral-support rank (15 at d=4).
    for dd in (6, 8):
        prob = 5 * dd * dd
        marg = n_obs * (dd - 1)
        # spectral rank: at d=4 each obs has |spec|=d/2 -> 9*(d/2) zero-identities, rank 15
        # (=9*(d/2)-3); same bookkeeping propagated as an ESTIMATE (flagged):
        spec_rank = n_obs * (dd // 2) - 3
        aff = prob - 5 - marg - spec_rank
        # vertex count: unrestricted torsor d^4; spectral cut factor 4 at d=4 (256->64).
        vlo, vhi = dd ** 3, dd ** 4
        print(f"  d={dd} (K33 analogue, one context deleted): raw prob coords {prob}, "
              f"affine dim ESTIMATE {aff} (=5d^2-5-9(d-1)-[9d/2-3]), "
              f"|L| in [{vlo}, {vhi}] (d^3..d^4; d=4 realized d^3=64).")
        # McMullen upper bound on facet count for n vertices in dim a:
        from math import comb
        n = vlo; a = aff
        ub = comb(n - (a + 1) // 2 - 1 + (a + 1) // 2, (a + 1) // 2)  # crude cyclic-type bound
        print(f"    facet-count scale: d=4 had 23,256 from (64 pts, dim 33); "
              f"UB(cyclic, n={n}, dim~{a}) ~ 10^{len(str(ub))-1} - enumeration is "
              f"OUT OF REACH for direct dual conversion.")
    print("  tooling: Normaliz NOT installed here; lrs wedges on these polytopes (pinned);")
    print("  d=4 needed Normaliz 3.10 + group-quotient adjacency decomposition (V43/V43b).")
    print("  RECOMMENDED COLAB PROTOCOL: (i) build vertex set + derived automorphism group")
    print("  exactly (integer L-solve + spectral filter; group by K33 graph autos x spectral")
    print("  shift torsor, V41 style); (ii) adjacency decomposition BFS modulo the group,")
    print("  Normaliz 3.10+ (DualMode + project-and-lift) per ridge subproblem, exact integer")
    print("  hyperplane recovery through rank-(aff-1) tight sets; (iii) recursive V43b")
    print("  decomposition for heavy ridges; (iv) independent cross-check: direct Normaliz on")
    print("  the full polytope ONLY at d=6 if |L|<~300 and aff<~60 (days of wall time, 50-100GB);")
    print("  at d=8 a full census is infeasible - target the ghost/tier-1 subfamily and the")
    print("  tier-2 orbit transport instead, and test the arithmetic conjecture on random")
    print("  exact supporting hyperplanes.")
    print(f"[part4] t={time.time()-T0:.1f}s")

if __name__ == '__main__':
    parts = sys.argv[1:] or ['1', '2', '3', '4']
    for p in parts:
        {'1': part1, '2': part2, '3': part3, '4': part4}[p]()
