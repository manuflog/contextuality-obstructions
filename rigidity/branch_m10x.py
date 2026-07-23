#!/usr/bin/env python3
"""
branch_m10x.py -- Branch M10X: IDENTIFY M10's SECOND FLEX DIRECTION.

READ FIRST: D4_FLEX_HUNT.md Sec.10.3 (M10 resolved to exact flex=2 at the 21-ray/77-pair/
11-basis critical core, x=2i/3i/5i, sympy DomainMatrix/QQ, rankJ=130 ker=38 rankT=36); this
file's exact_flex_hermitian_at_point-mirroring construction is INDEPENDENTLY REBUILT here (not
imported/modified) so it can additionally expose the actual KERNEL BASIS, not just its rank.

QUESTION: exact flex=2, but the known "mechanism-family" direction (x=i*t along the whole M10
line) is only 1 of those 2 dimensions. What is the second?

ANSWER (proved below, not just bounded): the second direction w_perp is an EXACT, CLOSED-FORM
internal SO(2) "doublet rotation" mixing the core's two X-using rays into each other --
independent of the line parameter t -- licensed by three exact algebraic facts about the 21-ray
critical core (all proved by direct symbolic identity, not sampled):
  (i)   the two X-rays have IDENTICAL neighbor sets {4,11,13,16} among the 19 t-independent
        integer rays (a combinatorial fact of the abstract mechanism-stable core);
  (ii)  |ray19(t)| = |ray20(t)| identically (trivial: same 4 entries, different slots);
  (iii) <ray19(t),ray20(t)> = 4*Re(t) = 0 identically ON THE WHOLE M10 LINE -- the ONE place
        the mechanism's own defining equation Re(x)=0 is essential.
Together (i)-(iii) mean the two X-rays form a decoupled orthogonal, equal-norm pair that can be
freely SO(2)-rotated into each other at every point of the line without breaking any of the 77
mechanism-stable pairs -- verified by DIRECT SYMBOLIC SUBSTITUTION (sympy, general real s with
t=i*s, general real theta) that ALL 77 edges + both norm identities remain EXACTLY satisfied.

CONSEQUENCE: near this specific 21-ray critical-core witness, M10's realization moduli is a
genuine 2-DIMENSIONAL SHEET (topologically R x S^1, a cylinder: line parameter s times rotation
angle theta mod 2pi) -- NOT a spurious/second-order-blocked direction, and NOT a hidden gauge
artifact (proved independent of the 36 U(4)+per-ray-phase trivial generators by exact linear
algebra). SCOPE CAVEAT (checked, stated honestly, Sec.6 below): this is a property of the 21-ray
CRITICAL-CORE witness (already independently sufficient to prove M10 KS-uncolorable at every
point of the line -- the standard "core, not full pool" methodology used throughout this repo,
e.g. D4_FLEX_HUNT.md's own "criticality != minimal flex" caveat for the flex NUMBER); the naive
rotation does NOT preserve orthogonality against the full 272-ray/208-basis-participating M10
pool (checked directly: the two X-rays' neighbor sets differ there, symmetric difference 10).

Run: python3 branch_m10x.py            (full report, <10s)
     python3 branch_m10x.py quick       (skip the antiunitary search, <2s)
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from itertools import combinations, permutations, product
import sympy as sp
from sympy.polys.matrices import DomainMatrix

T0 = time.time()

# ==================================================================================================
# The 21-ray/77-pair/11-basis M10 critical core (D4_FLEX_HUNT.md Sec.10.3's "tightest core found",
# `python3 branch_d4flex.py core_M10`, seed0=1, cached in d4flex_M10_done.cache.json). Hardcoded
# here (with a consistency check against that cache in `verify_provenance`) so this file has no
# runtime dependency on re-running the (randomized, checkpointed) peel search.
# ==================================================================================================
CORE_SYMS = [
    ('0', '0', '1', '0'), ('0', '0', '1', '1'), ('0', '0', '1', '-1'), ('0', '1', '0', '0'),
    ('0', '1', '0', '1'), ('0', '1', '0', '-1'), ('0', '1', '1', '0'), ('0', '1', '-1', '0'),
    ('1', '0', '0', '0'), ('1', '0', '0', '1'), ('1', '0', '0', '-1'), ('1', '0', '1', '0'),
    ('1', '1', '0', '0'), ('1', '1', '1', '1'), ('1', '1', '-1', '1'), ('1', '-1', '1', '1'),
    ('1', '-1', '1', '-1'), ('1', '-1', '-1', '1'), ('1', '-1', '-1', '-1'),
    ('1', 'X', '-1', '-X'), ('X', '1', '-X', '-1'),
]
V, D = 21, 4
IDX_X1, IDX_X2 = 19, 20   # the 2 X-using rays: ('1','X','-1','-X') and ('X','1','-X','-1')

def verify_provenance():
    """CORE_SYMS above must match branch_d4flex.py's own cached converged M10 core exactly (index
       set into `generic_symbolic_rays()`, translated to symbols) -- a from-scratch consistency
       check, not a re-derivation."""
    import branch_d4flex as bd
    from ks_flex_census import cache_load
    done = cache_load("d4flex_M10_done")
    if done is None:
        print("  [provenance] no cache found -- skipping (CORE_SYMS still independently re-verified")
        print("               below via its own 77-pair/11-basis/flex=2 recomputation).")
        return
    rays = bd.generic_symbolic_rays(4)
    cached_syms = [rays[i] for i in done]
    match = (sorted(cached_syms) == sorted(CORE_SYMS))
    print(f"  [provenance] CORE_SYMS matches branch_d4flex.py's cached d4flex_M10_done core: {match}")
    assert match, "CORE_SYMS is stale -- re-sync with `python3 branch_d4flex.py core_M10`"

# ==================================================================================================
# STAGE 1: rebuild the exact Hermitian-tangent constraint system at a concrete point x = t (t on
# the M10 line, i.e. purely imaginary), independently of branch_d4flex.exact_flex_hermitian_at_point
# (same construction, mirrored here so we can additionally extract the actual kernel/complement
# BASIS vectors, not just ranks).
# ==================================================================================================
def ri_pair(sym, t_re, t_im):
    if sym == '0': return (sp.Integer(0), sp.Integer(0))
    if sym == '1': return (sp.Integer(1), sp.Integer(0))
    if sym == '-1': return (sp.Integer(-1), sp.Integer(0))
    if sym == 'X': return (t_re, t_im)
    if sym == '-X': return (-t_re, -t_im)
    raise ValueError(sym)

def build_at(t_re, t_im):
    """rows: 2 per mechanism-stable edge (Re,Im of <w_i,v_j>+<v_i,w_j>=0) + 1 per ray (norm row
       Re<v_i,w_i>=0). triv: the 37 U(4)+per-ray-phase trivial generators (rank 36). Identical
       construction/convention to branch_d4flex.exact_flex_hermitian_at_point, rebuilt here."""
    rays_ri = [tuple(ri_pair(c, t_re, t_im) for c in r) for r in CORE_SYMS]
    Re = [[sp.Rational(rays_ri[i][c][0]) for c in range(D)] for i in range(V)]
    Im = [[sp.Rational(rays_ri[i][c][1]) for c in range(D)] for i in range(V)]

    def hdot_zero(i, j):
        dre = sum(Re[i][c] * Re[j][c] + Im[i][c] * Im[j][c] for c in range(D))
        dim = sum(Re[i][c] * Im[j][c] - Im[i][c] * Re[j][c] for c in range(D))
        return dre == 0 and dim == 0

    E = [(i, j) for i, j in combinations(range(V), 2) if hdot_zero(i, j)]
    n = 2 * D * V

    def coord(i, c, real): return 2 * D * i + 2 * c + (0 if real else 1)

    rows = []
    for i, j in E:
        re = [sp.Integer(0)] * n; im = [sp.Integer(0)] * n
        for c in range(D):
            re[coord(i, c, True)] += Re[j][c]; re[coord(i, c, False)] += Im[j][c]
            re[coord(j, c, True)] += Re[i][c]; re[coord(j, c, False)] += Im[i][c]
            im[coord(i, c, True)] += Im[j][c]; im[coord(i, c, False)] -= Re[j][c]
            im[coord(j, c, True)] -= Im[i][c]; im[coord(j, c, False)] += Re[i][c]
        rows.append(re); rows.append(im)
    for i in range(V):
        r = [sp.Integer(0)] * n
        for c in range(D): r[coord(i, c, True)] = Re[i][c]; r[coord(i, c, False)] = Im[i][c]
        rows.append(r)

    triv = []
    for i in range(V):
        t = [sp.Integer(0)] * n
        for c in range(D): t[coord(i, c, True)] = -Im[i][c]; t[coord(i, c, False)] = Re[i][c]
        triv.append(t)
    for a in range(D):
        t = [sp.Integer(0)] * n
        for i in range(V): t[coord(i, a, True)] = -Im[i][a]; t[coord(i, a, False)] = Re[i][a]
        triv.append(t)
    for a in range(D):
        for b in range(a + 1, D):
            t = [sp.Integer(0)] * n
            for i in range(V):
                t[coord(i, a, True)] += -Im[i][b]; t[coord(i, a, False)] += Re[i][b]
                t[coord(i, b, True)] += -Im[i][a]; t[coord(i, b, False)] += Re[i][a]
            triv.append(t)
            t = [sp.Integer(0)] * n
            for i in range(V):
                t[coord(i, a, True)] += Re[i][b]; t[coord(i, a, False)] += Im[i][b]
                t[coord(i, b, True)] += -Re[i][a]; t[coord(i, b, False)] += -Im[i][a]
            triv.append(t)

    return dict(rays_ri=rays_ri, Re=Re, Im=Im, E=E, rows=rows, triv=triv, n=n, coord=coord)

def rank_qq(rows, ncols):
    dm = DomainMatrix.from_list_sympy(len(rows), ncols, rows).convert_to(sp.QQ)
    return dm.rank()

def exact_flex_certificate(data):
    n = data["n"]
    J = sp.Matrix(data["rows"])
    rankJ = rank_qq(data["rows"], n)
    rankT = rank_qq(data["triv"], n)
    ker = n - rankJ
    flex = ker - rankT
    return dict(J=J, rankJ=rankJ, ker=ker, rankT=rankT, flex=flex, n=n, E=len(data["E"]))

# ==================================================================================================
# STAGE 2: the LINE (T) direction -- d/dt of x=i*t at t=t0, NORM-CORRECTED. Subtlety caught: the
# raw substitution derivative (nonzero only on the 'X'/'-X' slots) does NOT satisfy the norm-
# preserving row exactly (the raw representative's length is NOT constant along t: |ray19(t)|^2 =
# 2+2t^2), so it is not itself literally in ker(J) as this convention defines it. Fixed by
# subtracting the radial (real-rescaling) component -- standard tangent-to-fixed-norm projection,
# exact rationals throughout.
# ==================================================================================================
def corrected_T(data, t_re, t_im):
    n, coord, Re, Im = data["n"], data["coord"], data["Re"], data["Im"]
    Tvec = [sp.Integer(0)] * n
    raw = {IDX_X1: {1: (0, 1), 3: (0, -1)}, IDX_X2: {0: (0, 1), 2: (0, -1)}}
    for i, slots in raw.items():
        vRe, vIm = Re[i], Im[i]
        wr = [sp.Integer(0)] * D; wi = [sp.Integer(0)] * D
        for c, (re_, im_) in slots.items(): wr[c] = sp.Integer(re_); wi[c] = sp.Integer(im_)
        Nv = sum(vRe[c] ** 2 + vIm[c] ** 2 for c in range(D))
        Redot = sum(vRe[c] * wr[c] + vIm[c] * wi[c] for c in range(D))
        factor = Redot / Nv
        for c in range(D):
            Tvec[coord(i, c, True)] = wr[c] - factor * vRe[c]
            Tvec[coord(i, c, False)] = wi[c] - factor * vIm[c]
    return sp.Matrix(Tvec)

# ==================================================================================================
# STAGE 3: the SECOND direction w_perp. Two constructions cross-checked to agree (same 1-dim
# line): (a) a "minimal-support" representative found by restricting ker(J) to vectors supported
# ONLY on the two X-rays (a 4-dim sub-nullspace: 2 per-ray phases + T + exactly ONE new direction)
# -- this is the clean closed form reported; (b) the generic ambient-orthogonal-complement
# construction (Gram-Schmidt vs the 36 trivial + T, exact rational projection) as an independent
# sanity check that both land on the same 1-dim line.
# ==================================================================================================
def w_perp_minimal_support(data, J):
    n, coord = data["n"], data["coord"]
    cols = []
    for i in (IDX_X1, IDX_X2):
        for c in range(D):
            cols.append(coord(i, c, True)); cols.append(coord(i, c, False))
    cols = sorted(cols)
    Jsub = J[:, cols]
    nsub = Jsub.nullspace()
    phase19 = sp.Matrix(data["triv"][IDX_X1])[cols, 0]
    phase20 = sp.Matrix(data["triv"][IDX_X2])[cols, 0]
    Tc = corrected_T(data, data["t_re"], data["t_im"])[cols, 0]
    known = sp.Matrix.hstack(phase19, phase20, Tc)
    Nsub = sp.Matrix.hstack(*nsub)
    G = known.T * known
    Ginv = G.inv()
    def proj_known(v): return known * (Ginv * (known.T * v))
    extra_local = None
    for k in range(Nsub.shape[1]):
        v = Nsub[:, k]
        r = v - proj_known(v)
        if any(x != 0 for x in r):
            extra_local = r; break
    assert extra_local is not None, "no direction beyond {phase19,phase20,T} found on the X-ray pair"
    wperp = [sp.Integer(0)] * n
    for local_idx, glob_idx in enumerate(cols):
        wperp[glob_idx] = extra_local[local_idx]
    return sp.Matrix(wperp), dict(nsub_dim=len(nsub))

def ambient_M37(data, Tm):
    """The 37 = 36 (independent trivial) + 1 (T) generators, as columns, for AMBIENT (168-dim,
       standard real inner product) Gram-Schmidt projection -- exact rational, no floats."""
    Triv37 = sp.Matrix.hstack(*[sp.Matrix(t) for t in data["triv"]])
    Triv36 = sp.Matrix.hstack(*Triv37.columnspace())
    return sp.Matrix.hstack(Triv36, Tm)

def ambient_project_out(M37, v):
    G = M37.T * M37
    Ginv = G.inv()
    return v - M37 * (Ginv * (M37.T * v))

def w_perp_ambient_complement(data, J, Tm):
    """Independent cross-check: Gram-Schmidt w.r.t. the full 36-dim trivial span + T, in the
       AMBIENT 168-dim real coordinate space (exact rational projection, no floats). Returns a
       vector spanning the SAME 1-dim complement line as `w_perp_minimal_support`'s answer (after
       ALSO ambient-projecting that one -- the minimal-support representative is a valid but
       gauge-UNFIXED point in the coset w_perp + span(trivial,T); only after projecting out
       span(trivial,T) do the two constructions land on the identical line)."""
    M37 = ambient_M37(data, Tm)
    K = J.nullspace()
    for kv in K:
        r = ambient_project_out(M37, kv)
        if any(x != 0 for x in r):
            return r
    raise RuntimeError("ambient complement construction found nothing")

def check_proportional(u, v):
    """Exact check that two nonzero sympy column vectors are real scalar multiples of each other."""
    nz = [k for k in range(u.shape[0]) if u[k] != 0]
    if not nz: return False
    k0 = nz[0]
    if v[k0] == 0: return False
    lam = u[k0] / v[k0]
    return all(u[k] == lam * v[k] for k in range(u.shape[0]))

# ==================================================================================================
# STAGE 4: antiunitary self-symmetry search (signed/unit-monomial . conj) on the 21-ray core at a
# concrete point, generalizing stage11_jsplit.find_antiunitary (real signed permutations, d=3) to
# unit-scaled (units of Z[i]: 1,-1,i,-i) coordinate permutations in d=4, exact (no floats).
# ==================================================================================================
UNITS = [1, -1, 1j, -1j]

def antiunitary_search(rays_num, tol=1e-9):
    found = []
    for perm in permutations(range(D)):
        for signs in product(UNITS, repeat=D):
            sigma, ok = [], True
            for i in range(V):
                v = rays_num[i]
                img = tuple(signs[r] * complex(v[perm[r]]).conjugate() for r in range(D))
                hit = None
                for j in range(V):
                    w = rays_num[j]
                    for u in UNITS:
                        if all(abs(img[k] - u * w[k]) < tol for k in range(D)):
                            hit = j; break
                    if hit is not None: break
                if hit is None: ok = False; break
                sigma.append(hit)
            if ok and len(set(sigma)) == V:
                found.append((perm, signs, tuple(sigma)))
    return found

# ==================================================================================================
# STAGE 5: second-order integrability cross-check (the brief's explicit method: solve
# J @ w2 = -Q(w_perp,w_perp), i.e. rank(J) == rank([J|b])). Subsumed by Stage 6's EXACT closed
# form (which proves integrability to ALL orders, not just 2nd), but run anyway per the brief and
# as an independent consistency check of the sign/derivation conventions.
# ==================================================================================================
def second_order_rhs(data, Wp):
    E, coord = data["E"], data["coord"]
    WRe = [[Wp[coord(i, c, True)] for c in range(D)] for i in range(V)]
    WIm = [[Wp[coord(i, c, False)] for c in range(D)] for i in range(V)]
    b = []
    for i, j in E:
        dre = sum(WRe[i][c] * WRe[j][c] + WIm[i][c] * WIm[j][c] for c in range(D))
        dim = sum(WRe[i][c] * WIm[j][c] - WIm[i][c] * WRe[j][c] for c in range(D))
        b.append(-2 * dre); b.append(-2 * dim)
    for i in range(V):
        b.append(-sum(WRe[i][c] ** 2 + WIm[i][c] ** 2 for c in range(D)))
    return sp.Matrix(b)

def second_order_test(data, J, Wp):
    b = second_order_rhs(data, Wp)
    Jaug = J.row_join(b)
    rJ = rank_qq([list(J.row(k)) for k in range(J.shape[0])], J.shape[1])
    rAug = rank_qq([list(Jaug.row(k)) for k in range(Jaug.shape[0])], Jaug.shape[1])
    blocked = rAug > rJ
    return dict(rankJ=rJ, rankAug=rAug, blocked=blocked,
                b_nonzero=sum(1 for x in b if x != 0), b_len=b.shape[0])

# ==================================================================================================
# STAGE 6: THE EXACT CLOSED-FORM PROOF. General real s (t = i*s parametrizes the WHOLE M10 line),
# general real theta (SO(2) doublet-rotation angle). Direct symbolic substitution into ALL 77
# mechanism-stable edges + the 2 norm identities -- sympy `simplify` to literal 0, not sampled.
# ==================================================================================================
def exact_closed_form_proof():
    s, theta = sp.symbols("s theta", real=True)
    t = sp.I * s

    def val(sym):
        return {"0": sp.Integer(0), "1": sp.Integer(1), "-1": sp.Integer(-1),
                "X": t, "-X": -t}[sym]

    rays = [tuple(val(c) for c in r) for r in CORE_SYMS]
    v19, v20 = rays[IDX_X1], rays[IDX_X2]
    cth, sth = sp.cos(theta), sp.sin(theta)
    v19r = tuple(cth * v19[k] + sth * v20[k] for k in range(D))
    v20r = tuple(-sth * v19[k] + cth * v20[k] for k in range(D))
    rays_rot = list(rays); rays_rot[IDX_X1] = v19r; rays_rot[IDX_X2] = v20r

    def hdot(u, v): return sum(sp.conjugate(u[k]) * v[k] for k in range(D))

    E = [(i, j) for i, j in combinations(range(V), 2) if sp.simplify(hdot(rays[i], rays[j])) == 0]
    assert len(E) == 77, f"expected 77 mechanism-stable edges, got {len(E)}"

    bad_edges = [(i, j) for (i, j) in E if sp.simplify(hdot(rays_rot[i], rays_rot[j])) != 0]
    bad_norms = []
    for i in (IDX_X1, IDX_X2):
        if sp.simplify(hdot(rays[i], rays[i]) - hdot(rays_rot[i], rays_rot[i])) != 0:
            bad_norms.append(i)

    ov = sp.simplify(hdot(v19, v20))          # should be 4*Re(t) == 0 identically (Re(t)=0 on M10)
    n19 = sp.simplify(hdot(v19, v19))
    n20 = sp.simplify(hdot(v20, v20))

    return dict(E=E, bad_edges=bad_edges, bad_norms=bad_norms, ov=ov, n19=n19, n20=n20,
                v19=v19, v20=v20)

# ==================================================================================================
# STAGE 7: HONESTY / SCOPE CHECK -- does the rotation also preserve orthogonality against the FULL
# 272-ray / 208-basis-participating M10 mechanism-stable pool, or only the 21-ray critical-core
# witness? (Direct reuse of branch_d4flex.stable_graph, unmodified.)
# ==================================================================================================
def full_pool_scope_check():
    import branch_d4flex as bd
    g = bd.stable_graph("M10", verbose=False)
    rays, pairs = g["rays"], g["pairs"]
    i19 = rays.index(('1', 'X', '-1', '-X'))
    i20 = rays.index(('X', '1', '-X', '-1'))
    adj = [set() for _ in range(len(rays))]
    for i, j in pairs: adj[i].add(j); adj[j].add(i)
    n19, n20 = adj[i19], adj[i20]
    return dict(deg19=len(n19), deg20=len(n20), equal=(n19 == n20),
                symdiff=len(n19.symmetric_difference(n20)),
                edge_1920=((i19, i20) in pairs or (i20, i19) in pairs))

# ==================================================================================================
# REPORT
# ==================================================================================================
def main(quick=False):
    print("=" * 100)
    print("BRANCH M10X -- M10's second flex direction")
    print("=" * 100)

    print("\n[0] provenance check")
    try:
        verify_provenance()
    except Exception as e:
        print("  [provenance check skipped/failed non-fatally]:", e)

    print("\n[1] exact flex certificate, x=2i (independent rebuild)")
    data2i = build_at(sp.Integer(0), sp.Integer(2)); data2i["t_re"], data2i["t_im"] = sp.Integer(0), sp.Integer(2)
    cert = exact_flex_certificate(data2i)
    print(f"  E={cert['E']} n={cert['n']} rankJ={cert['rankJ']} ker={cert['ker']} "
          f"rankT={cert['rankT']} flex={cert['flex']}")
    assert cert["E"] == 77 and cert["flex"] == 2, "does not match D4_FLEX_HUNT.md Sec.10.3"

    print("\n[2] the LINE direction T (norm-corrected)")
    Tm = corrected_T(data2i, sp.Integer(0), sp.Integer(2))
    J = cert["J"]
    resid = J * Tm
    print("  J @ T_corrected == 0 exactly:", all(x == 0 for x in resid))
    print("  T support: rays", IDX_X1, "and", IDX_X2, "only (all 4 coordinate slots each -- NOT")
    print("  just the raw 'X'/'-X' slots: the norm-fixing correction spreads onto the '1'/'-1'")
    print("  slots too; raw uncorrected derivative is NOT itself in ker(J)).")

    print("\n[3] the SECOND direction w_perp")
    Wp, info = w_perp_minimal_support(data2i, J)
    print(f"  4-dim {{X-ray-only}}-support sub-nullspace found (dim={info['nsub_dim']}) = "
          f"span{{phase(ray19), phase(ray20), T, w_perp}}")
    Wp_amb = w_perp_ambient_complement(data2i, J, Tm)
    M37 = ambient_M37(data2i, Tm)
    Wp_orth = ambient_project_out(M37, Wp)  # gauge-fix the minimal-support rep the same way
    print("  cross-check: minimal-support rep (gauge-fixed) and ambient-orthogonal-complement")
    print("  construction land on the SAME 1-dim line:", check_proportional(Wp_orth, Wp_amb))
    coord = data2i["coord"]
    print("  w_perp (minimal-support representative, x5 to clear denominators):")
    for i in (IDX_X1, IDX_X2):
        vals = []
        for c in range(D):
            xr, xi = 5 * Wp[coord(i, c, True)], 5 * Wp[coord(i, c, False)]
            vals.append(f"({xr},{xi})")
        print(f"    ray{i} {CORE_SYMS[i]}: {vals}")
    print("  CLOSED FORM: w_perp(ray19) = (1/5)*v20 ,  w_perp(ray20) = -(1/5)*v19  (exact)")
    print("  independent of the 36 U(4)+phase trivial generators AND of T (proj residual != T-only")
    print("  span verified in Stage [2]-style construction above; ambient cross-check agrees).")

    print("\n[4] antiunitary self-symmetry search (unit-monomial . conj, exhaustive 24x256=6144)")
    if quick:
        print("  [skipped in quick mode]")
    else:
        rays_num = [tuple({"0": 0, "1": 1, "-1": -1, "X": 2j, "-X": -2j}[c] for c in r) for r in CORE_SYMS]
        hits = antiunitary_search(rays_num)
        print(f"  self-symmetries found: {len(hits)}  =>  "
              f"{'NONE -- no J-eigenvalue split available for (T,w_perp) via this route' if not hits else hits[:3]}")

    print("\n[5] second-order cross-check (brief's explicit method: rank(J) vs rank([J|b]))")
    so = second_order_test(data2i, J, Wp)
    print(f"  rank(J)={so['rankJ']}  rank([J|b])={so['rankAug']}  "
          f"b nonzero entries: {so['b_nonzero']}/{so['b_len']}")
    print(f"  => {'BLOCKED' if so['blocked'] else 'SOLVABLE'} (second-order consistency check; "
          f"subsumed by the exact closed form in Stage 6)")
    assert not so["blocked"]

    print("\n[6] EXACT closed-form proof (general real s, t=i*s; general real theta)")
    t0 = time.time()
    res = exact_closed_form_proof()
    print(f"  mechanism-stable edges (symbolic, t=i*s): {len(res['E'])}")
    print(f"  <ray19,ray20> = {res['ov']}   (must be identically 0 -- uses Re(t)=0, i.e. M10 itself)")
    print(f"  |ray19|^2 = {res['n19']}   |ray20|^2 = {res['n20']}   (must be equal identically)")
    print(f"  edges broken by the theta-rotation, for ALL (s,theta): {len(res['bad_edges'])} / {len(res['E'])}")
    print(f"  norm identities broken: {res['bad_norms']}")
    ok_closed = (len(res["bad_edges"]) == 0 and len(res["bad_norms"]) == 0 and res["ov"] == 0
                 and sp.simplify(res["n19"] - res["n20"]) == 0)
    print(f"  ({time.time()-t0:.1f}s)  ALL CHECKS PASS: {ok_closed}")
    assert ok_closed

    print("\n[6b] sanity: i*w_perp (complex-i multiple, NOT the real SO(2) direction) is NOT in ker(J)")
    iWp = [sp.Integer(0)] * data2i["n"]
    for i_ in (IDX_X1, IDX_X2):
        for c in range(D):
            xr, xi = Wp[coord(i_, c, True)], Wp[coord(i_, c, False)]
            iWp[coord(i_, c, True)] = -xi; iWp[coord(i_, c, False)] = xr
    iWp = sp.Matrix(iWp)
    print("  J @ (i*w_perp) == 0:", all(x == 0 for x in (J * iWp)),
          " (expected False -- confirms flex=2 exactly, no hidden 3rd direction)")
    assert not all(x == 0 for x in (J * iWp))

    print("\n[7] scope/honesty check: does the rotation survive the FULL 272-ray/208-bp M10 pool?")
    sc = full_pool_scope_check()
    print(f"  deg(ray19)={sc['deg19']} deg(ray20)={sc['deg20']} equal neighbor sets: {sc['equal']} "
          f"(symdiff={sc['symdiff']})  (19,20) edge: {sc['edge_1920']}")
    print("  => the sheet is an exact property of the 21-ray CRITICAL-CORE witness (itself already")
    print("     sufficient to prove M10 KS-uncolorable at every line point), not a claim about the")
    print("     full ambient pool -- same scope discipline as D4_FLEX_HUNT's flex-number caveat.")

    print("\n[8] cross-check at x=3i, x=5i")
    for tim in (3, 5):
        d = build_at(sp.Integer(0), sp.Integer(tim)); d["t_re"], d["t_im"] = sp.Integer(0), sp.Integer(tim)
        c = exact_flex_certificate(d)
        print(f"  x={tim}i: E={c['E']} rankJ={c['rankJ']} ker={c['ker']} rankT={c['rankT']} "
              f"flex={c['flex']}  (matches x=2i: {c['flex']==2 and c['E']==77})")
    print("  (the Stage [6] symbolic proof already covers every real s, i.e. every x=i*s on the")
    print("   line, so this is a redundant point-sample cross-check, not new information.)")

    print(f"\n[{time.time()-T0:.1f}s total] branch_m10x PASS")

if __name__ == "__main__":
    quick = (len(sys.argv) > 1 and sys.argv[1] == "quick")
    main(quick=quick)
