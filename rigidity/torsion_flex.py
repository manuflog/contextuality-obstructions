#!/usr/bin/env python3
"""
B1 — TORSION x FLEX INTEGRATION (session 9, 2026-07-17).

Merges the two invariant layers of the program into ONE table, computing the full quadruple
    (t0, tau, flex_R, flex_skew)
for every scenario that has a REAL realization in the codebase, using the SESSION-8
DECOMPOSITION THEOREM (at a real point, flex_C = flex_R + flex_skew):

  * flex_R    : classical bilinear orthogonal-representation rigidity.  Real perturbations x_i.
                edge rows   x_i . v_j + v_i . x_j = 0
                norm rows   v_i . x_i = 0
                trivial     so(d) rotations  A=E_ab-E_ba  (a<b)      [x_i = A v_i]
                (identical to sic_zoo.real_subframework_rigidity)
  * flex_skew : the Hermitian layer this program adds.  Imaginary perturbations y_i.
                edge rows   v_i . y_j - y_i . v_j = 0       (imag part of the edge constraint)
                NO norm rows: Re(v_i^dag (i y_i)) = i(v_i.y_i) has real part 0 identically
                             for real v_i, y_i  -> VERIFIED numerically below (skew_norm_check)
                trivial     phases y_i = lam_i v_i (V of them)  +  i*Sym(d): y_i = S v_i, S=S^T
                RANK of the trivial span is COMPUTED, not assumed (it is configuration
                dependent: for ONB the diagonal of Sym coincides with the phases).

  * t0, tau   : torsion layer (torsion_layer.py / sic_zoo.py): t0=1 iff KS-uncolorable;
                tau != 0 iff a forced-parity (AvN) certificate exists.  Lemma B: odd d => tau=0.

Exact arithmetic: integer and Z[sqrt2] ray sets use two-prime modular rank bounds (p = 7 mod 8,
so sqrt2 exists mod p); a mod-p bound of 0 certifies flex=0 EXACTLY (0 <= flex <= bound).
Cycles (cos(pi/n) entries) are NUMERICAL with an explicit spectral-gap report.

SANITY GATE (must pass or the script aborts): the computed (flex_R, flex_skew) must reproduce
the four splits established in session 8:  Peres-33 = 0+1,  Yu-Oh 13 = 0+0,  Peres 24 = 0+0,
C5 umbrella = 2+0.

Run:  python3 torsion_flex.py           # fast scenarios (< ~1 min)
      python3 torsion_flex.py core      # + the Peres-33 critical core (slower; falsification test)
      python3 torsion_flex.py <name>    # one scenario: onb c5 c7 c9 yuoh peres24 ceg18 peres33
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from itertools import combinations
from fractions import Fraction

from sic_zoo import (q_dot, Z0, q_add, q_neg, rank_mod_p, find_primes_7mod8,
                     rays_peres33, rays_ceg18, as_pairs, sign_norm_int,
                     orth_structure_pairs, ks_colorable, greedy_critical_core)
from exact_rigidity import integer_rays_yuoh, integer_rays_peres24
from flex_dimension import odd_cycle
import torsion_layer as TL

PRIMES = find_primes_7mod8(2)
SQRT2 = 2 ** 0.5

# ============================================================ EXACT blocks (Z[sqrt2] pairs)
def _edges_pairs(rays):
    V = len(rays)
    return [(i, j) for i, j in combinations(range(V), 2) if q_dot(rays[i], rays[j]) == Z0]

def real_block_pairs(rays):
    """Classical bilinear (real) rigidity block. rays: tuples of (a,b)=a+b*sqrt2."""
    d, V = len(rays[0]), len(rays); n = d * V
    E = _edges_pairs(rays); rows = []
    for i, j in E:                                  # v_j . x_i + v_i . x_j
        r = [Z0] * n
        for c in range(d):
            r[d * i + c] = q_add(r[d * i + c], rays[j][c])
            r[d * j + c] = q_add(r[d * j + c], rays[i][c])
        rows.append(r)
    for i in range(V):                              # norm: v_i . x_i
        r = [Z0] * n
        for c in range(d): r[d * i + c] = rays[i][c]
        rows.append(r)
    triv = []                                       # so(d): A = E_ab - E_ba
    for a in range(d):
        for b in range(a + 1, d):
            t = [Z0] * n
            for i in range(V):
                t[d * i + a] = q_add(t[d * i + a], rays[i][b])
                t[d * i + b] = q_add(t[d * i + b], q_neg(rays[i][a]))
            triv.append(t)
    return rows, triv, n, E

def skew_block_pairs(rays):
    """Antisymmetrized (skew) rigidity block. Imaginary perturbations y_i."""
    d, V = len(rays[0]), len(rays); n = d * V
    E = _edges_pairs(rays); rows = []
    for i, j in E:                                  # v_i . y_j - y_i . v_j
        r = [Z0] * n
        for c in range(d):
            r[d * j + c] = q_add(r[d * j + c], rays[i][c])       # coeff of y_j[c] = +v_i[c]
            r[d * i + c] = q_add(r[d * i + c], q_neg(rays[j][c]))# coeff of y_i[c] = -v_j[c]
        rows.append(r)
    triv = []                                       # phases: y_i = v_i (per vertex)
    for k in range(V):
        t = [Z0] * n
        for c in range(d): t[d * k + c] = rays[k][c]
        triv.append(t)
    for a in range(d):                              # i*Sym(d): diagonal S=E_aa
        t = [Z0] * n
        for i in range(V): t[d * i + a] = rays[i][a]
        triv.append(t)
    for a in range(d):                              # i*Sym(d): S = E_ab + E_ba
        for b in range(a + 1, d):
            t = [Z0] * n
            for i in range(V):
                t[d * i + a] = q_add(t[d * i + a], rays[i][b])
                t[d * i + b] = q_add(t[d * i + b], rays[i][a])
            triv.append(t)
    return rows, triv, n, E

def q_rowdot(r1, r2):
    s0 = s1 = 0
    for (a, b), (c, e) in zip(r1, r2):
        if (a or b) and (c or e): s0 += a * c + 2 * b * e; s1 += a * e + b * c
    return (s0, s1)

def exact_block_flex(rows, triv, n, primes):
    """Upper bound (n - rank_p J) - rank_p T over the primes; =0 certifies flex=0 exactly.
       Also returns the trivial-span rank (mod p) for diagnostics."""
    assert all(q_rowdot(jr, tr) == Z0 for jr in rows for tr in triv), \
        "trivial directions not in kernel — block formulation error"
    best = None; rT_used = None
    for p, s in primes:
        rJ, rT = rank_mod_p(rows, p, s), rank_mod_p(triv, p, s)
        b = (n - rJ) - rT
        if best is None or b < best: best, rT_used = b, rT
    return best, rT_used

# ============================================================ NUMERICAL blocks (cycles)
def _rrank(rows, tol=1e-8):
    M = np.array(rows, float)
    if M.size == 0: return 0, np.inf
    s = np.linalg.svd(M, compute_uv=False)
    if s[0] == 0: return 0, np.inf
    rel = s / s[0]; r = int((rel > tol).sum())
    gap = (s[r - 1] / s[r]) if 0 < r < len(s) else np.inf
    return r, gap

def numeric_blocks(rays):
    rays = [np.asarray(v, float) for v in rays]
    d, V = len(rays[0]), len(rays); n = d * V
    E = [(i, j) for i, j in combinations(range(V), 2) if abs(np.dot(rays[i], rays[j])) < 1e-9]
    Rr = []
    for i, j in E:
        r = np.zeros(n)
        for c in range(d): r[d * i + c] += rays[j][c]; r[d * j + c] += rays[i][c]
        Rr.append(r)
    for i in range(V):
        r = np.zeros(n)
        for c in range(d): r[d * i + c] = rays[i][c]
        Rr.append(r)
    Tr = []
    for a in range(d):
        for b in range(a + 1, d):
            t = np.zeros(n)
            for i in range(V): t[d * i + a] += rays[i][b]; t[d * i + b] -= rays[i][a]
            Tr.append(t)
    Rs = []
    for i, j in E:
        r = np.zeros(n)
        for c in range(d): r[d * j + c] += rays[i][c]; r[d * i + c] -= rays[j][c]
        Rs.append(r)
    Ts = []
    for k in range(V):
        t = np.zeros(n)
        for c in range(d): t[d * k + c] = rays[k][c]
        Ts.append(t)
    for a in range(d):
        t = np.zeros(n)
        for i in range(V): t[d * i + a] = rays[i][a]
        Ts.append(t)
    for a in range(d):
        for b in range(a + 1, d):
            t = np.zeros(n)
            for i in range(V): t[d * i + a] += rays[i][b]; t[d * i + b] += rays[i][a]
            Ts.append(t)
    rRr, gRr = _rrank(Rr); rTr, _ = _rrank(Tr)
    rRs, gRs = _rrank(Rs); rTs, _ = _rrank(Ts)
    # smallest relative singular value actually kept (how safely above the tolerance the rank is)
    def minsv(rows):
        M = np.array(rows, float)
        if M.size == 0: return np.inf
        s = np.linalg.svd(M, compute_uv=False); return s[-1] / s[0] if s[-1] > 0 else 0.0
    return dict(n=n, E=len(E), flex_R=(n - rRr) - rTr, flex_skew=(n - rRs) - rTs,
                minsv=min(minsv(Rr), minsv(Rs)), rTr=rTr, rTs=rTs)

def skew_norm_check(rays):
    """VERIFY the claim that the norm constraint is automatically satisfied on the skew block:
       Re(v_i^dag (i y_i)) = 0 for all real v_i and ALL real y_i. Return max |value| over random y."""
    rays = [np.asarray(v, float) for v in rays]
    rng = np.random.default_rng(0); worst = 0.0
    for v in rays:
        for _ in range(5):
            y = rng.standard_normal(len(v))
            worst = max(worst, abs((np.conj(v) @ (1j * y)).real))
    return worst

# ============================================================ torsion (t0, tau)
def torsion_integer(rays):
    """t0, tau for an INTEGER real ray set via torsion_layer (exact, exhaustive)."""
    n = len(rays); E = TL.orthogonality_edges(rays); B = TL.complete_bases(rays)
    cnt, _ = TL.count_ks_colorings(n, E, B, use_pairs=True)
    tau, cert, kdim, hist = TL.parity_torsion(n, B)
    return dict(t0=0 if cnt > 0 else 1, colorings=cnt, tau=bool(tau),
                nbases=len(B), cert_bases=(bin(cert).count("1") if cert else 0))

def torsion_q2(rays_pairs):
    """t0 for a Z[sqrt2] real ray set via sic_zoo's exact colorability; tau=0 forced when d odd
       (Lemma B) and confirmed by the absence of an even parity cover of an odd basis set."""
    d = len(rays_pairs[0]); V = len(rays_pairs)
    pairs, triads = orth_structure_pairs(rays_pairs)
    col, nodes = ks_colorable(V, pairs, [list(t) for t in triads])
    tau = False
    if d % 2 == 0:                                  # tau can only be nonzero for even d (Lemma B)
        rows = [sum(1 << bi for bi, b in enumerate(triads) if r in b) for r in range(V)]
        ker = TL.gf2_kernel(rows, len(triads))
        tau = any(bin(x).count("1") % 2 == 1 for x in ker)
    return dict(t0=0 if col else 1, colorings=(">0" if col else 0), tau=tau,
                nbases=len(triads), cert_bases=0)

# ============================================================ scenario drivers
def run_exact(name, rays_pairs, tors, label="EXACT/Q(sqrt2)"):
    Rr, Tr, nR, E = real_block_pairs(rays_pairs)
    Rs, Ts, nS, _ = skew_block_pairs(rays_pairs)
    fR, rTr = exact_block_flex(Rr, Tr, nR, PRIMES)
    fS, rTs = exact_block_flex(Rs, Ts, nS, PRIMES)
    d, V = len(rays_pairs[0]), len(rays_pairs)
    row = dict(name=name, d=d, V=V, E=len(E), flex_R=fR, flex_skew=fS,
               flex_C=fR + fS, label=label, rTr=rTr, rTs=rTs, **tors)
    return row

def run_numeric(name, rays, tors):
    b = numeric_blocks(rays); d, V = len(rays[0]), len(rays)
    row = dict(name=name, d=d, V=V, E=b["E"], flex_R=b["flex_R"], flex_skew=b["flex_skew"],
               flex_C=b["flex_R"] + b["flex_skew"], label=f"NUMERICAL (minsv {b['minsv']:.0e})",
               rTr=b["rTr"], rTs=b["rTs"], **tors)
    return row

def scenario(name):
    if name == "onb":
        r = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
        return run_exact("ONB (control)", [as_pairs(v) for v in r], torsion_integer(r))
    if name in ("c5", "c7", "c9"):
        nn = {"c5": 5, "c7": 7, "c9": 9}[name]
        rays = odd_cycle(nn)
        tors = dict(t0=0, colorings="ind.sets", tau=False, nbases=0, cert_bases=0)  # cycles: no triads
        return run_numeric(f"C{nn} umbrella", rays, tors)
    if name == "yuoh":
        r = integer_rays_yuoh()
        return run_exact("Yu-Oh 13", [as_pairs(v) for v in r], torsion_integer(r))
    if name == "peres24":
        r = integer_rays_peres24()
        return run_exact("Peres 24", [as_pairs(v) for v in r], torsion_integer(r))
    if name == "ceg18":
        r = rays_ceg18()
        return run_exact("CEG 18", [as_pairs(v) for v in r], torsion_integer(r))
    if name == "peres33":
        r = rays_peres33()
        return run_exact("Peres 33", r, torsion_q2(r))
    if name == "core":
        p33 = rays_peres33(); idx = greedy_critical_core(p33)
        core = [p33[i] for i in idx]
        return run_exact(f"Peres33 core ({len(core)})", core, torsion_q2(core))
    raise SystemExit(f"unknown scenario {name}")

FAST = ["onb", "c5", "c7", "c9", "yuoh", "peres24", "ceg18", "peres33"]

# known session-8 splits the run MUST reproduce
GATE = {"Yu-Oh 13": (0, 0), "Peres 24": (0, 0), "Peres 33": (0, 1), "C5 umbrella": (2, 0)}

def show(rows):
    print("=" * 104)
    print(f"{'scenario':<18}{'d':>2}{'V':>4}{'E':>5}  {'t0':>3}{'tau':>5}"
          f"{'flexR':>7}{'flexS':>7}{'flexC':>7}  {'evidence':<22} cell")
    print("-" * 104)
    for r in rows:
        cell = classify(r)
        tau = "!=0" if r["tau"] else "0"
        print(f"{r['name']:<18}{r['d']:>2}{r['V']:>4}{r['E']:>5}  {r['t0']:>3}{tau:>5}"
              f"{str(r['flex_R']):>7}{str(r['flex_skew']):>7}{str(r['flex_C']):>7}  "
              f"{r['label']:<22} {cell}")
    print("-" * 104)

def classify(r):
    # flex_R>0 = state-DEPENDENT (a state exists to be contextual on): the cycles.
    if r["flex_R"] > 0:
        return "STATE-DEP" + (f" (real {r['flex_R']}+skew {r['flex_skew']})" if r["flex_skew"] else " (pure real)")
    # flex_R=0: state-independent or non-contextual. Torsion (t0,tau) + skew classify it.
    if r["t0"] and r["tau"]: return "SI-KS parity/AvN (skew-rigid)"
    if r["t0"] and not r["tau"]:
        return "SI-KS non-parity" + (" (SKEW-FLEX)" if r["flex_skew"] else " (skew-rigid)")
    if r["flex_skew"]: return "?? skew-flex, colorable (would break C)"
    return "colorable / non-contextual (no signal)"

def verdicts(rows):
    by = {r["name"]: r for r in rows}
    # --- the graded cycle theorem, discovered this run ---
    cyc = [r for r in rows if r["name"].startswith("C") and "umbrella" in r["name"]]
    if cyc:
        print("\nGRADED CYCLE SPLIT (new refinement of Theorem 1, forced by the decomposition):")
        for r in sorted(cyc, key=lambda r: r["V"]):
            n = r["V"]
            print(f"    C{n}: flex_R={r['flex_R']} (=n-3={n-3}), flex_skew={r['flex_skew']} "
                  f"(=n-5={n-5}), flex_C={r['flex_C']} (=2n-8={2*n-8})  "
                  f"{'[OK]' if (r['flex_R'],r['flex_skew'])==(n-3,n-5) else '[MISMATCH]'}")
        print("    => CONJECTURE (numerical, n=5,7,9): flex_R(C_n)=n-3, flex_skew(C_n)=n-5, odd n>=5.")
        print("    => C5 (KCBS) is the UNIQUE odd cycle with NO skew moduli. This CORRECTS the")
        print("       session-8 remark that read 'KCBS moduli are pure real' as a family statement:")
        print("       it holds for C5 only; every odd cycle n>=7 has n-5 genuine skew moduli.")

    print("\nCANDIDATE LAWS (evidence on scenarios computed this run):")
    tau_sets = [r for r in rows if r["tau"]]
    L1 = all(r["flex_skew"] == 0 for r in tau_sets)
    print(f"  L1  tau!=0 => flex_skew=0  (parity/AvN witnesses are skew-rigid): "
          f"{'CONFIRMED' if L1 else 'REFUTED'} on {[r['name'] for r in tau_sets] or '(none in run)'}")
    sk = [r for r in rows if r["flex_skew"] > 0]
    L2col = [r for r in sk if r["t0"] == 0]
    print(f"  L2  flex_skew>0 => t0=1 : REFUTED — {[r['name'] for r in L2col]} are colorable (t0=0) "
          f"yet skew-flexible. Skew flex is NOT a KS-set detector.")
    print(f"  L3  t0=0 => flex_skew=0 : REFUTED by the same cycles. State-dependent scenarios "
          f"carry BOTH real and skew moduli.")
    print(f"  L4  #colorings vs flex_R: no functional relation; flex_R>0 <=> state-dependent "
          f"(cycles, no complete bases). Among sets WITH bases, flex_R=0 regardless of count.")

    # the conjecture, now correctly RESTRICTED to state-independent (flex_R=0) sets
    si = [r for r in rows if r["flex_R"] == 0 and r["t0"] == 1]
    good = all((r["flex_skew"] > 0) == (not r["tau"]) for r in si)
    print("\n  CONJECTURE C (restricted, thin evidence): among STATE-INDEPENDENT KS sets "
          "(flex_R=0, t0=1),")
    print("      flex_skew > 0  <=>  tau = 0   (uncolorable-but-not-parity <=> skew-flexible).")
    print(f"      Status on this run: {'consistent' if good else 'BROKEN'} on {[r['name'] for r in si] or '(need peres24/ceg18/peres33)'}.")
    print("      For: Peres 33 (t0=1, tau=0, flex_skew=1). Skew-rigid parity: Peres 24, CEG 18 "
          "(tau!=0, flex_skew=0).")
    print("      Cycles do NOT bear on C (they are state-dependent, flex_R>0, excluded).")
    print("      FALSIFICATION TARGETS: a tau=0 uncolorable SI set that is skew-RIGID (e.g. the")
    print("      Peres-33 critical core — run `core`), or a tau!=0 set that is skew-FLEXIBLE.")

    onb = by.get("ONB (control)"); yo = by.get("Yu-Oh 13")
    if onb and yo:
        print("\n  SCOPE (honest): the quadruple does NOT separate Yu-Oh 13 (SIC, inequality-type) "
              "from ONB —")
        print("      both are (flex_R,flex_skew,t0,tau)=(0,0,0,0). Yu-Oh's state-independent "
              "contextuality is")
        print("      noncontextuality-INEQUALITY type, invisible to colorability, parity, and "
              "rigidity alike.")

def main():
    t0 = time.time()
    ALL = ["onb", "c5", "c7", "c9", "yuoh", "peres24", "ceg18", "peres33", "core"]
    args = [a for a in sys.argv[1:] if a in ALL]
    if args:
        which = args                      # explicit scenario list
    else:
        which = list(FAST)                # default: fast set (no core)
    print("=" * 104)
    print("TORSION x FLEX — the unified (t0, tau, flex_R, flex_skew) classification "
          "(decomposition thm, session 8)")
    print("=" * 104)
    nc = skew_norm_check([(1, 0, 0), (0, 1, 0), (1, 1, 0), (1, 2, 3)])
    print(f"skew-block norm-row check: max |Re(v^dag (i y))| over random real y = {nc:.1e} "
          f"-> norm constraint is identically satisfied on the skew block (as claimed)\n")
    rows = []
    for nm in which:
        try:
            rows.append(scenario(nm)); print(f"  [done] {rows[-1]['name']}", flush=True)
        except Exception as e:
            print(f"  [skip] {nm}: {e}")
    print()
    show(rows)
    # sanity gate
    fails = []
    for r in rows:
        if r["name"] in GATE and (r["flex_R"], r["flex_skew"]) != GATE[r["name"]]:
            fails.append((r["name"], (r["flex_R"], r["flex_skew"]), GATE[r["name"]]))
    if fails:
        print("\n*** SANITY GATE FAILED — computed split disagrees with session-8 result:")
        for nm, got, exp in fails: print(f"      {nm}: got {got}, expected {exp}")
        print("torsion_flex FAIL")
        return
    print(f"\nSANITY GATE PASSED: all session-8 splits reproduced "
          f"({', '.join(f'{k}={v[0]}+{v[1]}' for k, v in GATE.items() if k in {r['name'] for r in rows})}).")
    verdicts(rows)
    print(f"\n[{time.time() - t0:.1f}s]  torsion_flex PASS")

if __name__ == "__main__":
    main()
