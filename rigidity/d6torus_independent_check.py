# INDEPENDENT verification of the d=6 grading 3-torus.
# Their witness (core symbols + gradings extracted by their code); MY verification of the
# all-orders identity, written from scratch:
#   For every edge (i,j): terms t_c = conj(u_c) v_c with u,v over {0,+-1,+-X}. Working in
#   Z[X,X^-1] with X formal (conj X = X^-1), bucket the coefficient of each term by the triple
#   (dX, dg1, dg2) where dX = X-power of the term and dgk = gk[j][c] - gk[i][c].
#   The rephasing v_ic -> e^{i(g1 s1 + g2 s2)} v_ic times the theta-dependence preserves the
#   edge for ALL (theta, s1, s2) iff every bucket's integer coefficient sum is ZERO.
# This is exactly "3393 coefficient sums vanish over Z" claimed by their torus stage, but
# recomputed here with independent bucketing code.
import sys
from collections import defaultdict
sys.path.insert(0, '/sessions/friendly-exciting-ptolemy/mnt/contextuality-obstructions/rigidity')
import branch_d6fourthorder as so4

f6, fc, CFG = so4.f6, so4.fc, so4.CFG
core_syms, _, core_pairs = f6.load_core(CFG)
V, d = CFG["exp_V"], CFG["d"]

def cache_load(name):
    import json, os
    p = f"/sessions/friendly-exciting-ptolemy/mnt/contextuality-obstructions/rigidity/{name}.cache.json"
    return json.load(open(p)) if os.path.exists(p) else None

pt = 'x5'
cert = cache_load(f"d6flexcert_cert_{pt}")
rays = fc.rays_at(core_syms, pt)
vt = fc.v_theta(core_syms, rays)
ws = {}
for nm in ("w3241", "w3243"):
    v = [0] * (2 * d * V)
    for i, val in cert["vectors"][nm]["entries"]:
        v[i] = val
    ws[nm] = v
G1 = so4._extract_grading(core_syms, rays, ws["w3241"], V, d)
G2 = so4._extract_grading(core_syms, rays, ws["w3243"], V, d)

# my own symbol parser: entry symbol -> (integer coeff, X-power) with coeff in {+1,-1}, or None
def parse(sym):
    if sym == '0': return None
    sign = -1 if sym.startswith('-') else 1
    b = sym.lstrip('-')
    if b == '1': return (sign, 0)
    if b == 'X': return (sign, 1)
    raise ValueError(sym)

bad = 0; buckets_total = 0
for (i, j) in core_pairs:
    acc = defaultdict(int)
    for c in range(d):
        pu = parse(core_syms[i][c]); pv = parse(core_syms[j][c])
        if pu is None or pv is None: continue
        (su, xu), (sv, xv) = pu, pv
        # conj(u_c) v_c = su*sv * X^(xv - xu); grading shift of the rephased edge term:
        key = (xv - xu, G1[j][c] - G1[i][c], G2[j][c] - G2[i][c])
        acc[key] += su * sv
    buckets_total += len(acc)
    for k, s in acc.items():
        if s != 0:
            bad += 1
            if bad <= 3: print("NONZERO BUCKET", (i, j), k, s)
print(f"edges checked: {len(core_pairs)}  buckets: {buckets_total}  nonzero buckets: {bad}")
print("TORUS IDENTITY " + ("VERIFIED: every (X-power, dg1, dg2) bucket sums to zero over Z"
      if bad == 0 else "FAILED"))
