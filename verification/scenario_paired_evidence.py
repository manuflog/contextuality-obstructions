# V48 - CONJECTURE 1 (scenario-paired classification) EVIDENCE HARNESS.
# Full run (16 chunked modes, ~5.6k records at d=4 and d=8, 0 mismatches) pinned in
# INDEX.md; records regenerable via CLI modes (drops/boundary/double/random/random2/
# pm2/eigen/magic). Default no-arg run = selftest. Expected: 'selftest: |L|=64, ...'
# c1_paired_probe.py -- Evidence probe for CONJECTURE 1 (scenario-paired classification).
#
# Conjecture (note v3, conj:pair): for every closed-triple Weyl family F at even d,
#     CF(F, rho) = 0   <=>   c(rho) in conv L(F)
# in the scenario-paired formulation of paper C.
#
# Formulation implemented (the maximal safe reading of paper C):
#   * L(F)  = spectral global sections: assignments lambda: O(F) -> Z_d with
#             lambda_v in spec(W_v) and sum_{v in C} lambda_v = s_C (mod d) for every
#             context C, where omega^{s_C} is the central phase of the closed triple
#             (pinned Weyl convention of verification/weyl.py).
#   * c(rho) = the scenario-PAIRED correlation vector: for every context C = (u, v, w)
#             and every character (a, b) in Z_d^2 \ {0}, the moment
#             sum_j omega^{-(a j1 + b j2)} e_C(j1, j2), evaluated within the host
#             context C (this is the pairing; each character carries the host
#             context's phase bookkeeping). At d=2 this reduces affinely to paper C's
#             c(rho) = (tr rho T_v) by the bijection lemma; at d=4 on the census
#             family it is the full moment space of V43 (faithful dim 33).
#   * CF     = sheaf-theoretic contextual fraction: LP over subnormalized mixtures of
#             deterministic global sections dominated by the empirical model, with
#             constraint rows over ALL spectral joint outcomes (including zero-
#             probability ones), per the pinned negative result of paper C that the
#             non-spectral relaxation under-computes CF.
#
# Tests: (a) CF via LP, (b) hull membership via LP (L_inf distance to conv c(L)),
# (c) for the census family (cert4 minus context 0) the exact-integer 23,256-facet
# classifier of d4_facet_classes.npz as an independent arbiter.
# A refutation of one direction = a state with CF = 0 strictly outside the hull;
# of the other = CF > 0 strictly inside.
#
# All outputs are written to the outputs directory as JSON lines; no repo file is
# touched. Chunkable: each invocation is bounded and appends.

import sys, os, json, itertools, argparse, time
import numpy as np
import scipy.optimize as so

REPO_VER = "/sessions/quirky-eloquent-babbage/mnt/contextuality-obstructions/verification"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "c1_records")
if not os.path.isdir(REPO_VER):
    REPO_VER = "/Users/manuflog/Developer/contextuality-obstructions/verification"
    OUT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_VER)
from weyl import build  # W(v) = tau^{-q(v)} kron_i X^{a_i} Z^{b_i}, per-site labels

D = int(os.environ.get("C1_D", "4"))   # local dimension (4 default; 8 supported)
X, Z, w4, tau, W, N = build(D, 2)      # N = D^2

CERT6 = [[tuple(v) for v in it["ctx"]]
         for it in json.load(open(os.path.join(REPO_VER, "cert4_min.json")))["items"]]
if D == 8:
    # Doubled certified family: v -> 2v mod 8. Closed triples (sums 0 mod 8),
    # commuting (symplectic forms 4*<u,v> = 0 mod 8 since <u,v> = 0 mod 4):
    # a genuine closed-triple Weyl family at d=8 in dimension 64.
    CERT6 = [[tuple((2 * np.array(v)) % 8) for v in C] for C in CERT6]

# Peres-Mermin doubled into d=4 (order-2 observables 2*v): a distinct closed-triple
# Weyl family at d=4 (AvN expected: exercises the L(F) = empty branch of the iff).
PM2 = [[(2,0,0,0),(0,0,2,0),(2,0,2,0)],[(0,0,0,2),(0,2,0,0),(0,2,0,2)],
       [(2,0,0,2),(0,2,2,0),(2,2,2,2)],[(2,0,0,0),(0,0,0,2),(2,0,0,2)],
       [(0,0,2,0),(0,2,0,0),(0,2,2,0)],[(2,0,2,0),(0,2,0,2),(2,2,2,2)]]


def s_of(C):
    M = np.eye(N, dtype=complex)
    for v in C:
        M = M @ W(np.array(v))
    t = np.trace(M)
    assert abs(t) > 1e-9, "context product not central"
    return int(np.round(np.angle(t / abs(t)) / (2 * np.pi / D))) % D


def spec_of(v):
    ev = np.linalg.eigvals(W(np.array(v)))
    return sorted(set(int(np.round(np.angle(x) / (2 * np.pi / D))) % D for x in ev))


class Family:
    """A closed-triple Weyl family (list of commuting triples summing to 0 mod d)."""

    def __init__(self, ctxs, name):
        self.name = name
        self.CTX = [list(map(tuple, C)) for C in ctxs]
        self.obs = sorted({t for C in self.CTX for t in C})
        self.oi = {t: k for k, t in enumerate(self.obs)}
        self.S = [s_of(C) for C in self.CTX]
        self.spec = {v: spec_of(v) for v in self.obs}
        # joint outcome structure per context: outcomes are (j1, j2) over
        # spec(C[0]) x spec(C[1]) with j3 = (s - j1 - j2) % D required in spec(C[2]).
        self.outcomes = []
        self.proj = []
        for ci, C in enumerate(self.CTX):
            Ws = [W(np.array(v)) for v in C]
            P1 = {j: sum((w4 ** (-a * j)) * np.linalg.matrix_power(Ws[0], a)
                         for a in range(D)) / D for j in self.spec[C[0]]}
            P2 = {j: sum((w4 ** (-a * j)) * np.linalg.matrix_power(Ws[1], a)
                         for a in range(D)) / D for j in self.spec[C[1]]}
            outs, prj = [], {}
            for j1 in self.spec[C[0]]:
                for j2 in self.spec[C[1]]:
                    j3 = (self.S[ci] - j1 - j2) % D
                    if j3 not in self.spec[C[2]]:
                        continue  # spectrally forbidden: projector provably 0
                    outs.append((j1, j2))
                    prj[(j1, j2)] = P1[j1] @ P2[j2]
            self.outcomes.append(outs)
            self.proj.append(prj)
        self.L = self._enum_L()
        self.VL = np.array([self._cvec_det(l) for l in self.L]) if len(self.L) else None
        self.CH = [(a, b) for a in range(D) for b in range(D) if (a, b) != (0, 0)]

    def _enum_L(self, cap=200000):
        """DFS enumeration of spectral global sections."""
        nC = len(self.CTX)
        order, seen = [], set()
        for C in self.CTX:
            for v in C:
                if v not in seen:
                    seen.add(v); order.append(v)
        pos = {v: k for k, v in enumerate(order)}
        # for each context, index of its last-assigned observable in DFS order
        last = [max(pos[v] for v in C) for C in self.CTX]
        by_last = {}
        for ci, lp in enumerate(last):
            by_last.setdefault(lp, []).append(ci)
        sols, assign = [], {}
        work = [0]

        class TooBig(Exception):
            pass

        def rec(k):
            work[0] += 1
            if work[0] > 3_000_000 or len(sols) > cap:
                raise TooBig
            if k == len(order):
                sols.append([assign[v] for v in self.obs_order_cache])
                return
            v = order[k]
            for val in self.spec[v]:
                assign[v] = val
                ok = True
                for ci in by_last.get(k, []):
                    tot = sum(assign[u] for u in self.CTX[ci]) % D
                    if tot != self.S[ci]:
                        ok = False
                        break
                if ok:
                    rec(k + 1)
            del assign[v]

        self.obs_order_cache = self.obs
        try:
            rec(0)
        except TooBig:
            raise AssertionError("section enumeration exceeds work budget")
        return [dict(zip(self.obs, s)) for s in sols]

    def _cvec_det(self, lam):
        """Scenario-paired correlation vector of a deterministic global section."""
        out = []
        for ci, C in enumerate(self.CTX):
            j1, j2 = lam[C[0]], lam[C[1]]
            for (a, b) in [(a, b) for a in range(D) for b in range(D) if (a, b) != (0, 0)]:
                z = np.exp(-2j * np.pi / D * (a * j1 + b * j2))
                out += [z.real, z.imag]
        return np.array(out)

    def emodel(self, rho):
        """Empirical model: per-context probabilities over spectral joint outcomes."""
        es = []
        for ci in range(len(self.CTX)):
            e = {j: float(np.real(np.trace(rho @ self.proj[ci][j])))
                 for j in self.outcomes[ci]}
            es.append(e)
        return es

    def cvec(self, es):
        """Scenario-paired correlation vector of an empirical model."""
        out = []
        for ci in range(len(self.CTX)):
            for (a, b) in self.CH:
                z = sum(np.exp(-2j * np.pi / D * (a * j1 + b * j2)) * p
                        for (j1, j2), p in es[ci].items())
                out += [z.real, z.imag]
        return np.array(out)

    def CF(self, es):
        """Sheaf contextual fraction: 1 - max total weight of a subnormalized
        mixture of global sections dominated by the empirical model."""
        if not self.L:
            return 1.0
        nL = len(self.L)
        Aub, bub = [], []
        for ci, C in enumerate(self.CTX):
            for j in self.outcomes[ci]:
                row = np.array([1.0 if (l[C[0]] == j[0] and l[C[1]] == j[1]) else 0.0
                                for l in self.L])
                Aub.append(row)
                bub.append(es[ci][j])
        r = so.linprog(-np.ones(nL), A_ub=np.array(Aub), b_ub=np.array(bub),
                       bounds=[(0, None)] * nL, method="highs")
        assert r.status == 0
        return float(1.0 + r.fun)

    def hull_dist(self, c):
        """L_inf distance from c to conv{c(lambda)} via LP; 0 <=> membership."""
        if not self.L:
            return np.inf
        nL, m = self.VL.shape
        # vars: p (nL), t (1); min t  s.t. |VL^T p - c| <= t, sum p = 1, p >= 0
        Aub = np.zeros((2 * m, nL + 1))
        Aub[:m, :nL] = self.VL.T
        Aub[m:, :nL] = -self.VL.T
        Aub[:, nL] = -1.0
        bub = np.concatenate([c, -c])
        Aeq = np.zeros((1, nL + 1)); Aeq[0, :nL] = 1.0
        cobj = np.zeros(nL + 1); cobj[nL] = 1.0
        r = so.linprog(cobj, A_ub=Aub, b_ub=bub, A_eq=Aeq, b_eq=[1.0],
                       bounds=[(0, None)] * nL + [(0, None)], method="highs")
        assert r.status == 0
        return float(r.fun)


# ---- census artifact (exact-integer arbiter, cert4 with context 0 deleted) ----
class Census:
    def __init__(self):
        d = np.load(os.path.join(REPO_VER, "d4_facet_classes.npz"))
        self.HN = d["full"]      # 23,256 x 34 exact integer facets [normal | offset]
        self.P = d["P"]          # faithful 33-coordinate selector out of 150

    def margin(self, c150):
        """Min facet slack (negative = violation = CF>0 per the V43 theorem)."""
        mu = c150[self.P]
        vals = self.HN[:, :33] @ mu + self.HN[:, 33]
        return float(vals.min())


# ------------------------- state generators -------------------------
def haar_pure(rng):
    v = rng.normal(size=N) + 1j * rng.normal(size=N)
    v /= np.linalg.norm(v)
    return np.outer(v, v.conj())


def ginibre_mixed(rng, rank):
    G = rng.normal(size=(N, rank)) + 1j * rng.normal(size=(N, rank))
    rho = G @ G.conj().T
    return rho / np.trace(rho).real


def joint_eigenstates(triple, nmax=8):
    """Joint eigenprojector states of a (deleted) commuting triple."""
    W1, W2 = W(np.array(triple[0])), W(np.array(triple[1]))
    M = W1 + np.pi * W2  # generic combination; eigenvectors are joint
    ev, U = np.linalg.eig(M)
    seen, rhos = set(), []
    for k in range(N):
        v = U[:, k]
        j1 = int(np.round(np.angle(v.conj() @ W1 @ v) / (2 * np.pi / D))) % D
        j2 = int(np.round(np.angle(v.conj() @ W2 @ v) / (2 * np.pi / D))) % D
        if (j1, j2) in seen:
            continue
        seen.add((j1, j2))
        v = v / np.linalg.norm(v)
        rhos.append((np.outer(v, v.conj()), (j1, j2)))
        if len(rhos) >= nmax:
            break
    return rhos


# ------------------------- probes -------------------------
def probe_states(fam, census, rng, npure, nmixed, tag, fh):
    recs = []
    def one(rho, kind):
        es = fam.emodel(rho)
        cf = fam.CF(es)
        c = fam.cvec(es)
        hd = fam.hull_dist(c)
        rec = {"family": fam.name, "state": kind, "CF": cf, "hull_dist": hd}
        if census is not None:
            rec["facet_margin"] = census.margin(c)
        recs.append(rec)
        fh.write(json.dumps(rec) + "\n")
    for k in range(npure):
        one(haar_pure(rng), f"pure{k}")
    ranks = [2, 4, 8, 16]
    for k in range(nmixed):
        one(ginibre_mixed(rng, ranks[k % 4]), f"mixed_r{ranks[k % 4]}_{k}")
    return recs


def probe_boundary(fam, census, rng, nseeds, fh, deltas=(1e-3, 1e-4, 1e-6)):
    """Bisect the classical boundary along rho(t) = t psi + (1-t) I/N and test
    both criteria just inside and just outside."""
    recs = []
    I16 = np.eye(N) / N
    found = 0
    while found < nseeds:
        rho1 = haar_pure(rng)
        if fam.CF(fam.emodel(rho1)) < 1e-4:
            continue
        found += 1
        lo, hi = 0.0, 1.0
        for _ in range(46):
            mid = (lo + hi) / 2
            rho = mid * rho1 + (1 - mid) * I16
            if fam.CF(fam.emodel(rho)) > 1e-11:
                hi = mid
            else:
                lo = mid
        tstar = (lo + hi) / 2
        for dl in deltas:
            for sgn in (+1, -1):
                t = min(1.0, max(0.0, tstar + sgn * dl))
                rho = t * rho1 + (1 - t) * I16
                es = fam.emodel(rho)
                cf = fam.CF(es)
                c = fam.cvec(es)
                hd = fam.hull_dist(c)
                rec = {"family": fam.name, "state": f"boundary_t*{tstar:.12f}_{sgn:+d}{dl:g}",
                       "tstar": tstar, "delta": sgn * dl, "CF": cf, "hull_dist": hd}
                if census is not None:
                    rec["facet_margin"] = census.margin(c)
                recs.append(rec)
                fh.write(json.dumps(rec) + "\n")
    return recs


def rand_family(rng, ntriples, max_obs=11):
    """Random CONNECTED closed-triple Weyl family at d=4 (shared observables)."""
    for _ in range(400):
        while True:
            u = tuple(int(x) for x in rng.integers(0, D, 4))
            if any(u):
                break
        fam, pool = [], [u]
        ok = True
        for _ in range(ntriples):
            done = False
            for _att in range(400):
                a = pool[rng.integers(len(pool))]
                # bias toward CYCLES: with prob 1/2 close over two pool observables
                if rng.random() < 0.5 and len(pool) >= 2:
                    v = pool[rng.integers(len(pool))]
                else:
                    v = tuple(int(x) for x in rng.integers(0, D, 4))
                sp = (a[0] * v[1] - a[1] * v[0] + a[2] * v[3] - a[3] * v[2]) % D
                wv = tuple(int(x) for x in (-(np.array(a) + np.array(v))) % D)
                trip = [a, v, wv]
                if (not any(v)) or (not any(wv)) or sp != 0 or len(set(trip)) != 3:
                    continue
                if any(set(trip) == set(C) for C in fam):
                    continue
                fam.append(trip)
                pool += [v, wv]
                done = True
                break
            if not done:
                ok = False
                break
        if not ok:
            continue
        obs = {t for C in fam for t in C}
        if len(obs) <= max_obs:
            return fam
    return None


def sform(u, v):
    return (u[0] * v[1] - u[1] * v[0] + u[2] * v[3] - u[3] * v[2]) % D


def rand_square(rng, want_class=None, tries=40000):
    """Random Weyl magic square (K_{3,3}): 4 seed observables, rest by closure.
    Row/column commutation reduces to 4 symplectic conditions plus one cross
    condition <o11,o22> + <o12,o21> = 0 (mod D). Optionally filter on the class
    value sum(S) mod D."""
    for _ in range(tries):
        o = {}
        o[(0, 0)], o[(0, 1)], o[(1, 0)], o[(1, 1)] = \
            [tuple(int(x) for x in rng.integers(0, D, 4)) for _ in range(4)]
        if any(sform(o[a], o[b]) for a, b in
               [((0, 0), (0, 1)), ((1, 0), (1, 1)), ((0, 0), (1, 0)), ((0, 1), (1, 1))]):
            continue
        if (sform(o[(0, 0)], o[(1, 1)]) + sform(o[(0, 1)], o[(1, 0)])) % D:
            continue
        for i in range(2):
            o[(i, 2)] = tuple(int(x) for x in
                              (-(np.array(o[(i, 0)]) + np.array(o[(i, 1)]))) % D)
            o[(2, i)] = tuple(int(x) for x in
                              (-(np.array(o[(0, i)]) + np.array(o[(1, i)]))) % D)
        o[(2, 2)] = tuple(int(x) for x in
                          (np.array(o[(0, 0)]) + np.array(o[(0, 1)])
                           + np.array(o[(1, 0)]) + np.array(o[(1, 1)])) % D)
        vals = list(o.values())
        if any(not any(v) for v in vals) or len(set(vals)) != 9:
            continue
        ctxs = [[o[(i, 0)], o[(i, 1)], o[(i, 2)]] for i in range(3)] + \
               [[o[(0, j)], o[(1, j)], o[(2, j)]] for j in range(3)]
        S = [s_of(C) for C in ctxs]
        if want_class is not None and sum(S) % D != want_class:
            continue
        return ctxs, S
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", nargs="?", default="selftest", choices=["selftest", "drops", "boundary", "double",
                                     "random", "random2", "pm2", "eigen", "magic"])
    ap.add_argument("--drops", default="0")
    ap.add_argument("--npure", type=int, default=30)
    ap.add_argument("--nmixed", type=int, default=20)
    ap.add_argument("--nseeds", type=int, default=3)
    ap.add_argument("--nfam", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--tag", default="x")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    t0 = time.time()
    outp = os.path.join(OUT, f"c1_results_{args.mode}_{args.tag}.jsonl")
    fh = open(outp, "a")

    def fam_drop(k):
        return Family([C for i, C in enumerate(CERT6) if i != k], f"cert4_drop{k}")

    if args.mode == "selftest":
        fam = fam_drop(0)
        cen = Census()
        assert len(fam.L) == 64, f"|L| = {len(fam.L)}"
        # deterministic vertices themselves: CF must be 0, hull distance 0, margin >= 0
        Yint = np.round(fam.VL[:, cen.P]).astype(np.int64)  # exact {-1,0,1}
        assert np.abs(fam.VL[:, cen.P] - Yint).max() < 1e-12
        vals = cen.HN[:, :33] @ Yint.T + cen.HN[:, 33][:, None]
        assert vals.min() >= 0 and (vals.min(axis=1) == 0).all()
        rho = haar_pure(rng)
        es = fam.emodel(rho)
        n = [sum(e.values()) for e in es]
        print(f"selftest: |L|=64, vertices exactly on 23,256 facets, "
              f"normalizations {min(n):.12f}..{max(n):.12f}, "
              f"CF(sample)={fam.CF(es):.6f}, hd={fam.hull_dist(fam.cvec(es)):.2e}, "
              f"margin={cen.margin(fam.cvec(es)):.4f}")
    elif args.mode == "drops":
        for k in [int(x) for x in args.drops.split(",")]:
            fam = fam_drop(k)
            cen = Census() if (k == 0 and D == 4) else None
            recs = probe_states(fam, cen, rng, args.npure, args.nmixed, args.tag, fh)
            agree = sum(((r["CF"] > 1e-8) == (r["hull_dist"] > 1e-9)) for r in recs)
            print(f"drop{k}: |L|={len(fam.L)}, {len(recs)} states, "
                  f"CF/hull agreement {agree}/{len(recs)}")
    elif args.mode == "boundary":
        for k in [int(x) for x in args.drops.split(",")]:
            fam = fam_drop(k)
            cen = Census() if (k == 0 and D == 4) else None
            recs = probe_boundary(fam, cen, rng, args.nseeds, fh)
            agree = sum(((r["CF"] > 1e-8) == (r["hull_dist"] > 1e-9)) for r in recs)
            print(f"boundary drop{k}: {len(recs)} probes, agreement {agree}/{len(recs)}")
    elif args.mode == "eigen":
        for k in [int(x) for x in args.drops.split(",")]:
            fam = fam_drop(k)
            cen = Census() if (k == 0 and D == 4) else None
            for rho, jj in joint_eigenstates(CERT6[k]):
                es = fam.emodel(rho)
                cf = fam.CF(es)
                hd = fam.hull_dist(fam.cvec(es))
                rec = {"family": fam.name, "state": f"jointeig{jj}", "CF": cf,
                       "hull_dist": hd}
                if cen:
                    rec["facet_margin"] = cen.margin(fam.cvec(es))
                fh.write(json.dumps(rec) + "\n")
                print(f"drop{k} joint eig {jj}: CF={cf:.6f} hd={hd:.3e}")
    elif args.mode == "double":
        pairs = [(i, j) for i in range(6) for j in range(i + 1, 6)]
        lo, hi = [int(x) for x in args.drops.split(",")]
        for (i, j) in pairs[lo:hi]:
            fam = Family([C for k, C in enumerate(CERT6) if k not in (i, j)],
                         f"cert4_drop{i}{j}")
            recs = probe_states(fam, None, rng, args.npure, args.nmixed, args.tag, fh)
            agree = sum(((r["CF"] > 1e-8) == (r["hull_dist"] > 1e-9)) for r in recs)
            print(f"drop({i},{j}): |L|={len(fam.L)}, agreement {agree}/{len(recs)}")
    elif args.mode == "pm2":
        fam = Family(PM2, "PM_doubled_d4")
        print(f"PM2: |L|={len(fam.L)} (empty = AvN branch)")
        recs = probe_states(fam, None, rng, args.npure, args.nmixed, args.tag, fh)
        ok = sum((r["CF"] > 1e-8 and np.isinf(r["hull_dist"])) for r in recs)
        print(f"PM2 AvN branch: CF>0 and hull empty on {ok}/{len(recs)}")
    elif args.mode == "magic":
        # Anomalous magic squares (class = D/2): full family must be CF>0 for
        # every state (L should be empty); each deletion is probed with joint
        # eigenstates of the deleted context, Haar states, and mixtures.
        made = 0
        while made < args.nfam:
            ctxs, S = rand_square(rng, want_class=D // 2)
            if ctxs is None:
                print("no anomalous square found in budget")
                break
            made += 1
            name = f"magic{D}_{args.seed}_{made}"
            try:
                famF = Family(ctxs, name + "_full")
            except AssertionError:
                made -= 1
                continue
            fh.write(json.dumps({"family": name, "ctxs": ctxs, "S": S,
                                 "nL_full": len(famF.L)}) + "\n")
            # full-family branch
            nbad = 0
            for k in range(4):
                rho = haar_pure(rng) if k % 2 == 0 else ginibre_mixed(rng, 4)
                es = famF.emodel(rho)
                cf = famF.CF(es)
                hd = famF.hull_dist(famF.cvec(es)) if famF.L else float("inf")
                ok = (cf > 1e-8) == ((not famF.L) or hd > 1e-9)
                nbad += not ok
                fh.write(json.dumps({"family": name + "_full", "state": f"s{k}",
                                     "CF": cf, "hull_dist": hd}) + "\n")
            # one random deletion, eigen + Haar + mixed
            k = int(rng.integers(6))
            fam = Family([C for i, C in enumerate(ctxs) if i != k],
                         f"{name}_drop{k}")
            recs = []
            for rho, jj in joint_eigenstates(ctxs[k], nmax=4):
                es = fam.emodel(rho)
                recs.append({"family": fam.name, "state": f"eig{jj}",
                             "CF": fam.CF(es),
                             "hull_dist": fam.hull_dist(fam.cvec(es))})
            for k2 in range(6):
                rho = haar_pure(rng) if k2 % 2 == 0 else ginibre_mixed(rng, 4)
                es = fam.emodel(rho)
                recs.append({"family": fam.name, "state": f"s{k2}",
                             "CF": fam.CF(es),
                             "hull_dist": fam.hull_dist(fam.cvec(es))})
            for r in recs:
                fh.write(json.dumps(r) + "\n")
            agree = sum(((r["CF"] > 1e-8) == (r["hull_dist"] > 1e-9)) for r in recs)
            npos = sum(r["CF"] > 1e-8 for r in recs)
            print(f"{name}: S={S} sum={sum(S) % D}, |L_full|={len(famF.L)}, "
                  f"full-branch bad {nbad}/4; drop{k}: |L|={len(fam.L)}, "
                  f"agreement {agree}/{len(recs)}, CF>0 on {npos}")
    elif args.mode == "random2":
        # Random families probed with joint eigenstates of their OWN contexts and
        # near-boundary mixtures: targets the CF>0 branch in varied scenarios.
        made = navn = ncontex = ntot = 0
        while made < args.nfam:
            ctxs = rand_family(rng, int(rng.integers(4, 7)),
                               max_obs=11 if D == 4 else 9)
            if ctxs is None:
                continue
            try:
                fam = Family(ctxs, f"rand2_{args.seed}_{made}")
            except AssertionError:
                continue
            if fam.L and len(fam.L) > 4000:
                continue
            made += 1
            if not fam.L:
                navn += 1
                # AvN branch: conv L empty; conjecture demands CF > 0 for all states
                bad = 0
                for k in range(6):
                    es = fam.emodel(haar_pure(rng))
                    cf = fam.CF(es)
                    ntot += 1
                    ncontex += cf > 1e-8
                    bad += cf <= 1e-8
                    fh.write(json.dumps({"family": fam.name, "state": f"avn{k}",
                                         "CF": cf, "hull_dist": float("inf")}) + "\n")
                print(f"{fam.name}: |L|=0 (AvN); CF>0 failures: {bad}/6")
                continue
            recs = []
            for ci, C in enumerate(fam.CTX):
                for rho, jj in joint_eigenstates(C, nmax=3):
                    es = fam.emodel(rho)
                    cf = fam.CF(es)
                    hd = fam.hull_dist(fam.cvec(es))
                    rec = {"family": fam.name, "state": f"eig_c{ci}_{jj}",
                           "CF": cf, "hull_dist": hd}
                    recs.append(rec)
                    fh.write(json.dumps(rec) + "\n")
                    # near-boundary mixture toward I/N if contextual
                    if cf > 1e-4:
                        I16 = np.eye(N) / N
                        lo, hi = 0.0, 1.0
                        for _ in range(30):
                            mid = (lo + hi) / 2
                            rm = mid * rho + (1 - mid) * I16
                            if fam.CF(fam.emodel(rm)) > 1e-11:
                                hi = mid
                            else:
                                lo = mid
                        for dl in (+1e-4, -1e-4):
                            t = min(1.0, max(0.0, (lo + hi) / 2 + dl))
                            rm = t * rho + (1 - t) * I16
                            es2 = fam.emodel(rm)
                            rec = {"family": fam.name,
                                   "state": f"eigbnd_c{ci}_{jj}_{dl:+g}",
                                   "CF": fam.CF(es2),
                                   "hull_dist": fam.hull_dist(fam.cvec(es2))}
                            recs.append(rec)
                            fh.write(json.dumps(rec) + "\n")
            agree = sum(((r["CF"] > 1e-8) == (r["hull_dist"] > 1e-9)) for r in recs)
            npos = sum(r["CF"] > 1e-8 for r in recs)
            ntot += len(recs)
            ncontex += npos
            print(f"{fam.name}: nC={len(fam.CTX)} nO={len(fam.obs)} |L|={len(fam.L)}"
                  f", agreement {agree}/{len(recs)}, CF>0 on {npos}")
            fh.write(json.dumps({"family": fam.name, "ctxs": ctxs,
                                 "S": fam.S, "nL": len(fam.L)}) + "\n")
        print(f"random2 summary: {made} families ({navn} AvN), "
              f"{ncontex}/{ntot} records with CF>0")
    elif args.mode == "random":
        made = 0
        while made < args.nfam:
            ctxs = rand_family(rng, int(rng.integers(4, 7)),
                               max_obs=11 if D == 4 else 9)
            if ctxs is None:
                continue
            try:
                fam = Family(ctxs, f"rand{args.seed}_{made}")
            except AssertionError:
                continue
            if fam.L and len(fam.L) > 4000:
                continue
            made += 1
            recs = probe_states(fam, None, rng, 6, 4, args.tag, fh)
            if fam.L:
                agree = sum(((r["CF"] > 1e-8) == (r["hull_dist"] > 1e-9)) for r in recs)
                print(f"{fam.name}: nC={len(fam.CTX)} nO={len(fam.obs)} "
                      f"|L|={len(fam.L)}, agreement {agree}/{len(recs)}")
            else:
                ok = sum(r["CF"] > 1e-8 for r in recs)
                print(f"{fam.name}: nC={len(fam.CTX)} nO={len(fam.obs)} |L|=0 (AvN), "
                      f"CF>0 on {ok}/{len(recs)}")
            fh.write(json.dumps({"family": fam.name, "ctxs": ctxs,
                                 "S": fam.S, "nL": len(fam.L)}) + "\n")
    fh.close()
    print(f"[{args.mode}] done in {time.time() - t0:.1f}s -> {outp}")


if __name__ == "__main__":
    main()
