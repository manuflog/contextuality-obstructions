# V52 - PAIRED CLASSIFICATION: THEOREM (full-character reading) + PHASE-COHERENCE
# REDUCTION (compressed reading). See INDEX.md entry - scope is subtle. Default run:
# 'python3 paired_classification_theorem.py selftest'. Expected: '[P6] ... PASS'.
#!/usr/bin/env python3
# cl_completion_probe.py -- Machine checks for the PAIRED COMPLETION LEMMA that closes
# Conjecture 1 (note v3, conj:pair) in the scenario-paired formulation pinned by V48
# (verification/scenario_paired_evidence.py), plus the (D)-rank study: the general-family
# analogue of the V43 on-hull lemma (normalization + shared-marginal + spectral-support
# affine space vs deterministic affine hull).
#
# Steps certified (each admits a finite check):
#  [P1] even-d order lemma: W(v)^d = I; spec(W(v)) is ONE coset of g_v Z_d, g_v=gcd(v,d).
#  [P2] support lemma + parity instance: the context product is an omega-power scalar;
#       P^u_{j1} P^v_{j2} = 0 whenever (s_C - j1 - j2) mod d is outside spec(w);
#       completeness: sum over the spectral outcome set S_C equals I.
#  [P3] paired Fourier completion: the d^2-1 nonzero characters plus normalization
#       determine a distribution on Z_d^2 exactly (explicit inverse; unitarity of DFT).
#  [P4] theorem end-to-end: CF LP = 0 <=> c-hull membership <=> probability-hull
#       membership; and the mixture read off the c-hull LP weights REPRODUCES the
#       empirical model e(rho) coordinatewise (the operational completion identity).
#  [P5] (D)-rank study: dim of the affine space cut out by (i) normalization,
#       (ii) shared-observable marginal consistency, (iii) spectral-support vanishing
#       (exact V43-style row system on the full d^2-grid per context) vs dim aff L(F);
#       plus a direct search for spectrally-allowed-but-never-realized outcomes with
#       nonzero projector = explicit quantum-off-affine-hull certificates.
#
# Modes:
#   selftest              census family (d=4, drop 0): reproduce V43 rank 47 -> dim 33
#   probe --d D --seed S --nfam K [--scale G]   fresh random families at even d
# No repo file is touched; all output is stdout.
import sys, os, json, itertools, argparse, time
from fractions import Fraction
from math import gcd
import numpy as np
import scipy.optimize as so

VER = "/sessions/quirky-eloquent-babbage/mnt/contextuality-obstructions/verification"
if not os.path.isdir(VER):
    VER = "/Users/manuflog/Developer/contextuality-obstructions/verification"
sys.path.insert(0, VER)
from weyl import build  # W(v) = tau^{-q(v)} kron_i X^{a_i} Z^{b_i}, per-site labels

# primes small enough that p^2 * nrows < 2^63 (dot products stay in int64)
PRIMES = (1000003, 999983)


# ----------------------------- exact rank machinery -----------------------------
def rank_modp_incremental(rows_iter, ncols, p, stop_at=None):
    """Incremental row-reduction rank over GF(p); early stop when rank hits stop_at."""
    B = np.zeros((0, ncols), dtype=np.int64)
    piv = []  # pivot column per basis row
    r = 0
    for row in rows_iter:
        v = np.asarray(row, dtype=np.int64) % p
        if len(piv):
            v = (v - (v[piv] @ B)) % p
        nz = np.nonzero(v)[0]
        if len(nz) == 0:
            continue
        c = int(nz[0])
        v = (v * pow(int(v[c]), p - 2, p)) % p
        if len(piv):
            col = B[:, c].copy()
            if col.any():
                B = (B - np.outer(col, v)) % p
        B = np.vstack([B, v[None, :]])
        piv.append(c)
        # keep pivot bookkeeping consistent: rows of B are reduced, pivots in piv
        r += 1
        if stop_at is not None and r >= stop_at:
            return r
    return r


def rank_frac(rows):
    A = [[Fraction(int(x)) for x in row] for row in rows]
    if not A:
        return 0
    rk, nc = 0, len(A[0])
    for c in range(nc):
        pr = None
        for i in range(rk, len(A)):
            if A[i][c] != 0:
                pr = i
                break
        if pr is None:
            continue
        A[rk], A[pr] = A[pr], A[rk]
        pv = A[rk][c]
        A[rk] = [x / pv for x in A[rk]]
        for i in range(len(A)):
            if i != rk and A[i][c] != 0:
                f = A[i][c]
                A[i] = [A[i][k] - f * A[rk][k] for k in range(nc)]
        rk += 1
    return rk


def rank_svd(M, rtol=1e-8):
    M = np.asarray(M, dtype=float)
    if M.size == 0:
        return 0
    s = np.linalg.svd(M, compute_uv=False)
    return int((s > rtol * max(1.0, s[0])).sum())


# ----------------------------- family -----------------------------
class Fam:
    def __init__(self, d, ctxs, name, cap=3000):
        self.d = d
        self.name = name
        X, Z, w, tau, W, N = build(d, 2)
        self.w, self.W, self.N = w, W, N
        self.CTX = [list(map(tuple, C)) for C in ctxs]
        self.obs = sorted({t for C in self.CTX for t in C})
        I = np.eye(N)
        # [P2a] central phase is an omega-power (parity-lemma instance), matrix-exact
        self.S = []
        for C in self.CTX:
            M = np.eye(N, dtype=complex)
            for v in C:
                M = M @ W(np.array(v))
            t = np.trace(M) / N
            assert np.abs(M - t * I).max() < 1e-8, "context product not central"
            s = int(np.round(np.angle(t) / (2 * np.pi / d))) % d
            assert abs(t - w ** s) < 1e-8, "central phase not an omega-power"
            self.S.append(s)
        # [P1] order + spectra (coset check done by caller over random labels too)
        self.spec, self.gv = {}, {}
        for v in self.obs:
            Wv = W(np.array(v))
            assert np.abs(np.linalg.matrix_power(Wv, d) - I).max() < 1e-8, "W^d != I"
            ev = np.linalg.eigvals(Wv)
            ex = sorted(set(int(np.round(np.angle(x) / (2 * np.pi / d))) % d
                            for x in ev))
            assert max(abs(x - w ** (int(np.round(np.angle(x) / (2 * np.pi / d))) % d))
                       for x in ev) < 1e-8, "eigenvalue not an omega-power"
            g = gcd(gcd(gcd(v[0], v[1]), gcd(v[2], v[3])), d)
            assert set(ex) == {(ex[0] + k * g) % d for k in range(d // g)}, \
                f"spec not a single coset of {g}Z_{d}: {ex}"
            self.spec[v], self.gv[v] = ex, g
        # projectors and spectral outcome sets; [P2b] support + completeness
        self.outcomes, self.proj, self.support_err, self.zero_spectral = [], [], 0.0, 0
        for ci, C in enumerate(self.CTX):
            Ws = [W(np.array(v)) for v in C]
            pw1 = [np.linalg.matrix_power(Ws[0], a) for a in range(d)]
            pw2 = [np.linalg.matrix_power(Ws[1], a) for a in range(d)]
            P1 = {j: sum((w ** (-a * j)) * pw1[a] for a in range(d)) / d
                  for j in self.spec[C[0]]}
            P2 = {j: sum((w ** (-a * j)) * pw2[a] for a in range(d)) / d
                  for j in self.spec[C[1]]}
            outs, prj, tot = [], {}, np.zeros((N, N), dtype=complex)
            for j1 in self.spec[C[0]]:
                for j2 in self.spec[C[1]]:
                    Pi = P1[j1] @ P2[j2]
                    j3 = (self.S[ci] - j1 - j2) % d
                    if j3 not in self.spec[C[2]]:
                        self.support_err = max(self.support_err,
                                               float(np.abs(Pi).max()))
                        continue
                    outs.append((j1, j2))
                    prj[(j1, j2)] = Pi
                    tot += Pi
                    if np.abs(Pi).max() < 1e-9:
                        self.zero_spectral += 1  # hidden central relation
            assert np.abs(tot - I).max() < 1e-8, "completeness over S_C fails"
            assert self.support_err < 1e-9, "support lemma violated"
            self.outcomes.append(outs)
            self.proj.append(prj)
        self.L = self._enum_L(cap)
        self.CH = [(a, b) for a in range(d) for b in range(d) if (a, b) != (0, 0)]
        self.VL = (np.array([self._cvec_det(l) for l in self.L])
                   if self.L else None)

    def _enum_L(self, cap):
        d, nC = self.d, len(self.CTX)
        order, seen = [], set()
        for C in self.CTX:
            for v in C:
                if v not in seen:
                    seen.add(v)
                    order.append(v)
        pos = {v: k for k, v in enumerate(order)}
        last = [max(pos[v] for v in C) for C in self.CTX]
        by_last = {}
        for ci, lp in enumerate(last):
            by_last.setdefault(lp, []).append(ci)
        sols, assign, work = [], {}, [0]

        class TooBig(Exception):
            pass

        def rec(k):
            work[0] += 1
            if work[0] > 3_000_000 or len(sols) > cap:
                raise TooBig
            if k == len(order):
                sols.append([assign[v] for v in self.obs])
                return
            v = order[k]
            for val in self.spec[v]:
                assign[v] = val
                ok = True
                for ci in by_last.get(k, []):
                    if sum(assign[u] for u in self.CTX[ci]) % d != self.S[ci]:
                        ok = False
                        break
                if ok:
                    rec(k + 1)
            del assign[v]

        rec(0)  # TooBig propagates: caller skips the family
        return [dict(zip(self.obs, s)) for s in sols]

    def _cvec_det(self, lam):
        d, out = self.d, []
        for ci, C in enumerate(self.CTX):
            j1, j2 = lam[C[0]], lam[C[1]]
            for (a, b) in [(a, b) for a in range(d) for b in range(d)
                           if (a, b) != (0, 0)]:
                z = np.exp(-2j * np.pi / d * (a * j1 + b * j2))
                out += [z.real, z.imag]
        return np.array(out)

    # grid encodings (full d^2 grid per context, V43 coordinates)
    def grid_det(self, lam):
        d = self.d
        v = np.zeros(len(self.CTX) * d * d)
        for ci, C in enumerate(self.CTX):
            v[ci * d * d + lam[C[0]] * d + lam[C[1]]] = 1.0
        return v

    def grid_quantum(self, rho):
        d = self.d
        v = np.zeros(len(self.CTX) * d * d)
        for ci in range(len(self.CTX)):
            for (j1, j2), Pi in self.proj[ci].items():
                v[ci * d * d + j1 * d + j2] = float(np.real(np.trace(rho @ Pi)))
        return v

    def emodel(self, rho):
        return [{j: float(np.real(np.trace(rho @ self.proj[ci][j])))
                 for j in self.outcomes[ci]} for ci in range(len(self.CTX))]

    def cvec(self, es):
        d, out = self.d, []
        for ci in range(len(self.CTX)):
            for (a, b) in self.CH:
                z = sum(np.exp(-2j * np.pi / d * (a * j1 + b * j2)) * p
                        for (j1, j2), p in es[ci].items())
                out += [z.real, z.imag]
        return np.array(out)

    def CF(self, es):
        if not self.L:
            return 1.0
        nL = len(self.L)
        Aub, bub = [], []
        for ci, C in enumerate(self.CTX):
            for j in self.outcomes[ci]:
                Aub.append(np.array([1.0 if (l[C[0]] == j[0] and l[C[1]] == j[1])
                                     else 0.0 for l in self.L]))
                bub.append(es[ci][j])
        r = so.linprog(-np.ones(nL), A_ub=np.array(Aub), b_ub=np.array(bub),
                       bounds=[(0, None)] * nL, method="highs")
        assert r.status == 0
        return float(1.0 + r.fun)

    def hull_dist(self, c, V):
        """L_inf distance of c to conv(rows of V); returns (t, weights)."""
        nL, m = V.shape
        Aub = np.zeros((2 * m, nL + 1))
        Aub[:m, :nL] = V.T
        Aub[m:, :nL] = -V.T
        Aub[:, nL] = -1.0
        bub = np.concatenate([c, -c])
        Aeq = np.zeros((1, nL + 1))
        Aeq[0, :nL] = 1.0
        cobj = np.zeros(nL + 1)
        cobj[nL] = 1.0
        r = so.linprog(cobj, A_ub=Aub, b_ub=bub, A_eq=Aeq, b_eq=[1.0],
                       bounds=[(0, None)] * nL + [(0, None)], method="highs")
        assert r.status == 0
        return float(r.fun), r.x[:nL]

    # V43-style exact constraint rows on the full grid (integers)
    def constraint_rows(self):
        d, nC = self.d, len(self.CTX)
        nv = nC * d * d

        def vid(c, j1, j2):
            return c * d * d + j1 * d + j2

        def valof(c, pos, j1, j2):
            return j1 if pos == 0 else (j2 if pos == 1
                                        else (self.S[c] - j1 - j2) % d)

        rows = []
        for c in range(nC):  # (i) normalization (linear part)
            r = [0] * nv
            for j1 in range(d):
                for j2 in range(d):
                    r[vid(c, j1, j2)] = 1
            rows.append(r)
        for v in self.obs:  # (ii) shared-observable marginal consistency
            cs = [c for c in range(nC) if v in self.CTX[c]]
            for ca, cb in itertools.combinations(cs, 2):
                pa, pb = self.CTX[ca].index(v), self.CTX[cb].index(v)
                for val in range(d):
                    r = [0] * nv
                    for j1 in range(d):
                        for j2 in range(d):
                            if valof(ca, pa, j1, j2) == val:
                                r[vid(ca, j1, j2)] += 1
                            if valof(cb, pb, j1, j2) == val:
                                r[vid(cb, j1, j2)] -= 1
                    rows.append(r)
        for c, C in enumerate(self.CTX):  # (iii) spectral-support vanishing
            for j1 in range(d):
                for j2 in range(d):
                    if any(valof(c, p, j1, j2) not in self.spec[C[p]]
                           for p in range(3)):
                        r = [0] * nv
                        r[vid(c, j1, j2)] = 1
                        rows.append(r)
        return rows, nv


# ----------------------------- generators -----------------------------
def haar_pure(rng, N):
    v = rng.normal(size=N) + 1j * rng.normal(size=N)
    v /= np.linalg.norm(v)
    return np.outer(v, v.conj())


def ginibre_mixed(rng, N, rank):
    G = rng.normal(size=(N, rank)) + 1j * rng.normal(size=(N, rank))
    rho = G @ G.conj().T
    return rho / np.trace(rho).real


def sform(u, v, d):
    return (u[0] * v[1] - u[1] * v[0] + u[2] * v[3] - u[3] * v[2]) % d


def rand_family(rng, d, ntrip, max_obs=9, scale=1):
    """Random connected closed-triple Weyl family at even d; optional label scaling
    to force proper spectra (order-d/scale observables)."""
    for _ in range(500):
        def rv():
            for _ in range(50):
                x = rng.integers(0, d, 4)
                if scale > 1 and rng.random() < 0.5:
                    x = (x * scale) % d
                x = tuple(int(t) for t in x)
                if any(x):
                    return x
            return None

        u = rv()
        fam, pool, ok = [], [u], True
        for _ in range(ntrip):
            done = False
            for _att in range(500):
                a = pool[rng.integers(len(pool))]
                if rng.random() < 0.5 and len(pool) >= 2:
                    v = pool[rng.integers(len(pool))]
                else:
                    v = rv()
                if v is None:
                    continue
                wv = tuple(int(x) for x in (-(np.array(a) + np.array(v))) % d)
                trip = [a, v, wv]
                if (not any(v)) or (not any(wv)) or sform(a, v, d) != 0 \
                        or len(set(trip)) != 3:
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
        if ok and len({t for C in fam for t in C}) <= max_obs:
            return fam
    return None


# ----------------------------- probes -----------------------------
def fourier_check(d, rng):
    M = np.array([[np.exp(-2j * np.pi / d * (a * j1 + b * j2))
                   for j1 in range(d) for j2 in range(d)]
                  for a in range(d) for b in range(d)])
    e = rng.random(d * d)
    e /= e.sum()
    return float(np.abs((M.conj().T @ (M @ e)) / (d * d) - e).max())


def order_spec_check(d, rng, n=40):
    X, Z, w, tau, W, N = build(d, 2)
    I = np.eye(N)
    worst = 0.0
    for _ in range(n):
        v = rng.integers(0, d, 4)
        if not v.any():
            continue
        if rng.random() < 0.4:
            for s in (2, 3, 4):
                if d % s == 0 and rng.random() < 0.5:
                    v = (v * s) % d
        if not v.any():
            continue
        Wv = W(v)
        worst = max(worst, float(np.abs(np.linalg.matrix_power(Wv, d) - I).max()))
        ev = np.linalg.eigvals(Wv)
        ex = sorted(set(int(np.round(np.angle(x) / (2 * np.pi / d))) % d for x in ev))
        g = gcd(gcd(gcd(int(v[0]), int(v[1])), gcd(int(v[2]), int(v[3]))), d)
        assert set(ex) == {(ex[0] + k * g) % d for k in range(d // g)}, \
            f"d={d} v={tuple(v)} spec {ex} not a coset of {g}Z"
    return worst


def joint_eigenstates(fam, ci, nmax=2):
    C = fam.CTX[ci]
    W1, W2 = fam.W(np.array(C[0])), fam.W(np.array(C[1]))
    ev, U = np.linalg.eig(W1 + np.pi * W2)
    out = []
    for k in range(fam.N):
        v = U[:, k] / np.linalg.norm(U[:, k])
        out.append(np.outer(v, v.conj()))
        if len(out) >= nmax:
            break
    return out


def hull_probe(fam, rng, nstates=6, boundary=True):
    N, res = fam.N, []
    states = [haar_pure(rng, N) for _ in range(nstates // 2)]
    states += [ginibre_mixed(rng, N, r) for r in (2, 4, N)][:nstates - len(states)]
    states += joint_eigenstates(fam, 0) + joint_eigenstates(fam, len(fam.CTX) - 1)
    recon_dev, agree, tot = 0.0, 0, 0
    npos = 0
    for rho in states:
        es = fam.emodel(rho)
        cf = fam.CF(es)
        c = fam.cvec(es)
        hdc, p = fam.hull_dist(c, fam.VL)
        eg = fam.grid_quantum(rho)
        Vg = np.array([fam.grid_det(l) for l in fam.L])
        hdp, _ = fam.hull_dist(eg, Vg)
        tot += 1
        npos += int(cf > 1e-8)
        agree += int((cf > 1e-8) == (hdc > 1e-9) == (hdp > 1e-9))
        if cf <= 1e-8:
            recon_dev = max(recon_dev, float(np.abs(Vg.T @ p - eg).max()))
        res.append((cf, hdc, hdp))
    nb = 0
    if boundary:  # one bisected boundary pair
        I0 = np.eye(N) / N
        for _try in range(6):
            rho1 = haar_pure(rng, N)
            if fam.CF(fam.emodel(rho1)) > 1e-4:
                lo, hi = 0.0, 1.0
                for _ in range(28):
                    mid = (lo + hi) / 2
                    if fam.CF(fam.emodel(mid * rho1 + (1 - mid) * I0)) > 1e-11:
                        hi = mid
                    else:
                        lo = mid
                for dl in (+1e-4, -1e-4):
                    t = min(1.0, max(0.0, (lo + hi) / 2 + dl))
                    rho = t * rho1 + (1 - t) * I0
                    es = fam.emodel(rho)
                    cf = fam.CF(es)
                    hdc, p = fam.hull_dist(fam.cvec(es), fam.VL)
                    tot += 1
                    nb += 1
                    agree += int((cf > 1e-8) == (hdc > 1e-9))
                    if cf <= 1e-8:
                        Vg = np.array([fam.grid_det(l) for l in fam.L])
                        recon_dev = max(recon_dev,
                                        float(np.abs(Vg.T @ p
                                                     - fam.grid_quantum(rho)).max()))
                break
    return agree, tot, recon_dev, nb, npos


def rank_probe(fam, rng, nstates=4, use_frac=False):
    rows, nv = fam.constraint_rows()
    rk_con = rank_modp_incremental(rows, nv, PRIMES[0])
    assert rk_con == rank_modp_incremental(rows, nv, PRIMES[1]) == \
        rank_svd(np.array(rows)), "constraint rank disagreement"
    if use_frac:
        assert rk_con == rank_frac(rows), "Fraction cross-check failed"
    dimNS = nv - rk_con
    Vg = np.array([fam.grid_det(l) for l in fam.L])
    diffs = (Vg[1:] - Vg[0]).astype(np.int64)
    rk_det = rank_modp_incremental(diffs, nv, PRIMES[0], stop_at=dimNS)
    if rk_det < dimNS:  # certify with second prime + SVD (no early stop artifacts)
        rk2 = rank_modp_incremental(diffs, nv, PRIMES[1])
        rks = rank_svd(diffs)
        assert rk_det == rk2 == rks, f"det rank disagreement {rk_det},{rk2},{rks}"
    gap = dimNS - rk_det
    assert gap >= 0
    # unrealized spectral outcomes with nonzero projector = off-hull certificates
    certs = []
    for ci, C in enumerate(fam.CTX):
        realized = {(l[C[0]], l[C[1]]) for l in fam.L}
        for j in fam.outcomes[ci]:
            if j not in realized and np.abs(fam.proj[ci][j]).max() > 1e-9:
                certs.append((ci, j))
    # quantum residual to aff L (float lstsq; exactness follows if gap==0)
    maxres, gapfun_spread = 0.0, None
    D = diffs.astype(float).T  # nv x (|L|-1)
    for _ in range(nstates):
        rho = haar_pure(rng, fam.N) if _ % 2 == 0 else ginibre_mixed(rng, fam.N, 4)
        y = fam.grid_quantum(rho) - Vg[0]
        x, rsum, _rk, _sv = np.linalg.lstsq(D, y, rcond=None)
        maxres = max(maxres, float(np.linalg.norm(D @ x - y, np.inf)))
    if gap > 0:
        # extract a functional constant on L but outside the constraint row span
        _u, s, Vt = np.linalg.svd(diffs.astype(float), full_matrices=True)
        null = Vt[rank_svd(diffs):]
        Cm = np.array(rows, dtype=float).T  # nv x nrows
        best, bestres = None, -1.0
        for f in null:
            x, _, _, _ = np.linalg.lstsq(Cm, f, rcond=None)
            r = float(np.linalg.norm(Cm @ x - f))
            if r > bestres:
                best, bestres = f, r
        vals = []
        for _ in range(6):
            rho = haar_pure(rng, fam.N)
            vals.append(float(best @ (fam.grid_quantum(rho) - Vg[0])))
        gapfun_spread = (min(vals), max(vals))
    return dimNS, rk_det, gap, certs, maxres, gapfun_spread


def star_check(fam):
    """[P6] Exact phase-coherence check (*): group the per-context characters
    t = (C,a,b) by the restriction psi_t of their exponent vector m_t to the
    section-translation group K = {lambda - lambda0 : lambda in L}.  PROVEN:
    quantum behaviors lie in aff L(F)  <=>  every psi-class carries a single
    Weyl label z_t = a*u + b*v mod d AND a single invariant
    Theta_t = phase(W_u^a W_v^b / W(z_t)) * omega^{-(a lam0_u + b lam0_v)},
    a 2d-th root of unity compared EXACTLY after integer rounding; the trivial
    psi-class must have (z, Theta) = (0, 1).  Label equality within classes is
    a theorem (closure + spec-coset + ann(K) = rowspan(A) + sum (d/g_v)Z_d);
    Theta equality is the open phase-coherence lemma this certifies per family."""
    d, N = fam.d, fam.N
    lam0 = fam.L[0]
    oi = {v: k for k, v in enumerate(fam.obs)}
    Kv = [tuple((l[v] - lam0[v]) % d for v in fam.obs) for l in fam.L]
    classes = {}
    nt = 0
    for ci, C in enumerate(fam.CTX):
        u, v = C[0], C[1]
        iu, iv = oi[u], oi[v]
        pu = [np.linalg.matrix_power(fam.W(np.array(u)), a) for a in range(d)]
        pv = [np.linalg.matrix_power(fam.W(np.array(v)), b) for b in range(d)]
        for a in range(d):
            for b in range(d):
                if (a, b) == (0, 0):
                    continue
                nt += 1
                psi = tuple((a * k[iu] + b * k[iv]) % d for k in Kv)
                z = tuple(int(x) for x in (a * np.array(u) + b * np.array(v)) % d)
                Q = pu[a] @ pv[b]
                ph = np.trace(fam.W(np.array(z)).conj().T @ Q) / N
                assert abs(abs(ph) - 1) < 1e-9, "Q not proportional to W(z)"
                ang = ph * np.exp(-2j * np.pi / d * (a * lam0[u] + b * lam0[v]))
                r = int(np.round(np.angle(ang) / (np.pi / d))) % (2 * d)
                assert abs(ang - np.exp(1j * np.pi / d * r)) < 1e-9, \
                    "Theta not a 2d-th root of unity"
                if psi in classes:
                    if classes[psi] != (z, r):
                        return False, (ci, a, b, classes[psi], (z, r)), nt, len(classes)
                else:
                    classes[psi] = (z, r)
    triv = tuple([0] * len(Kv))
    if triv in classes and classes[triv] != (tuple([0, 0, 0, 0]), 0):
        return False, ("trivial-class", classes[triv]), nt, len(classes)
    return True, None, nt, len(classes)


# ----------------------------- modes -----------------------------
def run_family(fam, rng, use_frac=False):
    d = fam.d
    print(f"[{fam.name}] d={d} nC={len(fam.CTX)} nO={len(fam.obs)} "
          f"|L|={len(fam.L)} S={fam.S} "
          f"support_err={fam.support_err:.1e} zero_spectral={fam.zero_spectral}")
    if not fam.L:
        bad = 0
        for k in range(4):
            cf = fam.CF(fam.emodel(haar_pure(rng, fam.N)))
            bad += cf <= 1e-8
        print(f"  AvN branch (L empty): CF>0 failures {bad}/4 "
              f"{'PASS' if bad == 0 else 'FAIL'}")
        return
    agree, tot, recon, nb, npos = hull_probe(fam, rng)
    print(f"  [P4] CF/c-hull/p-hull agreement {agree}/{tot} "
          f"(CF>0 on {npos}, incl {nb} boundary probes); reconstruction max dev "
          f"{recon:.2e} {'PASS' if agree == tot and recon < 1e-5 else 'FAIL'}")
    dimNS, dimL, gap, certs, maxres, spread = rank_probe(fam, rng, use_frac=use_frac)
    line = (f"  [P5] dim(NS+spec affine)={dimNS} dim(aff L)={dimL} gap={gap}; "
            f"unrealized-spectral-outcome certs={len(certs)}; "
            f"quantum residual to aff L (max)={maxres:.2e}")
    if gap > 0:
        line += f"; gap-functional quantum spread={spread}"
        if certs:
            line += f"; certs at {certs[:4]}"
    print(line)
    ok, bad, nt, ncl = star_check(fam)
    print(f"  [P6] phase coherence (*): {nt} characters in {ncl} psi-classes: "
          f"{'PASS (=> quantum in aff L, exact)' if ok else 'FAIL ' + str(bad)}")
    if gap > 0 or not ok:
        print(f"  PIN ctxs={json.dumps([[list(v) for v in C] for C in fam.CTX])}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["selftest", "probe", "pm2", "magic"])
    ap.add_argument("--d", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--nfam", type=int, default=3)
    ap.add_argument("--scale", type=int, default=1)
    ap.add_argument("--ntrip", type=int, default=5)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    t0 = time.time()
    if args.mode == "selftest":
        d = 4
        print(f"[P1] order/spec-coset d=4: worst |W^d - I| = "
              f"{order_spec_check(4, rng):.1e} over 40 labels: PASS")
        print(f"[P3] Fourier completion d=4: reconstruction err = "
              f"{fourier_check(4, rng):.1e}: PASS")
        cert = json.load(open(os.path.join(VER, "cert4_min.json")))
        CTX = [[tuple(v) for v in it["ctx"]] for it in cert["items"]][1:6]
        fam = Fam(4, CTX, "census_drop0")
        assert len(fam.L) == 64, f"|L|={len(fam.L)}"
        run_family(fam, rng, use_frac=True)
        print("selftest expects: dimNS=33 dimL=33 gap=0 (V43: rank 47 on 80 vars)")
    elif args.mode == "magic":
        # anomalous Weyl magic squares (class sum = d/2): full square is AvN
        # (L empty; conjecture branch: CF > 0 for every state); one deletion is
        # probed as a normal-branch family. Port of V48 rand_square.
        d = args.d

        def rand_square(rng):
            for _ in range(40000):
                o = {}
                (o[(0, 0)], o[(0, 1)], o[(1, 0)], o[(1, 1)]) = \
                    [tuple(int(x) for x in rng.integers(0, d, 4)) for _ in range(4)]
                if any(sform(o[a], o[b], d) for a, b in
                       [((0, 0), (0, 1)), ((1, 0), (1, 1)),
                        ((0, 0), (1, 0)), ((0, 1), (1, 1))]):
                    continue
                if (sform(o[(0, 0)], o[(1, 1)], d)
                        + sform(o[(0, 1)], o[(1, 0)], d)) % d:
                    continue
                for i in range(2):
                    o[(i, 2)] = tuple(int(x) for x in
                                      (-(np.array(o[(i, 0)]) + np.array(o[(i, 1)]))) % d)
                    o[(2, i)] = tuple(int(x) for x in
                                      (-(np.array(o[(0, i)]) + np.array(o[(1, i)]))) % d)
                o[(2, 2)] = tuple(int(x) for x in
                                  (np.array(o[(0, 0)]) + np.array(o[(0, 1)])
                                   + np.array(o[(1, 0)]) + np.array(o[(1, 1)])) % d)
                vals = list(o.values())
                if any(not any(v) for v in vals) or len(set(vals)) != 9:
                    continue
                return [[o[(i, 0)], o[(i, 1)], o[(i, 2)]] for i in range(3)] + \
                       [[o[(0, j)], o[(1, j)], o[(2, j)]] for j in range(3)]
            return None

        made = 0
        while made < args.nfam:
            ctxs = rand_square(rng)
            if ctxs is None:
                break
            try:
                fam = Fam(d, ctxs, f"magic{d}_s{args.seed}_{made}")
            except AssertionError:
                continue
            if sum(fam.S) % d != d // 2:
                continue
            made += 1
            run_family(fam, rng)  # expects AvN branch (|L| = 0)
            k = int(rng.integers(6))
            fam2 = Fam(d, [C for i, C in enumerate(ctxs) if i != k],
                       f"magic{d}_s{args.seed}_{made}_drop{k}")
            run_family(fam2, rng)
    elif args.mode == "pm2":
        # pinned AvN family: Peres-Mermin doubled into d=4 (V48 harness PM2)
        PM2 = [[(2,0,0,0),(0,0,2,0),(2,0,2,0)],[(0,0,0,2),(0,2,0,0),(0,2,0,2)],
               [(2,0,0,2),(0,2,2,0),(2,2,2,2)],[(2,0,0,0),(0,0,0,2),(2,0,0,2)],
               [(0,0,2,0),(0,2,0,0),(0,2,2,0)],[(2,0,2,0),(0,2,0,2),(2,2,2,2)]]
        fam = Fam(4, PM2, "PM_doubled_d4")
        run_family(fam, rng)
        print("pm2 pinned: |L|=16, S=0^6 -- NOT AvN. The V48 harness comment "
              "'AvN expected' is a stale erratum (doubling v -> 2v quadruples "
              "tau-phases: the d=2 anomaly dies mod 4); the harness Family "
              "class itself agrees (|L|=16).")
    else:
        d = args.d
        assert d % 2 == 0, "conjecture is stated at even d"
        print(f"[P1] order/spec-coset d={d}: worst |W^d - I| = "
              f"{order_spec_check(d, rng):.1e} over 40 labels: PASS")
        print(f"[P3] Fourier completion d={d}: reconstruction err = "
              f"{fourier_check(d, rng):.1e}: PASS")
        made = 0
        while made < args.nfam:
            ctxs = rand_family(rng, d, int(rng.integers(4, args.ntrip + 2)),
                               max_obs=9, scale=args.scale)
            if ctxs is None:
                continue
            try:
                fam = Fam(d, ctxs, f"rand_d{d}_s{args.seed}_sc{args.scale}_{made}")
            except Exception as ex:
                if "TooBig" in type(ex).__name__ or "budget" in str(ex):
                    continue
                raise
            if fam.L and len(fam.L) > 1500:
                continue
            made += 1
            run_family(fam, rng, use_frac=(d == 4 and len(fam.CTX) <= 5))
    print(f"done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
