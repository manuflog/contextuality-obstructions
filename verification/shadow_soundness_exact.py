# V54 - THE ODD-Q SHADOW: SOUND WITH NO EXCEPTIONS, AND INCOMPLETE IN THE OTHER DIRECTION.
#
# CORRECTS Paper B v1, Prop. 9, which stated that the mod-2 (odd-Q) shadow of the exact Z_d
# criterion can produce a FALSE CERTIFICATE, quoted 28.6% for the rate at d=8, and cited an
# explicit example in this repository. All three are wrong.
#
# THEOREM. Let d be EVEN. If M x = gamma (mod d) is solvable then M x = gamma (mod 2), so any
# lambda with lambda.M = 0 (mod 2) satisfies lambda.gamma = (lambda.M).x = 0 (mod 2). The shadow
# therefore cannot fire on a noncontextual system: the false-positive rate is exactly 0 at every
# even d, and no false certificate exists to be exhibited.
#
# THE REAL GAP. The shadow is INCOMPLETE, in the opposite direction: a Z_d-cycle certifies
# contextuality when lambda.gamma != 0 (mod d), and if that value is nonzero but EVEN its mod-2
# reduction vanishes, so the shadow is blind to a genuinely contextual system. A pinned
# non-degenerate d=8 witness is asserted below, and the miss rate is measured under a stated
# protocol.
#
# WHAT STANDS from Prop. 9: a minimal odd-Q SUPPORT need not be a Z_d-cycle, so certificate
# MINIMIZATION must search over Z_d. That is a false WITNESS, not a false VERDICT.
#
# SCOPE. Nothing here concerns Weyl families; for those see shadow_gap.py (V44).
#
# Expect: "shadow_soundness_exact PASS". Exit 1 on any failure. ~25 s.

import itertools
import sys

import numpy as np

D = 8


def kernel_mod2(M):
    """Basis of {lambda : lambda.M = 0 (mod 2)}."""
    A = (M % 2).T.copy() % 2
    rows, cols = A.shape
    piv, r = [], 0
    for c in range(cols):
        p = next((i for i in range(r, rows) if A[i, c]), None)
        if p is None:
            continue
        A[[r, p]] = A[[p, r]]
        for i in range(rows):
            if i != r and A[i, c]:
                A[i] ^= A[r]
        piv.append(c)
        r += 1
    basis = []
    for f in [c for c in range(cols) if c not in piv]:
        v = np.zeros(cols, dtype=int)
        v[f] = 1
        for i, c in enumerate(piv):
            v[c] = A[i, f]
        basis.append(v % 2)
    return basis


def shadow_fires(M, g):
    b = g % 2
    for lam in kernel_mod2(M):
        if (lam @ b) % 2 == 1:
            return True, lam
    return False, None


_XCACHE = {}


def _all_x(n, d):
    """All of Z_d^n as a single (d**n, n) array, cached. Exhaustive => exact, not sampled."""
    key = (n, d)
    if key not in _XCACHE:
        _XCACHE[key] = np.array(list(itertools.product(range(d), repeat=n)), dtype=np.int64)
    return _XCACHE[key]


def solvable(M, g, d):
    """Vectorised exhaustive search for x with M x = g (mod d). Returns a witness or None."""
    X = _all_x(M.shape[1], d)
    R = (X @ M.T) % d                      # (d**n, contexts)
    hit = np.nonzero(np.all(R == (g % d), axis=1))[0]
    return X[hit[0]] if hit.size else None


def zd_witness(M, g, d):
    """A Z_d-cycle certifying contextuality, or None. Vectorised exhaustive search."""
    L = _all_x(M.shape[0], d)
    closes = np.all((L @ M) % d == 0, axis=1)
    pairs = (L @ g) % d != 0
    hit = np.nonzero(closes & pairs)[0]
    return L[hit[0]] if hit.size else None


# ------------------------------------------------------- [1] soundness has no counterexample
def test_no_false_positive(trials=3000, k=4, n=4, seed=7):
    """The shadow must never fire on a solvable system. Theorem above says 0; verify it."""
    rng = np.random.default_rng(seed)
    fired = violations = 0
    for _ in range(trials):
        M = rng.integers(0, 2, size=(k, n))
        g = rng.integers(0, D, size=k)
        s, _ = shadow_fires(M, g)
        if not s:
            continue
        fired += 1
        if solvable(M, g, D) is not None:
            violations += 1
    print(f"  {trials} random incidence systems (k={k}, n={n}, d={D}); shadow fired on {fired}")
    print(f"  of those, solvable over Z_{D} (= a false certificate): {violations}")
    assert fired > 50, "sample too thin to be worth anything"
    assert violations == 0, "A FALSE POSITIVE WOULD REFUTE THE THEOREM ABOVE"
    print(f"  => false-certificate rate 0/{fired}. Paper B's quoted 28.6% is WRONG; the true rate")
    print("     is exactly 0, and not empirically but provably, for every even d.")
    return fired


# ------------------------------------------------------- [2] the real gap: incompleteness
HAND_WITNESS_M = np.array([
    [1, 0, 1, 0],
    [0, 1, 1, 1],
    [1, 1, 1, 0],
    [0, 1, 0, 0],
], dtype=int)
HAND_WITNESS_G = np.array([3, 0, 0, 1], dtype=int)


def test_incompleteness_witness():
    """A pinned, non-degenerate system that is Z_8-contextual with the shadow blind to it."""
    M, g = HAND_WITNESS_M, HAND_WITNESS_G
    assert (M.sum(axis=1) > 0).all(), "no empty context"
    assert (M.sum(axis=0) > 0).all(), "no unused observable"
    assert len(np.unique(M, axis=0)) == M.shape[0], "contexts distinct"

    lam = zd_witness(M, g, D)
    assert lam is not None, "must be Z_8-contextual"
    val = (lam @ g) % D
    fired, _ = shadow_fires(M, g)

    print(f"  M = {M.tolist()}")
    print(f"  gamma = {g.tolist()}   (contexts nonempty, distinct; every observable used)")
    print(f"  Z_{D} cycle lambda = {lam.tolist()}:  lambda.M mod {D} = {((lam @ M) % D).tolist()},"
          f"  lambda.gamma mod {D} = {val}  => CONTEXTUAL")
    print(f"  lambda.gamma mod 2 = {val % 2}  -- nonzero mod {D} but EVEN, so its mod-2 reduction dies")
    print(f"  shadow fires? {fired}   => the criterion is BLIND to a genuinely contextual system")

    assert solvable(M, g, D) is None, "must be unsolvable, i.e. genuinely contextual"
    assert val % 2 == 0, "the whole point is that the certifying value is even"
    assert not fired, "the shadow must be silent here"
    print("  => the shadow is INCOMPLETE. This is the real gap, and it is the OPPOSITE direction")
    print("     from the one Prop. 9 describes.")


def test_incompleteness_is_common(trials=20000, k=4, n=4, seed=19):
    """Incompleteness is not a freak: measure how often it happens, with the protocol stated."""
    rng = np.random.default_rng(seed)
    contextual = blind = 0
    for _ in range(trials):
        M = rng.integers(0, 2, size=(k, n))
        if (M.sum(axis=1) == 0).any() or (M.sum(axis=0) == 0).any():
            continue
        g = rng.integers(0, D, size=k)
        if solvable(M, g, D) is not None:
            continue
        contextual += 1
        s, _ = shadow_fires(M, g)
        if not s:
            blind += 1
    rate = blind / contextual if contextual else float('nan')
    print(f"  protocol: {trials} draws, k={k}, n={n}, d={D}, seed={seed}, degenerate systems skipped")
    print(f"  genuinely Z_{D}-contextual: {contextual};  of those the shadow missed: {blind}"
          f"  -> incompleteness rate {rate:.3f}")
    assert contextual > 100, "sample too thin"
    assert blind > 0, "expected the shadow to miss some contextual systems"
    return rate


if __name__ == "__main__":
    print("[1] SOUNDNESS: can the shadow ever fire on a noncontextual system?")
    test_no_false_positive()
    print()
    print("[2] INCOMPLETENESS: pinned non-degenerate witness")
    test_incompleteness_witness()
    print()
    print("[3] INCOMPLETENESS: how common is it?")
    rate = test_incompleteness_is_common()
    print()
    print("SUMMARY")
    print("  Soundness: the shadow cannot fire on a noncontextual system. Rate 0 at every even d,")
    print("  by the two-line proof in the header. Paper B v1's 28.6% and its cited false")
    print("  certificate are withdrawn in v2.")
    print(f"  Incompleteness: real, and in the other direction -- ~{rate:.0%} of contextual systems")
    print("  under the stated protocol, with a pinned non-degenerate witness above.")
    print("  Certificate minimization must still search over Z_d-cycles.")
    print("shadow_soundness_exact PASS")
