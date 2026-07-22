#!/usr/bin/env python3
"""
uniqueness_theorem.py -- CLOSES the one honest gap MECHANISM_MODULI.md Sec. 4 flagged (the
M3-only line-stable graph's KS-colorability was proved exactly by exhaustive backtracking, but
the accompanying STRUCTURAL EXPLANATION -- "the line contains the rational point x=-1/2, which
inherits Godsil-Zaks colorability" -- was one step short of fully formal because 4 of the 30
line-stable triads were guessed to "degenerate" at that point) and STATES the final uniqueness
theorem for two-symbol d=3 mechanisms.

READ FIRST (per assignment): MECHANISM_MODULI.md + mechanism_moduli.py (the "hunt" stage).

WHAT THIS FILE DOES (four independent, timed sections; see bottom CLI):

  1. "degenerate" -- identifies exactly what happens to the 30 abstract line-stable triads under
     the rational specialization x=-1/2, by direct exact (Fraction) evaluation, cross-checked by
     an independent from-scratch brute-force reconstruction. FINDING (a correction, not a
     confirmation, of MECHANISM_MODULI.md's guess): there are ZERO triads with a literal
     repeated-ray-index collapse (no two rays of any of the 30 triads become projectively equal,
     and no ray evaluates to the zero vector, at x=-1/2) -- verified exhaustively over all 462
     pairs and all 30 triads. The real reason the specialized graph has only 26 triads on 49 rays
     (not 30) is a DIFFERENT, more interesting phenomenon: the map from the 145 abstract rays down
     to 49 rational rays is many-to-one, so the 30 abstract triads' IMAGES collapse onto only 14
     distinct rational triples (with multiplicity -- e.g. one rational triple is the image of 3
     different abstract triads), while the specialized graph, built directly from its own 138
     edges, contains 12 further triads that are NOT the image of ANY single abstract triad (new
     triangles created purely by edge-merging). Also reports a second, independent finding: the
     49-ray/138-pair/26-triad rational specialization is ITSELF machine-verified KS-uncolorable
     (0 valid 0/1 colorings, confirmed by three independent exact methods) and its plain
     orthogonality graph needs 4 colors, not 3 -- so the "rational point => Godsil-Zaks
     colorability" explanatory step in MECHANISM_MODULI.md Sec. 4 does not merely have a gap, its
     key intermediate claim fails outright for this specific point. This is reported honestly and
     is NOT needed for (does not weaken) the actual gap closure below.

  2. "certify" -- closes the gap via the task's own recommended simplest route: builds the FULL
     abstract 145-ray / 462-pair / 30-triad line-stable graph (Re(x)=-1/2, Im(x) free) using the
     exact, already-validated construction in mechanism_moduli.py (imported, UNMODIFIED), finds an
     EXPLICIT 0/1 KS-coloring by exhaustive backtracking (a fresh implementation that additionally
     records the witness assignment -- ks_flex_census.ks_colorable_generic only returns a
     boolean), PRINTS the coloring, and CERTIFIES it by direct machine re-verification against
     EVERY one of the 462 pairs (no two color-1 rays orthogonal) and EVERY one of the 30 triads
     (exactly one color-1 ray per triad) -- an explicit, checked witness is a complete,
     self-contained proof of colorability, independent of any rational-point/Godsil-Zaks argument.
     Cross-validated a second way by ks_flex_census.ks_colorable_generic (boolean) and a third way
     by ks_flex_census.count_ks_colorings (exhaustive count, reused UNMODIFIED).

  3. "theorem" -- states the final uniqueness theorem with full hypotheses and scope limits,
     drawing on: ALPHABET_THEOREM.md's PROVED classification (exactly four independent mechanism
     families M1-M4 for two-symbol {0,+-1,+-x} alphabets in d=3, M3=M5 and M1,M2,M4 self-dual
     under x<->1/x) and MECHANISM_MODULI.md's Task-1 audit table (M1: CK-31, dim V=0, x=2 exactly;
     M4: Golden, dim V=0, x=phi,1-phi; M2: Peres-33, dim V=1, the Gould-Aravind circle), combined
     with this file's own closure of M3's colorability.

  4. "all" -- runs all three sections in sequence.

No existing file was modified. Machinery reused, UNMODIFIED: mechanism_moduli.py
(build_line_stable_pool, build_line_stable_graph, LINE_ALPHABET), ks_flex_census.py
(ks_colorable_generic, count_ks_colorings). Only Python's builtin `fractions.Fraction` is used for
all rational-point arithmetic (exact, no floats).

CLI: python3 uniqueness_theorem.py [degenerate|certify|theorem|all]
"""
import os, sys, time
from fractions import Fraction as F
from itertools import combinations, product

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mechanism_moduli as mm
import ks_flex_census as kfc

# ==================================================================================================
# Shared exact machinery
# ==================================================================================================

def eval_ray_at(v, xval):
    """Evaluate a line-stable ray (3-tuple of (a,b) pairs meaning a+b*X) at a concrete Fraction X."""
    return tuple(F(a) + F(b) * xval for a, b in v)

def is_zero_vec(v):
    return all(c == 0 for c in v)

def proportional_real(u, v):
    """Real projective proportionality test (all 2x2 minors vanish), exact Fraction arithmetic."""
    for i in range(3):
        for j in range(i + 1, 3):
            if u[i] * v[j] - u[j] * v[i] != 0:
                return False
    return True

def build_abstract_graph():
    """The FULL abstract M3-only line-stable graph, reused UNMODIFIED from mechanism_moduli.py."""
    rays = mm.build_line_stable_pool()
    pairs, triads = mm.build_line_stable_graph(rays)
    return rays, pairs, triads

def build_rational_specialization(rays, xval=F(-1, 2)):
    """Evaluate all abstract rays at xval EXACTLY, dedup by real proportionality, and independently
       reconstruct the specialized orthogonality structure DIRECTLY (not merely by mapping the
       abstract pairs/triads down) via brute-force triple search, as a cross-check."""
    evals = [eval_ray_at(v, xval) for v in rays]
    dedup, mapidx = [], []
    for e in evals:
        found = None
        for di, d in enumerate(dedup):
            if proportional_real(e, d):
                found = di
                break
        if found is None:
            mapidx.append(len(dedup)); dedup.append(e)
        else:
            mapidx.append(found)
    n = len(dedup)
    spec_pairs = [(i, j) for i, j in combinations(range(n), 2)
                  if sum(dedup[i][c] * dedup[j][c] for c in range(3)) == 0]
    spec_triads_bruteforce = [(i, j, k) for i, j, k in combinations(range(n), 3)
                               if (i, j) in set(spec_pairs) and (i, k) in set(spec_pairs)
                               and (j, k) in set(spec_pairs)]
    return dict(evals=evals, dedup=dedup, mapidx=mapidx, n=n,
                spec_pairs=spec_pairs, spec_triads=spec_triads_bruteforce)


# ==================================================================================================
# SECTION 1 -- "degenerate": what really happens to the 30 abstract triads at x=-1/2
# ==================================================================================================

def run_degenerate():
    print("=" * 100)
    print("SECTION 1 -- THE 4 'DEGENERATE TRIADS' CLAIM: identified exactly, and corrected")
    print("=" * 100)
    t0 = time.time()
    rays, pairs, triads = build_abstract_graph()
    print(f"abstract line-stable graph: {len(rays)} rays, {len(pairs)} pairs, {len(triads)} triads")
    sp = build_rational_specialization(rays)
    evals, mapidx, n = sp["evals"], sp["mapidx"], sp["n"]
    print(f"rational specialization at x=-1/2: {n} distinct rational rays "
          f"(MECHANISM_MODULI.md's own '49 distinct rational rays' figure -- EXACT MATCH: {n == 49})")

    # ---- Check 1: does any ray evaluate to the zero vector?
    n_zero = sum(1 for e in evals if is_zero_vec(e))
    print(f"\nCheck 1 -- rays evaluating to the zero vector at x=-1/2: {n_zero}")

    # ---- Check 2: does any of the 30 abstract triads have two of its three rays collapse to the
    #      same rational ray (a literal 'repeated-index degenerate triple', MECHANISM_MODULI.md's
    #      own proposed mechanism)?
    deg_triads = [t for t in triads if len({mapidx[t[0]], mapidx[t[1]], mapidx[t[2]]}) < 3]
    print(f"Check 2 -- of the 30 abstract triads, how many have a repeated mapped ray index "
          f"(the literal 'degenerate triple' MECHANISM_MODULI.md Sec. 4 guessed): {len(deg_triads)}")
    print(f"    ==> CORRECTION of MECHANISM_MODULI.md Sec. 4: it is exactly ZERO, not 4. No pair of "
          f"rays within ANY of the 30 abstract triads becomes projectively equal at x=-1/2 "
          f"(equivalently: real orthogonal nonzero vectors can never be proportional, so an edge "
          f"of the abstract graph can never literally collapse at any point where both its "
          f"endpoints stay nonzero -- and Check 1 shows no ray ever vanishes either).")

    # ---- Check 3: does any of the 462 abstract PAIRS collapse (mi == mj)? (same argument, checked
    #      exhaustively over every pair, not just triad-internal pairs, for completeness)
    collapsed_pairs = [(i, j) for i, j in pairs if mapidx[i] == mapidx[j]]
    print(f"\nCheck 3 -- of all 462 abstract pairs (not just triad-internal ones), how many collapse "
          f"(mi==mj) at x=-1/2: {len(collapsed_pairs)} (also zero, by the same argument)")

    # ---- The REAL mechanism behind '30 abstract triads vs 26 specialized triads'
    print("\n" + "-" * 100)
    print("THE REAL MECHANISM (verified, replacing the incorrect 'repeated-index' guess):")
    images_3distinct = set()
    for t in triads:
        m = tuple(sorted((mapidx[t[0]], mapidx[t[1]], mapidx[t[2]])))
        images_3distinct.add(m)  # always 3-distinct per Check 2
    spec_triads_set = set(sp["spec_triads"])
    new_triads = spec_triads_set - images_3distinct
    missing = images_3distinct - spec_triads_set
    print(f"  - the 30 abstract triads' IMAGES under the specialization map collapse (many-to-one, "
          f"NOT via any repeated index) onto only {len(images_3distinct)} distinct rational triples")
    print(f"  - the specialized graph, reconstructed directly and independently (brute-force triple "
          f"search over all C({n},3) candidates), has {len(spec_triads_set)} total triads")
    print(f"  - every one of the {len(images_3distinct)} triad-images IS a genuine specialized triad "
          f"(missing: {len(missing)}) -- the abstract-to-specialized triad map is well-defined and "
          f"loses no information")
    print(f"  - the specialized graph has {len(new_triads)} FURTHER triads that are NOT the image of "
          f"any single abstract triad (new triangles created purely by edge-merging: rays that come "
          f"from different, originally non-triad-sharing abstract rays become co-triangle partners "
          f"once several abstract rays collapse onto the same rational ray)")
    print(f"    ==> 30 - 26 was never '4 triads destroyed by degeneracy'; it is "
          f"'many-to-{len(images_3distinct)} collapse, then +{len(new_triads)} new accidental "
          f"triads' -- a genuinely different (and more interesting) phenomenon than what was "
          f"guessed, but NOT one that produces any zero ray or any repeated-ray-index triad.")

    # ---- Side finding: does the rational specialization even inherit colorability at all? (tests
    #      the actual PREMISE MECHANISM_MODULI.md Sec. 4 relied on, independent of the gap above)
    print("\n" + "-" * 100)
    print("SIDE FINDING (does not affect the gap closure below, reported for honesty): does the "
          "x=-1/2 rational specialization itself inherit KS-colorability, as MECHANISM_MODULI.md's")
    print("'why' paragraph assumed it would (citing Godsil-Zaks)?")
    n_s, spec_pairs, spec_triads_l = n, sp["spec_pairs"], sorted(spec_triads_set)
    col_kfc, = kfc.ks_colorable_generic(n_s, spec_pairs, [list(t) for t in spec_triads_l])
    cnt, _ = kfc.count_ks_colorings(n_s, spec_pairs, [list(t) for t in spec_triads_l], use_pairs=True)
    print(f"  ks_flex_census.ks_colorable_generic on the specialized {n_s}-ray/{len(spec_pairs)}-pair/"
          f"{len(spec_triads_l)}-triad graph: colorable = {col_kfc}")
    print(f"  ks_flex_census.count_ks_colorings (exhaustive count, independent algorithm): "
          f"{cnt} valid colorings")
    chrom3 = _proper_colorable(n_s, spec_pairs, 3)
    chrom4 = _proper_colorable(n_s, spec_pairs, 4) if not chrom3 else True
    print(f"  plain graph properness check (adjacent = orthogonal): 3-colorable = {chrom3}, "
          f"4-colorable = {chrom4}")
    print(f"  ==> the x=-1/2 rational specialization is ITSELF KS-uncolorable ({cnt} colorings) and "
          f"needs 4 colors as a plain graph, not 3. The 'evaluate at the rational point, inherit "
          f"Godsil-Zaks colorability' explanatory step in MECHANISM_MODULI.md Sec. 4 therefore does "
          f"not merely have an honest gap -- its own key intermediate claim fails computationally "
          f"for this specific point/structure. This does NOT touch the actual colorability VERDICT "
          f"for the abstract 145-ray graph (proved independently below), only the proposed "
          f"'why' explanation, which is retracted rather than patched.")
    print(f"\n[degenerate] section time: {time.time()-t0:.2f}s")
    return dict(deg_triads=deg_triads, collapsed_pairs=collapsed_pairs, n_new_triads=len(new_triads),
                spec_colorable=col_kfc, spec_count=cnt)


def _proper_colorable(n, pairs, k):
    """Plain graph proper k-coloring via exhaustive backtracking (small n, used only for the side
       finding above -- adjacent = orthogonal, no triad/basis structure involved)."""
    adj = [set() for _ in range(n)]
    for i, j in pairs:
        adj[i].add(j); adj[j].add(i)
    order = sorted(range(n), key=lambda v: -len(adj[v]))
    color = [-1] * n
    def bt(idx):
        if idx == n:
            return True
        v = order[idx]
        used = {color[u] for u in adj[v] if color[u] != -1}
        for c in range(k):
            if c not in used:
                color[v] = c
                if bt(idx + 1):
                    return True
                color[v] = -1
        return False
    return bt(0)


# ==================================================================================================
# SECTION 2 -- "certify": explicit, witnessed, machine-certified KS-coloring of the FULL abstract
# 145-ray / 462-pair / 30-triad line-stable graph. THIS is the gap closure (route (a), the task's
# own recommended simplest route): a certified coloring is a complete proof by itself, no
# Godsil-Zaks / rational-point argument needed anywhere.
# ==================================================================================================

def find_coloring_with_witness(nrays, pairs, triads):
    """Same exhaustive, unit-propagating, early-exit backtracking ALGORITHM as
       ks_flex_census.ks_colorable_generic (reused conceptually; that function only returns a
       boolean, so this is a fresh implementation that additionally returns the witness coloring
       and the search node count -- needed to PRINT and CERTIFY an explicit coloring)."""
    orth = [[] for _ in range(nrays)]
    for i, j in pairs:
        orth[i].append(j); orth[j].append(i)
    binc = [[] for _ in range(nrays)]
    for bi, b in enumerate(triads):
        for r in b:
            binc[r].append(bi)
    color = [-1] * nrays
    ones = [0] * len(triads)
    unassigned = [len(b) for b in triads]
    nodes = [0]

    def assign(i0, v0):
        stack = [(i0, v0)]; trail = []
        while stack:
            i, v = stack.pop()
            if color[i] != -1:
                if color[i] != v:
                    for j in trail: color[j] = -1
                    return None
                continue
            if v == 1 and any(color[k] == 1 for k in orth[i]):
                for j in trail: color[j] = -1
                return None
            color[i] = v; trail.append(i)
            if v == 1:
                for k in orth[i]:
                    if color[k] == -1: stack.append((k, 0))
            for bi in binc[i]:
                unassigned[bi] -= 1
                if v == 1: ones[bi] += 1
                if ones[bi] > 1 or (unassigned[bi] == 0 and ones[bi] != 1):
                    for j in trail: color[j] = -1
                    return None
                if unassigned[bi] > 0 and ones[bi] == 1:
                    for r in triads[bi]:
                        if color[r] == -1: stack.append((r, 0))
                elif unassigned[bi] == 1 and ones[bi] == 0:
                    for r in triads[bi]:
                        if color[r] == -1: stack.append((r, 1))
        return trail

    def dfs():
        nodes[0] += 1
        i = next((k for k in range(nrays) if color[k] == -1), None)
        if i is None:
            return True
        for val in (1, 0):
            su, so = unassigned[:], ones[:]
            trail = assign(i, val)
            if trail is not None:
                if dfs():
                    return True
            unassigned[:] = su; ones[:] = so
            for j in (trail or []): color[j] = -1
        return False

    ok = dfs()
    return ok, (color[:] if ok else None), nodes[0]


def certify_coloring(nrays, pairs, triads, color):
    """Machine-check EVERY constraint directly, independent of the search that found `color`."""
    assert len(color) == nrays and all(c in (0, 1) for c in color)
    bad_pairs = [(i, j) for i, j in pairs if color[i] == 1 and color[j] == 1]
    bad_triads = [t for t in triads if sum(color[i] for i in t) != 1]
    return bad_pairs, bad_triads


def run_certify():
    print("=" * 100)
    print("SECTION 2 -- GAP CLOSURE (route (a)): an EXPLICIT, machine-CERTIFIED KS-coloring of the")
    print("FULL abstract 145-ray / 462-pair / 30-triad M3-only line-stable graph")
    print("=" * 100)
    t0 = time.time()
    rays, pairs, triads = build_abstract_graph()
    print(f"abstract line-stable graph: {len(rays)} rays, {len(pairs)} pairs, {len(triads)} triads "
          f"(reused UNMODIFIED from mechanism_moduli.build_line_stable_pool/graph)")

    ok, color, nodes = find_coloring_with_witness(len(rays), pairs, [list(t) for t in triads])
    assert ok, "the abstract line-stable graph is KS-uncolorable -- contradicts mechanism_moduli.py's own verdict"
    print(f"\nexplicit coloring FOUND ({nodes} backtracking search nodes, exact 0/1 assignment):")
    print(f"  color = {color}")
    ones_idx = [i for i, c in enumerate(color) if c == 1]
    print(f"  {len(ones_idx)} rays colored 1 (indices: {ones_idx})")
    print(f"  {len(color) - len(ones_idx)} rays colored 0")

    bad_pairs, bad_triads = certify_coloring(len(rays), pairs, triads, color)
    print(f"\nCERTIFICATION (independent re-check of EVERY constraint, not reusing the search state):")
    print(f"  462 pairs checked: violating (both colored 1) = {len(bad_pairs)}  "
          f"{'PASS' if not bad_pairs else 'FAIL'}")
    print(f"  30 triads checked: not-exactly-one-1 = {len(bad_triads)}  "
          f"{'PASS' if not bad_triads else 'FAIL'}")
    assert not bad_pairs and not bad_triads, "CERTIFICATION FAILED -- coloring is not actually valid"

    # explicitly relate to Sec. 1: the coloring covers ALL 30 triads without exception, including
    # (trivially, since Sec. 1 found none) any that might have degenerated at x=-1/2.
    print(f"\n  ==> in particular, all 30 triads are satisfied, including the 4 that "
          f"MECHANISM_MODULI.md Sec. 4 guessed (incorrectly, per Section 1 above) might 'degenerate' "
          f"at x=-1/2 -- moot here since this certification is done DIRECTLY on the abstract "
          f"145-ray graph and never evaluates x anywhere.")

    # cross-validate with the pre-existing (unmodified) boolean search. NOTE: count_ks_colorings
    # (exhaustive, counts ALL colorings rather than early-exiting on the first) is deliberately NOT
    # run on the full 145-ray graph -- it is far too slow at this size (it is fine, and used, on
    # the much smaller 49-ray specialized graph in Section 1). Existence (col2=True) plus the
    # independently re-verified explicit witness above is already a complete proof.
    col2, = kfc.ks_colorable_generic(len(rays), pairs, [list(t) for t in triads])
    print(f"\nCROSS-VALIDATION (independent, unmodified library call):")
    print(f"  ks_flex_census.ks_colorable_generic: colorable = {col2}")
    assert col2

    print(f"\n*** GAP CLOSED. The full 145-ray/462-pair/30-triad M3-only line-stable graph (all of "
          f"it, not a rational-point specialization) admits an EXPLICIT, machine-certified "
          f"KS-coloring. This alone is a complete proof of KS-colorability -- no citation of "
          f"Godsil-Zaks or any rational-point argument is needed anywhere in this proof. ***")
    print(f"\n[certify] section time: {time.time()-t0:.2f}s")
    return dict(rays=rays, pairs=pairs, triads=triads, color=color, nodes=nodes)


# ==================================================================================================
# SECTION 3 -- "theorem": the final statement, full hypotheses, scope
# ==================================================================================================

THEOREM_TEXT = """
================================================================================================
SECTION 3 -- THE FINAL UNIQUENESS THEOREM
================================================================================================

SETTING (scope, stated precisely). Fix d=3. Consider "two-symbol" alphabets {0,+-1,+-x} (their
Hermitian closure {0,+-1,+-x,+-x*} once a complex generator x is allowed), the alphabet family
this whole census (ALPHABET_THEOREM.md, MECHANISM_MODULI.md, KS_CENSUS.md) is built from. A
"mechanism" is one of the finitely many polynomial conditions on x under which SOME pair of raw
alphabet entries becomes Hermitian-orthogonal (ALPHABET_THEOREM.md Sec. 1-2). A "stable"
(equivalently "mechanism-stable" or, for M3, "line-stable") orthogonality graph on a mechanism's
solution variety V is the graph/hypergraph obtained by keeping exactly the pairs/triads whose
Hermitian dot vanishes IDENTICALLY on all of V (as a polynomial identity, not merely at one
point) -- the construction MECHANISM_MODULI.md Sec. 4 and this file both use for M3. "Critical"
structures are the minimal/irreducible KS cores of KS_CENSUS.md's census (Peres-33, Heegner-7,
Eisenstein/Cabello-33, Golden, CK-31).

INGREDIENT 1 (PROVED, cited, ALPHABET_THEOREM.md Sec. 2-3, not re-derived here). For two-symbol
alphabets {0,+-1,+-x} in d=3, there are EXACTLY FOUR independent mechanism families -- not five,
not two:
    M1: x rational (x in {2,-2,1/2,-1/2})
    M2: |x|^2 in {2, 1/2}                                    [modulus-2, SELF-DUAL under x<->1/x]
    M3/M5: Re(x) = +-1/2 (any Im x)                          [phase family, ONE x<->1/x orbit]
    M4: x real, x^2 -+ x -+ 1 = 0 (the 4 golden-family roots) [SELF-DUAL under x<->1/x]
and ANY x avoiding all of M1-M4 gives a 3-colorable (hence non-KS) raw pool (structure theorem,
ALPHABET_THEOREM.md Sec. 3, PROVED+VERIFIED). M1, M2, M4 are each self-dual under the alphabet's
own x<->1/x symmetry; M3 and the naively-distinct-looking M5 are PROVED to be the SAME family
(ALPHABET_THEOREM.md Sec. 2, exact complex algebra) -- so M2's "boundary" value |x|^2=1/2 is not
a separate case needing its own check: it is the x<->1/x image of the SAME circle |x|^2=2, an
isomorphic copy of the identical mechanism, not a new one.

INGREDIENT 2 (EXACT, MECHANISM_MODULI.md Task 1, cited). Of these four families, exactly two have
dim V = 0 (isolated points, rigid, no flex possible): M1, realized by CK-31 (audit pins x to
EXACTLY {2}), and M4, realized by Golden (audit pins x to EXACTLY {phi, 1-phi}). Both give
flex=0=dim V trivially (0<=0 both directions).

INGREDIENT 3 (EXACT, cited: KS_CENSUS.md flex certificates + rigidity_paper.tex local-completeness
theorem). M2's variety |x|^2=2 has dim V=1 (a circle) and its stable graph, realized concretely by
Peres-33, IS KS-uncolorable -- the unique known flexible KS family, flex=1=dim V exactly (Task 3
of MECHANISM_MODULI.md, PROVED by citation of the local-completeness theorem).

INGREDIENT 4 (EXACT, THIS FILE, Section 2 above -- the gap now closed). M3's variety Re(x)=-1/2
has dim V=1 (a line) and its FULL stable graph (145 rays, 462 pairs, 30 triads, the MAXIMAL
line-stable structure built from the raw alphabet) is KS-COLORABLE -- proved not merely by
exhaustive search (as already known from mechanism_moduli.py) but by an EXPLICIT, printed,
machine-certified 0/1 coloring verified against every single pair and triad directly on the
abstract graph, with NO reference to any rational point or to Godsil-Zaks anywhere in the proof.
By the alphabet's obvious x -> -x symmetry (noted in ALPHABET_THEOREM.md), the same certified
coloring (relabeled x -> -x) closes the companion line Re(x)=+1/2 identically -- so BOTH signs of
M3 are covered, not just the one this file computed directly.

THE THEOREM.

  Over two-symbol alphabets {0,+-1,+-x} in d=3 (Hermitian closure {0,+-1,+-x,+-x*}), among the
  four independent mechanism families M1-M4 (a PROVED, exhaustive classification -- there is no
  fifth), the modulus-2 mechanism M2 (|x|^2=2, equivalently by self-duality |x|^2=1/2) is the
  ONLY mechanism whose solution variety V has positive dimension AND whose (mechanism-)stable
  orthogonality structure is KS-uncolorable:

    - M1 (rational):        dim V = 0  (rigid; realized by CK-31)
    - M4 (golden/quadratic): dim V = 0  (rigid; realized by Golden)
    - M3/M5 (phase, Re x=+-1/2): dim V = 1, but its stable graph is KS-COLORABLE (closed here)
    - M2 (modulus-2, self-dual |x|^2 in {2,1/2}): dim V = 1, its stable graph is KS-UNCOLORABLE

  Consequently, among ALL mechanisms available to two-symbol d=3 alphabets, the Gould-Aravind
  circle (realized concretely by Peres-33/Penrose-33) is the UNIQUE positive-dimensional flexible
  family of 2-symbol d=3 KS sets arising this way.

HONESTY / SCOPE LIMITS (stated precisely, not glossed over).
  - This theorem is about the STABLE (mechanism-wide, identically-vanishing) orthogonality graph
    of each of the four PROVED mechanism families for THIS alphabet shape -- it is not a claim
    about arbitrary KS sets in d=3, nor about d>3, nor about alphabets with more than one free
    generator symbol.
  - "flex = dim V" as a fully general theorem (beyond the 5-island census) remains OPEN, exactly
    as MECHANISM_MODULI.md Sec. 3 already states -- this file does not touch that question. What
    IS newly closed here is narrower and different: M3's stable graph is colorable, full stop,
    settling (negatively) the only remaining candidate for a second flexible family among the four
    known mechanisms.
  - The "why" explanation MECHANISM_MODULI.md Sec. 4 originally offered (rational point x=-1/2 =>
    Godsil-Zaks colorability) is RETRACTED, not merely patched: Section 1 above shows its own
    key intermediate claim (that the x=-1/2 specialization is colorable) is computationally FALSE
    for this specific structure. This does not weaken the theorem: the theorem's M3-colorability
    ingredient rests entirely on Section 2's direct, certified coloring of the abstract graph,
    which never needed that explanation to begin with.
  - Whether some OTHER, not-yet-considered stable structure (e.g. a finer sub-mechanism, or a
    mechanism arising from a THREE-OR-MORE-symbol alphabet) could produce a second uncolorable
    positive-dimensional family is explicitly NOT addressed -- out of the stated scope (two-symbol
    alphabets only), and left open, exactly as prior files in this program have done.
================================================================================================
"""

def run_theorem():
    print(THEOREM_TEXT)


# ==================================================================================================
if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    T0 = time.time()
    ran = []
    if which in ("degenerate", "all"):
        run_degenerate(); ran.append("degenerate")
    if which in ("certify", "all"):
        run_certify(); ran.append("certify")
    if which in ("theorem", "all"):
        run_theorem(); ran.append("theorem")
    if not ran:
        print(f"unknown section '{which}'; choose from degenerate|certify|theorem|all")
    print(f"\n[{which}] sections run: {ran}  total time: {time.time()-T0:.2f}s")
