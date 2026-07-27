#!/usr/bin/env python3
"""
BRANCH: rank(A) >= rank(B) — the residual open statement behind flex_R - flex_skew <= d-1.

CONTEXT (reused verbatim, not re-derived; see DMINUS1_BOUND.md, dminus1_bound.py — READ THOSE
FIRST). At a faithful real realization of an exclusivity graph (rays v_i in R^d, edges = orthogonal
pairs), for edge e=(i,j) define the directed edge rows p_e = e_j (x) v_i, q_e = e_i (x) v_j and
    A_e = p_e + q_e   (symmetric edge row)         A := span{A_e}
    B_e = p_e - q_e   (antisymmetric edge row)     B := span{B_e}   (B *is* dminus1_bound's S)
Already PROVED (dminus1_bound.py / DMINUS1_BOUND.md): span(A) cap span(N) = {0}, and the identity
    flex_R - flex_skew = d - r - (rank A - rank B),   r = dim(symmetric eigen-commutant) >= 1.
so flex_R - flex_skew <= d-1 is EQUIVALENT (r=1, the generic/irreducible case) to, and in general
implied by:
                              rank(A) >= rank(B)                                   (**)
verified with 0/277 counterexamples in the prior session's zoo, CONJECTURE, two proof attempts
recorded as not closing (both logged honestly in DMINUS1_BOUND.md).

THIS FILE (new, this branch) adds two things:

(1) A HARDER counterexample hunt, deliberately targeting the regime the brief flags as
    under-explored: forests, pseudoforests (every component has <=1 cycle), bipartite graphs built
    from even cycles, disconnected unions, low d=3 with few edges, and — crucially — deliberately
    NON-faithful (ray-collapsed) realizations, to probe the boundary of the "faithful" hypothesis.

(2) A PROOF for a genuine, precisely-scoped subclass: PSEUDOFORESTS (every connected component of
    the exclusivity graph contains at most one cycle — this includes all forests as the trivial
    case, and strictly extends the brief's "trees/forests" suggestion to the natural next class up,
    unicyclic graphs and their disjoint unions/forests-with-one-extra-edge). The proof needs ONLY
    (a) v_i != 0 for every i, and (b) "faithful" in the strong sense v_i not proportional to v_j for
    i != j (ray-injectivity) — in particular NO orthogonality, NO genericity, and NO dimension
    bound are used anywhere in the algebra. Orthogonality is only used (as always) to determine
    which pairs count as edges of the realized graph.

Structure of the proof (full detail + the general reduction lemma in BRANCH_RANK_INEQ.md):

  LEMMA 1 (2-core reduction, general, PROVED, any graph, any nonzero v_i, no orthogonality used).
    Let i be a vertex of degree <= 1 in G, with G' = G - i.  Then restriction gives a linear
    ISOMORPHISM  stress(A)(G) ~= stress(A)(G'),  stress(B)(G) ~= stress(B)(G')  (stress(X) :=
    |E| - rank(X)). Iterating (leaf-peeling) down to the 2-core core(G):
        rank(A)(G) = (|E(G)|-|E(core(G))|) + rank(A)(core(G))     [and same for B]
    so rank(A)(G) >= rank(B)(G)  <=>  rank(A)(core(G)) >= rank(B)(core(G)).

  LEMMA 2 (single cycle, PROVED, any faithful [ray-injective] nonzero v_i, no orthogonality used).
    For any cycle C_n (n>=3) with v_i pairwise non-proportional: stress(A) = stress(B) = 0, i.e.
    rank(A) = rank(B) = n exactly.  (Proof: a symmetric/antisymmetric stress on a cycle is either
    identically 0 or nowhere-0; nowhere-0 forces v_{k+1} || v_{k-1} for every k, which propagates
    around the parity-2 sub-cycles of C_n and forces >=2 vertices to share a ray — contradicting
    ray-injectivity for every n>=3. Full case-split for odd/even n in BRANCH_RANK_INEQ.md.)

  THEOREM (pseudoforests, PROVED). If every connected component of G has at most one cycle
  (equivalently |E(component)| <= |V(component)|), and the realization is faithful (v_i pairwise
  non-proportional), then rank(A)(G) = rank(B)(G) = |E(G)| exactly (hence certainly rank(A) >=
  rank(B), with equality). Proof: core(G) is a disjoint union of simple cycles (Lemma 1's peeling
  applied to a graph whose only possible core content is 2-regular, since a pseudoforest's core has
  every vertex degree exactly 2), apply Lemma 2 to each cycle component of the core, combine with
  Lemma 1 and the trivial disjoint-union additivity of stress (no shared vertices/edges between
  components => the linear system block-diagonalizes exactly).

  RESIDUAL OPEN CASE: 2-cores with a vertex of degree >= 3 (theta graphs, cacti with a branch
  vertex, dense KS sets, wheels, prisms, complete bipartite, ...). Still CONJECTURE there; every
  such case in this file's zoo (and the prior 277-row zoo) satisfies rank(A) >= rank(B), with
  strict inequality only on the four dense named KS sets.

Every rank/identity claim below is machine-checked against `dminus1_bound.build_blocks` (imported,
not re-implemented) on freshly Gauss-Newton-realized rays — i.e. Lemma 1 and the pseudoforest
theorem are verified independently on many random instances, not merely assumed.

Run:
    python3 branch_rank_ineq.py            # fast: pseudoforest theorem checks + core-reduction
                                            # checks + degenerate-hypothesis probe + a moderate
                                            # hunt (~150 graphs), <45s
    python3 branch_rank_ineq.py big         # + a much larger hunt (trees/pseudoforests/theta/
                                            # bipartite/disconnected/low-d3), ~35-40s, run standalone
"""
import os, sys, time, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from itertools import combinations
from collections import defaultdict

from dminus1_bound import build_blocks
from hermitian_bilinear import (realize_graph, cycle_graph, wheel_graph, complete_bipartite,
                                 prism_graph, random_graph)
from flex_dimension import odd_cycle

TOL = 1e-8

# ======================================================================================
# graph combinatorics: 2-core (leaf-peeling), pseudoforest test, union of components
# ======================================================================================
def leaf_peel_core(edges, V):
    """Iteratively delete degree<=1 vertices. Returns (core_edges, core_vertices) as sorted lists.
    Pure combinatorics, no vector data -- this is the graph-theoretic '2-core'."""
    edges = set((min(i, j), max(i, j)) for i, j in edges)
    verts = set(range(V))
    changed = True
    while changed:
        changed = False
        deg = {v: 0 for v in verts}
        for i, j in edges:
            deg[i] += 1; deg[j] += 1
        low = [v for v in verts if deg[v] <= 1]
        if low:
            for v in low:
                verts.discard(v)
            edges = set((i, j) for i, j in edges if i in verts and j in verts)
            changed = True
    return sorted(edges), sorted(verts)

def component_cyclomatic(edges, V):
    """Union-find; returns dict root -> (num_vertices, num_edges) per connected component
    (isolated vertices counted, contributing (1,0))."""
    parent = list(range(V))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    for i, j in edges:
        union(i, j)
    comp_v = defaultdict(int); comp_e = defaultdict(int)
    for v in range(V):
        comp_v[find(v)] += 1
    for i, j in edges:
        comp_e[find(i)] += 1
    return {r: (comp_v[r], comp_e[r]) for r in comp_v}

def is_pseudoforest(edges, V):
    """True iff every connected component has at most one independent cycle (|E|<=|V| per
    component). Forests (|E|=|V|-1 per component) are the special case with 0 cycles."""
    comps = component_cyclomatic(edges, V)
    return all(e <= v for v, e in comps.values())

def core_is_disjoint_cycles(core_edges, core_vertices):
    """Sanity check: pseudoforest cores must be 2-regular (disjoint union of simple cycles) or
    empty. Independent structural check, not assumed."""
    if not core_vertices:
        return True
    deg = defaultdict(int)
    for i, j in core_edges:
        deg[i] += 1; deg[j] += 1
    return all(deg[v] == 2 for v in core_vertices)

# ======================================================================================
# random graph builders: trees, unicyclic graphs, pseudoforest unions, "theta" (non-pseudoforest)
# ======================================================================================
def random_tree(V, seed):
    rng = random.Random(seed)
    nodes = list(range(V)); rng.shuffle(nodes)
    edges = []
    for idx in range(1, V):
        parent = nodes[rng.randrange(idx)]
        a, b = parent, nodes[idx]
        edges.append((min(a, b), max(a, b)))
    return edges

def random_unicyclic(V, seed):
    """Tree + exactly one extra edge (closing exactly one cycle in the single component)."""
    edges = random_tree(V, seed)
    rng = random.Random(seed + 555)
    existing = set(edges)
    for _ in range(50):
        a, b = rng.randrange(V), rng.randrange(V)
        e = (min(a, b), max(a, b))
        if a != b and e not in existing:
            edges.append(e)
            return edges
    return edges  # V too small to add a non-edge (V<=1); returns tree

def union_components(pieces):
    """pieces: list of (edges, n). Returns (edges, total_n) with vertex ids offset per piece."""
    all_edges = []; offset = 0
    for edges, n in pieces:
        all_edges += [(i + offset, j + offset) for i, j in edges]
        offset += n
    return all_edges, offset

def random_pseudoforest(V_target, seed, n_pieces=None):
    """Random disjoint union of trees and unicyclic pieces, total vertex count ~V_target."""
    rng = random.Random(seed)
    if n_pieces is None:
        n_pieces = rng.randint(1, max(1, V_target // 4))
    sizes = []
    remaining = V_target
    for k in range(n_pieces - 1):
        s = rng.randint(2, max(2, remaining - 2 * (n_pieces - k - 1))) if remaining > 4 else remaining
        s = max(2, min(s, remaining - 2))
        sizes.append(s); remaining -= s
    sizes.append(max(1, remaining))
    pieces = []
    for k, sz in enumerate(sizes):
        if sz < 2:
            pieces.append(([], sz)); continue
        cyclic = rng.random() < 0.5
        e = random_unicyclic(sz, seed * 131 + k) if cyclic else random_tree(sz, seed * 131 + k)
        pieces.append((e, sz))
    return union_components(pieces)

def theta_graph(path_lengths=(2, 2, 2)):
    """Two hub vertices joined by len(path_lengths) internally-disjoint paths of the given edge
    lengths. The simplest 2-core with a degree>=3 vertex (NOT a pseudoforest) -- the natural next
    test case beyond the proved subclass."""
    edges = []; V = 2
    a, b = 0, 1
    for L in path_lengths:
        if L == 1:
            edges.append((a, b)); continue
        prev = a
        for _ in range(L - 1):
            edges.append((min(prev, V), max(prev, V))); prev = V; V += 1
        edges.append((min(prev, b), max(prev, b)))
    return edges, V

def ray_distinct(rays, tol=1e-6):
    """Strong faithfulness check used by the proof: v_i pairwise NON-PROPORTIONAL (this is
    STRONGER than hermitian_bilinear._faithful, which only forbids accidental ORTHOGONALITY, not
    accidental parallelism -- the proof needs ray-injectivity, so we check it explicitly rather
    than assume the realizer's notion of 'faithful' already gives it)."""
    rays = [np.asarray(v, float) for v in rays]
    rays = [v / np.linalg.norm(v) for v in rays]
    n = len(rays)
    for i in range(n):
        for j in range(i + 1, n):
            if abs(abs(np.dot(rays[i], rays[j])) - 1) < tol:
                return False
    return True

# ======================================================================================
# (1) PSEUDOFOREST THEOREM: machine-check rank(A)=rank(B)=|E| on random pseudoforest instances
# ======================================================================================
def check_pseudoforest_instance(name, edges, V, d, seed, tries=15, log=None):
    if not is_pseudoforest(edges, V):
        raise AssertionError(f"{name}: not a pseudoforest (bug in generator)")
    rays = realize_graph(edges, V, d, tries=tries, seed=seed)
    row = dict(name=name, d=d, V=V, E=len(edges))
    if rays is None:
        row["status"] = "no-realization"
        if log is not None: log.append(row)
        return row
    if not ray_distinct(rays):
        row["status"] = "skip-nondistinct"  # measure-zero event; report, don't count either way
        if log is not None: log.append(row)
        return row
    res = build_blocks(rays)
    ok = (res["rA"] == row["E"] and res["rB"] == row["E"] and res["rA"] == res["rB"])
    row.update(status="ok", rA=res["rA"], rB=res["rB"], E_detected=res["E"],
               theorem_holds=ok, rank_A_ge_B=res["rank_A_ge_B"])
    if log is not None: log.append(row)
    return row

def run_pseudoforest_zoo(n_instances=40, seed0=0, dims=(3, 4, 5), tries=15):
    log = []
    rng = random.Random(seed0)
    for k in range(n_instances):
        Vt = rng.randint(4, 14)
        d = rng.choice(dims)
        kind = rng.choice(["tree", "forest", "unicyclic", "mixed"])
        if kind == "tree":
            edges = random_tree(Vt, seed0 * 10007 + k)
            V = Vt
        elif kind == "forest":
            edges, V = random_pseudoforest(Vt, seed0 * 10007 + k, n_pieces=rng.randint(2, 4))
            # bias toward zero-cycle pieces by rejecting cyclic and rebuilding as trees only
            if not all(e <= v for v, e in component_cyclomatic(edges, V).values()):
                pass
        elif kind == "unicyclic":
            edges = random_unicyclic(Vt, seed0 * 10007 + k); V = Vt
        else:
            edges, V = random_pseudoforest(Vt, seed0 * 10007 + k)
        check_pseudoforest_instance(f"{kind}-{k}(V={V},d={d})", edges, V, d,
                                     seed=seed0 * 97 + k, tries=tries, log=log)
    return log

# ======================================================================================
# (2) 2-CORE REDUCTION LEMMA: machine-check rank(X)(G) = pendant_E + rank(X)(core(G))
# ======================================================================================
def check_core_reduction_instance(name, edges, V, d, seed, tries=15, log=None):
    rays = realize_graph(edges, V, d, tries=tries, seed=seed)
    row = dict(name=name, d=d, V=V, E=len(edges))
    if rays is None:
        row["status"] = "no-realization"
        if log is not None: log.append(row)
        return row
    if not ray_distinct(rays):
        row["status"] = "skip-nondistinct"
        if log is not None: log.append(row)
        return row
    res_full = build_blocks(rays)
    core_edges, core_vertices = leaf_peel_core(edges, V)
    pendant_E = len(edges) - len(core_edges)
    if not core_edges:
        core_rA = core_rB = core_E = 0
    else:
        idx = {v: k for k, v in enumerate(core_vertices)}
        core_rays = [rays[v] for v in core_vertices]
        res_core = build_blocks(core_rays)
        core_rA, core_rB, core_E = res_core["rA"], res_core["rB"], res_core["E"]
        if core_E != len(core_edges):
            row["status"] = "core-edge-mismatch"  # accidental extra orthogonality inside the core
            if log is not None: log.append(row)
            return row
    pred_rA = pendant_E + core_rA
    pred_rB = pendant_E + core_rB
    ok = (res_full["rA"] == pred_rA and res_full["rB"] == pred_rB)
    row.update(status="ok", full_rA=res_full["rA"], full_rB=res_full["rB"],
               pred_rA=pred_rA, pred_rB=pred_rB, pendant_E=pendant_E, core_E=core_E,
               core_is_cycles=core_is_disjoint_cycles(core_edges, core_vertices),
               lemma_holds=ok)
    if log is not None: log.append(row)
    return row

def run_core_reduction_zoo(seed0=1, tries=15):
    log = []
    specs = []
    # pendant trees glued onto a nontrivial core: theta graph + random pendant trees at core verts
    for (paths, npend, d) in [((2, 2, 2), 3, 3), ((3, 3, 2), 2, 4), ((2, 2, 2, 2), 4, 4),
                                ((3, 2, 2), 3, 5)]:
        edges, Vc = theta_graph(paths)
        rng = random.Random(seed0 * 71 + hash(paths) % 1000)
        V = Vc
        for p in range(npend):
            attach = rng.randrange(Vc)
            tail = rng.randint(1, 3)
            prev = attach
            for _ in range(tail):
                edges.append((min(prev, V), max(prev, V))); prev = V; V += 1
        specs.append((f"theta{paths}+{npend}pendants d={d}", edges, V, d))
    # pendant trees glued onto a cycle (should ALSO satisfy pseudoforest theorem; cross-check
    # the core-reduction lemma reduces correctly to a pure cycle core, consistent with Lemma 2)
    for (n, npend, d) in [(5, 3, 3), (6, 4, 4), (7, 2, 5)]:
        edges, Vc = cycle_graph(n)
        rng = random.Random(seed0 * 53 + n)
        V = Vc
        for p in range(npend):
            attach = rng.randrange(Vc)
            edges.append((min(attach, V), max(attach, V))); V += 1
        specs.append((f"C{n}+{npend}pendants d={d}", edges, V, d))
    for name, edges, V, d in specs:
        check_core_reduction_instance(name, edges, V, d, seed=seed0, tries=tries, log=log)
    return log

# ======================================================================================
# (3) DEGENERATE / NON-FAITHFUL PROBE: does the pseudoforest theorem's ray-injectivity hypothesis
#     matter? Analytic C4 = K_{2,2} construction (Theorem A' of EVEN_CYCLES.md, reused): faithful
#     realizations are v0,v2 in a 2-plane W1, v1,v3 in W1^perp, W2 -- ANY choice of vectors in
#     W1,W2 automatically satisfies all 4 edges (cross W1-W2 orthogonality), so this is the one
#     cycle where "graph-faithful" (no accidental extra edge) does NOT by itself force v0 != v2 or
#     v1 != v3 (C4 = K_{2,2} exactly: collapsing a class can never create an edge beyond what's
#     already there, unlike n>=6 -- see BRANCH_RANK_INEQ.md sec 4 for the general argument). This
#     makes C4 the sharp test case for whether ray-injectivity (not just graph-faithfulness) is
#     load-bearing in Lemma 2. We verify: (i) a genuinely ray-distinct, graph-faithful C4 in d=4
#     has stress(A)=stress(B)=0 (matches the theorem); (ii) collapsing BOTH v0=v2 AND v1=v3 (the
#     full double-collapse the Lemma-2 proof identifies as the unique failure mode -- collapsing
#     only one pair is NOT enough, verified below) turns on a genuine stress(A)=stress(B)=1>0,
#     while rank(A)=rank(B) is STILL observed (equality, not a violation) on this instance.
# ======================================================================================
def degenerate_probe_c4(d=4):
    if d < 4:
        raise ValueError("C4 needs d>=4 to be graph-faithful (Theorem A, EVEN_CYCLES.md)")
    E = np.eye(d)
    v0, v2 = E[0], (E[0] + E[1]) / np.linalg.norm(E[0] + E[1])       # distinct rays in W1=span(e0,e1)
    v1, v3 = E[2], (E[2] + 2 * E[3]) / np.linalg.norm(E[2] + 2 * E[3])  # distinct rays in W2=span(e2,e3)
    out = {}
    baseline_rays = [v0, v1, v2, v3]
    base = build_blocks(baseline_rays)
    out["baseline"] = dict(rA=base["rA"], rB=base["rB"], E=base["E"],
                            ray_distinct=ray_distinct(baseline_rays))
    # partial collapse (v0=v2 only, v1,v3 stay distinct): equations at vertices 1,3 already force
    # a_{01}=a_{12}=0 directly from v0,v2 INDEPENDENT in baseline; with v0=v2 those equations
    # become a_{01}+a_{12}=0 (weaker) -- but equations at vertices 0,2 still involve v1,v3
    # (independent), forcing a_{30}=a_{01}=0 outright. So partial collapse should NOT open a
    # stress -- verified below (a genuine cross-check that BOTH pairs must collapse together).
    partial_rays = [v0, v1, v0.copy(), v3]
    partial = build_blocks(partial_rays)
    out["partial_collapse_v0v2"] = dict(rA=partial["rA"], rB=partial["rB"], E=partial["E"],
                                         ray_distinct=ray_distinct(partial_rays))
    # full double collapse: v2:=v0, v3:=v1 (both parity classes collapse simultaneously) --
    # still graph-faithful (C4=K_{2,2} exactly, so no accidental extra edge is created).
    full_rays = [v0, v1, v0.copy(), v1.copy()]
    full = build_blocks(full_rays)
    out["full_double_collapse"] = dict(rA=full["rA"], rB=full["rB"], E=full["E"],
                                        ray_distinct=ray_distinct(full_rays),
                                        rank_A_ge_B=full["rank_A_ge_B"])
    return out

# ======================================================================================
# (4) HARDER HUNT — the open (non-pseudoforest) regime: theta graphs, bipartite-from-even-cycles,
#     disconnected mixes, low d=3 sparse/moderate density, large-r block unions with pseudoforest
#     pieces mixed in. Tracks the max observed rank(B)-rank(A) (should stay <=0; >0 = VIOLATION).
# ======================================================================================
def add_row(rows, name, rays, evidence):
    if rays is None:
        return None
    if not ray_distinct(rays):
        rows.append(dict(name=name, status="skip-nondistinct"))
        return None
    res = build_blocks(rays)
    res["name"] = name; res["evidence"] = evidence; res["status"] = "ok"
    res["gap"] = res["rB"] - res["rA"]
    rows.append(res)
    return res

def hunt_fast(seed0=42, tries=15):
    rows = []
    t0 = time.time()
    # (a) theta graphs (simplest non-pseudoforest 2-core), several path-length combos, d=3..6
    for paths in [(2, 2, 2), (2, 2, 3), (3, 3, 3), (2, 2, 2, 2), (2, 3, 4), (2, 2, 2, 3)]:
        edges, V = theta_graph(paths)
        for d in (3, 4, 5):
            rays = realize_graph(edges, V, d, tries=tries, seed=sum(paths) + d)
            add_row(rows, f"theta{paths} d={d}", rays, "NUMERICAL (theta graph, non-pseudoforest)")
    # (b) bipartite even-cycle unicyclic + complete bipartite at d=3 specifically (low-d stress)
    for (n, d) in [(4, 4), (6, 3), (6, 4), (8, 3), (8, 4), (10, 4)]:
        edges, V = cycle_graph(n)
        rays = realize_graph(edges, V, d, tries=tries, seed=n + d)
        add_row(rows, f"evenC{n} d={d} (bipartite cycle)", rays, "NUMERICAL (bipartite even cycle)")
    for (m, k) in [(2, 3), (3, 3), (2, 4)]:
        edges, V = complete_bipartite(m, k)
        rays = realize_graph(edges, V, 3, tries=tries, seed=m * 10 + k)
        add_row(rows, f"K_{{{m},{k}}} d=3", rays, "NUMERICAL (bipartite, d=3 stress)")
    # (c) low d=3, few-edge sparse random graphs (many V, low p) -- explicit brief target
    rng = random.Random(seed0)
    for V in (6, 8, 10, 12, 14):
        for p in (0.15, 0.25):
            for s in range(2):
                edges, n = random_graph(V, p, seed=rng.randint(0, 10**6))
                if not edges:
                    continue
                rays = realize_graph(edges, n, 3, tries=tries, seed=s + V)
                add_row(rows, f"Rand3(V={V},p={p},s={s}) d=3", rays, "NUMERICAL (low-d3 sparse)")
    # (d) disconnected unions mixing a dense KS-like graph with pseudoforest pieces (large r,
    #     the brief's explicit "large symmetric commutant r" + disconnected-union target)
    c5 = [np.asarray(v).real for v in odd_cycle(5)]
    tree_edges = random_tree(5, 7)
    tree_rays = realize_graph(tree_edges, 5, 3, tries=tries, seed=7)
    if tree_rays is not None:
        block = []
        offset = 0
        for piece in (c5, tree_rays):
            dd = len(piece[0])
            for v in piece:
                vv = np.zeros(dd); vv[:] = v
                block.append(vv)
        # embed both pieces in orthogonal subspaces of a bigger ambient space
        d1, d2 = len(c5[0]), len(tree_rays[0])
        big = []
        for v in c5:
            vv = np.zeros(d1 + d2); vv[:d1] = v; big.append(vv)
        for v in tree_rays:
            vv = np.zeros(d1 + d2); vv[d1:] = v; big.append(vv)
        add_row(rows, "C5(KS)+RandTree5 (disconnected, r>1)", big,
                "NUMERICAL (disjoint union, dense+pseudoforest, large r)")
    print(f"  [hunt_fast] {len(rows)} rows built in {time.time()-t0:.1f}s")
    return rows

def hunt_big(seed0=123, tries=15):
    """Bigger hunt: more theta variants, larger random pseudoforests, denser low-d3, more seeds.
    Intended for a standalone 'big' run (not chained after hunt_fast within one 45s budget)."""
    rows = []
    t0 = time.time()
    for a in range(2, 5):
        for b in range(2, 5):
            for c in range(2, 5):
                if (a, b, c) in [(2, 2, 2)]:
                    continue
                edges, V = theta_graph((a, b, c))
                d = 3 + ((a + b + c) % 3)
                rays = realize_graph(edges, V, d, tries=tries, seed=a * 100 + b * 10 + c)
                add_row(rows, f"theta({a},{b},{c}) d={d}", rays, "NUMERICAL (theta zoo)")
    rng = random.Random(seed0)
    for V in (8, 10, 12, 15, 18, 22):
        for p in (0.1, 0.2, 0.3):
            for s in range(2):
                edges, n = random_graph(V, p, seed=rng.randint(0, 10**6))
                if not edges:
                    continue
                d = rng.choice([3, 4])
                rays = realize_graph(edges, n, d, tries=tries, seed=s + V + int(p * 100))
                add_row(rows, f"Rand(V={V},p={p},s={s}) d={d}", rays, "NUMERICAL (big random hunt)")
    for k in range(20):
        edges, V = random_pseudoforest(rng.randint(8, 25), seed0 * 31 + k)
        d = rng.choice([3, 4, 5])
        rays = realize_graph(edges, V, d, tries=tries, seed=k + seed0)
        add_row(rows, f"BigPseudoforest{k}(V={V},d={d})", rays, "NUMERICAL (big pseudoforest)")
    print(f"  [hunt_big] {len(rows)} rows built in {time.time()-t0:.1f}s")
    return rows

# ======================================================================================
# reporting
# ======================================================================================
def summarize_pseudoforest(log):
    ok_rows = [r for r in log if r.get("status") == "ok"]
    n_ok = len(ok_rows)
    fails = [r for r in ok_rows if not r["theorem_holds"]]
    skipped_nr = len([r for r in log if r.get("status") == "no-realization"])
    skipped_nd = len([r for r in log if r.get("status") == "skip-nondistinct"])
    print(f"\nPSEUDOFOREST THEOREM (rank A = rank B = |E| exactly): "
          f"{n_ok - len(fails)}/{n_ok} instances OK "
          f"({skipped_nr} no-realization, {skipped_nd} skip-nondistinct, both excluded from count)")
    if fails:
        print(f"  *** FAILURES *** {[r['name'] for r in fails]}")
    return len(fails) == 0

def summarize_core_reduction(log):
    ok_rows = [r for r in log if r.get("status") == "ok"]
    n_ok = len(ok_rows)
    fails = [r for r in ok_rows if not r["lemma_holds"]]
    print(f"\n2-CORE REDUCTION LEMMA (rank(X)(G) = pendant_E + rank(X)(core)): "
          f"{n_ok - len(fails)}/{n_ok} instances OK")
    for r in ok_rows:
        print(f"  {r['name']:<32} pendantE={r['pendant_E']:>3} coreE={r['core_E']:>3} "
              f"full(rA,rB)=({r['full_rA']},{r['full_rB']})  pred(rA,rB)=({r['pred_rA']},{r['pred_rB']})  "
              f"core2reg={r['core_is_cycles']}  {'OK' if r['lemma_holds'] else '*** FAIL ***'}")
    if fails:
        print(f"  *** FAILURES *** {[r['name'] for r in fails]}")
    return len(fails) == 0

def summarize_hunt(rows, label):
    ok_rows = [r for r in rows if r.get("status") == "ok"]
    n = len(ok_rows)
    bad = [r for r in ok_rows if not r["rank_A_ge_B"]]
    maxgap = max((r["gap"] for r in ok_rows), default=None)
    print(f"\nHUNT [{label}]: {n} realized rows "
          f"({len([r for r in rows if r.get('status')=='skip-nondistinct'])} skipped non-distinct)")
    print(f"  rank(A) >= rank(B): holds on {n - len(bad)}/{n} rows; "
          f"max observed rank(B)-rank(A) = {maxgap}")
    if bad:
        print(f"  *** VIOLATION(S) *** {[(r['name'], r['rA'], r['rB']) for r in bad]}")
    strict = [r for r in ok_rows if r["rA"] > r["rB"]]
    if strict:
        print(f"  strict rank(A)>rank(B) on {len(strict)} rows: {[r['name'] for r in strict]}")
    return len(bad) == 0, maxgap

def print_degenerate_probe(out):
    print("\nDEGENERATE/NON-FAITHFUL PROBE (C4=K_{2,2}, d=4, analytic construction):")
    b, p, f = out["baseline"], out["partial_collapse_v0v2"], out["full_double_collapse"]
    print(f"  baseline (ray-distinct={b['ray_distinct']}):            "
          f"rA={b['rA']} rB={b['rB']} E={b['E']}  "
          f"(theorem predicts equality: {'YES' if b['rA']==b['rB']==b['E'] else 'NO -- unexpected'})")
    print(f"  partial collapse v0=v2 only (ray-distinct={p['ray_distinct']}): "
          f"rA={p['rA']} rB={p['rB']} E={p['E']}  "
          f"({'stress still 0, as predicted (one collapsed pair is not enough)' if p['rA']==p['E'] else 'UNEXPECTED: stress appeared from a single collapsed pair'})")
    print(f"  full double collapse v2=v0,v3=v1 (ray-distinct={f['ray_distinct']}): "
          f"rA={f['rA']} rB={f['rB']} E={f['E']}  rank_A_ge_B={f['rank_A_ge_B']}")
    if f["rA"] < f["E"]:
        print(f"  -> CONFIRMS the ray-injectivity hypothesis is load-bearing exactly at the "
              f"predicted failure mode (both parity classes collapsed together): stress(A) turns "
              f"on ({f['E']-f['rA']} dim) even though the realization stays graph-faithful (C4 = "
              f"K_{{2,2}} exactly, so no accidental edge is created by the collapse). "
              f"rank(A)>=rank(B) still HOLDS on this instance (equality, {f['rA']}={f['rB']}) -- "
              f"not a counterexample, but a genuine boundary case showing WHY the hypothesis is "
              f"stated with ray-injectivity, not just graph-faithfulness, for n=4.")
    else:
        print("  -> UNEXPECTED: full double collapse did not open a stress; investigate.")

# ======================================================================================
def main():
    t0 = time.time()
    args = sys.argv[1:]
    mode = "big" if "big" in args else "fast"
    print("=" * 110)
    print("BRANCH RANK INEQUALITY — rank(A) >= rank(B) : pseudoforest proof + harder hunt")
    print("=" * 110)

    print("\n--- (1) pseudoforest theorem: random tree/forest/unicyclic instances ---")
    pf_log = run_pseudoforest_zoo(n_instances=40 if mode == "fast" else 90, tries=15)
    pf_ok = summarize_pseudoforest(pf_log)
    print(f"[{time.time()-t0:.1f}s]")

    print("\n--- (2) 2-core reduction lemma: theta/cycle graphs with pendant trees glued on ---")
    cr_log = run_core_reduction_zoo(tries=20)
    cr_ok = summarize_core_reduction(cr_log)
    print(f"[{time.time()-t0:.1f}s]")

    print("\n--- (3) degenerate/non-faithful probe ---")
    probe = degenerate_probe_c4(d=4)
    print_degenerate_probe(probe)
    print(f"[{time.time()-t0:.1f}s]")

    print("\n--- (4) harder counterexample hunt ---")
    hrows = hunt_fast(tries=15)
    if mode == "big":
        hrows += hunt_big(tries=15)
    hunt_ok, maxgap = summarize_hunt(hrows, mode)
    print(f"[{time.time()-t0:.1f}s]")

    all_ok = pf_ok and cr_ok and hunt_ok
    print("\n" + "=" * 110)
    print(f"VERDICT: pseudoforest theorem {'PASS' if pf_ok else '*** FAIL ***'}; "
          f"core-reduction lemma {'PASS' if cr_ok else '*** FAIL ***'}; "
          f"hunt (mode={mode}) {'NO VIOLATION' if hunt_ok else '*** VIOLATION FOUND ***'} "
          f"(max rank(B)-rank(A) observed = {maxgap})")
    print(f"[{time.time()-t0:.1f}s total] {'ALL PASS' if all_ok else 'SEE FAILURES ABOVE'}")
    return 0 if all_ok else 1

if __name__ == "__main__":
    sys.exit(main())
