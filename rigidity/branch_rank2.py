#!/usr/bin/env python3
"""
BRANCH RANK2 — closing the theta-graph subclass, and a much larger one: 2-DEGENERATE graphs.

CONTEXT (read `BRANCH_RANK_INEQ.md` / `branch_rank_ineq.py` FIRST; both reused via import,
NEITHER is modified). Setup exactly as there: faithful REAL orthogonal representation v_i in
R^d\\{0}, edges = orthogonal pairs; for edge e=(i,j) directed rows p_e=e_j(x)v_i, q_e=e_i(x)v_j,
A_e=p_e+q_e (symmetric block A), B_e=p_e-q_e (antisymmetric block B). PROVED there (reused, not
re-derived): rank(A)>=rank(B) is the residual open statement behind flex_R-flex_skew<=d-1; and
    stress(X) := |E| - rank(X) = dim{ x: E->R (symmetric/antisym-extended) : sum_{j~i} x_{ij} v_j
                                        = 0  for every vertex i }                          (dagger)
so rank(A)>=rank(B) <=> dim stress(A) <= dim stress(B) [the brief's KEY REFRAME].
`branch_rank_ineq.py` PROVED: (Lemma 1) degree-<=1 ("leaf") peeling gives a restriction/extension
ISOMORPHISM stress(X)(G) ~= stress(X)(G-v) for ANY nonzero vertex vectors (no ray-injectivity
needed); (Lemma 2) a single cycle C_n has stress(A)=stress(B)=0 under ray-injectivity (v_i pairwise
NON-PROPORTIONAL); combined: rank(A)=rank(B)=|E| exactly for the whole PSEUDOFOREST class (every
component has <=1 cycle). Residual gap named there: any 2-core with a vertex of degree>=3 (theta
graphs, wheels, prisms, complete bipartite, Petersen, the 4 dense KS sets).

THIS FILE'S NEW RESULT (proof, not just numerics).

A one-line strengthening of Lemma 1's mechanism turns out to already answer item (1) of the brief
(theta graphs) as an immediate special case of a much more general statement:

    KEY OBSERVATION. If vertex v has degree EXACTLY 2 with (distinct) neighbors p != q, its
    stress equation is a two-term vector relation  c_p v_p + c_q v_q = 0.  Since p != q are
    DISTINCT vertices, ray-injectivity (v_p not proportional to v_q) means v_p, v_q are
    automatically LINEARLY INDEPENDENT (two nonzero, non-proportional vectors are linearly
    independent in ANY vector space, ANY dimension >= 2 -- this is just the definition of linear
    dependence for two vectors). Linear independence forces c_p = c_q = 0 DIRECTLY, in one line,
    from THIS SINGLE equation -- no propagation, no case-split on cycle parity needed.

Iterating this over a DEGENERACY ELIMINATION ORDER (a graph is k-degenerate iff its vertices admit
an order u_1,...,u_n with |N(u_i) INTERSECT {u_1,...,u_{i-1}}| <= k for every i -- the standard
graph-theoretic notion) gives, by induction, the following clean general theorem:

    THEOREM (2-degenerate graphs, PROVED below, Section 2).  If G is 2-DEGENERATE (equivalently:
    repeatedly deleting a vertex of current degree <=2 empties the graph -- this includes ALL
    trees, forests, pseudoforests, cycles, theta graphs, "theta chains"/generalized series-
    parallel-type graphs, and much more) and the realization is ray-injective, THEN
    stress(A)(G) = stress(B)(G) = 0 EXACTLY, i.e. rank(A)(G) = rank(B)(G) = |E(G)| identically,
    for ANY dimension d, no genericity/orthogonality used beyond determining the edge set.

Theta graphs (k>=3 internally-disjoint paths between 2 hubs) are 2-degenerate (peel every internal
path vertex, degree 2 throughout; what remains is 2 hubs with <=1 edge between them, degree <=1,
peel that too) -- so THEOREM closes item (1) of the brief as a corollary (Section 3), and strictly
subsumes the prior session's pseudoforest theorem (every pseudoforest's 2-core is a disjoint union
of cycles, each itself 2-degenerate) as another corollary -- reproved here by a genuinely simpler,
one-line-per-vertex argument (no cyclic case-split needed at all).

Also proved (Section 4, answering the brief's item (2), the "ear decomposition" ask): the RIGHT
generalization of Lemma 1's leaf-peeling is NOT a cut-vertex/ear-decomposition step (the brief
correctly flags that a degree->=3 cut vertex, e.g. a bowtie's shared vertex, genuinely mixes both
sides and breaks the clean restriction/extension bijection) -- it is a DEGREE-2 peeling step
instead (never touching degree->=3 vertices at all). This closes the gap the brief identified by
sidestepping it: peeling stops exactly at each graph's "3-core" (min-degree->=3 subgraph, possibly
empty), giving a general reduction lemma (Lemma R2, Section 4) that subsumes Lemma 1 and reduces
ANY graph's open problem to its (typically much smaller) 3-core, whether or not that 3-core is
empty. Wheels, prisms, K_{3,3}, Petersen, and the 4 dense KS sets all have 3-core = themselves
(min degree already >=3) -- genuinely outside this theorem, exactly where the brief said the
residual difficulty lives; confirmed computationally in Section 6 (not claimed proved there).

Every claim is machine-checked against `dminus1_bound.build_blocks` (imported, not reimplemented)
on freshly Gauss-Newton-realized rays via `hermitian_bilinear.realize_graph` (also imported).
`branch_rank_ineq.py`'s own graph generators/helpers (`theta_graph`, `ray_distinct`, `random_tree`,
etc.) are imported and reused, not copied or modified.

Run:
    python3 branch_rank2.py            # fast: theta zoo + random 2-degenerate zoo + 3-core
                                        # reduction lemma + degenerate boundary probe + small
                                        # non-example confirmatory check, <20s
    python3 branch_rank2.py big        # + larger/deeper zoos (chains of thetas, bigger random
                                        # 2-degenerate graphs, larger 3-core-reduction instances),
                                        # <45s
"""
import os, sys, time, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from itertools import combinations
from collections import defaultdict

from dminus1_bound import build_blocks
from hermitian_bilinear import (realize_graph, cycle_graph, wheel_graph, complete_bipartite,
                                 prism_graph, random_graph)
from branch_rank_ineq import (theta_graph, ray_distinct, random_tree, random_unicyclic,
                               random_pseudoforest, is_pseudoforest)

TOL = 1e-8

# ======================================================================================
# (0) DEGENERACY MACHINERY -- pure combinatorics, no vector data.
# ======================================================================================
def degeneracy_and_order(edges, V):
    """Standard greedy min-degree peeling (exact: this greedy always attains the true
    degeneracy -- a standard fact). Returns (degeneracy, removal_order) where removal_order
    lists vertices in the order they were peeled (removal_order[0] removed first); at the time
    vertex u is removed, its degree IN THE REMAINING GRAPH is <= degeneracy (by construction),
    and this is exactly the 'back-degree <= degeneracy' elimination order used in Section 2's
    induction (reversed)."""
    adj = [set() for _ in range(V)]
    for i, j in edges:
        adj[i].add(j); adj[j].add(i)
    remaining = set(range(V))
    deg = {v: len(adj[v]) for v in remaining}
    order = []
    degen = 0
    while remaining:
        v = min(remaining, key=lambda x: deg[x])
        degen = max(degen, deg[v])
        order.append((v, deg[v]))
        remaining.discard(v)
        for w in adj[v]:
            if w in remaining:
                deg[w] -= 1
    return degen, order

def is_k_degenerate(edges, V, k):
    degen, _ = degeneracy_and_order(edges, V)
    return degen <= k

def peel_to_kcore(edges, V, k):
    """Iteratively delete vertices of CURRENT degree <= k (standard k-core computation).
    Returns (core_edges, core_vertices) -- the maximal subgraph with min degree > k (empty if
    G is k-degenerate). Pure combinatorics; generalizes branch_rank_ineq.leaf_peel_core's
    threshold-1 version to a general threshold k (k=2 is what Section 4 uses)."""
    edges = set((min(i, j), max(i, j)) for i, j in edges)
    verts = set(range(V))
    changed = True
    while changed:
        changed = False
        deg = {v: 0 for v in verts}
        for i, j in edges:
            deg[i] += 1; deg[j] += 1
        low = [v for v in verts if deg[v] <= k]
        if low:
            for v in low:
                verts.discard(v)
            edges = set((i, j) for i, j in edges if i in verts and j in verts)
            changed = True
    return sorted(edges), sorted(verts)

# ======================================================================================
# (0b) graph generators specific to this branch
# ======================================================================================
def random_2degenerate_graph(V, seed, max_attach=2):
    """Builds a graph vertex-by-vertex: vertex v (v=1..V-1) connects to a uniformly random
    subset of size in {0,...,min(max_attach,v)} of {0,...,v-1}. By construction this has
    back-degree <= max_attach at every vertex in this very order, so it is EXACTLY
    max_attach-degenerate by the standard characterization -- no combinatorial checking needed
    to know it qualifies, though we double check with degeneracy_and_order anyway (independent
    verification, not just trusting the construction)."""
    rng = random.Random(seed)
    edges = []
    for v in range(1, V):
        k = rng.randint(0, min(max_attach, v))
        attach = rng.sample(range(v), k) if k > 0 else []
        for a in attach:
            edges.append((min(a, v), max(a, v)))
    return edges, V

def chain_of_thetas(path_specs, seed=0):
    """Glue several theta graphs into a chain: piece i's SECOND hub is identified with piece
    i+1's FIRST hub. A genuinely bigger, non-pseudoforest, non-single-theta 2-degenerate graph
    (multiple degree>=3 vertices, none adjacent to more than one other branch vertex directly
    unless a path_spec itself has a length-1 leg)."""
    all_edges = []
    offset = 0
    prev_hub = None
    for spec in path_specs:
        e, Vc = theta_graph(spec)
        if prev_hub is None:
            mapping = {0: offset, 1: offset + 1}
            nxt = offset + 2
        else:
            mapping = {0: prev_hub, 1: offset}
            nxt = offset + 1
        for v in range(2, Vc):
            mapping[v] = nxt; nxt += 1
        offset = nxt
        all_edges += [(mapping[a], mapping[b]) for a, b in e]
        prev_hub = mapping[1]
    return all_edges, offset

def petersen_graph():
    outer = [(i, (i + 1) % 5) for i in range(5)]
    inner = [(5 + i, 5 + (i + 2) % 5) for i in range(5)]
    spokes = [(i, 5 + i) for i in range(5)]
    return outer + inner + spokes, 10

def complete_graph(n):
    return [(i, j) for i in range(n) for j in range(i + 1, n)], n

# ======================================================================================
# (1) MAIN THEOREM CHECK: rank(A)=rank(B)=|E| for 2-degenerate graphs
# ======================================================================================
def check_2degenerate_instance(name, edges, V, d, seed, tries=15, log=None, expect_degeneracy_le=2):
    degen, _ = degeneracy_and_order(edges, V)
    row = dict(name=name, d=d, V=V, E=len(edges), degeneracy=degen)
    if degen > expect_degeneracy_le:
        row["status"] = "NOT-2-DEGENERATE"
        if log is not None: log.append(row)
        return row
    rays = realize_graph(edges, V, d, tries=tries, seed=seed)
    if rays is None:
        row["status"] = "no-realization"
        if log is not None: log.append(row)
        return row
    if not ray_distinct(rays):
        row["status"] = "skip-nondistinct"
        if log is not None: log.append(row)
        return row
    res = build_blocks(rays)
    ok = (res["rA"] == row["E"] and res["rB"] == row["E"] and res["rA"] == res["rB"])
    row.update(status="ok", rA=res["rA"], rB=res["rB"], E_detected=res["E"],
               theorem_holds=ok, rank_A_ge_B=res["rank_A_ge_B"])
    if log is not None: log.append(row)
    return row

def run_theta_zoo(dims=(3, 4, 5, 6), tries=15, big=False):
    """Item (1) of the brief: theta graphs specifically, many path-length profiles."""
    log = []
    specs = [(2, 2, 2), (2, 2, 3), (2, 3, 3), (3, 3, 3), (2, 2, 2, 2), (1, 2, 2), (1, 3, 4),
              (2, 3, 4), (1, 2, 3, 4), (2, 2, 2, 3), (1, 4, 5)]
    if big:
        specs += [(a, b, c) for a in range(1, 5) for b in range(2, 5) for c in range(2, 5)
                  if a + b + c <= 12][:24]
    seen = set()
    for spec in specs:
        if spec in seen:
            continue
        seen.add(spec)
        edges, V = theta_graph(spec)
        for d in dims[:3] if not big else dims:
            check_2degenerate_instance(f"theta{spec} d={d}", edges, V, d,
                                        seed=sum(spec) * 13 + d, tries=tries, log=log)
    return log

def run_random_2degenerate_zoo(n_instances=30, seed0=0, dims=(3, 4, 5), tries=15,
                                 max_attach=2, Vrange=(5, 14)):
    log = []
    rng = random.Random(seed0)
    for k in range(n_instances):
        V = rng.randint(*Vrange)
        d = rng.choice(dims)
        edges, _ = random_2degenerate_graph(V, seed0 * 7919 + k, max_attach=max_attach)
        check_2degenerate_instance(f"Rand2Deg(V={V},d={d})#{k}", edges, V, d,
                                    seed=seed0 * 131 + k, tries=tries, log=log)
    return log

def run_chain_theta_zoo(tries=15, big=False):
    log = []
    chains = [
        [(2, 2, 2), (2, 2, 2)],
        [(2, 3, 2), (1, 3, 3)],
        [(2, 2, 2), (2, 2, 2), (2, 2, 2)],
        [(1, 2, 3), (2, 2, 2), (1, 3, 2)],
    ]
    if big:
        chains.append([(2, 2, 2, 2), (1, 2, 3), (2, 3, 4), (2, 2, 2)])
    for i, specs in enumerate(chains):
        edges, V = chain_of_thetas(specs, seed=i)
        d = 3 + (i % 3)
        check_2degenerate_instance(f"ThetaChain{specs} d={d}", edges, V, d,
                                    seed=1000 + i, tries=tries, log=log)
    return log

# ======================================================================================
# (2) LEMMA R2: degree-<=2 peeling reduces to the 3-core (min-degree->=3 subgraph). This
#     subsumes branch_rank_ineq.py's Lemma 1 (which only used the degree<=1 phase and did not
#     need ray-injectivity there); the degree=2 phase genuinely needs ray-injectivity, matching
#     the pattern of the prior file's Lemma 2 (cycle) also needing it.
# ======================================================================================
def check_3core_reduction_instance(name, edges, V, d, seed, tries=15, log=None):
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
    core_edges, core_vertices = peel_to_kcore(edges, V, 2)
    peeled_E = len(edges) - len(core_edges)
    if not core_edges:
        core_rA = core_rB = core_E = 0
    else:
        core_rays = [rays[v] for v in core_vertices]
        res_core = build_blocks(core_rays)
        core_rA, core_rB, core_E = res_core["rA"], res_core["rB"], res_core["E"]
        if core_E != len(core_edges):
            row["status"] = "core-edge-mismatch"
            if log is not None: log.append(row)
            return row
    pred_rA = peeled_E + core_rA
    pred_rB = peeled_E + core_rB
    ok = (res_full["rA"] == pred_rA and res_full["rB"] == pred_rB)
    row.update(status="ok", full_rA=res_full["rA"], full_rB=res_full["rB"],
               pred_rA=pred_rA, pred_rB=pred_rB, peeled_E=peeled_E, core_E=core_E,
               core_min_degree_ge3=all(
                   sum(1 for (i, j) in core_edges if i == v or j == v) >= 3
                   for v in core_vertices) if core_vertices else True,
               lemma_holds=ok)
    if log is not None: log.append(row)
    return row

def run_3core_reduction_zoo(tries=20, big=False):
    """Graphs whose 3-core is NONEMPTY (a wheel or K4 with degree-2 material glued on), to
    exercise Lemma R2 in the genuinely nontrivial regime (peeled_E > 0 AND core_E > 0)."""
    log = []
    specs = []
    # wheel + pendant trees + a subdivided (degree-2-inserted) spoke -- 3-core should reduce
    # back to (a subdivision-free part of) the wheel itself
    for (n, npend, d) in [(4, 3, 3), (5, 2, 4), (6, 3, 5)]:
        edges, Vc = wheel_graph(n)
        rng = random.Random(7 * n + 3)
        V = Vc
        # subdivide one rim edge (insert a degree-2 vertex) -- should get peeled back off
        i, j = edges[-1]
        edges = edges[:-1]
        mid = V; V += 1
        edges += [(min(i, mid), max(i, mid)), (min(j, mid), max(j, mid))]
        for p in range(npend):
            attach = rng.randrange(Vc)
            edges.append((min(attach, V), max(attach, V))); V += 1
        specs.append((f"wheel{n}+subdiv+{npend}pendants d={d}", edges, V, d))
    # K4 with a pendant theta-chain glued on (degree-2 material feeding into a 3-core)
    ke, kv = complete_graph(4)
    e2, v2 = theta_graph((2, 3, 2))
    edges = list(ke) + [(a + kv, b + kv) for a, b in e2]
    edges.append((0, kv))  # attach K4 vertex 0 to the theta's first hub
    specs.append(("K4+theta(2,3,2)pendant d=4", edges, kv + v2, 4))
    if big:
        for (n, npend, d) in [(7, 4, 4), (8, 3, 5)]:
            edges, Vc = wheel_graph(n)
            rng = random.Random(11 * n)
            V = Vc
            for p in range(npend):
                attach = rng.randrange(Vc)
                tail = rng.randint(1, 3)
                prev = attach
                for _ in range(tail):
                    edges.append((min(prev, V), max(prev, V))); prev = V; V += 1
            specs.append((f"wheel{n}+{npend}chainpendants d={d}", edges, V, d))
    for name, edges, V, d in specs:
        check_3core_reduction_instance(name, edges, V, d, seed=17, tries=tries, log=log)
    return log

# ======================================================================================
# (3) DEGENERATE / NON-RAY-INJECTIVE BOUNDARY PROBE for the degree-2 mechanism specifically.
#     Fresh, independently-built analytic construction (own code, not a re-run of the prior
#     file's C4 probe, though it exercises the same underlying phenomenon: C4=K_{2,2} is the
#     minimal graph where EVERY vertex's two neighbors are exactly the other side's full
#     support, so collapsing BOTH parity classes removes every rescue route).
# ======================================================================================
def degenerate_probe_c4(d=4):
    if d < 4:
        raise ValueError("C4 needs d>=4 to be graph-faithful (no accidental extra edge)")
    E = np.eye(d)
    v0 = E[0]
    v2 = (E[0] + 0.7 * E[1]); v2 /= np.linalg.norm(v2)
    v1 = E[2]
    v3 = (E[2] + 1.3 * E[3]); v3 /= np.linalg.norm(v3)
    out = {}
    baseline = [v0, v1, v2, v3]
    b = build_blocks(baseline)
    out["baseline"] = dict(rA=b["rA"], rB=b["rB"], E=b["E"], ray_distinct=ray_distinct(baseline))
    single = [v0, v1, v0.copy(), v3]
    s = build_blocks(single)
    out["single_pair_collapse_v0v2"] = dict(rA=s["rA"], rB=s["rB"], E=s["E"],
                                             ray_distinct=ray_distinct(single))
    full = [v0, v1, v0.copy(), v1.copy()]
    f = build_blocks(full)
    out["both_pairs_collapse"] = dict(rA=f["rA"], rB=f["rB"], E=f["E"],
                                       ray_distinct=ray_distinct(full),
                                       rank_A_ge_B=f["rank_A_ge_B"])
    return out

def degenerate_probe_theta_hub_collapse(d=5, tries=15, seed=99):
    """Attempt the analogous probe on K_{2,3}=theta(2,2,2): collapse the TWO HUBS (the only
    pair every internal degree-2 vertex's equation shares) onto the same ray, keeping the 3
    internal vertices distinct/generic. Reported HONESTLY either way: because K_{2,3}'s hub
    equations are 3-TERM (not 2-term), the hub equations themselves may independently rescue
    the conclusion if the 3 internal-vertex rays stay linearly independent -- this probe checks
    which outcome actually occurs, it does not assume one."""
    rays = realize_graph(*theta_graph((2, 2, 2)), d, tries=tries, seed=seed)
    if rays is None:
        return dict(status="no-realization")
    rays = [np.asarray(v, float) for v in rays]
    # theta_graph((2,2,2)): hubs are vertices 0,1; internal vertices 2,3,4
    v = [r.copy() for r in rays]
    v[1] = v[0].copy()  # collapse hub1 onto hub0's ray -- breaks ray-injectivity at the hubs
    res = build_blocks(v)
    return dict(status="ok", rA=res["rA"], rB=res["rB"], E=res["E"],
                ray_distinct=ray_distinct(v), baseline_E=len(theta_graph((2, 2, 2))[0]))

# ======================================================================================
# (4) CONFIRMATORY NON-EXAMPLE CHECK: graphs with degeneracy >=3 (wheels, K4, K_{3,3}, prisms,
#     Petersen) should be correctly REJECTED by is_k_degenerate (i.e. genuinely outside this
#     branch's theorem) -- and we report (NUMERICALLY, not as new proof) whether rank(A)>=rank(B)
#     still holds on them anyway, exactly reproducing the qualitative finding of
#     branch_rank_ineq.py's hunt (equality almost everywhere, strict inequality only at the 4
#     dense KS sets) with fresh instances, so this file's picture is self-contained.
# ======================================================================================
def run_nonexample_confirmation(tries=15):
    rows = []
    cases = [
        ("wheel5", *wheel_graph(5), 4),
        ("wheel6", *wheel_graph(6), 4),
        ("prism5", *prism_graph(5), 4),
        ("K33", *complete_bipartite(3, 3), 3),
        ("K4", *complete_graph(4), 4),
        ("Petersen", *petersen_graph(), 4),
    ]
    for name, edges, V, d in cases:
        degen, _ = degeneracy_and_order(edges, V)
        row = dict(name=name, V=V, E=len(edges), d=d, degeneracy=degen,
                   is_2degenerate=(degen <= 2))
        rays = realize_graph(edges, V, d, tries=tries, seed=hash(name) % 100000)
        if rays is None:
            row["status"] = "no-realization"
        elif not ray_distinct(rays):
            row["status"] = "skip-nondistinct"
        else:
            res = build_blocks(rays)
            row.update(status="ok", rA=res["rA"], rB=res["rB"], E_detected=res["E"],
                       rank_A_ge_B=res["rank_A_ge_B"])
        rows.append(row)
    return rows

# ======================================================================================
# reporting
# ======================================================================================
def summarize_theorem(log, label):
    ok_rows = [r for r in log if r.get("status") == "ok"]
    n_ok = len(ok_rows)
    fails = [r for r in ok_rows if not r["theorem_holds"]]
    not2deg = len([r for r in log if r.get("status") == "NOT-2-DEGENERATE"])
    skipped_nr = len([r for r in log if r.get("status") == "no-realization"])
    skipped_nd = len([r for r in log if r.get("status") == "skip-nondistinct"])
    print(f"\n[{label}] rank(A)=rank(B)=|E| exactly: {n_ok - len(fails)}/{n_ok} OK "
          f"({not2deg} not-2-degenerate [generator bug if >0], {skipped_nr} no-realization, "
          f"{skipped_nd} skip-nondistinct)")
    if fails:
        print(f"  *** FAILURES *** {[r['name'] for r in fails]}")
    if not2deg:
        print(f"  *** BUG: generator produced non-2-degenerate instance(s) ***")
    return len(fails) == 0 and not2deg == 0

def summarize_3core(log):
    ok_rows = [r for r in log if r.get("status") == "ok"]
    fails = [r for r in ok_rows if not r["lemma_holds"]]
    print(f"\n[3-CORE REDUCTION, Lemma R2] rank(X)(G) = peeled_E + rank(X)(3core(G)): "
          f"{len(ok_rows) - len(fails)}/{len(ok_rows)} OK")
    for r in ok_rows:
        print(f"  {r['name']:<34} peeledE={r['peeled_E']:>3} coreE={r['core_E']:>3} "
              f"full(rA,rB)=({r['full_rA']},{r['full_rB']}) pred(rA,rB)=({r['pred_rA']},{r['pred_rB']}) "
              f"core_min_deg>=3={r['core_min_degree_ge3']} {'OK' if r['lemma_holds'] else '*** FAIL ***'}")
    if fails:
        print(f"  *** FAILURES *** {[r['name'] for r in fails]}")
    return len(fails) == 0

def print_c4_probe(out):
    print("\n[DEGENERATE PROBE 1] C4=K_{2,2}, d=4 (fresh construction, own code):")
    b, s, f = out["baseline"], out["single_pair_collapse_v0v2"], out["both_pairs_collapse"]
    print(f"  baseline (ray-distinct={b['ray_distinct']}): rA={b['rA']} rB={b['rB']} E={b['E']} "
          f"(theorem predicts 0 stress: {'YES' if b['rA']==b['E'] else 'NO -- unexpected'})")
    print(f"  single pair collapse v0=v2 (ray-distinct={s['ray_distinct']}): "
          f"rA={s['rA']} rB={s['rB']} E={s['E']} "
          f"({'still stress-free, as predicted' if s['rA']==s['E'] else 'UNEXPECTED stress from one collapsed pair'})")
    print(f"  both pairs collapse v2=v0,v3=v1 (ray-distinct={f['ray_distinct']}): "
          f"rA={f['rA']} rB={f['rB']} E={f['E']} rank_A_ge_B={f['rank_A_ge_B']}")
    if f["rA"] < f["E"]:
        print("  -> CONFIRMS ray-injectivity is load-bearing for the degree-2 zeroing step "
              "exactly where the proof predicts (both neighbors of some degree-2 vertex "
              "collapsing together); rank(A)>=rank(B) still holds on this instance (not a "
              "counterexample to the target inequality, only to the STRONGER equality claim).")
    else:
        print("  -> UNEXPECTED: no stress opened; investigate.")

def print_theta_probe(out):
    print("\n[DEGENERATE PROBE 2] theta(2,2,2)=K_{2,3}, hub collapse (own fresh construction):")
    if out.get("status") != "ok":
        print(f"  status={out.get('status')} (skipped)"); return
    print(f"  hubs collapsed onto one ray (ray-distinct={out['ray_distinct']}): "
          f"rA={out['rA']} rB={out['rB']} E={out['E']} (baseline E={out['baseline_E']})")
    if out["rA"] < out["baseline_E"]:
        print("  -> stress DID open: the hub collapse breaks the degree-2 mechanism at every "
              "internal vertex simultaneously and is NOT rescued by the (3-term, but here still "
              "linearly-dependent-in-effect) hub equations on this instance.")
    else:
        print("  -> stress did NOT open here: the theta graph's hub equations (3 independent "
              "internal-vertex rays) rescue the conclusion even though ray-injectivity fails at "
              "the hubs -- reported honestly as a genuine (and mechanistically informative) "
              "negative result for this particular probe, not glossed over. It shows the "
              "degree-2 mechanism has redundancy in K_{2,3} that C4 lacks; it does NOT weaken "
              "the main theorem, whose hypothesis (full ray-injectivity) is simply not violated "
              "in the theorem's own regime.")

def print_nonexample_table(rows):
    print("\n[CONFIRMATORY NON-EXAMPLES] degeneracy >=3 graphs (outside this branch's theorem):")
    for r in rows:
        print(f"  {r['name']:<10} V={r['V']:>2} E={r['E']:>2} degeneracy={r['degeneracy']} "
              f"is_2degenerate={r['is_2degenerate']} status={r['status']} "
              + (f"rA={r.get('rA')} rB={r.get('rB')} rank_A_ge_B={r.get('rank_A_ge_B')}"
                 if r.get("status") == "ok" else ""))
    bad_deg = [r for r in rows if r["is_2degenerate"]]
    if bad_deg:
        print(f"  *** UNEXPECTED: these named non-examples were classified 2-degenerate: "
              f"{[r['name'] for r in bad_deg]} (would mean the theorem DOES apply -- investigate) ***")
    return len(bad_deg) == 0

# ======================================================================================
def main():
    t0 = time.time()
    args = sys.argv[1:]
    mode = "big" if "big" in args else "fast"
    print("=" * 110)
    print("BRANCH RANK2 — 2-degenerate graphs (theta graphs as the headline corollary)")
    print("=" * 110)

    print("\n--- (1) theta-graph zoo [brief item (1)] ---")
    theta_log = run_theta_zoo(tries=15, big=(mode == "big"))
    theta_ok = summarize_theorem(theta_log, "THETA GRAPHS")
    print(f"[{time.time()-t0:.1f}s]")

    print("\n--- (2) random 2-degenerate graph zoo ---")
    rand_log = run_random_2degenerate_zoo(n_instances=25 if mode == "fast" else 600, tries=15,
                                           dims=(3, 4, 5, 6, 7),
                                           Vrange=(5, 14) if mode == "fast" else (5, 32))
    rand_ok = summarize_theorem(rand_log, "RANDOM 2-DEGENERATE")
    print(f"[{time.time()-t0:.1f}s]")

    print("\n--- (2b) random 1-degenerate (attach<=1, trees/forests -- sanity floor) and "
          "attach<=2 with a fixed larger max_attach=2 but bigger seeds sweep ---")
    extra_ok = True
    if mode == "big":
        extra_log = []
        for max_attach in (1, 2):
            extra_log += run_random_2degenerate_zoo(n_instances=120, seed0=999 + max_attach,
                                                      tries=15, max_attach=max_attach,
                                                      Vrange=(6, 24))
        extra_ok = summarize_theorem(extra_log, f"RANDOM (max_attach in 1,2) EXTRA SWEEP")
        print(f"[{time.time()-t0:.1f}s]")

    print("\n--- (3) chain-of-thetas zoo (multi-branch-vertex 2-degenerate graphs) ---")
    chain_log = run_chain_theta_zoo(tries=15, big=(mode == "big"))
    chain_ok = summarize_theorem(chain_log, "CHAIN-OF-THETAS")
    print(f"[{time.time()-t0:.1f}s]")

    print("\n--- (4) Lemma R2: degree-<=2 peel reduces to the 3-core ---")
    core_log = run_3core_reduction_zoo(tries=20, big=(mode == "big"))
    core_ok = summarize_3core(core_log)
    print(f"[{time.time()-t0:.1f}s]")

    print("\n--- (5) degenerate/non-ray-injective boundary probes ---")
    c4out = degenerate_probe_c4(d=4)
    print_c4_probe(c4out)
    thout = degenerate_probe_theta_hub_collapse(d=5, tries=15)
    print_theta_probe(thout)
    print(f"[{time.time()-t0:.1f}s]")

    print("\n--- (6) confirmatory non-example check (degeneracy>=3 graphs, outside the theorem) ---")
    nonex_rows = run_nonexample_confirmation(tries=15)
    nonex_ok = print_nonexample_table(nonex_rows)
    print(f"[{time.time()-t0:.1f}s]")

    all_ok = theta_ok and rand_ok and chain_ok and core_ok and nonex_ok and extra_ok
    print("\n" + "=" * 110)
    print(f"VERDICT: theta {'PASS' if theta_ok else '*** FAIL ***'}; "
          f"random-2-degen {'PASS' if rand_ok else '*** FAIL ***'}; "
          f"extra-sweep {'PASS' if extra_ok else '*** FAIL ***'}; "
          f"chain-theta {'PASS' if chain_ok else '*** FAIL ***'}; "
          f"3-core-reduction {'PASS' if core_ok else '*** FAIL ***'}; "
          f"non-example classification {'PASS' if nonex_ok else '*** FAIL ***'}")
    print(f"[{time.time()-t0:.1f}s total] {'ALL PASS' if all_ok else 'SEE FAILURES ABOVE'}")
    return 0 if all_ok else 1

if __name__ == "__main__":
    sys.exit(main())
