#!/usr/bin/env python3
# line30_color.py -- TASK 2: KS-colorability of the LINE-STABLE hypergraph (SAT +
# independent DPLL backtracker), then TASK 3 prologue: greedy peel to a critical core.
#
# KS coloring: assign {0,1} to rays; every orthogonal PAIR at most one 1; every TRIAD
# (complete basis) at least one 1 (with the pair clauses => exactly one).  UNSAT =
# KS-UNCOLORABLE.
#
#   stage color : SAT (Glucose3) + independent DPLL on the full stable graph.
#   stage peel  : greedy ray deletion while uncolorability persists (SAT-tested),
#                 checkpointed; then per-ray criticality verification + DPLL on the core.
#
# Cache: line30_core.cache.json.
import json, os, sys, time
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from itertools import combinations

T0 = time.time()
HERE = os.path.dirname(os.path.abspath(__file__))


def load(tag):
    p = os.path.join(HERE, f"line30_stable_{tag}.cache.json")
    return json.load(open(p)) if os.path.exists(p) else None


pool = load("pool")
assert pool, "run line30_stable.py pool first"
RAYS = [tuple(r) for r in pool["rays"]]
E_ALL = [tuple(e) for e in pool["pairs"]]
T_ALL = [tuple(t) for t in pool["triads"]]
N = len(RAYS)


def induced(keep):
    ks = set(keep)
    E = [(i, j) for i, j in E_ALL if i in ks and j in ks]
    T = [t for t in T_ALL if all(v in ks for v in t)]
    return E, T


def sat_colorable(E, T):
    from pysat.solvers import Glucose3
    s = Glucose3()
    for i, j in E:
        s.add_clause([-(i + 1), -(j + 1)])
    for t in T:
        s.add_clause([v + 1 for v in t])
    res = s.solve()
    s.delete()
    return res


def dpll_colorable(E, T):
    """Independent backtracker: DPLL with unit propagation on the same clause set."""
    clauses = [frozenset((-(i + 1), -(j + 1))) for i, j in E] + \
              [frozenset((v + 1 for v in t)) for t in T]
    varset = sorted({abs(l) for c in clauses for l in c})

    def solve(assign):
        # unit propagation
        while True:
            changed = False
            for c in clauses:
                vals = [assign.get(abs(l), None) if l > 0 else
                        (None if assign.get(abs(l), None) is None else not assign[abs(l)])
                        for l in c]
                # literal truth: l>0 true iff assign True; l<0 true iff assign False
                lits = list(c)
                truth = []
                for l in lits:
                    a = assign.get(abs(l))
                    truth.append(None if a is None else (a if l > 0 else (not a)))
                if any(v is True for v in truth):
                    continue
                un = [l for l, v in zip(lits, truth) if v is None]
                if not un:
                    return False
                if len(un) == 1:
                    l = un[0]
                    assign[abs(l)] = (l > 0)
                    changed = True
            if not changed:
                break
        for v in varset:
            if v not in assign:
                for val in (False, True):
                    a2 = dict(assign)
                    a2[v] = val
                    if solve(a2):
                        return True
                return False
        return True

    sys.setrecursionlimit(100000)
    return solve({})


def stage_color():
    E, T = E_ALL, T_ALL
    s = sat_colorable(E, T)
    print(f"[color] STABLE graph ({N} rays / {len(E)} pairs / {len(T)} triads): "
          f"SAT says {'COLORABLE' if s else 'UNCOLORABLE'}")
    d = dpll_colorable(E, T)
    print(f"[color] independent DPLL backtracker says "
          f"{'COLORABLE' if d else 'UNCOLORABLE'}  ({time.time()-T0:.1f}s)")
    assert s == d, "SAT and DPLL disagree!"
    json.dump(dict(colorable=bool(s)), open(os.path.join(HERE, "line30_color.cache.json"), "w"))


def stage_peel():
    cp = os.path.join(HERE, "line30_core.cache.json")
    st = json.load(open(cp)) if os.path.exists(cp) else None
    if st and st.get("complete"):
        print(f"[peel] already complete: core {len(st['core'])} rays")
        return
    if st:
        keep = st["keep"]
        pos = st["pos"]
        rounds = st["rounds"]
    else:
        keep = list(range(N))
        pos = 0
        rounds = 0
    E, T = induced(keep)
    assert not sat_colorable(E, T), "stable graph colorable -- nothing to peel"
    changed_any = st.get("changed_any", False) if st else False
    while True:
        if pos >= len(keep):
            rounds += 1
            if not changed_any:
                break
            changed_any = False
            pos = 0
        r = keep[pos]
        trial = [v for v in keep if v != r]
        E, T = induced(trial)
        if not sat_colorable(E, T):
            keep = trial          # still uncolorable without r: delete r
            changed_any = True
        else:
            pos += 1
        if time.time() - T0 > 32:
            json.dump(dict(keep=keep, pos=pos, rounds=rounds, changed_any=changed_any,
                           complete=False), open(cp, "w"))
            print(f"[peel] checkpoint: {len(keep)} rays, pos {pos} -- re-run to continue")
            return
    # criticality check: removing ANY remaining ray must make it colorable
    crit = True
    for r in keep:
        E, T = induced([v for v in keep if v != r])
        if not sat_colorable(E, T):
            crit = False
            print(f"[peel] WARNING: ray {r} still removable (greedy order artifact)")
    E, T = induced(keep)
    assert not sat_colorable(E, T)
    d = dpll_colorable(E, T)
    assert not d, "DPLL disagrees on core uncolorability!"
    core_syms = [RAYS[v] for v in keep]
    print(f"[peel] CRITICAL CORE: {len(keep)} rays / {len(E)} pairs / {len(T)} triads; "
          f"all-critical={crit}; SAT and DPLL agree UNCOLORABLE  ({time.time()-T0:.1f}s)")
    json.dump(dict(keep=keep, core=keep, core_syms=[list(s) for s in core_syms],
                   core_pairs=[list(e) for e in E], core_triads=[list(t) for t in T],
                   all_critical=bool(crit), uncolorable=True, complete=True),
              open(cp, "w"))


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "color"
    dict(color=stage_color, peel=stage_peel)[which]()
