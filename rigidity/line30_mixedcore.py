#!/usr/bin/env python3
# line30_mixedcore.py -- the crux question behind the verdict: does ANY critical core of
# the line-stable graph actually DEFORM along t?
#
# Facts so far: the greedy core (43 rays) contains only t-homogeneous rays (all entries
# t-degree 0, or all t-degree 1), which are PROJECTIVELY CONSTANT in t, so v_t = 0 there.
# Mixed rays (both degrees present) are the only rays that move projectively.
#
#   (a) the t-constant subpool (all homogeneous rays): colorable or not?  If uncolorable,
#       the KS-uncolorable content of the line exists t-INDEPENDENTLY.
#   (b) the mixed-only-protecting peel: delete homogeneous rays first (greedy), keep
#       mixed rays; does an all-critical core CONTAINING mixed rays exist?
#   (c) the complement test: delete ALL mixed rays -> colorable?  (= are mixed rays
#       jointly necessary?)  and delete ALL homogeneous triad rays -> colorable?
import json, os, sys, time
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

T0 = time.time()
HERE = os.path.dirname(os.path.abspath(__file__))
pool = json.load(open(os.path.join(HERE, "line30_stable_pool.cache.json")))
RAYS = [tuple(r) for r in pool["rays"]]
E_ALL = [tuple(e) for e in pool["pairs"]]
T_ALL = [tuple(t) for t in pool["triads"]]
N = len(RAYS)


def kind(s):
    a = sum(1 for x in s if x in ("1", "-1"))
    b = sum(1 for x in s if x in ("X", "-X", "Y", "-Y"))
    return "mixed" if (a and b) else "const"


KINDS = [kind(r) for r in RAYS]


def induced(keep):
    ks = set(keep)
    E = [(i, j) for i, j in E_ALL if i in ks and j in ks]
    T = [t for t in T_ALL if all(v in ks for v in t)]
    return E, T


def colorable(keep):
    from pysat.solvers import Glucose3
    E, T = induced(keep)
    s = Glucose3()
    for i, j in E:
        s.add_clause([-(i + 1), -(j + 1)])
    for t in T:
        s.add_clause([v + 1 for v in t])
    r = s.solve()
    s.delete()
    return r


# (a) t-constant subpool
const_rays = [i for i in range(N) if KINDS[i] == "const"]
ca = colorable(const_rays)
E, T = induced(const_rays)
print(f"[a] t-CONSTANT subpool: {len(const_rays)} rays / {len(E)} pairs / {len(T)} triads "
      f"-> {'COLORABLE' if ca else 'UNCOLORABLE'}")

# (c) delete all mixed rays is (a); now delete all homogeneous-degree-1 rays but keep units+mixed:
unit_mixed = [i for i in range(N)
              if KINDS[i] == "mixed" or all(s in ("0", "1", "-1") for s in RAYS[i])]
cb = colorable(unit_mixed)
E, T = induced(unit_mixed)
print(f"[c] units+mixed only ({len(unit_mixed)} rays / {len(E)} pairs / {len(T)} triads) "
      f"-> {'COLORABLE' if cb else 'UNCOLORABLE'}")

# (b) mixed-protecting greedy peel: try to delete homogeneous rays first, mixed last
keep = list(range(N))
order = sorted(range(N), key=lambda i: (KINDS[i] == "mixed",))  # const first
assert not colorable(keep)
changed = True
while changed:
    changed = False
    for r in order:
        if r not in keep:
            continue
        trial = [v for v in keep if v != r]
        if not colorable(trial):
            keep = trial
            changed = True
mixed_in_core = [r for r in keep if KINDS[r] == "mixed"]
E, T = induced(keep)
print(f"[b] mixed-protecting peel core: {len(keep)} rays / {len(E)} pairs / {len(T)} triads; "
      f"mixed rays in core: {len(mixed_in_core)}")
crit = all(colorable([v for v in keep if v != r]) for r in keep)
print(f"[b] all-critical: {crit}")
if mixed_in_core:
    print("    mixed core rays:", [RAYS[r] for r in mixed_in_core])
json.dump(dict(const_subpool_colorable=bool(ca), units_mixed_colorable=bool(cb),
               protect_core=keep, protect_core_mixed=mixed_in_core,
               protect_core_syms=[list(RAYS[r]) for r in keep],
               all_critical=bool(crit)),
          open(os.path.join(HERE, "line30_mixedcore.cache.json"), "w"))
print(f"done ({time.time()-T0:.1f}s)")
