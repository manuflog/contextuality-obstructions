#!/usr/bin/env python3
# line30_muses.py -- COMPLETE enumeration of ALL minimal uncolorable cores (MUSes over
# rays) of the line-stable hypergraph, MARCO-style, to decide RIGOROUSLY whether any
# critical core contains a mixed-degree (t-deforming) ray.
#
# Universe: only rays covered by triads can belong to an all-critical core (a ray in no
# triad of the induced subgraph never flips colorability: color the rest, set it to 0).
# MARCO loop with a SAT map over the 69 selector variables:
#   seed maximal (phases=True) -> test KS-colorability of the induced subgraph:
#     uncolorable -> shrink to a MUS (greedy deletion, SAT per step), block supersets;
#     colorable   -> grow to an MSS, block subsets (require a ray outside the MSS).
# Map UNSAT => every MUS has been listed.  Checkpointed (this box kills at ~45 s).
import json, os, sys, time
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

T0 = time.time()
HERE = os.path.dirname(os.path.abspath(__file__))
CK = os.path.join(HERE, "line30_muses.cache.json")

pool = json.load(open(os.path.join(HERE, "line30_stable_pool.cache.json")))
RAYS = [tuple(r) for r in pool["rays"]]
E_ALL = [tuple(e) for e in pool["pairs"]]
T_ALL = [tuple(t) for t in pool["triads"]]


def kind(s):
    a = sum(1 for x in s if x in ("1", "-1"))
    b = sum(1 for x in s if x in ("X", "-X", "Y", "-Y"))
    return "mixed" if (a and b) else "const"


UNIV = sorted({v for t in T_ALL for v in t})            # 69 triad-covered rays
IDX = {g: k for k, g in enumerate(UNIV)}                # global ray -> selector 0..68
NU = len(UNIV)
E_U = [(a, b) for a, b in E_ALL if a in IDX and b in IDX]

from pysat.solvers import Glucose3


def colorable(sel):
    """KS-colorability of the induced subgraph on {UNIV[k] : k in sel}."""
    ks = {UNIV[k] for k in sel}
    s = Glucose3()
    for i, j in E_U:
        if i in ks and j in ks:
            s.add_clause([-(i + 1), -(j + 1)])
    for t in T_ALL:
        if all(v in ks for v in t):
            s.add_clause([v + 1 for v in t])
    r = s.solve()
    s.delete()
    return r


def shrink(sel):
    sel = sorted(sel)
    for r in list(sel):
        trial = [v for v in sel if v != r]
        if not colorable(trial):
            sel = trial
    return sel


def grow(sel):
    sel = set(sel)
    for r in range(NU):
        if r not in sel and colorable(sel | {r}):
            sel.add(r)
    return sorted(sel)


st = json.load(open(CK)) if os.path.exists(CK) else dict(
    blocks=[], muses=[], msses=0, done=False)
if st.get("done"):
    print(f"[muses] already complete: {len(st['muses'])} MUSes")
else:
    mapS = Glucose3()
    for cl in st["blocks"]:
        mapS.add_clause(cl)
    mapS.set_phases([k + 1 for k in range(NU)])          # prefer maximal seeds
    while True:
        if not mapS.solve():
            st["done"] = True
            json.dump(st, open(CK, "w"))
            print(f"[muses] MAP UNSAT -- enumeration COMPLETE: {len(st['muses'])} MUSes, "
                  f"{st['msses']} MSSes  ({time.time()-T0:.1f}s)")
            break
        model = mapS.get_model()
        seed = [k for k in range(NU) if model[k] > 0]
        if not colorable(seed):
            M = shrink(seed)
            st["muses"].append([UNIV[k] for k in M])
            cl = [-(k + 1) for k in M]
            mapS.add_clause(cl)
            st["blocks"].append(cl)
        else:
            S = grow(seed)
            st["msses"] += 1
            cl = [(k + 1) for k in range(NU) if k not in S]
            assert cl, "the full universe is colorable?!"
            mapS.add_clause(cl)
            st["blocks"].append(cl)
        if time.time() - T0 > 32:
            json.dump(st, open(CK, "w"))
            print(f"[muses] checkpoint: {len(st['muses'])} MUSes, {st['msses']} MSSes so "
                  f"far -- re-run to continue  ({time.time()-T0:.1f}s)")
            sys.exit(0)

# ---- analysis ------------------------------------------------------------------------
muses = st["muses"]
sizes = sorted(len(m) for m in muses)
with_mixed = [m for m in muses if any(kind(RAYS[g]) == "mixed" for g in m)]
print(f"[muses] total MUSes: {len(muses)}; sizes min/max = "
      f"{(sizes[0], sizes[-1]) if sizes else None}")
print(f"[muses] MUSes containing a MIXED (t-deforming) ray: {len(with_mixed)}")
if with_mixed:
    m = with_mixed[0]
    print(f"[muses] example mixed MUS ({len(m)} rays): "
          f"{[RAYS[g] for g in m if kind(RAYS[g]) == 'mixed']} + const rays")
else:
    print("[muses] => EVERY minimal uncolorable core consists of t-CONSTANT rays only:")
    print("         the KS-uncolorable content of the 30-degree line NEVER deforms.")
