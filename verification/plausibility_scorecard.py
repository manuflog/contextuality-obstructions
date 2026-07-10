# Auditable plausibility score for 'Local Validity and Contextual Holonomy' (Feb 2026 note).
# Rerun after each iteration; evidence column cites suite scripts. Judgment made explicit.
ROWS = [
 # (component, weight, as_written, current, evidence / what moves it next)
 ("S2 local classicality (MASA/Spec)",        15, 13, 13, "standard; V23 pending for full"),
 ("S3 groupoid + cocycle core",               35,  8, 27, "as-written: wrong groupoid (PaperA Thm1) + "
   "'encodes' false in strong form (V13,V25). patched: PU(n) fix + Thm Q/J weak claim, V10-V13. "
   "V24 DONE: unsolvability<=>odd-cycle, 2100 random families + pinned PM/cert4, 0 mismatches. remaining 8: true equivalence thm beyond AvN (open)"),
 ("S4 collapse / Lueders",                    25, 15, 20, "V22 pins axioms (i)-(iii), efficiency shown "
   "necessary. next: instrument-level generality, Leifer-Spekkens embedding -> 23"),
 ("operational consistency (no new preds)",   10, 10, 10, "by construction"),
 ("Copenhagen adequacy (observer/WF)",         15,  6,  9, "V12 HW-CONFIRMED (ibm_fez): loop +1, -i excluded ~56sig at ray level; PM 35.4sig over tight bound. "
   "note must pin section. ceiling ~11 without the observer-context groupoid paper"),
]
def show(tag, idx):
    tot=sum(r[idx] for r in ROWS); W=sum(r[1] for r in ROWS)
    print(f"\n{tag}: {tot}/{W}")
    for r in ROWS: print(f"  {r[0]:<42} {r[idx]:>2}/{r[1]}")
    return tot
a=show("AS WRITTEN (Feb 2026 note)",2); b=show("CURRENT (with this session's patches+certs)",3)
print(f"\nscore: {a} -> {b}   (+/-5 judgment band; ceiling w/o open problems ~80)")
