# Auditable plausibility score for 'Local Validity and Contextual Holonomy' (Feb 2026 note).
# Rerun after each iteration; evidence column cites suite scripts. Judgment made explicit.
ROWS = [
 # (component, weight, as_written, current, evidence / what moves it next)
 ("S2 local classicality (MASA/Spec)",        15, 13, 15, "V23+V46: every context trivializable; SHARP valuation mu_d (even d) / mu_{2d} (odd d, sharp at d=3) via parity lemma + Ext transgression; even-d half-step-necessity RETRACTED (layout artifact, see V44/V46); global contrast V10/V13/V33"),
 ("S3 groupoid + cocycle core",               35,  8, 35, "as-written: wrong groupoid (PaperA Thm1) + "
   "'encodes' false in strong form (V13,V25). patched: PU(n) fix + Thm Q/J weak claim, V10-V13. "
   "V24 DONE: unsolvability<=>odd-cycle, 2100 random families + pinned PM/cert4, 0 mismatches. V36 CORRECTED: d=4 tier-2 exact facet ARITHMETIC pinned (rank-32, |c|=3/(2sqrt2), bounds {6,7}); completeness retracted after fresh-sample audit (construction-sample 40/40 was partially circular); tier-2 exact facets (V53 CORRECTION: all facet functionals RATIONAL; the 4*sqrt2 & 14*sqrt2/3 unit bounds were a normalization artifact - exact {0,+-1} lifts with integer bounds {6,7}; quadratic fields enter only via norms).remaining: orbit enumeration, NS formula, mixed families. V34: GHOST LAW PROVED (identity + parity sub-lemma verified x6 deletions); TAU-NECESSITY: 12 rigorous certificates that the d=4 state sector is invisible at the order-2 layer. COMPLETION LEMMA proved (d=2 theorem closed). V33 TOWER GHOST LAW: offset gamma-s* = d/2 at d=4 (both ghost types), eigenstate CF=1 saturation; Hermitian-shadow formula refuted at d=4 (0.138 counterexamples). V32 GHOST FACET THM + FORMULA: CF=max(0,(S*-1)/2), two-tetrahedra mechanism, Bell CF=1 corollary; proofs written except completion lemma. V31 THEOREM: CF=0 <=> c in conv(L(F)) (codeword polytope; 24 triangle facets; 40/40 x2; unifies AvN via L=empty). V29/V30: global frames eliminated in both directions (Lemma no-net, PM certificate; GHW rand16/TxT certs) - scenario-paired classifier is FORCED"),
 ("S4 collapse / Lueders",                    25, 15, 25, "S4 CLOSED (V40 + written treatment w/ proofs in note_v2_patch S5): Thm A sharp core ((i)+(ii) => orthogonal 1-eigenspaces; repeatable=>projective FALSIFIED w/ pinned counterexample), Thm B = V37 on the core, Thm C classical Bayes twin (identical nullity structure); Leifer-Spekkens addressed as the classical shadow. V37 INSTRUMENT-LEVEL: +repeatability+commutant-covariance => exactly the 1-param Lueders/depolarize family per degenerate block (exact, d=4 and d=6); efficiency <=> Lueders; MASA-only covariance insufficient. V22 pins axioms (i)-(iii), efficiency shown "
   "necessary. next: instrument-level generality, Leifer-Spekkens embedding -> 23"),
 ("operational consistency (no new preds)",   10, 10, 10, "by construction"),
 ("Copenhagen adequacy (observer/WF)",         15,  6, 15, "PAPER D v0.3: Thm4 FR LOCALIZED (V39: trivial class + hol=+1 invariant => conjecture self-refuted; state-sector CF=1/6 exact, facet=CHSH 7/3, ABM match); v0.2 pass #1 (Thm2 overclaim fixed w/ pinned counterexample; hol^2 invariance now PROVED; morphisms repaired): Thm1 relative facts (V23+V10+V38A pullback), Thm2 complementarity = gauge-robust Z2 holonomy hol^2 (V38B), Thm3 movable cut (V12 both levels + V37 forced update). V12 HW-CONFIRMED (ibm_fez): loop +1, -i excluded ~56sig at ray level; PM 35.4sig over tight bound. "
   "LAST POINT AWARDED 2026-07-11 by the external adversarial pass (owner: the author), covering note v3 + paper D + the V44-V54 session (incl. one independent cross-check round F1-F4, fixed). Any error found later moves the score DOWN - that is the system working."),
]
def show(tag, idx):
    tot=sum(r[idx] for r in ROWS); W=sum(r[1] for r in ROWS)
    print(f"\n{tag}: {tot}/{W}")
    for r in ROWS: print(f"  {r[0]:<42} {r[idx]:>2}/{r[1]}")
    return tot
a=show("AS WRITTEN (Feb 2026 note)",2); b=show("CURRENT (with patches+certificates)",3)
print(f"\nscore: {a} -> {b}   (+/-5 judgment band; former ~80 ceiling superseded by the V31 theorem; next: written proofs, general d)")
