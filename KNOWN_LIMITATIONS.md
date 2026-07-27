# KNOWN LIMITATIONS

Stated plainly, so no claim is read at more than its established strength.

## Mathematical scope
- **Paper A stack-cohomology layer still needs a specialist proof audit.** The self-contained
  appendix (`papers/paperA_foundations.tex`) now covers the objects, vacuity, order-n, Morita, the
  H²=Z/n⊕Z₂ computation, and (new) the even-n generation (§"Resolution of the even-n generation gap").
  What remains for an equivariant-topology / stack-cohomology specialist to confirm: the Morita
  invariance hypotheses, the LHS-spectral-sequence differentials/edge maps in the smooth setting, and
  exclusion of non-module/discontinuous splittings. "Order exactly n" and "generates Z/n" are now
  argued in full; they await that external rubber-stamp.
- **Pontryagin-square / type-III anomaly identification is not constructed.** The carry invariant `Q`
  has the *algebraic form* of a Pontryagin-square quadratic refinement; the cochain complex, coefficient
  sequence, cohomology operation, and the map from Weyl data are not built here. Language has been
  hedged accordingly.
- **Detection-equivalence composite-d step** (SNF double-annihilator over composite d) is proved in
  outline and corroborated computationally; the fully general write-up with all closure/commutation
  hypotheses is still owed.
- **Local valuation** odd-d sharpness is currently shown by search + the d=3 witness; a general
  analytic sharpness statement is preferable.
- **d=4 facet census** is reproduced by two engines *within this repository*; it has not been re-run
  by an independent third party.
- **Finite-dimensional scope.** Results are for finite Weyl/Heisenberg systems and finite-dimensional
  Hilbert spaces; no claim is made in infinite dimensions.
- **Several generalizations remain open** (labelled in `verification/INDEX.md`): general-d census, the
  Θ-coherence congruence, the tower doubling-law generativity closed form, the KCBS cohomological
  home, the observer-category architecture (double category vs fibration).

## Observer / interpretation
- The observer object is a **category** (non-invertible refinement isometries), not a groupoid;
  holonomy is defined only on the invertible switch subgroupoid.
- The **movable-cut** result is exactly: probability-level invariance + section-phase bookkeeping +
  cohomology-class invariance. The broader "the cut may be moved without empirical consequence" is an
  interpretive reading it *supports*, not a stronger theorem.
- **No interpretation-uniqueness.** The formalism supports a Copenhagen-compatible reading; it does
  **not** prove Copenhagen is the unique interpretation, does not derive that collapse/outcomes occur,
  and does **not** solve the measurement / definite-outcome problem. Rival interpretations can use the
  same formal results.

## Hardware
- The 80σ is **shot-noise-only and overstates the combined significance**. The three devices FAIL a
  homogeneity test (χ²≈266 on 2 d.o.f., Birge ratio ≈11.5): the chip-to-chip scatter is ~20× the
  combined shot-noise SE, so a systematics-aware (Birge-scaled) combination gives **≈7σ, not 80σ**.
  Each device individually is 32–61σ above the bound (shot noise), so the qualitative violation is
  robust, but the combined 80σ is not a legitimate single-S significance (`hardware/COMPATIBILITY.md`).
  It is **not** a loophole-free certification. Compatibility/nondisturbance diagnostics (marginal consistency across
  contexts, order-reversal, repeatability, disturbance-corrected bound) and systematic-uncertainty
  accounting (calibration drift, readout/gate error, backend selection, compilation variability,
  multiple-testing) are **not yet reported**. Inverse-variance combination assumes independent
  statistical errors and does not remove common systematics.

## Process
- **No external specialist review yet.** Internal consistency and independent (same-author)
  reimplementation are not a substitute for domain-expert scrutiny in (i) group cohomology / stacks,
  (ii) contextuality, (iii) quantum foundations, (iv) quantum-information experiment. This is the
  single largest remaining step to publication-grade confidence.
- **Novelty / priority concern (must resolve before posting the flagship).** A directly on-topic,
  currently *uncited* paper — Abramsky, Cercelescu & Constantin, *Commutation Groups and
  State-Independent Contextuality*, FSCD 2024 (arXiv:2603.12197) — studies state-independent
  contextuality over Z_d via commutation groups and "contextual words," i.e. the flagship's exact
  topic. A precise comparison of their parity/value results with the spectrum {0,d/2} is **required**;
  until done, the novelty of the obstruction-spectrum result is not established. See `NOVELTY.md`.
- **Other novelty positioning** is not yet audited against the full literature; presenting a known
  result as new is a reputational risk that specialist review should close (see `NOVELTY.md`).

## On the score
No score of 100 is claimed. A healthy research program stays falsifiable; errors found move the score
down (the ledger in `verification/INDEX.md` records this, including the corrections of 2026-07-11).

## Consolidated proof appendices — honest gap register (2026-07-12)
The new `papers/paperA_foundations.tex`, `papers/obstruction_spectrum_master.tex`, and
`papers/observer_category.tex` state, per step, whether each is a complete proof, a cited sketch, or a
gap. The gaps they flag:
- **Paper A (order n at even n) — ✅ CLOSED (2026-07-12).** The canonical class generates the Z/n
  summand of H²(PN(T);U(1)) for **every n, including n=4 (Peres–Mermin)**. Argument (in
  `papers/paperA_foundations.tex` §"Resolution of the even-n generation gap", checked in
  `verification/paperA_evenN.py`): S_n lifts to honest permutation matrices (P_σ P_τ = P_{στ} exactly),
  so the extension splits over S_n and res_{S_n}(c)=0, i.e. c's Schur (Z_2) component vanishes at every
  n; together with order n this forces c into the standard Z/n summand as a generator. One textbook
  step remains for a specialist to rubber-stamp: that res_{S_n} is the Z_2-projection (retracts the
  inflation) and kills the E∞^{1,1}=Z/n part — the standard LHS edge behaviour for a split extension.
- **Paper B attainment.** A self-contained parametric construction proves attainment for all
  d≡2 (mod 4); the d≡0 (mod 4) generality is **inherited from ROZF** and explicit certs cover d=8,16.
  The exact value-bit carry coefficients are corroborated computationally, not yet hand-certified.
- **Paper D observer category.** The fiberwise U(1) cocycle is complete; assembling it into a
  **single class on the whole category** (Baues–Wirsching/Thomason H² / a U(1)-gerbe over the
  enlargement category) is open. Holonomy and the movable cut need only the fiberwise section.

## Harness failure, found and fixed 2026-07-27 (recorded, not quietly patched)

For roughly one day the verification suite reported green on nine scripts that never ran.

**What happened.** A commit whose message was "harden the suite" appended a bare module-level
`sys.exit(0)` to `verification/criterion.py`. Python executes a module's entire body on import,
so every script doing `import criterion` died *during the import*: it executed none of its own
checks, printed `criterion PASS` (criterion's own token, emitted before the exit), and returned
status 0. Affected, directly or through `evend_frame_probe`:

`holonomy_vs_solvability.py`, `d3_gap_certificate.py`, `evend_frame_probe.py`,
`pauli_slice_bridge.py`, `ghost_facet_theorem.py`, `paired_frame_construction.py`,
`ghw_net_necessity.py`, `gf4_net_necessity.py`, `net_robust_negativity.py`.

Between them these carry the Paper C equivalence stress test, the d=3 gap certificate, the ghost
facet theorem, Lemma 1's net necessity, and the 2^15 frame sweep.

**Why neither existing gate caught it.** `run_all.sh` checks exit code *and* a verdict token. The
exit code was 0 and the token was present — it just belonged to a different script. A gate that
greps for a string cannot tell which process printed it.

**What the failure did NOT do.** It did not make any published claim wrong. All nine scripts were
re-run after the fix and every quoted number reproduced exactly (1500/600 families, 0 mismatches;
0 coboundary hits over 19683 cochains; 24 = 20 + 4 facets; 40/40; rank 10 over F2; 1024 nets).
The mathematics was sound; the harness was lying about having checked it.

**Fix.** The bare exits are removed (falling off the end of a script already exits 0), and
`verification/check_import_safety.py` now runs FIRST in `run_all.sh`. It AST-parses every script,
works out which modules are imported by another, and fails on any exit-like effect reachable at
IMPORT time in one of them: `sys.exit` / `exit` / `quit` / `os._exit` / `raise SystemExit`,
including behind an `if`, inside a `try`, via an alias (`import sys as s`), or through a
module-level call into a module-level helper. Exits inside a real `if __name__ == '__main__':`
guard are allowed, and the guard is resolved structurally.

**That last point is itself a corrected error.** The first version of this gate, written the same
day, tested for the guard with `"__main__" in src` — a bare substring test that skipped the scan
entirely for any file merely containing the string. Seven of the thirteen imported modules were
exempted by accident, including `arf_global`, which sits one link further up the same import chain
as `criterion`. A gate with a hole in exactly the place it was written to cover is worse than no
gate, because it is trusted. Ten evasion cases are now checked explicitly.

**Token bleed — the underlying surface — is now closed too.** The import-safety gate stops a
module from *killing* its importer, but the deeper problem was that a module could *speak for*
its importer: `criterion.py` printed `criterion PASS` into the stdout of its four importers, and
`evend_frame_probe`, `state_sector_probe` and `d4_odd_sector_facets` did the same to theirs. That
is what let nine dead scripts satisfy a token gate. Every importer now wraps the noisy import in
`contextlib.redirect_stdout`, so each script's stdout contains only its own verdict. Verified:
`d3_gap_certificate` went from emitting `criterion PASS` to three clean lines of its own, and no
canonical script emits another's token. 21/21 green.

The principle worth keeping: **a verdict is only evidence if it is attributable to the thing being
judged.** Both silent-failure incidents in this program came from violating it — once by a script
that printed no verdict of its own, once by a script that printed someone else's.

**The general lesson, which is the reason this section exists.** Two of the three silent-failure
incidents in this program were introduced by the commit that was *supposed* to eliminate silent
failures. Hardening is itself a change, and changes need the gate applied to them. A verdict
token is only evidence if it is attributable to the process being judged.
