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


## Verification-suite defects found and fixed, 2026-07-27/28

Recorded because the suite's value depends on its failures being visible.

**1. Import-time exits disabled nine scripts.** A bare module-level `sys.exit(0)` in
`criterion.py` terminated every script that imported it, directly or through
`evend_frame_probe`: `holonomy_vs_solvability`, `d3_gap_certificate`, `evend_frame_probe`,
`pauli_slice_bridge`, `ghost_facet_theorem`, `paired_frame_construction`, `ghw_net_necessity`,
`gf4_net_necessity`, `net_robust_negativity`. Each exited 0 having run none of its own checks.
No published claim was affected: all nine were re-run after the fix and every quoted number
reproduces. Fixed, and `check_import_safety.py` now runs first in `run_all.sh` and rejects
import-time exits in any imported module.

**2. Token bleed.** Four modules printed at import time, so an importer's stdout could carry
another script's verdict token — which is how the failure above stayed invisible to a gate that
greps for tokens. Every importer now suppresses import-time output, so each script's stdout
carries only its own verdict.

**3. Tier-2 arithmetic.** The header of `d4_tier2_catalogue.py` asserted that tier-2 facet
normals live on a `Z[1/sqrt2]` grid with coefficient magnitude `3/(2 sqrt2)`, and that
"Tsirelson's sqrt2 sits in the bound". That presentation was a normalization artifact and is
withdrawn: the stored lift has coefficients in `{0,+-1}` with integer bounds `{6,7}`, every facet
functional in the census is rational, and `Q(sqrt2)` is not among the eighteen canonical fields.
Certificate: `d4_arithmetic_profile.py`. The retraction is carried in Paper C.

**4. Paper B Prop. 9.** See the correction note in `verification/INDEX.md`. The claimed
false-certificate rate of 28.6% is wrong; the true rate is 0, provably, at every even d. The
criterion is incomplete in the opposite direction. Corrected in Paper B v2.

**5. Bibliography.** A self-citation in Paper 2 named a venue and identifier that do not exist.
Every self-citation across all papers was then audited against what is actually published, and
stale "in preparation" references, missing identifiers, and version-instead-of-concept DOIs were
corrected. Verification scripts do not check bibliographies; this class of defect needs a reading
pass, and now gets one before any deposit.
