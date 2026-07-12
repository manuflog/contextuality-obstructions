# KNOWN LIMITATIONS

Stated plainly, so no claim is read at more than its established strength.

## Mathematical scope
- **Paper A stack-cohomology layer is the largest open item.** `thmG_general.py` verifies the
  load-bearing *equivariant-homomorphism lemma*, not the full geometric theorem. The precise
  cohomology theory, the canonical-class construction, Morita invariance, the exact order-n argument
  (including exclusion of non-module/discontinuous splittings), and the `H²=Z/n⊕Z₂` computation need a
  self-contained proof appendix checkable by an equivariant-topology / stack-cohomology specialist
  **without** consulting the scripts. Until then, "order exactly n" is *proved at the algebraic level
  and under review at the geometric level*.
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
- The 80σ is **shot-noise statistical significance under the implemented witness model**, not a
  loophole-free certification. Compatibility/nondisturbance diagnostics (marginal consistency across
  contexts, order-reversal, repeatability, disturbance-corrected bound) and systematic-uncertainty
  accounting (calibration drift, readout/gate error, backend selection, compilation variability,
  multiple-testing) are **not yet reported**. Inverse-variance combination assumes independent
  statistical errors and does not remove common systematics.

## Process
- **No external specialist review yet.** Internal consistency and independent (same-author)
  reimplementation are not a substitute for domain-expert scrutiny in (i) group cohomology / stacks,
  (ii) contextuality, (iii) quantum foundations, (iv) quantum-information experiment. This is the
  single largest remaining step to publication-grade confidence.
- **Novelty positioning is not yet audited** against the full literature; presenting a known result as
  new is a reputational risk that a specialist review should close (see `CLAIMS.md`).

## On the score
No score of 100 is claimed. A healthy research program stays falsifiable; errors found move the score
down (the ledger in `verification/INDEX.md` records this, including the corrections of 2026-07-11).
