# CLAIMS — status of every major claim

Each headline claim, stated at the strength actually established, with its evidence type and any
**external dependency** (a result whose correctness this claim inherits). Status vocabulary:
**proved** (self-contained analytic proof), **proved+computed** (analytic proof + machine
corroboration), **computational** (established by exact/finite computation), **numerical** (finite
floating-point stress test), **hardware** (experimental, assumption-dependent), **structural
interpretation** (a reading the math supports but does not uniquely force), **open**.

| Claim | Status | Evidence | Where | External dependency |
|---|---|---|---|---|
| **Obstruction spectrum** `2S≡0 (mod d)`; value set `{0,d/2}` (even d), `{0}` (odd) | proved+computed | analytic (telescoping) + `spectrum_test2.py` + independent value-bit check | Paper B §2 | **priority**: compare vs Abramsky–Cercelescu–Constantin (FSCD 2024) — see `NOVELTY.md` |
| **Commutator-carry form** `b(C)=Σ⟨u,v⟩/d` | proved+computed | analytic + `criterion.py` | Paper B §3 | none |
| **Attainment** of `d/2` at every even d | proved (inherited) + computed | ROZF construction translated + `verify_cert8/16.py` + independent re-derivation | Paper B §6 | **ROZF, Quantum 7, 979 (2023)** — attainment generality is theirs |
| **Detection equivalence** (no NC assignment ⇔ kernel pairing d/2 ⇔ odd carry) | proved+computed | SNF double-annihilator + `solvability_equivalence.py` + brute-force d=2 | note Thm 3 / Paper B | none (composite-d SNF step should be written out in full — see limitations) |
| **Value-bit formula** (Thm W) | computational | `Wformula.py` 0/550 | Paper B §5 | analytic write-up still terse |
| **Q ≈ Pontryagin-square** | **algebraic-form only** | resemblance of formulas; cohomology operation **not** constructed | Paper B §7 | **open**: precise identification with the Pontryagin square / type-III anomaly |
| **Sharp local valuation** μ_d (even) / μ_{2d} (odd), sharp at d=3 | proved+computed | parity lemma + transgression + `mu_d_valuation.py` + independent check | note Thm 1 / V46 | none (general odd-d sharpness by analytic example, not only search) |
| **Canonical MASA-stack class, order exactly n; generates Z/n (all n, incl. n=4)** | **proved (algebra) / under review (geometry)** | order-n via equivariant-hom lemma + `thmG_general.py`; even-n generation closed in `paperA_foundations.tex` + `paperA_evenN.py` | Paper A | stack/cohomology layer still needs a specialist audit (Morita hypotheses; LHS edge maps in the smooth setting) |
| **H²(PN(T);U(1)) ≅ Z/n ⊕ Z₂** | proved (spectral sequence) / under review | LHS spectral sequence argument in Paper A §7 | Paper A | differentials/extension problems should be checked by a specialist |
| **d=4 facet census**: 61 classes / 23,256 facets | computational (two engines) | adjacency decomposition + Normaliz, class-for-class agreement | Paper C / V43 | reproduced by two engines *in this codebase*; not yet re-run by a third party |
| **Instrument-level Lüders** = affine depolarizing line, CP interval `−1/(r²−1)≤p≤1`, unique single-Kraus member | proved+computed | Schur + Choi bound + `lueders_instrument.py` (numerical) + `lueders_cp_interval.py` (exact rational) | note §collapse / V37 | none |
| **Sharp-core POVM** (repeatable⇒projective false, witness ≥3-dim) | proved+computed | support lemma + qutrit example (`s4_povm_bayes.py`) + independent check | V40 | none |
| **Classical twin** affine family `−1/(m−1)≤t≤1`; swap at m=2 | proved+computed | linear algebra + `s4_povm_bayes.py` | V40 | none |
| **Observer structure** = category with switch subgroupoid | proved (definitional) | Paper D §2 | Paper D | needs full category-theoretic architecture (double category / fibration choice) |
| **Movable cut** (ray-level invariance; section-phase bookkeeping; class invariance) | proved (3 levels) + interpretation | `observer_groupoid.py`, `fr_cycle.py` | Paper D §3 | the broad "cut may be moved without empirical consequence" is a **structural interpretation** |
| **Frauchiger–Renner localized** (trivial switch holonomy; CF=1/6 on one CHSH facet) | proved+computed | `fr_cycle.py` + independent LP (CF=1/6) | Paper D §4 | none |
| **Hardware PM witness** (per-device 32–61σ; combined 80σ **shot-noise-only**) | hardware, **assumption-dependent** | 3-device replication; `verify_combined.py` + `compatibility_analysis.py` | hardware/ | devices fail homogeneity (χ²≈266/2) ⇒ systematics-aware combined ≈**7σ**, not 80σ; nondisturbance controls not reported |
| **Copenhagen commitments are theorems** | **structural interpretation** | the five tenets map to the results above | note | the formalism *supports* a Copenhagen-compatible reading; it does not prove Copenhagen is the unique interpretation, nor solve the measurement problem |

See `KNOWN_LIMITATIONS.md` for the scope boundaries and `PROOF_DEPENDENCIES.md` for the dependency
graph. The plausibility scorecard (`verification/plausibility_scorecard.py`) is an internal
self-assessment; by a stricter publication-readiness rubric (independent replication + specialist
review still pending) the program sits lower, and closing that gap is the current priority.
