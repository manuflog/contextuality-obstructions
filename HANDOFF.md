# HANDOFF — Contextuality & Contextual Holonomy Program
**Date:** July 10, 2026 · **State:** plausibility score **99/100** · **Prepared for continuation in Claude Cowork**

## 1. What this is
A multi-paper research program formalizing the thesis of Manu's Feb 2026 note
*"Local Validity and Contextual Holonomy: A Structural Formulation of the Copenhagen
Principle."* The thesis: the Copenhagen commitments are structural theorems of the
context groupoid PU(n)⋉X with its canonical U(1) 2-cocycle. The program's method is
adversarial: every claim is a theorem in a companion paper, a machine-verified
certificate in the suite, or an explicitly labeled conjecture. A running, auditable
plausibility scorecard (`verification/plausibility_scorecard.py`) tracks the note's
standing: **52 (as written) → 99 (current)**. The final point is reserved for Manu's
external adversarial pass — do not self-award it.

## 2. Repos and workflow
- `manuflog/contextuality-obstructions` — papers + verification suite (source of truth).
- `manuflog/qkernel` — the kernel/hardware package (holonomy module `holonomy.py`;
  the Peres–Mermin noncontextuality bound was patched to the exact deterministic
  bound 4).
- Manu works via **Google Colab + GitHub Desktop** (no command-line git). Handoffs are
  **complete whole-repo replacement zips**, always pre-flighted from a clean unzip
  (two past upload accidents motivated this rule). Claude-prepared commits are stamped
  *"pending Manu's adversarial review."*

## 3. The papers
| File | Title | Status |
|---|---|---|
| `papers/paperA.pdf` | The Canonical Class of the MASA Context Stack and the Peres–Mermin Obstruction | working draft (Thm 1 vacuity; order-n class; B_n split iff n odd; PM pairing −1) |
| `papers/paperB.pdf` | The Obstruction Spectrum of Weyl Context Families and the 2-adic Tower | working draft (Thm 1 spectrum {0,d/2}, any multiplicity — Rem. 3; Thm 4 carry criterion; Thm 7 attainment) |
| `papers/paperC.tex/pdf` | What Contextual Holonomy Detects | drafted; carries d=2 classification, d=4 tower laws, τ-necessity, tier-2 retraction ledger, complete census paragraph |
| `papers/paperD.tex/pdf` | The Observer-Context Groupoid (v0.3) | 4 theorems (relative facts; hol² Z₂ invariant; movable cut; FR localized) — **awaiting Manu's adversarial pass** |
| `papers/note_v3_local_validity.tex/pdf` | The original note, v3 full rewrite | complete; self-passed (6 issues found & fixed); Thm 3 upgraded to a proven equivalence (V44) — **awaiting Manu's adversarial pass** |

The old CMP submission ("Contextual Holonomy, Lifting Bundle Gerbes, …", v2_FINAL,
March) is a separate track with its own standing instruction (see §6).

## 4. Theorem ledger (headline results, with certificates)
- **Local classicality** (S2): every context's restricted cocycle is a coboundary,
  explicit λ ∈ μ_{4d}, half-τ steps necessary — `local_classicality.py` (V23).
- **Canonical class**: vacuous on U(n)⋉X; order exactly n on PU(n)⋉X — paper A.
- **Detection equivalence** (NEW, this session): no NC assignment ⟺ kernel vector
  pairing d/2 ⟺ odd commutator carry; proof = constructive SNF double-annihilator +
  paper B Thm 1 + Thm 4; includes the **gauge-integrality lemma** —
  `solvability_equivalence.py` (V44). Gap certificates both directions:
  `d3_gap_certificate.py`, `kcbs_converse.py`.
- **d=2 classification**: CF = max(0,(S*−1)/2); CF=0 ⟺ c(ρ) ∈ conv L(F) —
  V31/V32 (completion lemma + ABM duality).
- **d=4 complete facet census**: **61 classes, 23,256 facets**, derived twice
  (adjacency decomposition with exact-integer certified pivots; Normaliz direct),
  class-for-class agreement; **on-hull lemma proven** (24 spectral-support identities,
  exact rank 47 → affine dim 33) ⇒ **CF(ρ)>0 ⟺ a facet is violated**, unconditional —
  `d4_facet_census.py` (V43), data `d4_facet_classes.npz` (+ `d4_group_seed.pkl`).
- **Tower laws**: ghost law γ−s*≡d/2; τ-necessity — V33/V34.
- **Collapse**: instrument-level Lüders (Schur proof, `note_v2_patch.md`; V37);
  sharp core + "repeatable⇒projective" falsified (diag(1,c)); classical Bayes twin —
  V40.
- **Observer sector** (paper D): relative facts (PM⊗I at d=8 keeps signs); hol² Z₂
  invariant (MUB −1); movable cut; **FR localized** (trivial switch holonomy — the
  paper's own initial conjecture self-refuted; CF=1/6 exact on one CHSH facet at 7/3) —
  V38/V39.
- **Hardware** (ibm_fez, job d986q62f47jc73a7hm2g): loop −2.93°±1.55° (ray +1, −i
  excluded ~56σ); PM S=4.6125±0.0173 (35.4σ over bound 4) —
  `hardware/results/ibm_fez_holonomy_20260709.json`.

## 5. The verification suite
`verification/` — ~32 V-numbered scripts; **`INDEX.md` is the source of truth** (every
claim → script → pinned one-line output). Run any script standalone
(`python3 <script>.py`); expect its pinned PASS line. Python deps: numpy, scipy,
sympy, pycddlib (pip, `--break-system-packages` on Ubuntu). The census artifacts ship
in-repo, so **no heavy compute is required to verify anything** — reruns of the
*construction* (not needed) would want: pycddlib, lrslib (warning: **lrs wedges
reproducibly** on several tight-set polytopes — pinned), Normaliz 3.10 (old input
format: `nrows ncols / matrix / polytope`, run `normaliz -c -s -x=4 file` **without**
the `.in` extension).

## 6. Standing instructions (culture — do not relax)
1. **Adversarial honesty**: failures are findings and get pinned in INDEX.md;
   retractions are recorded (see tier-2 completeness retraction; the 121→61 key-type
   duplication caught only by the independent cross-check — "the cross-check is not
   optional").
2. **CMP paper rule**: Claude has twice prematurely declared the CMP paper ready
   (false coboundary ω=δλ; v3 "CMP-ready" with five uncaught errors). **Never declare
   that paper ready without a full adversarial self-read.** The same discipline now
   applies to note v3 and paper D: readiness is *ready-for-Manu's-review*, never
   submission-ready by Claude's word alone.
3. **Scorecard rules**: moves only for genuine theorems; external review owns the last
   point; errors found move the score down and that's the system working.
4. **Packaging**: whole-repo replacement zips only, pre-flighted from a clean unzip.

## 7. What remains (the road past 99, and beyond)
**The last point:** Manu's adversarial pass on **paper D** and **note v3**. Named
attack surfaces for note v3: (i) the Theorem 3 equivalence proof (check the
double-annihilator step and paper B Rem. 3's multiplicity scope); (ii) whether V23's
symmetric-bicharacter argument satisfies as a general proof; (iii) Principle (iii)
wording.

**Open mathematics** (in rough value order):
- Conjecture 1 (scenario-paired classification, even d) — global-frame versions
  refuted V28–V30.
- Conjecture 2 (tower: holonomies ∈ μ_{2d}).
- General-d facet census; arithmetic profile of the 61 classes (33-coord integer
  normals vs 150-dim mixed grid ¼ℤ ∪ 1/(2√2)ℤ — Tsirelson's √2 appears in tier-2).
- τ-general (non-integral-gauge-free) statement of the V44 equivalence — currently a
  proof-sketch remark; the gauge lemma reduces it, write it out.
- Cohomological home of KCBS-type (class-invisible) contextuality — paper C's open
  problem.
- V14–V21 roadmap items (unstarted; see earlier INDEX plans).
- CMP paper repair track (separate; approved repair plan on file).

## 8. Gotchas ledger (hard-won)
- **weyl.py τ-convention**: PM contexts are τ-odd under it — integrality is a *gauge
  choice* (V44's lemma). Never filter triples by integrality; gauge instead.
- **lrs wedges** on these polytopes (identical partial output under different budgets);
  cdd is insertion-order sensitive (one 60-point polytope defeated all shuffles);
  **Normaliz is the reliable engine** here.
- Class dictionaries: one key encoding only (the 121-vs-61 lesson).
- `/tmp` does not persist across environments; everything needed ships in-repo
  (`d4_facet_classes.npz`, `d4_group_seed.pkl`).
- Colab: `pip install --break-system-packages` not needed there; plain pip.

## 9. First moves in Cowork
1. Unzip both repos (replace working trees), open `verification/INDEX.md`.
2. `python3 verification/plausibility_scorecard.py` → expect 99.
3. `python3 verification/d4_facet_census.py` → expect V43 PASS (few minutes).
4. Read `papers/note_v3_local_validity.pdf` and `papers/paperD.pdf` adversarially —
   that pass is the last point, and anything found goes to INDEX.md first.
