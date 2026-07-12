# NOVELTY — claim-by-claim assessment vs. prior art

Honest novelty matrix for the seven headline results of the program. The goal here is to find
**overlap**, not to inflate novelty: where a result looks likely-known, it is marked
`medium`/`uncertain` with the reason. Confidence = confidence that the *stated* result is genuinely
new relative to the closest prior art found by literature search (July 2026), **not** confidence
that the math is correct.

All prior-art URLs are collected in **Sources** at the end. Dependencies the papers already
acknowledge (ORBR, ROZF, Gross, Lisoněk–Raussendorf–Singh, de Boutray–Holweck–Saniga, ABCP,
Kim–Abramsky) are noted; the single most important prior work the papers do **not** currently cite
is **Abramsky–Cercelescu–Constantin, *Commutation Groups and State-Independent Contextuality*,
FSCD 2024** (arXiv:2603.12197), which directly anticipates the two flagship Paper-B claims.

---

## Summary matrix

| # | Headline claim | Closest prior art | Novelty confidence |
|---|---|---|---|
| 1 | Obstruction spectrum / value grading `{0,d/2}` (even d), `{0}` (odd); `2S≡0 (mod d)` | Abramsky–Cercelescu–Constantin, FSCD 2024, Thm 16 & 18; Gross; ROZF | **uncertain (high overlap)** |
| 2 | Commutator-carry closed-form criterion `b(C)=Σ⟨u,v⟩/d`; odd-Q cycle test | Abramsky–Cercelescu–Constantin Thm 19–20; Saniga–Holweck–de Boutray | **medium** |
| 3 | Exact-order-`n` canonical MASA-stack class on `[X/PU(n)]` | Standard `H²(PU(n))≅Z/n` central-extension / Weil-rep obstruction; ORBR; Wirthmüller | **medium (geometry) / uncertain (order-n core)** |
| 4 | Observer-context category + U(1) cocycle; relative facts | Frembs–Döring context category; Rovelli/Di Biagio relational "relative facts" | **medium** |
| 5 | Frauchiger–Renner localization / exact taxonomy (trivial holonomy; CF=1/6 on one CHSH facet) | FR-as-Hardy-contextuality (Abramsky); refined-FR-via-strong-contextuality (2409.05491) | **medium** |
| 6 | Sharp local valuation `μ_d` (even) / `μ_{2d}` (odd), even/odd inversion, sharp at d=3 | Sheaf-theoretic "local sections always exist" (Abramsky–Brandenburger); Gross `2⁻¹ mod d` | **medium-high (specific refinement) / low (qualitative core)** |
| 7 | Instrument-level Lüders affine depolarizing family; exact CP interval; unique single-Kraus; sharp core | Lüders/Ozawa/Busch repeatability theory; covariant-channel (Schur) depolarizing form | **medium (exact packaging) / low (scaffolding)** |

---

## Claim 1 — Obstruction spectrum / exact value grading `{0, d/2}`

- **Repository result.** For `m`-qudit Weyl–Heisenberg families, the achievable state-independent
  AvN value set is *exactly* `{0, d/2}` for even `d` and `{0}` for odd `d`; every certificate satisfies
  `2S ≡ 0 (mod d)` (Paper B, Obstruction Spectrum Theorem). Attainment at every even `d` is credited to
  ROZF; the upper bound and the value classification are claimed as the paper's own.
- **Closest known result.** **Abramsky, Cercelescu & Constantin, *Commutation Groups and
  State-Independent Contextuality*, FSCD 2024** (arXiv:2603.12197). Working over `Z_d` for all `d ≥ 2`
  with a skew-symmetric commutator matrix `μ`, they prove: **Thm 16** — a contextual word can exist
  only if `d` is even, and its global phase value `k` satisfies `2k ≡ 0 (mod d)` (they state
  explicitly "the global phase factor for a contextual word over `Z_{2k}` must be `k`", i.e. `d/2`);
  **Thm 18** — for odd `d` a non-contextual value assignment always exists. Separately, the even/odd
  *existence* dichotomy for Weyl/Heisenberg state-independent contextuality is long established (Gross's
  discrete Hudson theorem, quant-ph/0602001; "no SIC proof generalizes to the odd Heisenberg–Weyl
  group") and the cohomological even-`d` obstruction / general attainment is ROZF (Quantum 7, 979).
- **Precise difference.** In `Z_d` the solution set of `2k ≡ 0` is exactly `{0, d/2}`, so
  Abramsky–Cercelescu–Constantin Thm 16 **already yields the value grading `{0, d/2}` and the
  `2S≡0` relation**, and Thm 18 already yields the odd-`d` `{0}` result. Paper B's genuine residue
  over this is narrow: (a) the concrete realization as `m`-qudit *Weyl* families with the explicit
  symmetric lift and per-context carry, (b) the systematic *attainment* at every even `d` (which the
  paper itself attributes to ROZF), and (c) the "spectrum" framing plus the depth/value-bit corollaries.
  The core spectral statement — the headline of the program — appears substantially anticipated.
- **Novelty confidence: uncertain (high overlap).** The flagship theorem is largely a re-derivation,
  in a Weyl realization, of Abramsky–Cercelescu–Constantin Thm 16/18 (2024), combined with the ROZF
  attainment it already credits. A specialist should treat the value grading `{0, d/2}` as *likely
  known*.

## Claim 2 — Commutator-carry closed-form criterion

- **Repository result.** Contextuality of an even-`d` family is decided by a single integer invariant,
  the commutator carry `b(C)=Σ_{{u,v}} ⟨u,v⟩/d (mod 2)`; a family is contextual iff some cycle has odd
  symplectic self-pairing `Q` ("odd-Q" criterion, Paper B Thm cycle/carry). Marketed as a closed-form
  per-cycle evaluation replacing a search.
- **Closest known result.** (i) Same Abramsky–Cercelescu–Constantin paper: their **Thm 19** gives a
  cluster-graph criterion (non-contextual iff the commutation graph minus its center is a cluster
  graph) and **Thm 20** a definitive `Z_2` characterization by three forbidden induced subgraphs, all
  driven by the same `μ`. (ii) The **Saniga–Holweck–de Boutray–Giorgetti** program (arXiv:2105.13798;
  Muller et al. 2407.02928) casts observable-based contextuality as an unsatisfiable `GF(2)` linear
  system on the symplectic polar space `W_n` with a computed contextuality degree — which Paper B's
  Related Work already acknowledges as the `Z_2` shadow of its `A`,`b`. (iii) The symplectic form as
  the Heisenberg commutator / 2-cocycle is textbook (finite Weil representation).
- **Precise difference.** Paper B's contribution is a genuinely *closed-form, per-cycle* arithmetic
  criterion (`odd Q`) over `Z_d`, rather than a graph/`GF(2)`-search decision, and it lifts the degree
  notion to even `d` via the carry vector. That is a real sharpening, but it operates on the same
  object (the symplectic commutator matrix) and reaches a decision already characterized (existence of
  a contextual word) by the 2024 commutation-groups work.
- **Novelty confidence: medium.** The *form* (closed per-cycle carry) plausibly new; the *criterion it
  decides* is not. Overlap risk concentrated on Abramsky–Cercelescu–Constantin Thm 19–20.

## Claim 3 — Exact-order-`n` canonical MASA-stack class

- **Repository result.** The lifting class of `U(1)→U(n)→PU(n)`, Morita-reduced to the stack
  `[X/PU(n)]` of MASAs of `M_n(C)`, is a canonical degree-2 class of **order exactly `n`**
  (Paper A, Thm 8/Prop 2), with `H²(PU(n)⋉X;U(1))` nontrivial witnessed by Peres–Mermin, and a real
  `Z_2` shadow through the signed-permutation group `B_n` that splits iff `n` is odd.
- **Closest known result.** That `H²(PU(n);U(1)) ≅ H²(BPU(n);Z) ⊃ Z/n` and that the class of
  `U(1)→U(n)→PU(n)` is a generator of **order exactly `n`** is standard (projective-unitary-group
  cohomology; the canonical generator `ν ∈ H²(BPU(n);Z/n) ≅ Z/n`; see the PU(n) cohomology literature,
  arXiv:1710.09222, 2103.03523, and the projective-representation obstruction theory). The obstruction
  to lifting a `PU(n)`/projective representation to a linear `U(n)` one **is** this central-extension
  class; restricted to the finite Heisenberg subgroup it is the standard symplectic 2-cocycle of the
  Weil/Schrödinger representation. The Pauli/Peres–Mermin instantiation as a group-cohomology
  obstruction is ORBR (1701.01888) and the "homological invariants of stabilizer states" line
  (Wirthmüller, quant-ph/0611131; Comm. Math. Phys. 2024).
- **Precise difference.** The arithmetic core "order exactly `n`" is, on its face, the standard order
  of the generator of `H²(PU(n))`. What is not obviously in the literature: (a) the *stack* packaging
  `[X/PU(n)]` over the MASA space with Morita invariance as the home of the class, and (b) the **real
  orthogonal shadow** via `B_n = N(T)∩O(n)` with the mechanism "`−I` is a rotation square iff `n`
  even," giving the qubit-vs-odd-qudit `Z_2` split. Item (b) is the least standard and is the plausible
  novel kernel; item (a) is repackaging; the order-`n` statement itself is at high risk of being a
  known cohomological fact re-derived.
- **Novelty confidence: medium (stack/real-shadow geometry) / uncertain (the order-`n` class itself).**
  The repository already flags the stack/cohomology layer as its "largest open item" needing specialist
  audit — consistent with this.

## Claim 4 — Observer-context category + cocycle (relative facts, Z₂ switch invariant, movable cut)

- **Repository result.** An "observer" is formalized as an isometric enlargement plus a context on the
  enlarged algebra, organized into a category whose invertible switches form a subgroupoid carrying the
  `U(1)` cocycle; relative facts (local classical sections exist, no global assignment), a gauge-robust
  `Z_2` switch-loop invariant, and a "movable cut" follow (Paper D).
- **Closest known result.** The **context category** (poset/category of commutative subalgebras with a
  presheaf of local sections) is exactly the **Frembs–Döring** framework (arXiv:1910.09591,
  "Contextuality and the fundamental theorems of quantum mechanics"; Gleason for composite systems) and
  the earlier Isham–Butterfield/Döring topos program. **"Relative facts"** is the core notion of
  **Rovelli's relational quantum mechanics** (Di Biagio–Rovelli, "Stable Facts, Relative Facts";
  "Relational Quantum Mechanics and Contextuality," Found. Phys. 2024, arXiv:2308.08922). The
  sheaf-theoretic "local sections always exist, global section obstructed" is Abramsky–Brandenburger
  (1102.0264).
- **Precise difference.** Novelty is in the *specific* categorical object — a category whose
  invertible switches form a subgroupoid and which carries the program's `U(1)` cocycle, plus the
  gauge-robust `Z_2` switching invariant (`+1` on Weyl-switch loops, `−1` on the MUB triangle) tied to
  the Wigner-friend `−i`. The notion "relative facts" and "context category" are not new; their fusion
  with the extension cocycle and the concrete `Z_2` loop invariant is the contribution.
- **Novelty confidence: medium.** The architecture is a new assembly of established components; the
  repository itself marks the category-theoretic architecture as still under construction.

## Claim 5 — Frauchiger–Renner localization / exact taxonomy

- **Repository result.** In the observer-context category the FR cycle carries **trivial** class and
  exactly-invariant switch holonomy `+1` (refuting "paradox measures holonomy"), while its contradiction
  is state-sector with **contextual fraction exactly `1/6`** carried by a single CHSH facet (CHSH value
  `7/3`) (Paper D, Note).
- **Closest known result.** That the FR argument is **built on Hardy's logical contextuality** and can
  be read as a contextuality/sheaf paradox is established (Abramsky, "Contextuality: At the Borders of
  Paradox"; FR itself uses Hardy's paradox). A recent explicit recasting is **"A refined
  Frauchiger–Renner paradox based on strong contextuality"** (arXiv:2409.05491, Quantum 2026). The
  observer-independent-facts / Local-Friendliness no-go family (Brukner 1804.00749; Bong et al.,
  Nature Physics 2020, 1907.05607) frames the same scenarios.
- **Precise difference.** The *qualitative* "FR localizes to state-sector (Hardy-type) contextuality"
  is anticipated. Paper D's specific, checkable residue is quantitative: **the exact contextual
  fraction `1/6` on the identified CHSH facet** and the **triviality of the switch holonomy**
  (the "paradox is not holonomy" claim). Those specific numbers/decompositions are not something the
  search surfaced elsewhere.
- **Novelty confidence: medium.** New exact taxonomy of a known reduction; low risk of being *verbatim*
  prior work, but the framing overlaps a crowded FR-as-contextuality literature.

## Claim 6 — Sharp local valuation `μ_d` (even) / `μ_{2d}` (odd)

- **Repository result.** Every single commuting context trivializes over the value group `μ_d` for even
  `d` and `μ_{2d}` for odd `d` (sharp, with explicit `d=3` example); "local classicality is exact,"
  with an **even/odd inversion** of the naive expectation (odd `d` needs the *larger* group).
- **Closest known result.** The qualitative statement — *a single context always admits a classical
  (noncontextual) section; contextuality is only the obstruction to gluing* — is the foundational
  content of sheaf-theoretic contextuality (Abramsky–Brandenburger 1102.0264; also explicit in
  Abramsky–Cercelescu–Constantin, where every maximal clique carries local sections). The `2⁻¹ mod d`
  arithmetic that distinguishes even/odd is Gross / Raussendorf et al. (1511.08506).
- **Precise difference.** The *qualitative* "local validity" is standard and not novel. What the search
  did **not** find stated anywhere is the specific **graded value group** `μ_d`/`μ_{2d}` with the
  counterintuitive even/odd *inversion* and the sharpness witness at `d=3`. If correct and correctly
  attributed, that quantitative refinement looks new; but it is a refinement of a known qualitative fact.
- **Novelty confidence: medium-high for the specific value-group/inversion refinement; low for the
  qualitative "local classicality" core** (which is textbook sheaf theory and should be cited as such).

## Claim 7 — Instrument-level Lüders affine family / sharp core

- **Repository result.** Under Born statistics + repeatability + full commutant covariance, each
  projective-block instrument lies in a one-parameter **affine depolarizing family** with exact CP
  interval `−1/(r²−1) ≤ p ≤ 1` per rank-`r` block, with **Lüders the unique single-Kraus member**;
  plus a "sharp core" POVM result (repeatable⇒projective is false, witness must be ≥3-dim) and a
  classical Bayesian twin (Note, V37/V40).
- **Closest known result.** Repeatability theory is classical: **Ozawa (1984)** — repeatability forces
  discrete spectrum; **Busch et al.** — for sharp observables, repeatable ⟺ first-kind + value
  reproducible; **Lüders (1951)** and the generalized-Lüders-theorem literature (e.g.
  Int. J. Theor. Phys. 2014) — Lüders instrument is the unique ideal/efficient instrument of a sharp
  observable. The **depolarizing form of a covariant channel on a degenerate block** is standard
  representation theory (Schur's lemma / covariant-channel structure), and the CP interval of a
  depolarizing channel is a standard Choi-positivity computation.
- **Precise difference.** The individual ingredients are all classical. The contribution is the
  *explicit, exact packaging*: the rational CP interval `−1/(r²−1) ≤ p ≤ 1` per rank-`r` block, the
  "unique single-Kraus = Lüders" statement inside that family, and the ≥3-dimensional counterexample to
  "repeatable⇒projective" (correcting a false qubit folk example). This reads as a careful re-derivation
  and correction, not a new phenomenon.
- **Novelty confidence: medium for the exact interval/uniqueness packaging; low for the surrounding
  claims,** which restate Ozawa/Busch/Lüders. Overlap risk is that a measurement-theory specialist
  recognizes the whole affine family as the known covariant-instrument classification.

---

## Where a specialist must confirm (overlap-risk register, highest first)

1. **HIGHEST — the flagship spectrum + criterion vs. Abramsky–Cercelescu–Constantin (FSCD 2024,
   arXiv:2603.12197).** Their Thm 16 (`contextual word ⇒ d even`, value `= d/2`, `2k≡0 mod d`) and
   Thm 18 (odd-`d` noncontextuality) appear to **already contain the `{0,d/2}` value grading and the
   `2S≡0` relation** over `Z_d`, and their Thm 19–20 give commutation-graph contextuality criteria
   close to the commutator-carry test. **This paper is not cited anywhere in the repository.** A
   specialist must read it and decide how much of Paper B's headline survives as new (candidate residue:
   the Weyl realization, the per-cycle *closed-form* `odd-Q` arithmetic, the ROZF-based attainment
   spectrum, and the 2-adic tower). *This is the single reference to check first.*

2. **HIGH — central extension of the finite Heisenberg / Weil–metaplectic overlap (Paper A & the whole
   cocycle backbone).** The program's `U(1)` cocycle on `PU(n)`/the Heisenberg group is, up to Morita
   reduction, the standard obstruction 2-cocycle of the Weil/Schrödinger projective representation of
   the finite Heisenberg group, whose class generates `H²(PU(n))≅Z/n` of order `n`. Risk: the
   "canonical order-`n` MASA-stack class" and "`H²(PN(T);U(1))≅Z/n⊕Z₂`" are the known central-extension
   / metaplectic obstruction re-derived under stack language. A specialist in finite Weil
   representations and `H*(BPU_n)` (see 1710.09222, 2103.03523; Gurevich–Hadani 0705.4556;
   "Splitting of Clifford groups," 2603.24743) should confirm what, if anything, beyond the **real
   `B_n` shadow** (the `Z_2` even/odd split) is genuinely new.

3. **HIGH — Pontryagin-square / type-III-anomaly framing (Paper B §7).** The repository *already*
   downgrades `Q ≈ Pontryagin square` to "algebraic-form only / open" and does not construct the
   cohomology operation — this is the honest call. Risk remains that the invariant `Q` (a quadratic
   refinement of a symplectic pairing with an Arf/Brown-type flavor) simply **is** the known Pontryagin
   square / quadratic refinement of the Weyl–Heisenberg cocycle, in which case even the "resemblance"
   language understates a real identification. A specialist in Pontryagin squares / quadratic
   refinements (and the Arf-invariant transfer used in the doubling law) should adjudicate; keep the
   "we do not construct the operation" hedge until they do.

4. **MEDIUM — the context-category / relative-facts machinery vs. Frembs–Döring and relational QM.**
   Ensure Paper D and the Note cite Frembs–Döring (context category, 1910.09591) and Rovelli/Di Biagio
   (relative facts) so the categorical packaging is clearly positioned as *new assembly of known
   notions*, not new notions.

5. **MEDIUM — "local validity" vs. sheaf-theoretic local sections.** The qualitative claim that every
   single context is classical is definitional in Abramsky–Brandenburger and should be cited as such;
   only the `μ_d/μ_{2d}` graded value group and the even/odd inversion should be advanced as new.

6. **MEDIUM — instrument classification vs. Ozawa/Busch/Lüders.** A quantum-measurement-theory
   specialist should confirm the affine depolarizing family per block is not already the standard
   covariant-instrument classification; advance only the exact CP interval and the corrected ≥3-dim
   sharp-core witness as the specific new content.

---

## Bottom line (for triage)

- **Most likely genuinely new (highest confidence):** (6) the specific `μ_d/μ_{2d}` value group with
  even/odd inversion; (5) the exact FR taxonomy (CF `=1/6` on one CHSH facet + provably trivial switch
  holonomy); (7) the exact CP interval `−1/(r²−1)≤p≤1` with unique single-Kraus Lüders. Each is a
  *specific, checkable refinement* the literature search did not surface elsewhere — though each sits on
  standard scaffolding.
- **Biggest overlap risk (lowest confidence):** (1) the obstruction spectrum `{0,d/2}` / `2S≡0`
  (substantially anticipated by Abramsky–Cercelescu–Constantin Thm 16/18, plus Gross and ROZF);
  (2) the commutator-carry criterion (same paper's Thm 19–20 + Saniga–Holweck); (3) the order-`n`
  MASA-stack class (standard `H²(PU(n))≅Z/n` / Weil-rep obstruction).
- **Single reference a specialist should check first:** **S. Abramsky, Ş.-I. Cercelescu, C.-M.
  Constantin, *Commutation Groups and State-Independent Contextuality*, FSCD 2024 (LIPIcs vol. 299,
  art. 28; arXiv:2603.12197)** — it is the closest prior art to the two flagship Paper-B claims, works
  over `Z_d` already, and is currently uncited.

---

## Sources

- Abramsky, Cercelescu, Constantin — *Commutation Groups and State-Independent Contextuality*, FSCD 2024: https://drops.dagstuhl.de/storage/00lipics/lipics-vol299-fscd2024/LIPIcs.FSCD.2024.28/LIPIcs.FSCD.2024.28.pdf · arXiv: https://arxiv.org/pdf/2603.12197
- Okay, Roberts, Bartlett, Raussendorf — *Topological proofs of contextuality in quantum mechanics* (ORBR): https://arxiv.org/abs/1701.01888
- Raussendorf, Okay, Zurel, Feldmann (ROZF) — *The role of cohomology in quantum computation with magic states*, Quantum 7, 979 (2023): https://quantum-journal.org/papers/q-2023-04-13-979/ · https://arxiv.org/abs/2110.11631
- Gross — *Hudson's theorem for finite-dimensional quantum systems* (discrete Wigner / odd-d stabilizer nonnegativity): https://arxiv.org/abs/quant-ph/0602001
- Raussendorf, Browne, Delfosse, Okay, Bermejo-Vega — *Contextuality and Wigner function negativity in qubit quantum computation*, PRA 95, 052334: https://arxiv.org/pdf/1511.08506
- Abramsky, Brandenburger — *The sheaf-theoretic structure of non-locality and contextuality*: https://arxiv.org/abs/1102.0264
- Abramsky, Barbosa, Mansfield — *The contextual fraction as a measure of contextuality*, PRL 2017: https://arxiv.org/pdf/1705.07918
- Abramsky, Barbosa, Kishida, Lal, Mansfield — *Contextuality, cohomology and paradox*: https://arxiv.org/abs/1502.03097
- Abramsky, Soares Barbosa, Carù, Perdrix — *A complete characterisation of All-versus-Nothing arguments for stabiliser states* (ABCP): https://arxiv.org/abs/1705.08459
- Kim, Abramsky — *State-independent all-versus-nothing arguments*: https://arxiv.org/pdf/2311.11218
- Lisoněk, Raussendorf, Singh — *Parity proofs of the Kochen–Specker theorem* (Z₂ parity proofs): https://arxiv.org/abs/1401.3035
- de Boutray, Holweck, Giorgetti, Masson, Saniga — *Contextuality degree of quadrics in multi-qubit symplectic polar spaces*: https://arxiv.org/abs/2105.13798
- Muller, Saniga, Giorgetti, Holweck, Kelleher — *A new heuristic approach for contextuality degree estimates*: https://arxiv.org/abs/2407.02928
- Holweck et al. / Saniga et al. — *The many mathematical faces of Mermin's proof of the Kochen–Specker theorem*: https://arxiv.org/pdf/1511.00950
- Okay et al. — *Simplicial quantum contextuality*: https://arxiv.org/pdf/2204.06648
- *Comparing two cohomological obstructions for contextuality*: https://arxiv.org/pdf/2212.09382
- Wirthmüller — *Homological invariants of stabilizer states*: https://arxiv.org/pdf/quant-ph/0611131
- *Homological Invariants of Pauli Stabilizer Codes*, Comm. Math. Phys. (2024): https://link.springer.com/article/10.1007/s00220-024-04991-y
- *Splitting of Clifford groups associated to finite abelian groups*: https://arxiv.org/pdf/2603.24743
- Projective unitary group (H²(PU(n)) ≅ Z/n, Euler-class generator): https://en.wikipedia.org/wiki/Projective_unitary_group
- Gu — *On H*(BPU_n;Z) and Weyl group invariants*: https://arxiv.org/pdf/2103.03523
- *The cohomology of projective unitary groups*: https://arxiv.org/pdf/1710.09222
- Gurevich, Hadani — *Quantization of symplectic vector spaces over finite fields* (finite Weil representation): https://arxiv.org/pdf/0705.4556
- Gurevich, Howe — *Combinatorics of finite abelian groups and Weil representations*: https://arxiv.org/pdf/1010.3528
- Appleby et al. — *A remarkable representation of the Clifford group* (Weil/metaplectic, qudit): https://arxiv.org/pdf/1202.3559
- Pontryagin square (unstable cohomology operation H^n(-;Z_2)→H^{2n}(-;Z_4)) — nLab / cohomology-operations references (see arXiv:1308.2926 for the TFT usage): https://arxiv.org/pdf/1308.2926
- Frembs, Döring — *Contextuality and the fundamental theorems of quantum mechanics* (context category / presheaves over contexts): https://arxiv.org/abs/1910.09591
- Rovelli / Di Biagio — *Stable Facts, Relative Facts* (relational QM): https://philpapers.org/rec/ROVSFR
- *Relational Quantum Mechanics and Contextuality*, Found. Phys. 54 (2024): https://arxiv.org/pdf/2308.08922
- Frauchiger, Renner — *Quantum theory cannot consistently describe the use of itself*, Nat. Commun. 9, 3711: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6346061/
- Brukner — *A no-go theorem for observer-independent facts*: https://arxiv.org/pdf/1804.00749
- Bong, Utreras-Alarcón, et al. — *A strong no-go theorem on the Wigner's friend paradox* (Local Friendliness, Nature Physics 2020): https://arxiv.org/pdf/1907.05607
- *A "thoughtful" Local Friendliness no-go theorem* (Wiseman, Cavalcanti, Rieffel): https://arxiv.org/pdf/2209.08491
- *A refined Frauchiger–Renner paradox based on strong contextuality*: https://arxiv.org/abs/2409.05491
- Lüders — translation of *Über die Zustandsänderung durch den Messprozess*: https://arxiv.org/pdf/quant-ph/0403007
- *Lüders Instruments, Generalised Lüders Theorem, and Some Aspects of Sufficiency*, Int. J. Theor. Phys. (2014): https://link.springer.com/article/10.1007/s10773-014-2485-y
- *Lüders channels and the existence of symmetric-informationally-complete measurements* (depolarizing/Lüders structure): https://arxiv.org/pdf/1907.10999
