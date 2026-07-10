# Contextuality as cohomological obstructions — research program

Reproducibility and artifact companion for a research program framing quantum contextuality as
central-extension / stack-cohomological obstructions. The companion papers, their machine-checkable
verifications, hardware validation, and the supporting software package.

**Author:** Manuel Flores Gordillo · [ORCID 0009-0006-8246-0343](https://orcid.org/0009-0006-8246-0343)

## The papers (`papers/`)
- **Paper A — *The Canonical Class of the MASA Context Stack and the Peres–Mermin Obstruction*.**
  The qubit-vs-odd-qudit contextuality divide is given one geometric origin: a canonical degree-2
  class of **order exactly $n$** on the stack $[X/\mathrm{PU}(n)]$ of maximal abelian subalgebras of
  $M_n(\mathbb{C})$, with an even/odd real shadow through the signed-permutation group $B_n$, and full
  $H^2=\mathbb{Z}/n\oplus\mathbb{Z}_2$.
- **Paper B — *The Obstruction Spectrum of Weyl Context Families and the 2-adic Tower*.**
  The achievable state-independent AvN obstruction value is classified **exactly** as $\{0,d/2\}$ for
  even $d$ (and $\{0\}$ for odd), via a commutator-carry criterion equal to a Pontryagin-square /
  type-III anomaly, with matrix-checked certificates and a 3-device hardware replication.
- **Paper C — *What Contextual Holonomy Detects*.**
  The state sector: the $d=2$ codeword-polytope classification with the exact contextual-fraction
  formula, the $d=4$ tower laws, $\tau$-necessity, and the complete $d=4$ facet census
  (61 classes, 23,256 facets, derived twice by independent methods).
- **Paper D — *The Observer-Context Groupoid*.**
  Relative facts, the gauge-robust $\mathbb{Z}_2$ switching invariant, the movable cut, and the
  localization of the Frauchiger–Renner cycle.
- **Note — *Local Validity and Contextual Holonomy* (v3).**
  The synthesis: the Copenhagen commitments stated as five structural tenets, each a theorem of the
  companion papers, with every claim mapped to a verification script.

## What's here
- `papers/` — the papers (PDF; sources where applicable). All are **working drafts / preprints**
  (not yet submitted or posted to arXiv); cite as work in progress.
- `verification/` — ~50 self-contained Python scripts (V1–V46); **`INDEX.md` is the source of
  truth: it maps every load-bearing claim to its analytic proof and/or its verification script with
  the expected one-line result**, and it records corrections and retractions rather than erasing
  them. Heavy census artifacts (`d4_facet_classes.npz`, `d4_group_seed.pkl`) ship in-repo, so no
  verification requires heavy compute.
- `hardware/` — the $d=2$ Peres–Mermin protocol, 3-device results, and `verify_combined.py` (recomputes
  the $80\sigma$ combined figure from raw per-context correlators).
- `qkernel/` — pointer to the public package repo.

## Selected headline results (see `verification/INDEX.md` for the full ledger)
- **Obstruction spectrum** (Paper B): achievable AvN values exactly $\{0,d/2\}$ (even $d$), $\{0\}$
  (odd $d$); commutator-carry criterion; attainment at every even $d$.
- **Detection equivalence** (note, Thm 3; V44/V45): no noncontextual $\mathbb{Z}_d$ assignment
  $\iff$ a kernel-vector pairing equals $d/2$ $\iff$ odd commutator carry — a theorem, not a
  verified coincidence.
- **Complete $d=4$ facet census** (Paper C; V43): 61 classes / 23,256 facets, derived twice
  independently; with the on-hull lemma, $\mathrm{CF}(\rho)>0 \iff$ a facet is violated,
  unconditionally.
- **Sharp local valuation** (V46): every single commuting context trivializes over $\mu_d$ (even
  $d$) and $\mu_{2d}$ (odd $d$, sharp) — local classicality is exact, with the sharp value group
  and an even/odd inversion of the naive expectation.
- **Collapse** (V37/V40): statistics + repeatability + commutant-covariance pin the update to the
  one-parameter Lüders/depolarize family per block; the classical twin of the same axioms is
  Bayesian conditioning.
- **Observer sector** (Paper D; V38/V39): relative facts survive Wigner enlargement; the
  Frauchiger–Renner cycle is localized (trivial switch holonomy; the state-sector exit is a single
  CHSH facet, $\mathrm{CF}=1/6$ exactly).

## Quick reproduce
```bash
cd verification
python verify_cert8.py      # Paper B attainment, d=8:  ALL CHECKS PASS (S=4)
python verify_cert16.py     # Paper B attainment, d=16: PASS (S=8)
python criterion.py         # Paper B Thm Q: closed-form odd-Q criterion
python Wformula.py          # Paper B Thm W: value-bit formula, 0 violations
python thmG_general.py      # Paper A Thm G: order exactly n (dimension-independent)
python d4_facet_census.py   # Paper C census: 61 classes / 23,256 facets, on-hull lemma
python solvability_equivalence.py  # Note Thm 3: detection equivalence (V44)
python mu_d_valuation.py    # Sharp local valuation, even/odd inversion (V46)
cd ../hardware && python verify_combined.py   # 3-device combined S = 4.76 (80 sigma)
```

## Contact & citation
Manuel Flores Gordillo — manuel.flores@columbia.edu · [ORCID 0009-0006-8246-0343](https://orcid.org/0009-0006-8246-0343).
See `CITATION.cff`. The `qkernel` package is at https://github.com/manuflog/qkernel.

## License
This repository is dual-licensed:
- **Code** (`verification/`, `hardware/*.py`) — **Apache-2.0** (`LICENSE`), matching the `qkernel`
  package.
- **Papers** (`papers/*.pdf`) — **CC-BY-4.0** (`papers/LICENSE`): free to share and adapt with
  attribution. The author retains copyright.

## Requirements
Python 3.10+, `numpy`, `sympy`; a few scripts additionally use `scipy`
(`verification/requirements.txt`). The core canonical set runs with
`bash verification/run_all.sh` (expect `9/9 canonical verifications passed`); every other script is
standalone with its expected output pinned in `INDEX.md`.

## Status & honesty
Claims are labelled verified / analytic / open in `verification/INDEX.md` and in the papers.
Open problems are flagged as such there (e.g. the scenario-paired classification and tower
conjectures; Paper B generativity closed form; Paper A Yu–Oh scope / gerbe refinement; the
general-$d$ facet census; the cohomological home of KCBS-type contextuality). Retractions and
corrections are recorded, not erased — the ledger includes retracted claims and the artifacts that
caused them. This companion contains no internal working notes — only the papers, the canonical
verifications, and the hardware artifacts.
