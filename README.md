# Contextuality as cohomological obstructions — research program

Reproducibility and artifact companion for a research program framing quantum contextuality as
central-extension / stack-cohomological obstructions. Two papers, their machine-checkable
verifications, hardware validation, and the supporting software package.

**Author:** Manuel Flores Gordillo · [ORCID 0009-0006-8246-0343](https://orcid.org/0009-0006-8246-0343)

## The two papers (`papers/`)
- **Paper A — *The Canonical Class of the MASA Context Stack and the Peres–Mermin Obstruction*.**
  The qubit-vs-odd-qudit contextuality divide is given one geometric origin: a canonical degree-2
  class of **order exactly $n$** on the stack $[X/\mathrm{PU}(n)]$ of maximal abelian subalgebras of
  $M_n(\mathbb{C})$, with an even/odd real shadow through the signed-permutation group $B_n$, and full
  $H^2=\mathbb{Z}/n\oplus\mathbb{Z}_2$.
- **Paper B — *The Obstruction Spectrum of Weyl Context Families and the 2-adic Tower*.**
  The achievable state-independent AvN obstruction value is classified **exactly** as $\{0,d/2\}$ for
  even $d$ (and $\{0\}$ for odd), via a commutator-carry criterion equal to a Pontryagin-square /
  type-III anomaly, with matrix-checked certificates and a 3-device hardware replication.

## What's here
- `papers/` — both papers (PDF).
- `verification/` — self-contained Python scripts; **`INDEX.md` maps every load-bearing claim to its
  analytic proof and/or its verification script with the expected result.** All 9 canonical scripts run
  on Python 3 + numpy/sympy.
- `hardware/` — the $d=2$ Peres–Mermin protocol, 3-device results, and `verify_combined.py` (recomputes
  the $80\sigma$ combined figure from raw per-context correlators).
- `qkernel/` — pointer to the public package repo.

## Quick reproduce
```bash
cd verification
python verify_cert8.py      # Paper B attainment, d=8:  ALL CHECKS PASS (S=4)
python verify_cert16.py     # Paper B attainment, d=16: PASS (S=8)
python criterion.py         # Paper B Thm Q: closed-form odd-Q criterion
python Wformula.py          # Paper B Thm W: value-bit formula, 0 violations
python thmG_general.py      # Paper A Thm G: order exactly n (dimension-independent)
cd ../hardware && python verify_combined.py   # 3-device combined S = 4.76 (80 sigma)
```

## Papers status
The two PDFs in `papers/` are **working drafts / preprints** (not yet submitted or posted to arXiv).
The mathematical results are complete and machine-verified as indexed; the drafts still await final
submission formatting. Please cite as work in progress.

## Contact & citation
Manuel Flores Gordillo — manuel.flores@columbia.edu · [ORCID 0009-0006-8246-0343](https://orcid.org/0009-0006-8246-0343).
See `CITATION.cff`. The `qkernel` package is at https://github.com/manuflog/qkernel.

## License
This repository is dual-licensed, the standard split for a research-artifact companion:
- **Code** (`verification/`, `hardware/*.py`) — **Apache-2.0** (`LICENSE`). Permissive; includes an
  explicit patent grant, matching the `qkernel` package.
- **Papers** (`papers/*.pdf`) — **CC-BY-4.0** (`papers/LICENSE`). Anyone may share and adapt with
  attribution; this is the standard open-access license and is accepted by arXiv and most journals,
  and it does not impede later journal submission (you retain copyright).

## Requirements
Python 3.10+, `numpy` and `sympy` (`verification/requirements.txt`). Run everything with
`bash verification/run_all.sh` (expect `9/9 canonical verifications passed`).

## Status & honesty
Claims are labelled verified / analytic / open in `verification/INDEX.md` and in the papers. Two
problems are open and flagged as such (Paper B generativity closed form; Paper A Yu–Oh scope / gerbe).
This companion contains no internal working notes — only the papers, the canonical verifications, and
the hardware artifacts.
