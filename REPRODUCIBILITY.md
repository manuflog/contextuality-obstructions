# REPRODUCIBILITY

## Environment
- Python 3.10+ (CI pins 3.11).
- `pip install -r verification/requirements.txt` → `numpy>=1.24`, `sympy>=1.12`, `scipy>=1.10`.
- No heavy compute required: census artifacts (`d4_facet_classes.npz`, `d4_group_seed.pkl`) ship in
  the repo, so no verification step needs `pycddlib`/`lrs`/`Normaliz` to *check* (only a full *re-run*
  of the census construction would).
- Randomized scripts use fixed seeds (`numpy.random.default_rng(<fixed>)`); reruns are deterministic.

## One-command check
```bash
bash verification/run_all.sh        # 12/12 canonical verifications; exits nonzero on any failure
bash verification/check_stale_claims.sh   # guards against retracted phrases in active sources
python3 independent_verification/independent_checks.py   # independent re-derivation + checksum
```
CI (`.github/workflows/verification.yml`) runs the canonical suite + the stale-claim guard on every
push, so each commit carries a green/red status.

## Per-claim reproduction
Every load-bearing claim maps to a script with a pinned one-line result in `verification/INDEX.md`.
Highlights (all standalone: `python3 <script>.py`):
```
verify_cert8.py / verify_cert16.py   Paper B attainment certificates (d=8 S=4, d=16 S=8)
spectrum_test2.py                    odd-d ⇒ value {0}
criterion.py                         closed-form odd-Q criterion
Wformula.py                          value-bit formula (0/550)
thmG_general.py                      Paper A order-n equivariant lemma (n=2..8)
mu_d_valuation.py                    sharp local valuation μ_d / μ_{2d}
d4_facet_census.py                   Paper C census (61 classes / 23,256 facets)
lueders_cp_interval.py               exact CP interval −1/(r²−1)≤p≤1
s4_povm_bayes.py                     sharp-core qutrit POVM + classical twin
hardware/verify_combined.py          3-device PM combined S=4.76 (shot-noise σ)
```

## Independent reproduction (no shared code)
`independent_verification/independent_checks.py` rebuilds the Weyl operators from raw matrices
(imports nothing from `verification/`) and re-derives the core outputs, emitting
`independent_checksums.json`. Current checksum and results are in that directory's `README.md`.

## What "reproduced" does and does not mean here
- **Does:** the finite Weyl/AvN core, CP interval, sharp-core POVM, and FR contextual fraction are
  reproduced by an independent implementation; the d=4 census is reproduced by two engines.
- **Does not (yet):** independent third-party replication; a formal-proof (Lean/Coq) layer; a
  compatibility-controlled hardware re-run; specialist proof audit of Paper A's stack cohomology.
  See `KNOWN_LIMITATIONS.md`.

## For a citable snapshot
Before citing exact numbers in the papers, tag a release and pin the commit:
- record the commit hash the papers cite;
- archive `independent_checksums.json` and the census artifacts;
- (recommended) mint a Zenodo DOI for the tagged release and cite it in each paper's data-availability
  line. *(DOI minting is a manual step; not done in-repo.)*
