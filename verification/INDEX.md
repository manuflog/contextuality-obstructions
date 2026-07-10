# Claim → verification index

Every load-bearing claim in the two papers, mapped to how it is established: an **analytic proof**
in the paper, a **computational verification** script here, or both. Scripts are self-contained
(Python 3 + numpy/sympy); run from this directory. Each entry lists the one-line result to expect.

## Paper A — *The Canonical Class of the MASA Context Stack and the Peres–Mermin Obstruction*

| Claim | Statement | How established |
|---|---|---|
| Thm A | $\mathrm{U}(1)$ lifting class is vacuous on $\mathrm{U}(n)\ltimes X$ | analytic (paper §1) |
| Prop (Morita) | reduction to $\mathrm{PU}(n)\ltimes X$ | analytic (paper §2) |
| **Thm G** | canonical class has **order exactly $n$** | analytic (equivariant-hom lemma) + `thmG_general.py` (dimension-independent stress test) |
| Thm B | real shadow splits **iff $n$ odd** ($-I$ is a rotation square iff $n$ even) | analytic (paper §3) |
| Thm E / F | $n=4$ Pauli slice: PM pairing $=-1$, equals the obstruction | analytic (paper §4–5) |
| Prop H2 | $H^2(\mathrm{PN}(T);\mathrm{U}(1))=\mathbb{Z}/n\oplus\mathbb{Z}_2$, canonical class generates $\mathbb{Z}/n$ | analytic (paper §7, spectral sequence) |

## Paper B — *The Obstruction Spectrum of Weyl Context Families and the 2-adic Tower*

| Claim | Statement | How established |
|---|---|---|
| **Thm J** | Obstruction Spectrum: $2S\equiv0\ (\mathrm{mod}\ d)$; achievable value set $\{0,d/2\}$ (even $d$), $\{0\}$ (odd) | analytic + `spectrum_test2.py` (odd-$d$ ⇒ only 0) |
| Thm J′ | commutator-carry form of the criterion | analytic + folded into `criterion.py` |
| Thm K | Depth Rigidity: obstruction lives at top 2-adic layer, no single lift carries it | analytic (paper §4) |
| **Thm W** | exact value-bit formula for lifted certificates | `Wformula.py` → `violations 0/550` |
| **Thm Q** | closed-form criterion: contextual ⟺ some cycle has odd $Q=\sum_{a<b}\langle v_a,v_b\rangle/d$ (Pontryagin square) | `criterion.py` |
| **Thm attain** | attainment at every even $d$; explicit certificates | `verify_cert8.py` → `ALL CHECKS PASS` ($S{=}4$); `verify_cert16.py` → `PASS` ($S{=}8$) |
| Doubling law (geometric step) | $T_{\mathrm{mix}}\equiv0$; $\mathbb{F}_2$ system is $d$-independent | `close_T2_proof.py` (structural) + `tmix_dindep.py` (byte-identical across $d\equiv2\bmod4$) |

## Open (honestly labelled in the papers)
- Paper B: base-level closed form for **generativity** of the doubling law (governed by the standard
  odd-carry-cycle criterion; no slicker base-level invariant found).
- Paper A: **Yu–Oh** monomial-shadow scope; Dixmier–Douady gerbe refinement.

## Hardware
See `../hardware/` — 3-device $d=2$ Peres–Mermin replication, combined $S=4.76$ at $80\sigma$;
`verify_combined.py` recomputes it from the raw per-context correlators.

## Session additions (2026-07-09) — Copenhagen-note bridge, state sector, hardware
| Claim | Script | Expected one-liner |
|---|---|---|
| PM sign product −1, all 512 gauges | `pm_gauge_invariance.py` | `PASS`, product {−1} |
| Thm E/F machine-checked (carry ≡ matrix) | `pauli_slice_bridge.py` | `PASS: routes AGREE` |
| WF loop: ray-level +1, det-section −i (FINDING) | `wf_loop_holonomy.py` | `PASS`, {−1j}, 500/500 |
| class ⇏ AvN (d=3, exhaustive) | `d3_gap_certificate.py` | `GAP CERTIFIED` |
| Lüders unique under axioms (i)–(iii) | `lueders_uniqueness.py` | `PASS`, efficiency necessary |
| contextual with zero class shadow (KCBS) | `kcbs_converse.py` | `PASS`, 2.23607 > 2 |
| unsolvability ⟺ odd cycle, d=2,4 | `holonomy_vs_solvability.py` | 0 mismatches + pinned positives |
| state sector: CF(PM)=1, CF(KCBS)=0.4721, discrete Hudson d=3 | `state_sector_probe.py` | `PASS`, strange −1/3 |
| hardware: ibm_fez two-loop run | `../hardware/results/ibm_fez_holonomy_20260709.json` | loop +1; S=4.6125, 35.4σ over tight bound 4 |
