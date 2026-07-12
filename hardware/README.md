# Hardware validation — Peres–Mermin ($d=2$) on IBM Quantum

The $d=2$ endpoint of the obstruction spectrum (value $d/2=1$, the Peres–Mermin obstruction) run on
three IBM Heron devices, as reported in Paper B §"Hardware validation at the base of the tower".

## Protocol (important)
Each of the six contexts is realized as a **sequential** non-destructive measurement: one shared
ancilla, Hadamard-test readout of each commuting Pauli into its own bit, reset between. A *single joint*
measurement of the reconstructed product is **algebraically pinned** (returns the exact sign on any
data, even noise) and so tests nothing — the sequential protocol is what makes the statistic genuinely
device-dependent. Contexts are pinned to one low-error qubit triple per device; dynamical decoupling and
measurement twirling on; 8192 shots/context.

## Result
| device | S | σ above bound (4) |
|---|---|---|
| ibm_marrakesh | 4.94 ± 0.02 | 61 |
| ibm_fez | 4.56 ± 0.02 | 32 |
| ibm_kingston | 4.74 ± 0.02 | 44 |
| **combined (inverse-variance)** | **4.761 ± 0.010** | **80** |

Chip-to-chip spread (4.56–4.94) is genuine variation; every device is far above the classical bound.

> **Two distinct `ibm_fez` measurements — not a discrepancy.** The `ibm_fez` value above ($S=4.56$)
> is from this **three-device replication** (`results/pm_d2_3device_results.json`), cited in Paper B.
> A separate **single-device holonomy job** (`results/ibm_fez_holonomy_20260709.json`, job
> `d986q62f47jc73a7hm2g`) reports its own Peres–Mermin witness $S=4.6125\pm0.0173$ (35.4σ) alongside
> the interferometric loop phase; that figure is cited in the synthesis note and Paper C. Same backend,
> two different runs.

## Reproduce
- `python verify_combined.py` — recomputes each device's $S$ from its six per-context correlators and
  the inverse-variance-weighted combined figure from `results/pm_d2_3device_results.json`.
- `PeresMermin_d2_v4_multibackend.ipynb` — the runnable Colab notebook (needs an IBM Quantum token;
  set `QISKIT_IBM_TOKEN`). Emits the sequential protocol, auto-selects the best qubit triple per device.

## Scope and significance caveat (read before quoting the σ)
The quoted **80σ is relative to shot-noise statistical uncertainty under the implemented witness
model** — it is *not* a loophole-free contextuality certification. Specifically the figure does **not**
incorporate: calibration drift, gate/model systematics, measurement disturbance, inconsistent
marginals across contexts, look-elsewhere/backend-selection effects, or device-dependent compatibility
loopholes. A Peres–Mermin witness above 4 under *sequential* measurements certifies contextuality only
under compatibility/nondisturbance assumptions; sequential disturbance or context-dependent
implementations can mimic an apparent violation. Inverse-variance combination across the three devices
assumes independent statistical errors and does **not** remove common systematics.

Recommended phrasing: *"80 standard deviations relative to shot-noise uncertainty under the implemented
witness model,"* not *"loophole-free 80σ contextuality."* Compatibility diagnostics that would harden
the claim (same-observable marginal across every context containing it; order-reversal tests;
repeated-measurement agreement; disturbance bounds; per-context calibration spread) are **not yet
reported here** and are an open experimental to-do.

This run certifies the **presence** of the $d=2$ obstruction under those assumptions. It does not test
the value grading; by the value–bit corollary the $d=4$ obstruction phase is the same $-1$, so the
grading is an algebraic (software-certified) fact and $d=4$ hardware would add no new physics at
feasible gate counts. Kept separate: (i) hardware replication of the PM witness; (ii) algebraic
verification of the obstruction grading; (iii) any empirical test of the higher-dimensional tower.
