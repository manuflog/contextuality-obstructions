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

## Reproduce
- `python verify_combined.py` — recomputes each device's $S$ from its six per-context correlators and
  the inverse-variance-weighted combined figure from `results/pm_d2_3device_results.json`.
- `PeresMermin_d2_v4_multibackend.ipynb` — the runnable Colab notebook (needs an IBM Quantum token;
  set `QISKIT_IBM_TOKEN`). Emits the sequential protocol, auto-selects the best qubit triple per device.

## Scope
This certifies the **presence** of the $d=2$ obstruction. It does not test the value grading; by the
value–bit corollary the $d=4$ obstruction phase is the same $-1$, so the grading is an algebraic
(software-certified) fact and $d=4$ hardware would add no new physics at feasible gate counts.
