# Independent verification (Priority 6)

A second implementation that **shares no code with `verification/`** — it does not import `weyl.py`
or any repository helper. Weyl operators are built from raw `d×d` matrices with an independently
written composition, and the symplectic form / commutation are recomputed from scratch. This guards
against the failure mode where agreement among scripts in one codebase reflects a shared bug or
convention rather than the mathematics.

## Run
```bash
python3 independent_checks.py     # prints PASS/FAIL per check; writes independent_checksums.json
```
Requires `numpy` and `scipy` (the latter only for check 8).

## What it re-derives, and the result (2026-07-12)
| # | Check | Independent result |
|---|-------|--------------------|
| 1 | Weyl multiplication ⇔ symplectic commutation | 200/200 agree at d=4 |
| 2 | Peres–Mermin context products | rows +1, third column −1, product −1 ⇒ AvN |
| 3 | Obstruction spectrum value-bit | lands in **exactly {0, d/2}** for d=2,4,6,8 |
| 4 | Detection equivalence at d=2 (brute force) | 0 of 512 assignments satisfy all 6 lines ⇒ AvN |
| 5 | Local valuation μ_d (even) / μ_{2d} (odd) | trivializer found at d=2,3 |
| 6 | Exact CP interval −1/(r²−1) ≤ p ≤ 1 | endpoints + just-outside rejections, r=2,3,4 |
| 7 | Qutrit repeatable non-projective POVM | both outcomes repeatable, E₁ unsharp |
| 8 | Frauchiger–Renner contextual fraction | **CF = 0.166667 = 1/6** (independent LP) |

`ALL INDEPENDENT CHECKS PASS`, checksum recorded in `independent_checksums.json`.

## Honest scope
This is an independent *numerical* reimplementation, not a formal proof or a different CAS. It covers
the finite Weyl/AvN core, the CP interval, the sharp-core POVM, and the FR contextual fraction. It does
**not** independently reproduce: the full d=4 facet census (heavy), the stack-cohomology order-n
theorem of Paper A (that needs a proof audit, not a recomputation), or the hardware analysis. During
development it caught two genuine bugs *in the test harness itself* (a wrong Peres–Mermin sign and a
conflation of context scalars with the value-bit) — recorded here as evidence the checks do real work.
