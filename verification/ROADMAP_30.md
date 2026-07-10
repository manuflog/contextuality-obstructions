# Roadmap: canonical verifications 9 → 30

Goal: grow the machine-checked core from 9 to 30 canonical scripts, each self-contained
(numpy/sympy), each mapping one load-bearing claim to a one-line expected result, extending
`INDEX.md`. Ordering is by leverage: repair the Copenhagen note first, then convert
analytic-only rows of both papers to verified, then attack the open problems, then harden.

Status: **18 / 30** (V1–V9 = existing suite; V10–V13, V22, V24–V27 shipped, all PASS).

## Tier I — Copenhagen-note §3 bridge (DONE this iteration)
| # | Script | Claim | Result |
|---|---|---|---|
| V10 | `pm_gauge_invariance.py` | PM six-sign product = −1, invariant under all 512 sign gauges | PASS |
| V11 | `pauli_slice_bridge.py` | carry route (Q=1) ≡ matrix route (−1): Thm E/F machine-checked | PASS |
| V12 | `wf_loop_holonomy.py` | WF loop (H,S): ray-level holonomy **+1**; det/metaplectic-section holonomy **−i** exactly (500/500 SU(2) closers); (−i)² = −1 section-independent | PASS + FINDING |
| V13 | `d3_gap_certificate.py` | order-3 Heisenberg class nontrivial (exhaustive 3⁹ coboundaries) AND qutrit AvN value {0} → class ⇏ AvN contextuality, certified | PASS |

V12 finding (action item for the CMP draft): the repair-plan value −i is a τ-level
(2-adic-tower) invariant of the determinant section, invisible at the projective/Bargmann
level. The draft must pin the section; interferometry targets the ray level (+1 for this
loop) or the section-independent square (−1). Known nit: V13's m=1 exhaustive slab has no
cycles (the 4 qutrit lines are observable-disjoint); the certificate's weight is the
2000-family m=2 sample (707 cycles, all even) plus Thm J analytic.

## Tier II — Paper B strengthening
| # | Script | Claim / expected |
|---|---|---|
| V14 | `d3_null_protocol.py` | emit hardware-ready qutrit protocol JSON; predicted S = 0 (risky null) |
| V15 | `cert_even_general.py` | attainment certificates at d=6 and d=12 (even, non-2-power); S = d/2 |
| V16 | `spectrum_exhaustive_d4.py` | exhaustive slab d=4, m≤2: value set exactly {0, 2} |
| V17 | `depth_rigidity_check.py` | Thm K computational: no single lift carries the obstruction (cert8) |
| V18 | `generativity_search.py` | open problem: falsify candidate base-level invariants systematically |

## Tier III — Paper A strengthening
| # | Script | Claim / expected |
|---|---|---|
| V19 | `yu_oh_shadow.py` | Yu–Oh (d=3): Z₂-shadow splits (n odd) ⇒ class-invisible; scope pinned |
| V20 | `H2_finite_models.py` | Prop H2 computational: H²(PN(T);U(1)) = Z/n ⊕ Z₂ for n = 2,3,4 |
| V21 | `Bn_splitting.py` | Thm B certificates: −I rotation-square iff n even (explicit roots / exhaustive nonexistence) |

## Tier IV — Copenhagen-note §2/§4
| # | Script | Claim / expected |
|---|---|---|
| V22 | `lueders_uniqueness.py` | Lüders = unique CP minimally-disturbing update; axioms pinned, numeric on random instances |
| V23 | `local_classicality.py` | Spec(C) classical structure per MASA; cross-context failure exhibited |

## Tier V — the equivalence program (the hard open gap)
| # | Script | Claim / expected |
|---|---|---|
| V24 | `holonomy_vs_contextual_fraction.py` | LP contextual fraction vs carry invariant over random families, d=3,4; covariation map + counterexample hunt |
| V25 | `converse_search.py` | certified example: LP-contextual family with trivial class (proves the CMP weakening was necessary, not cautious) |

## Tier VI — hardware bridge
| # | Script | Claim / expected |
|---|---|---|
| V26 | `pm_holonomy_from_hardware.py` | re-express 3-device S = 4.7614 as measured loop sign −1 with CI |
| V27 | `cert8_compile.py` | qiskit-ready circuits for the d=8 certificate (predicted S = 4) |
| V28 | `wf_interferometry_spec.py` | Pancharatnam protocol for V12: per-strand ±π/4, section-square −1 |

## Tier VII — adversarial hardening
| # | Script | Claim / expected |
|---|---|---|
| V29 | `convention_fuzz.py` | fuzz τ-sign / ordering conventions; certify which invariants are convention-independent (also: move `criterion.py` demo under `__main__` to silence import noise) |
| V30 | `property_harness.py` | randomized property tests (symp bilinearity, cocycle identity, gauge invariance, carry additivity), pinned seeds |

## Iteration protocol
Each iteration: implement 3–5 scripts → run full suite (old + new) → update INDEX.md rows
(analytic-only → verified where applicable) → log any FINDING that changes a paper claim.
Failures are reported as findings, never silently repaired.
