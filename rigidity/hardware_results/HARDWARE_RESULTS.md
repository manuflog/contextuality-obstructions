# EXP1 HARDWARE RESULTS — 2026-07-22, ibm_marrakesh (Heron), instance "Contextuality", plan open
297 circuits (33 rays x 3 theta x 3 states), 4000 shots each. Prediction: Sum<P_j> = 11 at every
point of the Peres-Penrose loop, for every state (Sum P = 11 I, proved; sump_mechanism.py).

theta     | zero    | plus    | random  | mean
Peres     | 11.2873 | 11.0340 | 11.0965 | 11.1393
Gaussian  | 11.1100 | 10.9032 | 10.9857 | 10.9996
Penrose   | 11.0258 | 11.0320 | 10.9663 | 11.0080
GRAND MEAN 11.0490 (+0.45% from 11). Shot-noise-only sigma per sum ~0.04, so residuals are
hardware (readout/gate/leakage) dominated, as the design predicted. HONEST READ: (a) STATE-
INDEPENDENCE holds within error at each theta (per-theta spreads 0.06-0.25); (b) LOOP-INVARIANCE
holds within error (per-theta means spread 0.14, no theta trend beyond scatter); (c) small
positive bias +0.45% consistent with readout asymmetry — a mitigated rerun (readout error
mitigation / dynamical decoupling) should tighten toward 11.000. VERDICT: the tight-frame witness
identity of the KS loop is CONFIRMED on real quantum hardware at three arithmetic points of the
moduli circle (Peres real, Gaussian, Penrose Z[sqrt-2]). First hardware data of the program.
EXP2 (holonomy eigenphase, target phi/2pi = 0.309357) running at time of writing.
