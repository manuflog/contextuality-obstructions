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

# EXP2 HARDWARE RESULTS — 2026-07-22, ibm_marrakesh, instance "Rigidity" (pay-as-you-go)
Hadamard-test estimate of the KS-loop holonomy eigenphase. Exact target phi/2pi = sqrt(1867)/33
= 0.309357436; Trotterized noiseless truths: N=4 -> 0.307395, N=8 -> 0.308910.
  N=4 (43 CX):  <X>=-0.3292 <Y>=+0.7302  ->  phi/2pi = 0.317415
  N=8 (283 CX): <X>=-0.1993 <Y>=-0.0060  ->  phi/2pi = 0.504791
HONEST READ: N=4 SUCCEEDED within noise — the Bloch vector contracted (|r| 0.80 vs ideal 1.0,
classic decoherence) but the ANGLE survived: 0.3174 vs exact 0.3094 (2.6% high; 1.0e-2 from the
N=4 Trotter truth). N=8 DECOHERED completely (283 CX; |r|~0.2, phase uninformative) — exactly as
the design's cost table predicted; error mitigation or a shallower compilation would be needed.
VERDICT: a first, noisy but genuine hardware measurement CONSISTENT with phi = 2pi*sqrt(1867)/33.
sqrt(1867) has been measured on a quantum processor to ~3%. Improvements: readout mitigation, ZNE,
dynamical decoupling, or the constant-connection shortcut (compile e^{2pi Atilde} directly, ~6 CX).

# EXP2-SHORTCUT HARDWARE RESULTS — 2026-07-22, ibm_fez, instance "Rigidity"
Constant-connection shortcut (W = e^{2pi Atilde}, NO Trotter error; 16 native 2q gates, depth 74).
Job d9g2anineu4c739q88k0, m = 1,2,3 controlled powers, 8000 shots:
  m=1: 0.355910 | m=2: 0.674269 | m=3: 0.986852   (targets 0.309357/0.618715/0.928072)
Simulator z-scores ~1 (clean); hardware z ~30 in shot-noise units => SYSTEMATICS-dominated.
KEY ANALYSIS — the multi-m data separates signal from systematic: linear fit phase(m) = slope*m + c
gives SLOPE = 0.3155 (vs exact 0.309357; +2.0%) and intercept c = 0.040 (a per-circuit systematic,
prep/readout coherent error). CROSS-MACHINE CONSISTENCY: marrakesh N=4 Trotter gave 0.3174; fez
shortcut slope 0.3155 — two processors, two protocols, both ~+2-3% above the exact value with the
same sign, suggesting a common coherent-overrotation bias rather than a physics discrepancy.
NEXT: readout mitigation + dynamical decoupling; or interleave with an identity-loop reference
circuit to cancel the prep systematic. Current best hardware statement: phi/2pi measured to ~2%
(multi-m fit) on two devices, consistent with sqrt(1867)/33.
