# Rigidity–contextuality verification suite

Verification scripts for the paper **"Exact rigidity certificates for Kochen–Specker sets, the
graded flex dictionary, and the Peres–Penrose circle"** (M. Flores Gordillo, preprint 2026).
Every claim in the paper carries an evidence label (PROVED / EXACT / NUMERICAL / LITERATURE / OPEN);
each label is backed by one of the scripts here. All exact certificates use rational / ℤ[√2] /
ℤ[ω] arithmetic (sympy ranks over ℚ, or two-prime modular rank bounds at primes where the relevant
quadratic is a residue) — no floating-point tolerances. Numerical results report explicit spectral
gaps.

Requirements: Python 3 with `numpy` and `sympy` (`scipy` for two scripts). Every script is
self-checking and prints PASS/FAIL. Typical runtimes: under 10 s each; a few census stages up to
~40 s (staged via CLI args).

## Core engines
| script | role |
|---|---|
| `flex_dimension.py` | numerical flex-dimension engine (spectral-gap-checked ranks) |
| `exact_rigidity.py` | exact flex certificates over ℚ (integer rays, extended Jacobian) |
| `sic_zoo.py` | ℤ[√2] exact machinery, KS-colorability, Peres-33 / CEG-18 rays |
| `torsion_layer.py` | torsion invariants t₀ (KS-colorability) and τ (forced parity) |
| `torsion_flex.py` | the real/skew block decomposition of the Hermitian rigidity matrix |
| `peres_penrose.py` | the exact Gould–Aravind circle (Laurent arithmetic over ℤ[√2]) |
| `even_cycles.py` | faithful cycle realizations; the C₄/CHSH exception |

## Theorem certificates (script → paper claim)
| script | certifies |
|---|---|
| `cycle_flex_theorem.py` | flex(Cₙ)=2n−8, odd n≥5 (Theorem, every proof ingredient) |
| `cycle_split_theorem.py` | graded split flex_ℝ=n−3, flex_skew=n−5 |
| `cycle_split_general.py` | general-d, both-parity graded split (13 realizations) |
| `sump_mechanism.py` | ΣP=11·I forced by four B₃-orbit tight frames (Schur) |
| `no_degeneration.py` | exact no-degeneration of the circle (456 non-edge Grams > 0 ∀θ) |
| `z2island.py` | the ℤ[√−2] island = Penrose-33; island identification |
| `cabello33.py` | Cabello's 14-basis set is exactly rigid over ℚ(√3) |
| `ks_flex_census.py` | exact rigidity census: Kernaghan's islands, CK-31, a new d=4 set |
| `graph_survey.py` | broad (graph, d) flex survey |
| `hermitian_bilinear.py` | the real block = classical bilinear OR-rigidity (two code paths) |
| `flex_criterion.py` | Maxwell/stress counts do not predict the skew flex |
| `dminus1_bound.py` | flex_ℝ−flex_skew ≤ d−1 reduced to rank(A)≥rank(B) |
| `branch_rank2.py` | rank(A)=rank(B)=|E| proved for 2-degenerate graphs; 3-core reduction |
| `branch_second_order.py` | second-order criterion; Peres-33's flex provably integrates |
| `branch_global.py` | arithmetic points of the circle (real / ℤ[√−2] / Gaussian / no Eisenstein) |

Run any script directly, e.g. `python3 cycle_split_theorem.py`. Scripts with heavy stages take a
CLI argument to run one stage at a time (see each script's docstring).

## Circle geometry / arithmetic / open-problem suite (second wave)
| script | certifies |
|---|---|
| `branch_arith.py` | ring points of the circle exist iff 2 ramifies; 8 special points; d=7 excluded |
| `branch_berry.py` | exact rational per-ray Berry phases; sum zero; abelian holonomy trivial |
| `phi_hunt.py` | the non-abelian eigenphase phi = 2*pi*sqrt(1867)/33 (rotating-frame proof) |
| `lattice_holonomy.py` | section-lattice holonomy map; 1867 = 43^2+3^2+3^2; rank-29 sublattice |
| `branch_imag.py` | Im v(theta) = sin(theta)*y identically; exact imaginarity profiles |
| `alphabet_theorem.py` | d=3 alphabet theorem: four mechanisms, x<->1/x duality, structure theorem |
| `alphabet_d4.py` | d=4 extension: term-alphabet lemma, 16 new mechanisms, no colorable regime |
| `theta_proof.py` | Lemma Z + coset-shift theorem (partial proof of the Theta-coherence lemma) |
branch_theta2.py -> gamma_t closed form (dependency of theta_proof.py)
