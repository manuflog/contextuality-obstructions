# PROOF DEPENDENCIES

Which theorems rest on which lemmas / external results. `→` means "depends on". External inputs are in
**bold**; the largest unaudited node is marked ⚠.

## Paper B — obstruction spectrum (self-contained core)
```
Weyl composition law  c(u,v) = -q(u)-q(v)+2β(u,v)+q(u+v)      [computed vs matrices d=4,6,9,16]
        │
        ▼
Spectrum Theorem  2S ≡ 0 (mod d)  →  value set {0, d/2} (even), {0} (odd)
        │  (telescoping; doubled phase is a coboundary with potential -q)
        ├──► Corollary 2 (obstruction is one bit b(C))
        ├──► Theorem 4 (commutator-carry form)
        ├──► Theorem 8 (cycle value = symplectic self-pairing Q)  [Q≈Pontryagin square: algebraic form only]
        ├──► Proposition 9 (odd-Q sound, possibly incomplete; Weyl faithfulness)
        └──► Theorem 5 (Depth Rigidity)
Attainment (Theorem 7)  →  **ROZF, Quantum 7, 979 (2023)**  +  explicit certs d=8,16 (this work)
```

## Detection equivalence (note Thm 3)
```
Spectrum Thm 1 + Thm 4  +  constructive SNF double-annihilator over Z_d  →  detection equivalence
        (composite-d step: proved in outline, corroborated; full write-up owed)
```

## Local valuation (note Thm 1 / V46)
```
parity lemma (c even on commuting pairs, even d)  →  c = 2c' symmetric Z_d-cocycle
        │            H²_sym = Ext, additive over cyclic decomposition
        ▼
cyclic transgression Σ c'(jg,g) divisible by ord(g)  →  trivializer in μ_d (even) / μ_{2d} (odd, sharp d=3)
```

## Paper A — canonical stack class  ⚠ (largest unaudited node)
```
central extension 1→U(1)→U(n)→PU(n)→1
        │  vacuity on U(n)⋉X ;   Morita reduction to PN(T)   **needs named Morita theorem + hypotheses**
        ▼
canonical class on PU(n)⋉X
        ├──► order exactly n   ←  S_n-equivariant homomorphism lemma  (thmG_general.py)   ⚠ geometry-to-lemma
        │                          reduction + exclusion of non-module splittings NOT fully written
        ├──► real shadow B_n  →  even/odd dichotomy (splits iff n odd)
        ├──► n=4 Pauli slice  →  PM pairing = -1  = obstruction
        └──► H²(PN(T);U(1)) = Z/n ⊕ Z₂   ←  LHS spectral sequence   ⚠ differentials/extensions to be checked
```

## Paper D — observer category
```
observer-context CATEGORY (switch subgroupoid + non-invertible refinements)
        ├──► Thm 1 relative facts  ←  local classicality (V23) + pullback monotonicity (V38)
        ├──► Thm 2 hol² Z₂ invariant (V38)      [necessity, not equivalence]
        ├──► Thm 3 movable cut  =  prob-invariance ⊕ section-phase ⊕ class-invariance  [+ interpretation]
        │            update rule  ←  V37 affine Lüders line + CP interval
        └──► Thm 4 FR localized  ←  fr_cycle.py  +  CF = 1/6 (independent LP)
```

## Cross-cutting computational lemmas (independently re-checked in `independent_verification/`)
- Weyl multiplication ⇔ symplectic commutation
- value-bit ∈ {0, d/2}
- CP interval −1/(r²−1) ≤ p ≤ 1
- sharp-core qutrit POVM
- FR contextual fraction 1/6

**Reading:** the finite Weyl/AvN core (Paper B, detection equivalence, local valuation, S4 instrument
layer) has the shortest, best-corroborated dependency chains. The geometric layer of Paper A is the
node most in need of an independent specialist proof audit.
