# Compatibility / systematics analysis — Peres–Mermin ($d=2$) hardware

Honest audit of what the **stored** hardware artifacts do and do not establish about
contextuality. Reproduce every number below with:

```bash
cd hardware && python3 compatibility_analysis.py   # numpy + json only
```

Inputs (read-only): `results/pm_d2_3device_results.json` (3-device replication, Paper B / README)
and `results/ibm_fez_holonomy_20260709.json` (single-device holonomy job, Paper C).
Noncontextual/classical bound $S\le 4$; quantum/algebraic maximum $6$.

The four things are kept **strictly separate**: (1) raw witness, (2) statistical (shot-noise)
uncertainty, (3) systematic uncertainty, (4) compatibility/nondisturbance status.

---

## 0. What the stored data actually contains (this determines what is testable)

- **`pm_d2_3device_results.json`** — per device: the **six signed per-context triple-product
  correlators** $\langle ABC\rangle$ for `{R1,R2,R3,C1,C2,C3}`, plus `S` and `SE`. **No
  per-observable marginals and no per-context joint bit-counts are stored.**
- **`ibm_fez_holonomy_20260709.json`** — **12 "pub" single-bit count distributions** (`{"0","1"}`
  over 8192 shots each), with `circuit_metadata = {}` (**empty — the pubs carry no observable/context
  labels**), plus the pinned figures `pm_S = 4.6125 ± 0.0173` and `loop_phase_deg`.

Consequence, stated up front: **neither file stores the per-observable marginals**, so the central
compatibility diagnostic (same observable's marginal compared across the two contexts containing it)
**cannot be computed from the existing data**. See §4.

---

## 1. Raw witness (bucket 1) — computable, and internally consistent

Each device's $S$ recomputed from its own six correlators reproduces the stored $S$ (same check as
`verify_combined.py`): **internal consistency PASS.**

| device | $S$ | shot-noise SE | $\sigma$ above 4 (shot-noise) |
|---|---|---|---|
| ibm_marrakesh | 4.9370 | 0.0154 | 60.8 |
| ibm_fez | 4.5583 | 0.0176 | 31.7 |
| ibm_kingston | 4.7380 | 0.0166 | 44.5 |

Holonomy job pinned `pm_S = 4.6125 ± 0.0173`. All values exceed the noncontextual bound 4.

---

## 2. Statistical (shot-noise) uncertainty (bucket 2) — the "80σ"

Inverse-variance combination of the three devices:

$$S_\text{comb} = 4.7614 \pm 0.0095\ (\text{stat}) \;\Rightarrow\; 80.1\,\sigma > 4.$$

**This is the "80σ."** It is **shot-noise-only** and assumes the three devices share one true $S$
whose only scatter is Poisson counting noise. That assumption is testable — and it fails (§3).

---

## 3. Systematic uncertainty (bucket 3) — the honest, computable finding

**Between-device homogeneity test** (the one real systematics probe the stored data supports):

- Homogeneity $\chi^2$ of the three $S$ about the weighted mean: **$\chi^2 = 265.2$ on $2$ dof**
  ($\chi^2/\text{dof} = 132.6$; exact $p = e^{-\chi^2/2} \approx 10^{-57.6}$).
- **Birge ratio** $\sqrt{\chi^2/\text{dof}} = 11.5$.

The three devices are **not statistically consistent with a common $S$**: the $\pm0.0095$
shot-noise bar is too small by $\sim\!12\times$. The chip-to-chip spread is a **systematic**, not
statistics:

- chip-to-chip sample SD of $S$ = **0.189** (≈ **20×** the quoted combined SE of 0.0095);
- Birge/PDG error-scaled combination: $S = 4.7614 \pm 0.1094 \Rightarrow \mathbf{7.0\,\sigma} > 4$;
- ensemble mean from the observed scatter: $S = 4.7444 \pm 0.1094 \Rightarrow 6.8\,\sigma > 4$.

**Interpretation (honest):**
- Each device **individually** is far above 4 on its own shot noise (31.7–60.8σ); that per-device
  statement is unaffected by the between-device inconsistency.
- The **combined "80σ" is not statistically legitimate** — its inputs fail homogeneity
  ($\chi^2=265/2$). Once the real between-device systematic (~0.19) is honored, the defensible
  combined significance is **~7σ, not 80σ**, and even that treats the observed scatter as the *whole*
  systematic story.
- The **qualitative** conclusion "every device well above 4" is **robust**: systematic ~0.19 ≪
  gap-to-bound ~0.74.

Other systematics — calibration drift, backend selection / look-elsewhere, compilation variability,
multiple testing — are **not quantified** (§5).

### 3b. The holonomy `pm_S` is not reproducible from its own stored counts

The 12 single-bit marginal correlators (all $N=8192$): four pubs (`1,5,7,10`) are consistent with
zero at 3·shot-SE; the rest are $\approx\!0.46$–$0.65$ in magnitude. Attempting to reproduce the
pinned witness:

- naive $\sum|\text{corr}|$ over all 12 pubs = **4.5662** (shot-SE 0.0340);
- pinned − naive = **+0.0463** (≈ **2.7 × pinned SE**).

With **empty metadata** (no pub → observable/context map) and no documented reconstruction recipe,
**`pm_S = 4.6125` cannot be independently verified from the stored counts** (the ~0.046 gap could be
readout mitigation or a specific pub subset — unknown from the file).

---

## 4. Compatibility / nondisturbance status (bucket 4) — NOT computable from stored data

**Target diagnostic:** each PM observable lives in one row and one column context; compare its
marginal $\langle A\rangle_\text{row}$ vs $\langle A\rangle_\text{col}$. A gap beyond shot noise is
signaling/disturbance — a **compatibility loophole** (the sequential PM witness certifies
contextuality only under nondisturbance).

**Status: NOT COMPUTABLE from the existing artifacts.**
- `pm_d2_3device_results.json` stores only the triple product $\langle ABC\rangle$ per context; the
  per-observable marginals were discarded and cannot be recovered from $\langle ABC\rangle$.
- `ibm_fez_holonomy_20260709.json` stores 12 **unlabeled** single-bit pubs (empty metadata) — no
  shared observable can even be identified across contexts.
- The notebook forms $\langle ABC\rangle$ shot-by-shot (`ctx_value`) but the **raw per-context 3-bit
  joint counts — which contain the marginals — are never persisted.**

**Minimum additional data to compute it (no new physics, just save/label more):** re-run and store
the full per-context 3-bit joint count distributions (or at least each observable's marginal within
each context), labeled by `(context, observable)`. Then for each of the 9 observables compare its
marginal across the two contexts containing it and report $\max|\langle A\rangle_\text{row}-\langle
A\rangle_\text{col}|$ vs shot noise. **Until then, nondisturbance is UNTESTED and cannot be asserted.**

---

## 5. Diagnostics that require NEW experiments — not yet available

None of these are in the stored data:

- **Order-reversal test** — measure $(A\text{ then }B)$ vs $(B\text{ then }A)$; nondisturbance
  requires both orders to agree.
- **Repeated-measurement agreement** — measure $(A\text{ then }A)$; a sharp/compatible $A$ must
  reproduce ($P(\text{agree})\approx1$).
- **Disturbance-corrected noncontextual bound** — the signaling/contextuality-by-default
  (Kujala–Dzhafarov) inflated bound $4+\Delta$; needs the marginals to estimate $\Delta$.
- **Calibration-drift systematics** — interleaved/time-stamped recalibration; no per-shot timing or
  drift data stored.
- **Backend-selection / look-elsewhere** — the best-qubit-triple is auto-selected per device (an
  unpenalized selection); not corrected.
- **Compilation variability** — repeats under different transpile seeds / optimization levels /
  layouts; only a single compilation is stored.
- **Multiple-testing correction** — 3 devices × 6 contexts (+ the holonomy job) were tested; no
  family-wise / look-elsewhere penalty is applied.

---

## Bottom line

- **The 80σ is shot-noise-only under the implemented witness model — not a loophole-free
  contextuality certification.**
- Every device is **individually** well above the noncontextual bound (31.7–60.8σ shot-noise), and
  that **qualitative** violation is robust to the observed systematics.
- But the three devices **fail homogeneity** ($\chi^2 = 265/2$, Birge 11.5), so the combined 80σ is
  not a legitimate statistical statement; the defensible combined significance, once the ~0.19
  between-device systematic is honored, is **~7σ**.
- **No compatibility / nondisturbance diagnostic has been performed** — marginal consistency,
  order-reversal, repeatability, and the disturbance-corrected bound are **not computable from the
  stored data** and require re-running with the per-context marginals saved (§4) plus the new
  experiments in §5.

*Consistent with `hardware/README.md` §"Scope and significance caveat" and `KNOWN_LIMITATIONS.md`
§Hardware; this document supplies the numbers behind those caveats. Generated by
`compatibility_analysis.py`; no existing file was modified.*
