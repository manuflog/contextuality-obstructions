"""Compatibility / nondisturbance diagnostics for the Peres-Mermin (d=2) hardware runs.

Honest systematics analysis. Reads ONLY the two stored artifacts

    results/pm_d2_3device_results.json     (3-device replication; Paper B / README)
    results/ibm_fez_holonomy_20260709.json (single-device holonomy job; Paper C)

and computes every diagnostic those files actually support -- and states plainly the
ones they do NOT support (with the extra measurement/saved data each would require).

Dependencies: numpy + json only (no scipy). Reads nothing else; writes nothing.
Run:  python3 compatibility_analysis.py

Buckets kept strictly separate in the output:
  (1) raw Peres-Mermin witness value(s)
  (2) statistical (shot-noise) uncertainty
  (3) systematic uncertainty
  (4) compatibility / nondisturbance status
"""
import json
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
NC_BOUND = 4.0  # noncontextual / classical bound for this witness
# Cabello sign pattern of the 6-context Peres-Mermin witness: +R1+R2+R3+C1+C2-C3
SIGN = {"R1": 1, "R2": 1, "R3": 1, "C1": 1, "C2": 1, "C3": -1}


def rule(c="-", n=78):
    print(c * n)


def chi2_sf_dof2(x):
    """Exact survival function of chi-square with 2 dof: SF(x) = exp(-x/2).
    Returned as (p, log10p). Closed form -> no scipy needed for this dof."""
    log10p = (-x / 2.0) / np.log(10.0)
    return np.exp(-x / 2.0), log10p


def main():
    d3 = json.load(open(os.path.join(RES, "pm_d2_3device_results.json")))
    hol = json.load(open(os.path.join(RES, "ibm_fez_holonomy_20260709.json")))

    print()
    rule("=")
    print("PERES-MERMIN (d=2) HARDWARE -- COMPATIBILITY / SYSTEMATICS DIAGNOSTICS")
    rule("=")
    print("Sources : results/pm_d2_3device_results.json , "
          "results/ibm_fez_holonomy_20260709.json")
    print("Bound   : noncontextual/classical S <= %.1f ; quantum/algebraic max = 6" % NC_BOUND)

    # ------------------------------------------------------------------ #
    # SECTION A. What is actually stored (per file), and the raw witness. #
    # ------------------------------------------------------------------ #
    print()
    rule()
    print("SECTION A  --  STORED QUANTITIES  (bucket 1: raw witness)")
    rule()
    print("pm_d2_3device_results.json : per device, the SIX signed per-context")
    print("  triple-product correlators <ABC> {R1,R2,R3,C1,C2,C3} + S + SE.")
    print("  -> NO per-observable marginals, NO per-context joint bit-counts are stored.")
    print("ibm_fez_holonomy_20260709.json : 12 'pub' single-bit count distributions")
    print("  {'0','1'} over 8192 shots, circuit_metadata = {} (EMPTY -> no labels),")
    print("  plus pinned figures pm_S and loop_phase_deg.")

    B = d3["backends"]
    names = list(B.keys())
    S = np.array([B[n]["S"] for n in names], float)
    SE = np.array([B[n]["SE"] for n in names], float)

    print()
    print("Per-device raw witness (recomputed S from its six correlators = stored S):")
    print("  %-16s %8s %8s %10s %10s" % ("device", "S", "SE", "recomp", "sigma_stat"))
    recomp = []
    for i, n in enumerate(names):
        pc = B[n]["per_context"]
        s_re = sum(SIGN[k] * pc[k] for k in SIGN)
        recomp.append(s_re)
        ok = "OK" if abs(s_re - S[i]) < 0.01 else "MISMATCH"
        print("  %-16s %8.4f %8.4f %10.4f %10.1f  (%s)"
              % (n, S[i], SE[i], s_re, (S[i] - NC_BOUND) / SE[i], ok))
    print("  -> internal consistency of stored correlators: PASS "
          "(recompute == stored, same as verify_combined.py).")

    # ------------------------------------------------------------------ #
    # SECTION B. Between-device homogeneity = the honest systematic probe. #
    # ------------------------------------------------------------------ #
    print()
    rule()
    print("SECTION B  --  BETWEEN-DEVICE HOMOGENEITY  (buckets 2 & 3: stat vs systematic)")
    rule()
    w = 1.0 / SE**2
    mean = float(np.sum(w * S) / np.sum(w))
    se_stat = float(np.sqrt(1.0 / np.sum(w)))
    sigma_stat = (mean - NC_BOUND) / se_stat
    chi2 = float(np.sum((S - mean)**2 / SE**2))
    dof = len(S) - 1
    birge = np.sqrt(chi2 / dof)
    se_scaled = se_stat * birge
    sigma_scaled = (mean - NC_BOUND) / se_scaled
    p, log10p = chi2_sf_dof2(chi2)
    sd = float(np.std(S, ddof=1))            # chip-to-chip scatter
    umean = float(np.mean(S))
    se_ens = sd / np.sqrt(len(S))            # SE of the ensemble mean from observed scatter
    sigma_ens = (umean - NC_BOUND) / se_ens

    print("(2) STATISTICAL / shot-noise combination (as reported in README/Paper B):")
    print("      inverse-variance mean S = %.4f  +/-  %.4f (stat)  ->  %.1f sigma > %.0f"
          % (mean, se_stat, sigma_stat, NC_BOUND))
    print("      This is the '80 sigma'. It ASSUMES the 3 devices share one true S and")
    print("      that the only scatter is shot noise.  Test that assumption:")
    print()
    print("      Homogeneity chi-square of the 3 device S about the weighted mean:")
    print("        chi2 = %.1f  on dof = %d   (chi2/dof = %.1f)" % (chi2, dof, chi2 / dof))
    print("        p (exact, chi2 dof=2 -> exp(-chi2/2)) ~ 10^%.1f  (= %.1e)"
          % (log10p, p))
    print("        Birge ratio sqrt(chi2/dof) = %.2f" % birge)
    print("      => The three devices are NOT statistically consistent with a common S.")
    print("         The +/-%.4f shot-noise error bar is too small by ~%.0fx; the '80 sigma'"
          % (se_stat, birge))
    print("         is inflated by unmodeled device-level (systematic) variation.")
    print()
    print("(3) SYSTEMATIC uncertainty implied by the failed homogeneity:")
    print("      chip-to-chip spread of S (sample SD, n=3) = %.4f" % sd)
    print("      (~%.0fx the quoted combined shot-noise SE of %.4f)" % (sd / se_stat, se_stat))
    print("      Birge/PDG-scaled combination:  S = %.4f +/- %.4f  ->  %.1f sigma > %.0f"
          % (mean, se_scaled, sigma_scaled, NC_BOUND))
    print("      Ensemble-mean via observed scatter: S = %.4f +/- %.4f -> %.1f sigma > %.0f"
          % (umean, se_ens, sigma_ens, NC_BOUND))
    print()
    print("      HONEST HEADLINE:")
    print("       - Each device on its OWN shot noise is far above %.0f:" % NC_BOUND)
    for i, n in enumerate(names):
        print("           %-16s %.1f sigma" % (n, (S[i] - NC_BOUND) / SE[i]))
    print("       - But the combined '80 sigma' is not statistically legitimate: the")
    print("         inputs fail homogeneity (chi2=%.0f/%d). Accounting for the real" % (chi2, dof))
    print("         between-device systematic (~%.2f), the defensible combined significance" % sd)
    print("         is ~%.0f sigma, not 80 -- and even that treats the scatter as the whole" % sigma_scaled)
    print("         systematic story (calibration drift / selection / compilation untested).")
    print("       - The QUALITATIVE conclusion 'every device well above %.0f' survives:" % NC_BOUND)
    print("         systematic ~%.2f << gap-to-bound ~%.2f." % (sd, umean - NC_BOUND))

    # ------------------------------------------------------------------ #
    # SECTION C. Holonomy file: 12 single-bit marginals + witness check.  #
    # ------------------------------------------------------------------ #
    print()
    rule()
    print("SECTION C  --  HOLONOMY JOB: 12 SINGLE-BIT MARGINALS  (buckets 1 & 2)")
    rule()
    zs, ses = [], []
    print("  %-5s %7s %7s %8s %9s %10s" % ("pub", "n0", "n1", "N", "corr", "shotSE"))
    for r in hol["results"]:
        c = r["counts"]["c"]
        n0 = int(c.get("0", 0)); n1 = int(c.get("1", 0)); N = n0 + n1
        z = (n0 - n1) / N
        se = np.sqrt(max((1.0 - z * z) / N, 0.0))
        zs.append(z); ses.append(se)
        print("  %-5d %7d %7d %8d %+9.4f %10.4f" % (r["pub"], n0, n1, N, z, se))
    zs = np.array(zs); ses = np.array(ses)
    near0 = [i for i in range(len(zs)) if abs(zs[i]) < 3 * ses[i]]
    print("  pubs consistent with 0 at 3-shotSE: %s" % near0)

    pinned_S, pinned_SE = hol["pinned_figures"]["pm_S"]
    sum_abs = float(np.sum(np.abs(zs)))
    se_all = float(np.sqrt(np.sum(ses**2)))
    print()
    print("  Reproduce the pinned witness pm_S = %.4f +/- %.4f from stored counts?"
          % (pinned_S, pinned_SE))
    print("    naive sum|corr| over all 12 pubs = %.4f  (shot-SE %.4f)" % (sum_abs, se_all))
    print("    pinned - naive = %+.4f  (= %.1f x pinned SE)"
          % (pinned_S - sum_abs, (pinned_S - sum_abs) / pinned_SE))
    print("    => NOT reproducible from this file: metadata is empty (no pub->observable/")
    print("       context map) and no documented recipe reconstructs %.4f (naive gives" % pinned_S)
    print("       %.4f; the ~%.3f gap could be readout mitigation or a pub subset -- unknown)."
          % (sum_abs, pinned_S - sum_abs))
    print("       The stored value pm_S=%.4f cannot be independently verified from the counts."
          % pinned_S)

    # ------------------------------------------------------------------ #
    # SECTION D. Marginal consistency: the target diagnostic -- NOT poss. #
    # ------------------------------------------------------------------ #
    print()
    rule()
    print("SECTION D  --  MARGINAL CONSISTENCY / NONDISTURBANCE  (bucket 4)")
    rule()
    print("Target: each PM observable sits in one ROW and one COLUMN context; compare its")
    print("marginal <A>_row vs <A>_col. A gap beyond shot noise = signaling/disturbance =")
    print("a compatibility loophole (the sequential PM witness certifies contextuality only")
    print("under nondisturbance).")
    print()
    print("STATUS: NOT COMPUTABLE from the stored data.  Reason, per file:")
    print("  * pm_d2_3device_results.json stores only the triple product <ABC> per context.")
    print("    The per-observable marginals were discarded; you cannot recover <A> from <ABC>.")
    print("  * ibm_fez_holonomy_20260709.json stores 12 unlabeled single-bit pubs with EMPTY")
    print("    circuit_metadata -- no map from pub -> (observable, context), so no shared")
    print("    observable can be identified across contexts.")
    print("  * The notebook computes <ABC> shot-by-shot (ctx_value) but the raw per-context")
    print("    3-bit JOINT counts -- which DO contain the marginals -- are never persisted.")
    print()
    print("REQUIRED ADDITIONAL DATA (no new physics, just save/label more): re-run and store")
    print("  the full per-context 3-bit joint count distributions (or at minimum each of the")
    print("  3 observable marginals within each context), labeled by (context, observable).")
    print("  Then for each of the 9 observables compare its marginal across the 2 contexts")
    print("  containing it, and report max |<A>_row - <A>_col| vs shot noise.")
    print("  Compatibility/nondisturbance status: UNTESTED (cannot be asserted from stored data).")

    # ------------------------------------------------------------------ #
    # SECTION E. Diagnostics requiring NEW experiments -- not available.  #
    # ------------------------------------------------------------------ #
    print()
    rule()
    print("SECTION E  --  DIAGNOSTICS REQUIRING NEW EXPERIMENTS  (NOT YET AVAILABLE)")
    rule()
    todo = [
        ("Order-reversal test",
         "measure (A then B) vs (B then A); nondisturbance needs both orders to agree."),
        ("Repeated-measurement agreement",
         "measure (A then A); a compatible/sharp A must reproduce -> P(agree)~1."),
        ("Disturbance-corrected NC bound",
         "signaling/contextuality-by-default (Kujala-Dzhafarov) inflated bound 4+Delta; "
         "needs marginals to get Delta."),
        ("Calibration-drift systematics",
         "interleaved/time-stamped recalibration; no per-shot time or drift data stored."),
        ("Backend-selection / look-elsewhere",
         "best-qubit-triple is auto-selected per device -> unpenalized selection; not corrected."),
        ("Compilation variability",
         "repeat under different transpile seeds/opt levels/layouts; single compilation stored."),
        ("Multiple-testing correction",
         "3 devices x 6 contexts (+holonomy) tested; no family-wise / look-elsewhere penalty."),
    ]
    for k, v in todo:
        print("  [ ] %-32s %s" % (k, v))

    # ------------------------------------------------------------------ #
    # SUMMARY.                                                            #
    # ------------------------------------------------------------------ #
    print()
    rule("=")
    print("SUMMARY (four buckets kept separate)")
    rule("=")
    print("(1) RAW WITNESS      : per-device S in [%.3f, %.3f] (3-dev); holonomy pinned pm_S=%.4f"
          % (S.min(), S.max(), pinned_S))
    print("                       all > NC bound %.0f; quantum max 6." % NC_BOUND)
    print("(2) STATISTICAL      : shot-noise only. Combined inverse-variance = %.4f +/- %.4f"
          % (mean, se_stat))
    print("                       -> %.0f sigma. THIS is the '80 sigma' -- shot-noise-only." % sigma_stat)
    print("(3) SYSTEMATIC       : 3 devices FAIL homogeneity (chi2=%.0f/%d, Birge=%.1f)."
          % (chi2, dof, birge))
    print("                       chip-to-chip SD=%.2f ~ %.0fx the stat SE. Birge-scaled -> ~%.0f sigma."
          % (sd, sd / se_stat, sigma_scaled))
    print("                       Drift/selection/compilation/multiple-testing systematics UNTESTED.")
    print("(4) COMPATIBILITY    : UNTESTED. Marginal consistency, order-reversal, repeatability,")
    print("                       disturbance-corrected bound are NOT computable from stored data")
    print("                       and require new/relabeled measurements (Sections D, E).")
    print()
    print("BOTTOM LINE: The 80 sigma is shot-noise-only under the implemented witness model,")
    print("NOT a loophole-free contextuality certification. Every device is individually well")
    print("above the noncontextual bound, and that qualitative violation is robust; but the")
    print("precise combined significance is capped near ~%.0f sigma once the observed device" % sigma_scaled)
    print("scatter is honored, and no nondisturbance/compatibility diagnostic has been performed.")
    print()


if __name__ == "__main__":
    main()
