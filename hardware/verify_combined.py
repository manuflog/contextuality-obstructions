"""Independently recompute the 3-device Peres-Mermin (d=2) combined witness from results/.
Each device S is recomputed from its six per-context correlators; devices are combined by
inverse-variance weighting. Reproduces the value reported in Paper B, section 'Hardware validation'."""
import json, math, os
d = json.load(open(os.path.join(os.path.dirname(__file__), "results", "pm_d2_3device_results.json")))
B = d["backends"]; Ss = []
print(f"{'device':16}{'S':>8}{'SE':>8}{'sigma>4':>9}")
for name, b in B.items():
    S, SE = b["S"], b["SE"]; pc = b["per_context"]
    # sign pattern of the Cabello witness: +R1 +R2 +R3 +C1 +C2 -C3
    Sre = pc["R1"]+pc["R2"]+pc["R3"]+pc["C1"]+pc["C2"]-pc["C3"]
    assert abs(Sre - S) < 0.01, (name, Sre, S)
    Ss.append((S, SE)); print(f"{name:16}{S:>8.3f}{SE:>8.4f}{(S-4)/SE:>9.1f}")
num = sum(S/SE**2 for S, SE in Ss); den = sum(1/SE**2 for S, SE in Ss)
Sc, SEc = num/den, math.sqrt(1/den)
print(f"\nCOMBINED (inverse-variance weighted): S = {Sc:.4f} +/- {SEc:.4f}  ({(Sc-4)/SEc:.1f} sigma > NC bound 4)")
print("Each device's S recomputed from its six per-context correlators: all consistent.")
