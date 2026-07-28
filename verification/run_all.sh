#!/usr/bin/env bash
# Run every canonical verification and report pass/fail.
# Exits nonzero if any script fails, so CI catches regressions.
#
# A script counts as passing only if ALL of these hold: exit code 0, its expected verdict token
# present in stdout, no failure token present, and (first entry) the suite is import-safe.
# Exit code alone is not sufficient -- a script can print a negative result for its own premise,
# contain no assertion, and still exit 0. A verdict token alone is not sufficient either, unless
# it is attributable to the script being run; see check_import_safety.py.
set -uo pipefail
cd "$(dirname "$0")"

# "script|regex that MUST appear in the output for the run to count as a pass"
SCRIPTS=(
  "check_import_safety|check_import_safety PASS"
  "shadow_soundness_exact|shadow_soundness_exact PASS"
  "verify_cert8|ALL CHECKS PASS"
  "verify_cert16|VERDICT: PASS"
  "spectrum_test2|spectrum_test2 PASS"
  "criterion|criterion PASS"
  "Wformula|Wformula PASS"
  "close_T2_proof|^PASS$"
  "tmix_dindep|tmix_dindep PASS"
  "arf_global|arf_global PASS"
  "thmG_general|thmG_general PASS"
  "lueders_cp_interval|V37b PASS"
  "lueders_instrument|^PASS$"
  "s4_povm_bayes|^PASS$"
  "paperA_evenN|paperA_evenN PASS"
  "acc_correspondence|ACC-correspondence PASS"
  "d3_gap_certificate|GAP CERTIFIED"
  "holonomy_vs_solvability|pinned cert4 d=4: unsolvable=True, odd-cycle=True, agree=True"
  "evend_frame_probe|PASS\(E4\)"
  "ghost_facet_theorem|PASS\(completion\)"
  "paired_frame_construction|PASS\(unified\)"
  "pauli_slice_bridge|PASS: routes AGREE"
)

# any of these in the output is a hard failure even when the exit code is 0
FAILTOKEN='FAILED CHECK|^FAIL$|VERDICT: FAIL|[A-Za-z0-9_-]+ FAIL$|Traceback \(most recent call last\)'

# Optional: `bash run_all.sh spectrum_test2 criterion` runs only the named scripts
# (handy for chunked / time-boxed reruns). No argument = the full canonical suite.
want_only=" $* "

pass=0; tot=0; failed=""
for entry in "${SCRIPTS[@]}"; do
  s="${entry%%|*}"; want="${entry#*|}"
  if [ "$#" -gt 0 ] && [ "${want_only#* $s }" = "$want_only" ]; then continue; fi
  tot=$((tot+1))
  tmp=$(mktemp)
  rc=0
  timeout 300 python3 "$s.py" >"$tmp" 2>&1 || rc=$?
  why=""
  if [ "$rc" -ne 0 ]; then
    why="nonzero exit ($rc)"
  elif ! grep -Eq "$want" "$tmp"; then
    why="MISSING VERDICT TOKEN: expected /$want/ (silent script, or the verdict changed)"
  elif grep -Eq "$FAILTOKEN" "$tmp"; then
    why="FAILURE TOKEN in output: $(grep -Eo "$FAILTOKEN" "$tmp" | head -1)"
  fi
  if [ -z "$why" ]; then
    echo "  ok   $s   [verdict: $(grep -Eo "$want" "$tmp" | head -1)]"; pass=$((pass+1))
  else
    echo "  FAIL $s   <-- $why"
    failed="$failed $s"
    sed 's/^/       | /' "$tmp"   # surface the failing output for debugging
  fi
  rm -f "$tmp"
done
echo "== $pass/$tot canonical verifications passed (exit code AND verdict token checked) =="
if [ "$pass" -ne "$tot" ]; then
  echo "== FAILING:$failed =="
  exit 1
fi
exit 0
