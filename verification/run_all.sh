#!/usr/bin/env bash
# Run every canonical verification and report pass/fail.
# Exits nonzero if any script fails, so CI catches regressions.
#
# HARDENED 2026-07-24. Exit code alone is NOT enough: a script can print "P1: False"
# for its own load-bearing premise, contain no assert, and still exit 0 -- which is
# exactly how close_T2_proof.py sat inside a suite reporting "14/14 passed".
# Each script must now ALSO print its expected verdict token on stdout, and must NOT
# print any failure token. All three conditions are checked; any one failing is loud.
set -uo pipefail
cd "$(dirname "$0")"

# "script|regex that MUST appear in the output for the run to count as a pass"
SCRIPTS=(
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
