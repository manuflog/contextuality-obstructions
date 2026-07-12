#!/usr/bin/env bash
# Run every canonical verification and report pass/fail.
# Exits nonzero if any script fails, so CI catches regressions.
set -uo pipefail
cd "$(dirname "$0")"
pass=0; tot=0
for s in verify_cert8 verify_cert16 spectrum_test2 criterion Wformula close_T2_proof \
         tmix_dindep arf_global thmG_general lueders_cp_interval lueders_instrument s4_povm_bayes paperA_evenN; do
  tot=$((tot+1))
  tmp=$(mktemp)
  if timeout 120 python3 "$s.py" >"$tmp" 2>&1; then
    echo "  ok   $s"; pass=$((pass+1))
  else
    echo "  FAIL $s"; sed 's/^/       | /' "$tmp"   # surface the failing output for debugging
  fi
  rm -f "$tmp"
done
echo "== $pass/$tot canonical verifications passed =="
[ "$pass" -eq "$tot" ] || exit 1
exit 0
