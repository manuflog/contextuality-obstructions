#!/usr/bin/env bash
# Run every canonical verification and report pass/fail.
set -u; cd "$(dirname "$0")"; pass=0; tot=0
for s in verify_cert8 verify_cert16 spectrum_test2 criterion Wformula close_T2_proof tmix_dindep arf_global thmG_general; do
  tot=$((tot+1))
  if timeout 120 python3 "$s.py" >/dev/null 2>&1; then echo "  ok   $s"; pass=$((pass+1)); else echo "  FAIL $s"; fi
done
echo "== $pass/$tot canonical verifications passed =="
