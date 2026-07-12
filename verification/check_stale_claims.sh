#!/usr/bin/env bash
# Fail if active source files contain retracted/obsolete claims OUTSIDE a labeled
# correction/retraction/ledger context. Guards against the recurring stale-claim problem.
set -uo pipefail
cd "$(dirname "$0")/.."
patterns=(
  "PM is tau-odd" "PM is .-odd" "PM contexts are .-odd"
  "integrality is a gauge choice"
  "half tau-steps are genuinely required" "half ..tau. steps are genuinely required"
  "machine-exact at"
  "diag(1,c) is repeatable"
  "repeatable => projective\" is nevertheless false as stated: a pinned"
  "observer-context groupoid" "observer--context groupoid"
  "Three results follow"
)
# files to scan: active sources (papers, note, readmes, scripts) but not this script,
# not INDEX.md's ledger, not note_v2_patch (patch history), not gitignored trees.
mapfile -t files < <(git ls-files '*.tex' '*.md' '*.py' | grep -vE 'check_stale_claims.sh')
bad=0
for pat in "${patterns[@]}"; do
  while IFS= read -r hit; do
    [ -z "$hit" ] && continue
    # allow lines that are clearly historical/retraction ledger text
    if echo "$hit" | grep -qiE 'retract|correction|ledger|superseded|was invalid|was a layout|obsolete|no longer|artifact'; then continue; fi
    echo "STALE: $hit"; bad=1
  done < <(grep -rnI "$pat" "${files[@]}" 2>/dev/null)
done
if [ "$bad" -ne 0 ]; then echo "== stale-claim check FAILED =="; exit 1; fi
echo "== stale-claim check passed =="; exit 0
