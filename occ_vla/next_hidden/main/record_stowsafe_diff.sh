#!/usr/bin/env bash
# Record StowSafe v5 as a DIFF against the frozen v4 (the code itself stays outside this public repo) + fingerprints.
# Usage: bash main/record_stowsafe_diff.sh   -> main/stowsafe_v5.patch, main/stowsafe_fingerprints.txt
set -euo pipefail
cd "$(dirname "$0")/.."
TP=/home/ubuntu/slocal1/third_party
fp() { (cd "$1" && find stowsafe integration -name "*.py" -not -path "*__pycache__*" | sort | xargs sha256sum | sha256sum | cut -c1-64); }
(cd "$TP" && diff -ruN -x __pycache__ -x .pytest_cache -x results -x "*.pyc" stowsafe_kit_v4/stowsafe stowsafe_kit_v5/stowsafe || true) > main/stowsafe_v5.patch
{ echo "v4 (frozen, unchanged): $(fp $TP/stowsafe_kit_v4/stowsafe)"
  echo "v5 (working copy):      $(fp $TP/stowsafe_kit_v5/stowsafe)"
  echo "patch sha256:           $(sha256sum main/stowsafe_v5.patch | cut -c1-64)"
  echo "recorded:               $(date -Is)"; } > main/stowsafe_fingerprints.txt
cat main/stowsafe_fingerprints.txt; grep -c '^+++' main/stowsafe_v5.patch | xargs echo "files changed:"
