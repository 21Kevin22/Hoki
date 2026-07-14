#!/usr/bin/env bash
# Clones the two reference implementations this project builds on, into
# third_party/. Not committed (see .gitignore) — rerun this after a
# fresh checkout instead.
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p third_party

if [ ! -d third_party/openpi/.git ]; then
    git clone --depth 1 https://github.com/Physical-Intelligence/openpi.git third_party/openpi
    # third_party/libero inside openpi is what LIBERO-Occ (src/occ_vla/eval) wraps
    git -C third_party/openpi submodule update --init third_party/libero
fi

if [ ! -d third_party/mmada/.git ]; then
    git clone --depth 1 https://github.com/gen-verse/mmada.git third_party/mmada
fi

echo "Done. See README.md for how these are installed/used."
