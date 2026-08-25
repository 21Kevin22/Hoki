#!/usr/bin/env bash
# occ_vla addition (2026-08-25): reproduces the vendored LIBERO + LIBERO-Occ
# checkouts the openvla-oft occluded-suite scripts (run_libero_occluded_oracle_headroom.py
# et al.) depend on, pinned to the exact commits this project was developed/
# validated against. Run once after a fresh `git clone` of this repo, before
# any occ_vla/scripts/run_libero_occluded_oracle_headroom.py invocation.
#
# Not the same script as occ_vla/scripts/setup_third_party.sh, which sets up
# the SIBLING pi0.5/MMaDA project's own third_party/{openpi,mmada} deps under
# a different directory (third_party/, not occ_vla/thirdparty/) -- unrelated
# to this project.
set -euo pipefail
cd "$(dirname "$0")/../thirdparty"

LIBERO_COMMIT=8f1084e3132a39270c3a13ebe37270a43ece2a01
LIBERO_OCC_COMMIT=25cc040025c5001d75a5bfb3fd3bae1759d887b0

if [ ! -d LIBERO/.git ]; then
    git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git LIBERO
    git -C LIBERO checkout "$LIBERO_COMMIT"
fi

if [ ! -d Libero-Occ/.git ]; then
    git clone https://github.com/litsh/Libero-Occ.git Libero-Occ
    git -C Libero-Occ checkout "$LIBERO_OCC_COMMIT"
fi

# Copies benchmark_assets/{bddl_files,init_files}/libero_*_occluded into the
# fresh LIBERO checkout -- the exact prerequisite
# scripts/register_libero_occ_suites.py's own docstring already documents.
LIBERO_ROOT="$(pwd)/LIBERO" bash Libero-Occ/scripts/setup/install_libero_occ_assets.sh

echo "Done -- LIBERO ($LIBERO_COMMIT) + LIBERO-Occ ($LIBERO_OCC_COMMIT) assets installed into thirdparty/LIBERO."
echo ""
echo "Still needed manually (NOT reproduced by this script):"
echo "  - Python env (occ_vla/.venv_openvla_oft, uv-managed): try 'uv venv"
echo "    occ_vla/.venv_openvla_oft --python 3.10 && uv pip install -e occ_vla'"
echo "    from occ_vla/pyproject.toml (not verified end-to-end -- no uv.lock"
echo "    exists). If that doesn't reproduce cleanly, rsync/scp the already-"
echo "    built .venv_openvla_oft (~11GB) directly instead of debugging a"
echo "    fresh resolve."
echo "  - Checkpoints under occ_vla/checkpoints/ -- openvla-7b-oft-libero10-vjepa"
echo "    (~15GB) is a LOCALLY FINE-TUNED checkpoint from this project's own"
echo "    vjepa_predictor work, not on any public hub. Must be copied/rsynced"
echo "    from the machine that trained it."
