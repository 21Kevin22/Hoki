"""Runs in third_party/openpi/.venv (has torch+torchvision; the base/
system Python that drives LIBERO does not -- see occ_vla/CLAUDE.md's
"three incompatible dependency stacks" note). Loads RAFT once, then
serves PKLP's three-frame patch-flow calls over the file-based RPC in
rpc.py, the same pattern as pi05_worker.py / mmada_worker.py.
"""

import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import rpc  # noqa: E402

from occ_vla.pklp.optical_flow import RaftFlowEstimator  # noqa: E402

RPC_DIR = os.environ.get("RAFT_WORKER_RPC_DIR", str(_ROOT / ".rpc" / "raft"))


def main():
    estimator = RaftFlowEstimator(variant="raft_large")
    estimator.load()
    print(f"[raft worker] model loaded, serving {RPC_DIR}", flush=True)

    def handler(arrays, fields):
        flow_earlier, flow_latest = estimator.three_frame_patch_flow(
            arrays["frame_t2"], arrays["frame_t1"], arrays["frame_t0"]
        )
        resp_arrays = {
            "patch_centers": flow_latest.patch_centers,
            "flow_earlier": flow_earlier.flow,
            "flow_latest": flow_latest.flow,
        }
        resp_fields = {"grid_rows": flow_latest.grid_shape[0], "grid_cols": flow_latest.grid_shape[1]}
        return resp_arrays, resp_fields

    rpc.serve(RPC_DIR, handler)


if __name__ == "__main__":
    main()
