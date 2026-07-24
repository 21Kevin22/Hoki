"""Runs in third_party/openpi/.venv. Loads pi0.5 once, then serves
Pi05Policy.step() calls over the file-based RPC in rpc.py."""

import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import rpc  # noqa: E402

from occ_vla.control.pi05_policy import Pi05Observation, Pi05Policy  # noqa: E402


# Not /tmp: this environment has shown /tmp can be cleared out from under
# a long-running process mid-session (killed a worker + lost its RPC
# dir once already) -- .rpc/ persists with the rest of the project.
# Overridable so a second pi0.5 worker (a dedicated GPU, e.g. for
# BASELINE while another GPU pair handles PROPOSED+MMaDA -- see
# run_demo_t08.py::run_parallel) can serve a different RPC dir.
RPC_DIR = os.environ.get("PI05_WORKER_RPC_DIR", str(_ROOT / ".rpc" / "pi05"))
# Off by default -- every experiment before 2026-07-20 used the stock
# path; set to "1" to load with control/occ_vla_policy_config.py's
# OccVlaLiberoInputs wiring instead, so a generated subgoal_image
# actually reaches the model (see PI05_WORKER_RPC_DIR's sibling comment
# in pi05_policy.py for why this is opt-in, not the default).
USE_OCC_VLA_INPUTS = os.environ.get("PI05_WORKER_USE_OCC_VLA_INPUTS", "0") == "1"


def main():
    policy = Pi05Policy(
        config_name="pi05_libero",
        checkpoint_dir=str(Path.home() / ".cache/openpi/openpi-assets/checkpoints/pi05_libero"),
        use_occ_vla_inputs=USE_OCC_VLA_INPUTS,
    )
    policy.load()
    print(f"[pi05 worker] policy loaded (use_occ_vla_inputs={USE_OCC_VLA_INPUTS}), serving {RPC_DIR}", flush=True)

    def handler(arrays, fields):
        obs = Pi05Observation(
            base_image=arrays["base_image"],
            wrist_image=arrays["wrist_image"],
            state=arrays["state"],
            prompt=fields["prompt"],
            subgoal_image=arrays.get("subgoal_image"),
            cot_anchor=fields.get("cot_anchor"),
        )
        actions = policy.step(obs)
        return {"actions": actions}, {}

    rpc.serve(RPC_DIR, handler)


if __name__ == "__main__":
    main()
