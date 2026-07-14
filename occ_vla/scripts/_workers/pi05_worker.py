"""Runs in third_party/openpi/.venv. Loads pi0.5 once, then serves
Pi05Policy.step() calls over the file-based RPC in rpc.py."""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import rpc  # noqa: E402

from occ_vla.control.pi05_policy import Pi05Observation, Pi05Policy  # noqa: E402

RPC_DIR = "/tmp/occ_vla_rpc/pi05"


def main():
    policy = Pi05Policy(
        config_name="pi05_libero",
        checkpoint_dir=str(Path.home() / ".cache/openpi/openpi-assets/checkpoints/pi05_libero"),
    )
    policy.load()
    print("[pi05 worker] policy loaded", flush=True)

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
