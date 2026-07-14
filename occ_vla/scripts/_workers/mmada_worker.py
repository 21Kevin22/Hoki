"""Runs in .venv_mmada. Loads MMaDA-8B + MAGVIT-v2 once, then serves
ArmFreeSubgoalGenerator.sample_arm_free_image() calls over the
file-based RPC in rpc.py."""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import rpc  # noqa: E402

from occ_vla.world_model.arm_free_subgoal import ArmFreeSubgoalGenerator  # noqa: E402
from occ_vla.world_model.mmada import MMaDAWorldModel  # noqa: E402
from occ_vla.world_model.tokenizer import MagvitV2Tokenizer  # noqa: E402

RPC_DIR = "/tmp/occ_vla_rpc/mmada"


def main():
    tokenizer = MagvitV2Tokenizer(checkpoint_path="showlab/magvitv2", device="cuda:0")
    tokenizer.load()
    world_model = MMaDAWorldModel(checkpoint_path="Gen-Verse/MMaDA-8B-MixCoT", tokenizer=tokenizer, device="cuda:0")
    world_model.load()
    generator = ArmFreeSubgoalGenerator(world_model)
    print("[mmada worker] models loaded", flush=True)

    def handler(arrays, fields):
        result = generator.sample_arm_free_image(
            image=arrays["image"],
            arm_pixel_mask=arrays["arm_pixel_mask"].astype(bool),
            instruction=fields["instruction"],
            horizon=fields.get("horizon", 5),
        )
        return {"image": result.image}, {}

    rpc.serve(RPC_DIR, handler)


if __name__ == "__main__":
    main()
