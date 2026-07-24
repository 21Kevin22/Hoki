"""Runs in .venv_mmada. Like mmada_worker.py, but loads the trained
arm-removal LoRA adapter (scripts/train_arm_removal_lora_tiny.py's
output, arm_removal_lora_tiny_adapter/) on top of the base MMaDA-8B
checkpoint before serving -- a preview-evaluation worker, not meant to
replace mmada_worker.py. Uses its own RPC dir (.rpc/mmada_lora) so it
can run alongside (or instead of) the plain worker without colliding.

Inference-only, so no 4-bit/gradient tricks needed here (those were
specifically for fitting backward-pass memory during training) -- loads
bf16 like the plain worker, then wraps with peft.PeftModel.
"""

import os
import sys
from pathlib import Path

from peft import PeftModel

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import rpc  # noqa: E402

from occ_vla.world_model.arm_free_subgoal import ArmFreeSubgoalGenerator  # noqa: E402
from occ_vla.world_model.mmada import MMaDAWorldModel  # noqa: E402
from occ_vla.world_model.tokenizer import MagvitV2Tokenizer  # noqa: E402

RPC_DIR = os.environ.get("MMADA_LORA_WORKER_RPC_DIR", str(_ROOT / ".rpc" / "mmada_lora"))
# Default (tiny, r=8) unchanged for backward compat; override to the
# full r=16 adapter (arm_removal_lora_full_adapter/) via env var.
# Set to "none" (2026-07-21) to skip loading any adapter at all -- runs
# the bare base MMaDA-8B-MixCoT checkpoint, to attribute the full-frame
# "subgoal" mode's abstract-color-field failure to the arm-removal
# LoRA's narrow specialization (catastrophic forgetting) vs. a
# limitation of the base model itself.
ADAPTER_PATH = os.environ.get("MMADA_LORA_ADAPTER_PATH", str(_ROOT / "arm_removal_lora_tiny_adapter"))


def main():
    tokenizer = MagvitV2Tokenizer(checkpoint_path="showlab/magvitv2", device="cuda:0")
    tokenizer.load()
    world_model = MMaDAWorldModel(checkpoint_path="Gen-Verse/MMaDA-8B-MixCoT", tokenizer=tokenizer, device="cuda:0")
    world_model.load()
    if ADAPTER_PATH.lower() != "none":
        world_model._model = PeftModel.from_pretrained(world_model._model, ADAPTER_PATH)  # noqa: SLF001
    generator = ArmFreeSubgoalGenerator(world_model)
    print(f"[mmada-lora worker] models loaded (adapter={ADAPTER_PATH}), serving {RPC_DIR}", flush=True)

    def handler(arrays, fields):
        # mode="subgoal" (2026-07-21, per user decision to move on from
        # arm-removal inpainting): full-frame future-state generation,
        # no arm mask/current image involved at all. Default unchanged
        # for backward compat with existing arm-removal callers.
        if fields.get("mode") == "subgoal":
            result = generator.sample_subgoal_image(
                instruction=fields["instruction"],
                horizon=fields.get("horizon", 5),
            )
        else:
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
