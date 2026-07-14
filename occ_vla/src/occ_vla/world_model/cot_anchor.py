"""Chain-of-thought logical anchor, used when visual information is
completely unavailable (not just self-occluded). Provides a text-level
prediction of the expected next state to keep the policy from stalling.

Grounded on MMadaModelLM.mmu_generate (models/modeling_mmada.py:377),
MMaDA's text-generation path (block-wise diffusion decoding, distinct
from t2i_generate's image-token unmasking used in arm_free_subgoal.py).
Stage-3/4 checkpoints in third_party/mmada/configs/mmada_pretraining_stage3_llada_instruct_512_cot.yaml
are the CoT-tuned variants this is meant to run against.
"""

from occ_vla.world_model.mmada import MMaDAWorldModel


class CotAnchorGenerator:
    def __init__(self, world_model: MMaDAWorldModel):
        self.world_model = world_model

    def generate(self, instruction: str, history: list[str]) -> str:
        """Returns a natural-language anchor, e.g. "the pot handle should
        now be under the gripper"."""
        prompt = "\n".join([*history, instruction])
        return self.world_model.generate_text(prompt)
