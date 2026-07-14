"""Arm-free subgoal generation for self-occlusion.

Triggered when the arm's occupancy of the target region exceeds
ARM_OCC_THRESHOLD. Encodes the current frame to MAGVIT-v2 tokens, masks
only the tokens covering the arm (MASK_TOKEN_ID), and lets MMaDA's
t2i_generate fill them in — i.e. inpainting via the same call the repo
uses for full generation, see mmada.py's docstring.

Plausibility scoring of the result belongs to
integration/uncertainty.py:PlausibilityChecker, not here — this module's
job is only to produce the candidate image.
"""

import math
from dataclasses import dataclass

import cv2
import numpy as np
import torch

from occ_vla.world_model.mmada import MASK_TOKEN_ID, MMaDAWorldModel
from occ_vla.world_model.tokenizer import SUBGOAL_IMAGE_SIDE, SUBGOAL_NUM_TOKENS

ARM_OCC_THRESHOLD = 0.30
TOKEN_GRID_SIDE = int(math.isqrt(SUBGOAL_NUM_TOKENS))  # 32x32 for 1024 tokens
DEFAULT_SUBGOAL_HORIZON = 5  # frames ahead the "success state" is imagined at


@dataclass
class SubgoalResult:
    image: np.ndarray  # arm-free, K-steps-ahead success-state image


class ArmFreeSubgoalGenerator:
    def __init__(self, world_model: MMaDAWorldModel):
        self.world_model = world_model

    def should_trigger(self, arm_s_occ: float) -> bool:
        return arm_s_occ > ARM_OCC_THRESHOLD

    def _arm_token_mask(self, arm_pixel_mask: np.ndarray) -> np.ndarray:
        """Downsample a HxW boolean arm mask to the TOKEN_GRID_SIDE^2
        token grid: a token is "arm" if >50% of its receptive field is
        arm pixels."""
        h, w = arm_pixel_mask.shape
        ph, pw = h // TOKEN_GRID_SIDE, w // TOKEN_GRID_SIDE
        pooled = arm_pixel_mask[: ph * TOKEN_GRID_SIDE, : pw * TOKEN_GRID_SIDE].reshape(
            TOKEN_GRID_SIDE, ph, TOKEN_GRID_SIDE, pw
        )
        return pooled.mean(axis=(1, 3)).reshape(-1) > 0.5

    def _build_subgoal_prompt(self, instruction: str, horizon: int) -> str:
        """Visual CoT framing: MMaDA's only conditioning channel into
        t2i_generate is text (training/prompting_utils.py::t2i_gen_prompt
        takes text_ids + image_ids, no trajectory/video input), so the
        "K steps ahead" horizon is expressed in the prompt itself rather
        than as a separate model input — this asks for the completed
        subgoal state, not a same-instant arm removal."""
        cleaned = instruction.strip().rstrip(".")
        return (
            f"{cleaned}. Imagine {horizon} steps from now, after the "
            f"robot arm has completed this sub-step: the ideal resulting "
            f"scene, with the arm out of the way and the target object "
            f"clearly visible in its new state."
        )

    def sample_arm_free_image(
        self,
        image: np.ndarray,
        arm_pixel_mask: np.ndarray,
        instruction: str,
        horizon: int = DEFAULT_SUBGOAL_HORIZON,
    ) -> SubgoalResult:
        # MagvitV2Tokenizer.encode requires exactly SUBGOAL_IMAGE_SIDE^2 input
        # (verified against the live model: token count scales with input
        # resolution, 1024 only comes out of a 512x512 input) — the caller's
        # camera frame is whatever size the robot's camera produces, so
        # resize here rather than pushing this constraint out to every caller.
        if image.shape[0] != SUBGOAL_IMAGE_SIDE or image.shape[1] != SUBGOAL_IMAGE_SIDE:
            image = cv2.resize(image, (SUBGOAL_IMAGE_SIDE, SUBGOAL_IMAGE_SIDE), interpolation=cv2.INTER_AREA)
        if arm_pixel_mask.shape[0] != SUBGOAL_IMAGE_SIDE or arm_pixel_mask.shape[1] != SUBGOAL_IMAGE_SIDE:
            arm_pixel_mask = (
                cv2.resize(
                    arm_pixel_mask.astype(np.uint8),
                    (SUBGOAL_IMAGE_SIDE, SUBGOAL_IMAGE_SIDE),
                    interpolation=cv2.INTER_NEAREST,
                )
                > 0
            )

        image_ids = self.world_model.tokenizer.encode(image)  # (SUBGOAL_NUM_TOKENS,), raw 0..codebook_size-1
        token_mask = self._arm_token_mask(arm_pixel_mask)
        # Non-mask positions must be offset into MMaDA's combined vocab
        # (image_token_offset's docstring); MASK_TOKEN_ID itself is a
        # reserved id, not part of the codebook range, so it's untouched.
        offset_image_ids = image_ids + self.world_model.image_token_offset
        masked_ids = np.where(token_mask, MASK_TOKEN_ID, offset_image_ids)

        prompt = self._build_subgoal_prompt(instruction, horizon)
        batch = self.world_model.build_prompt(prompt, torch.from_numpy(masked_ids).long().unsqueeze(0))
        resolved_ids = self.world_model.denoise(batch)
        image_out = self.world_model.tokenizer.decode(resolved_ids[0].cpu().numpy())
        return SubgoalResult(image=image_out)
