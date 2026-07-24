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

    def _arm_bbox_token_mask(self, arm_pixel_mask: np.ndarray, pad_tokens: int = 1) -> np.ndarray:
        """Bounding-box variant of _arm_token_mask: masks the full
        rectangular block of the token grid spanning the arm's pixel
        bounding box (+pad_tokens margin), instead of only the tokens
        the arm's silhouette actually touches.

        Motivation: the arm's silhouette is thin and diagonal, so
        _arm_token_mask leaves very few *contiguous* same-object tokens
        for MAGVIT-v2 to reconstruct a coherent held-object shape from
        -- most masked tokens only border 1-2 other masked tokens. A
        solid rectangular region gives the inpainter a normal
        "object-sized" contiguous area instead. Logged visual audit
        (t08_mmada_log/, 2026-07-15): 6/6 generations with the
        silhouette mask (schedule-fixed denoise, CFG on/off, two
        prompt styles) produced the same blob artifact regardless of
        conditioning -- pointing at mask geometry, not sampling
        hyperparameters, as the likely cause."""
        h, w = arm_pixel_mask.shape
        ph, pw = h // TOKEN_GRID_SIDE, w // TOKEN_GRID_SIDE
        rows = np.any(arm_pixel_mask, axis=1)
        cols = np.any(arm_pixel_mask, axis=0)
        if not rows.any():
            return np.zeros(TOKEN_GRID_SIDE * TOKEN_GRID_SIDE, dtype=bool)
        r0, r1 = np.where(rows)[0][[0, -1]]
        c0, c1 = np.where(cols)[0][[0, -1]]
        t_r0 = max(0, r0 // ph - pad_tokens)
        t_r1 = min(TOKEN_GRID_SIDE - 1, r1 // ph + pad_tokens)
        t_c0 = max(0, c0 // pw - pad_tokens)
        t_c1 = min(TOKEN_GRID_SIDE - 1, c1 // pw + pad_tokens)
        mask = np.zeros((TOKEN_GRID_SIDE, TOKEN_GRID_SIDE), dtype=bool)
        mask[t_r0 : t_r1 + 1, t_c0 : t_c1 + 1] = True
        return mask.reshape(-1)

    def _gripper_end_bbox_token_mask(
        self, arm_pixel_mask: np.ndarray, row_fraction: float = 0.35, pad_tokens: int = 1
    ) -> np.ndarray:
        """Bbox of only the arm's lowest `row_fraction` of rows (the
        gripper/terminal end, where the held object actually sits),
        instead of `_arm_bbox_token_mask`'s bbox of the *whole* arm
        silhouette. Motivation (occ_vla/CLAUDE.md, "MMaDA arm-free
        generation quality investigation"): the whole-arm bbox reaches
        diagonally from the mount (top-center) to the gripper
        (bottom-left), covering ~43-49% of the frame -- masking that
        much reintroduced a new artifact near an unrelated, correctly-
        visible object elsewhere in frame. This is the untested
        alternative flagged there: bound only the terminal end, which
        is the region that actually needs to look "resolved," not the
        arm's full diagonal reach."""
        h, w = arm_pixel_mask.shape
        rows = np.any(arm_pixel_mask, axis=1)
        if not rows.any():
            return np.zeros(TOKEN_GRID_SIDE * TOKEN_GRID_SIDE, dtype=bool)
        r_first, r_last = np.where(rows)[0][[0, -1]]
        cutoff = r_first + int(round((r_last - r_first) * (1.0 - row_fraction)))
        gripper_end_mask = arm_pixel_mask.copy()
        gripper_end_mask[:cutoff, :] = False
        return self._arm_bbox_token_mask(gripper_end_mask, pad_tokens=pad_tokens)

    def _gripper_end_area_token_mask(
        self, arm_pixel_mask: np.ndarray, target_area_fraction: float, pad_tokens: int = 0
    ) -> np.ndarray:
        """Square token-grid region of approximately `target_area_fraction`
        of the frame (e.g. 0.10 -> ~10% of 1024 tokens), anchored at the
        arm's gripper end (its lowest masked row, centered on that row's
        column extent) and extending upward from there.

        Unlike `_gripper_end_bbox_token_mask` (whose area is whatever the
        arm's own silhouette happens to span at a given row_fraction --
        data-dependent, varies frame to frame), this fixes the area
        directly so a mask-area sweep (5%/10%/15%/20%, see occ_vla/
        CLAUDE.md "Mask-area & temperature-schedule investigation") is
        comparable across frames with different arm geometry/camera
        angle/object size, instead of conflating area with shape."""
        h, w = arm_pixel_mask.shape
        ph, pw = h // TOKEN_GRID_SIDE, w // TOKEN_GRID_SIDE
        rows = np.any(arm_pixel_mask, axis=1)
        if not rows.any():
            return np.zeros(TOKEN_GRID_SIDE * TOKEN_GRID_SIDE, dtype=bool)
        r_last = np.where(rows)[0][-1]
        cols_at_bottom = np.where(arm_pixel_mask[r_last])[0]
        c_center = int(round(cols_at_bottom.mean())) if len(cols_at_bottom) else w // 2

        target_tokens = max(1, round(target_area_fraction * TOKEN_GRID_SIDE * TOKEN_GRID_SIDE))
        side_tokens = max(1, round(target_tokens**0.5))

        t_r_last = min(TOKEN_GRID_SIDE - 1, r_last // ph)
        t_c_center = min(TOKEN_GRID_SIDE - 1, c_center // pw)

        t_r1 = min(TOKEN_GRID_SIDE - 1, t_r_last + pad_tokens)
        t_r0 = max(0, t_r1 - side_tokens + 1)
        t_c0 = max(0, t_c_center - side_tokens // 2 - pad_tokens)
        t_c1 = min(TOKEN_GRID_SIDE - 1, t_c_center + (side_tokens - side_tokens // 2) - 1 + pad_tokens)

        mask = np.zeros((TOKEN_GRID_SIDE, TOKEN_GRID_SIDE), dtype=bool)
        mask[t_r0 : t_r1 + 1, t_c0 : t_c1 + 1] = True
        return mask.reshape(-1)

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

    def sample_subgoal_image(self, instruction: str, horizon: int = DEFAULT_SUBGOAL_HORIZON) -> SubgoalResult:
        """Full-frame counterpart to `sample_arm_free_image` (2026-07-21,
        per user decision to stop pursuing arm-removal inpainting):
        `sample_arm_free_image` only ever masks the arm's ~13% silhouette
        region, so 87% of the output is pinned to the CURRENT frame's real
        tokens even though `_build_subgoal_prompt` explicitly asks for a
        scene "steps from now, after the robot arm has completed this
        sub-step" -- the model is asked to imagine a future state while
        being simultaneously forced to keep most of the present pixels
        fixed. That contradiction is a plausible contributor to the
        washed-out/blob failure mode (see CLAUDE.md, "Best-of-N ruled
        out"), separate from any inference-time sampling issue.

        This masks the ENTIRE image (all SUBGOAL_NUM_TOKENS positions),
        so generation is driven purely by the text prompt (still the same
        "K steps ahead, ideal resulting scene" framing) with no
        current-frame content pinned at all -- ordinary from-scratch
        generation, same regime `MMadaModelLM.t2i_generate` and
        `MMaDAWorldModel.denoise`'s schedule fix were originally designed
        for (initial_mask_count == SUBGOAL_NUM_TOKENS here, so the fix is
        a no-op in this mode, not a special case).

        No `arm_pixel_mask`/current `image` input at all -- there's
        nothing to inpaint around. PlausibilityChecker's background-MSE
        heuristic doesn't apply to this mode's output (a full
        reimagining is *supposed* to differ from the current frame
        everywhere, not just where the arm was); a different acceptance
        criterion (or none, for a first look) is needed for this path.
        """
        prompt = self._build_subgoal_prompt(instruction, horizon)
        masked_ids = np.full(SUBGOAL_NUM_TOKENS, MASK_TOKEN_ID, dtype=np.int64)
        batch = self.world_model.build_prompt(prompt, torch.from_numpy(masked_ids).long().unsqueeze(0))
        resolved_ids = self.world_model.denoise(batch)
        image_out = self.world_model.tokenizer.decode(resolved_ids[0].cpu().numpy())
        return SubgoalResult(image=image_out)
