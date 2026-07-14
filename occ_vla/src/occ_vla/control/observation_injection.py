"""Feature reinjection: getting the world model's arm-free subgoal
image and PKLP's kinematic state into pi0.5's attention, at the actual
extension points openpi exposes — not by patching vendored code.

Where the "Prefix Mask Strategy" the pi0.5 architecture description
calls for really lives: `Pi0.embed_prefix`
(third_party/openpi/src/openpi/models/pi0.py:106) loops over
`obs.images` (a dict) and embeds *every* entry through SigLIP, each
contributing image tokens to the prefix with full bidirectional
attention alongside the language tokens — so, architecturally, feeding
the generated subgoal image as one of those image entries already *is*
prefix-token-level injection, no model surgery required.

The catch: `Pi0Config.inputs_spec()` (pi0_config.py) hardcodes exactly
three image keys (`base_0_rgb`, `left_wrist_0_rgb`,
`right_wrist_0_rgb`), and a pi0.5-libero checkpoint was trained with
that shape — you can't add a fourth key at inference time and expect
it to mean anything, but you *can* repurpose `right_wrist_0_rgb`, which
`LiberoInputs` (policies/libero_policy.py:53-72) already always
zero-fills with `image_mask=False` for the standard LIBERO single/dual
camera setup. Swapping that unused slot's content and mask is the
non-invasive injection point.

`LiberoInputs.__call__` hardcodes that zero-fill though — it doesn't
read a third image from the input dict at all, so this can't be done
purely via `repack_transforms` (the public
`policy_config.create_trained_policy(..., repack_transforms=...)` hook
runs *before* `LiberoInputs`, and `LiberoInputs` would still stomp the
result). The real fix, and the one `LiberoInputs`'s own docstring
invites ("you can copy this class and modify the keys"), is this
`OccVlaLiberoInputs` — a copy of `LiberoInputs` with the hardcoded zero
replaced by a real conditional read. Wiring it in needs either:
  (a) a custom TrainConfig entry (copy of the "pi05_libero" entry in
      third_party/openpi/src/openpi/training/config.py, swapping in
      `OccVlaLiberoInputs` as the data transform) — cleanest, but
      touches how the vendored config is registered, so it lives in
      occ_vla as a *new* config, not an edit to the vendored file; or
  (b) constructing `openpi.policies.policy.Policy` directly (bypassing
      `create_trained_policy`) with `transforms=[OccVlaLiberoInputs(...), ...]`
      built from the same pieces `create_trained_policy` assembles.
Neither is implemented here yet — this module only provides the
transform itself, real and unit-testable independent of which wiring
path is chosen.
"""

import dataclasses

import einops
import numpy as np


def _parse_image(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class OccVlaLiberoInputs:
    """Same contract as openpi.policies.libero_policy.LiberoInputs, plus:
    - `observation/subgoal_image`, if present, fills `right_wrist_0_rgb`
      (mask=True) instead of the zero-fill LiberoInputs always does.
    - `cot_anchor`, if present, is appended to the prompt text.

    `is_pi0_fast` mirrors LiberoInputs' own `model_type ==
    _model.ModelType.PI0_FAST` check (right_wrist_0_rgb's mask is True
    for Pi0-FAST regardless of subgoal presence) — passed as a plain
    bool so this module has no import-time dependency on openpi/jax.
    """

    is_pi0_fast: bool = False

    def __call__(self, data: dict) -> dict:
        base_image = _parse_image(data["observation/image"])
        wrist_image = _parse_image(data["observation/wrist_image"])
        subgoal_image = data.get("observation/subgoal_image")

        inputs = {
            "state": data["observation/state"],
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                "right_wrist_0_rgb": _parse_image(subgoal_image) if subgoal_image is not None else np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_ if (subgoal_image is not None or self.is_pi0_fast) else np.False_,
            },
        }

        if "actions" in data:
            inputs["actions"] = data["actions"]

        prompt = data.get("prompt", "")
        cot_anchor = data.get("cot_anchor")
        if cot_anchor:
            prompt = f"{prompt} {cot_anchor}".strip()
        if prompt:
            inputs["prompt"] = prompt

        return inputs
