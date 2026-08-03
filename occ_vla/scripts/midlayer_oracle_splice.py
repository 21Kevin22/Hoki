"""
midlayer_oracle_splice.py

DIAGNOSTIC ONLY. Tests the "ViT mid-layer injection" hypothesis: does
splicing ground-truth features at an INTERMEDIATE transformer block (instead
of the final output, as the earlier wrist_partial_oracle test did) let the
REMAINING blocks' self-attention "smooth over" the boundary inconsistency
between clean-image-context tokens and corrupted-image-context tokens?

Verified real internals (occ_vla, 2026-07-31) before writing this, not
guessed (see scripts/run_oft_camera_dropout_eval.py's session history):
  - vision_backbone.featurizer (DINOv2): 24 blocks, num_prefix_tokens=5
    (1 CLS + 4 register tokens -- NOT spatial, never masked/spliced)
  - vision_backbone.fused_featurizer (SigLIP): 27 blocks, num_prefix_tokens=0
  - OpenVLA-OFT's existing PrismaticVisionBackbone._create_featurizer already
    extracts at block (num_blocks - 2) -- the SECOND-TO-LAST block, not the
    model's true final layer. So "later blocks after the splice point" means
    "up to num_blocks-2", not all the way to the architecture's true end.
  - Both DINOv2 and SigLIP use patch14 @ 224px -> 16x16=256 patch grid, same
    spatial alignment -- one occlusion_mask (256,) applies to both.
  - PrismaticVisionBackbone.forward's multi-image split: each image
    contributes 6 channels (3 -> self.featurizer/DINOv2, 3 ->
    self.fused_featurizer/SigLIP); for num_images_in_input=2, image 0 =
    agentview, image 1 = wrist (torch.split(pixel_values, [6,6], dim=1)).

Implementation: monkey-patches `model.vision_backbone.forward` to reproduce
that exact multi-image logic, except for the WRIST image's DINOv2/SigLIP
computation, which runs through `_run_vit_with_midlayer_splice` instead of
a plain call. Agentview is completely unmodified. Reads clean wrist pixel
values from `model._diagnostic_clean_wrist_pixel_values` (a per-step
attribute the eval script sets before each get_action call) -- avoids
touching modeling_prismatic.py / openvla_utils.py / robot_utils.py at all
for what is explicitly a throwaway diagnostic, not a real deployment path.
"""

import torch


def _vit_prep(featurizer, x):
    x = featurizer.patch_embed(x)
    x = featurizer._pos_embed(x)
    x = featurizer.patch_drop(x)
    x = featurizer.norm_pre(x)
    return x


def run_vit_with_midlayer_splice(featurizer, x_corrupted_pixels, x_clean_pixels, split_layer, patch_mask_256):
    """Returns (B, 256, D) patch-token-only features, matching
    get_intermediate_layers(n={num_blocks-2})'s output convention, but with
    a ground-truth splice injected partway through at `split_layer`."""
    num_blocks = len(featurizer.blocks)
    extraction_layer = num_blocks - 2  # matches _create_featurizer's existing monkeypatch
    assert 0 <= split_layer < extraction_layer, f"split_layer={split_layer} must be in [0, {extraction_layer})"
    num_prefix = featurizer.num_prefix_tokens

    x_corrupted = _vit_prep(featurizer, x_corrupted_pixels)  # corrupted-image-context stream
    x_clean = _vit_prep(featurizer, x_clean_pixels)  # clean-image-context stream (ground truth)

    for i, blk in enumerate(featurizer.blocks):
        x_corrupted = blk(x_corrupted)
        if i <= split_layer:
            x_clean = blk(x_clean)  # only need the clean stream up through the splice point

        if i == split_layer:
            mask = patch_mask_256.to(dtype=torch.bool, device=x_corrupted.device).reshape(1, -1, 1)
            patch_corrupted = x_corrupted[:, num_prefix:]
            patch_clean = x_clean[:, num_prefix:]
            spliced_patch = torch.where(mask, patch_clean, patch_corrupted)
            x_corrupted = torch.cat([x_corrupted[:, :num_prefix], spliced_patch], dim=1)
            del x_clean  # not needed past the splice point

        if i == extraction_layer:
            break

    return x_corrupted[:, num_prefix:]  # (B, 256, D)


def make_midlayer_splice_forward(vision_backbone, split_frac, patch_mask_256_wrist):
    """Returns a replacement for `vision_backbone.forward` (bind via
    `vision_backbone.forward = types.MethodType(fn, vision_backbone)` or a
    plain closure assignment) that reproduces PrismaticVisionBackbone's
    multi-image logic, splicing only the wrist image (index 1) via
    run_vit_with_midlayer_splice, reading clean wrist pixels off
    vision_backbone._diagnostic_clean_wrist_pixel_values.
    """

    def patched_forward(pixel_values, occlusion_mask=None, proprio_for_dynamics=None):
        # occlusion_mask/proprio_for_dynamics accepted for call-signature
        # compatibility with PrismaticForConditionalGeneration._process_vision_features
        # (which now always passes them through) but unused here -- this
        # diagnostic's own ground-truth splice mechanism (patch_mask_256_wrist,
        # closed over above) is a separate, standalone substitute for the
        # trainable predictor path.
        assert vision_backbone.use_fused_vision_backbone
        num_images = vision_backbone.num_images_in_input
        clean_wrist_pixels = getattr(vision_backbone, "_diagnostic_clean_wrist_pixel_values", None)
        assert clean_wrist_pixels is not None, "set vision_backbone._diagnostic_clean_wrist_pixel_values before calling"

        if num_images == 1:
            images = [pixel_values]
        else:
            images = torch.split(pixel_values, [6] * num_images, dim=1)

        all_patches = []
        for img_idx, img in enumerate(images):
            img_regular, img_fused = torch.split(img, [3, 3], dim=1)

            if img_idx == 1:  # wrist -- splice mid-layer ground truth
                clean_regular, clean_fused = torch.split(clean_wrist_pixels, [3, 3], dim=1)
                num_blocks_dino = len(vision_backbone.featurizer.blocks)
                num_blocks_siglip = len(vision_backbone.fused_featurizer.blocks)
                split_layer_dino = int(num_blocks_dino * split_frac)
                split_layer_siglip = int(num_blocks_siglip * split_frac)
                patches = run_vit_with_midlayer_splice(
                    vision_backbone.featurizer, img_regular, clean_regular, split_layer_dino, patch_mask_256_wrist
                )
                patches_fused = run_vit_with_midlayer_splice(
                    vision_backbone.fused_featurizer, img_fused, clean_fused, split_layer_siglip, patch_mask_256_wrist
                )
            else:
                patches = vision_backbone.featurizer(img_regular)
                patches_fused = vision_backbone.fused_featurizer(img_fused)

            all_patches.append(torch.cat([patches, patches_fused], dim=2))

        return torch.cat(all_patches, dim=1)

    return patched_forward
