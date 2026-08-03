"""
vjepa_latent_predictor.py

V-JEPA-style 1-step latent dynamics predictor for OpenVLA-OFT's fused vision
features, used to in-place overwrite occluded/lost-camera patch tokens.

Not V-JEPA 2 itself (no self-supervised joint-embedding pretraining, no EMA
target encoder) -- a small, purpose-built conditional predictor that borrows
V-JEPA's framing ("predict the current latent state of a masked region from
context + a dynamics cue") rather than its actual training recipe. Call it
that in code/docs, not "V-JEPA 2", to avoid mischaracterizing the technique.

Empirical motivation (occ_vla, moka_pots / libero_10 task 8, n=10 per
condition, OpenVLA-OFT checkpoint moojink/openvla-7b-oft-finetuned-libero-10):
  baseline                 8/10
  agentview/wrist blackout 0/10  (gray fill)
  agentview/wrist "hold"   0/10  (frozen real frame -- rules out OOD-pixel-shock
                                   as the sole cause; a temporally static but
                                   *valid* frame is equally fatal)
  agentview/wrist partial  0/10  (35%-area centered patch, 65% still live)
All non-baseline conditions collapsed identically -- the failure tracks
*any* deviation from the training distribution's continuously-updating,
both-cameras-live input, not missing information specifically. This is why
the predictor is conditioned on `proprio` (a cheap dynamics cue correlated
with how the scene should be changing) and is designed as a residual
correction applied every call (not a one-shot fill), rather than a static
content-recovery module.
"""

import torch
import torch.nn as nn


class VJEPA_LatentDynamicsPredictor(nn.Module):
    """
    Predicts a per-token RESIDUAL correction (not an absolute replacement) for
    occluded vision patch tokens, conditioned on the previous timestep's full
    token set (`past_latents`) and current proprioception.

    Design constraints (see occ_vla review history):
      - Residual/velocity output, not absolute value: `f_final = f_original +
        mask * predictor(...)`. Combined with zero-init on `out_proj`, this
        guarantees `f_final == f_original` EXACTLY at init, for every token,
        occluded or not -- unlike a `(1-mask)*orig + mask*pred` replace
        formula, which zero-inits occluded tokens to a hard-zero vector (an
        even more extreme distribution deviation than any condition tested
        above, and empirically all of them were already fatal).
      - NFE=1: single forward pass, no iterative denoising loop.
      - Small hidden dim relative to feature_dim (2176 for OpenVLA-OFT's
        fused DINOv2+SigLIP backbone) to keep the forward pass cheap.
    """

    def __init__(self, feature_dim: int, proprio_dim: int = 8, hidden_dim: int = 256, num_heads: int = 4):
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim

        # Proprioception -> FiLM (gamma, beta) applied to the past-latent tokens.
        # proprio: [B, proprio_dim] -> proprio_emb: [B, hidden_dim]
        self.proprio_proj = nn.Sequential(
            nn.Linear(proprio_dim, hidden_dim),
            nn.SiLU(),
        )
        # proprio_emb: [B, hidden_dim] -> gamma_beta: [B, 2 * feature_dim]
        self.film = nn.Linear(hidden_dim, 2 * feature_dim)

        # Token projection down to hidden_dim for the attention step.
        self.in_proj = nn.Linear(feature_dim, hidden_dim)

        # Query = "what should the occluded/dynamic tokens be" (from the
        # FiLM-modulated temporal prior). Key/Value = current real vision
        # tokens (informative at unoccluded positions, and still the most
        # recent real signal available even where masked).
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)

        # Zero-init output projection: out_proj(...) == 0 for any input until
        # trained, so `f_final = f_original + mask * 0 == f_original` always.
        self.out_proj = nn.Linear(hidden_dim, feature_dim)

        self.reset_parameters()

    def reset_parameters(self):
        """Explicitly (re-)initialize every parameter this module owns.

        Called both from `__init__` AND explicitly again by `get_vla()`
        right after `from_pretrained(...)` returns -- HF's `low_cpu_mem_usage=True`
        loading path materializes weights not present in the checkpoint's
        state dict (i.e. this whole module, since it's newly added) via a
        meta-device "fast init" that does NOT reliably re-run a submodule's
        own `__init__`-time `nn.init.*` calls, and generic per-module hooks
        like `PreTrainedModel._init_weights` only know how to handle the
        layer types the original codebase used (`nn.Linear`, `nn.Embedding`)
        -- `nn.MultiheadAttention`'s raw `in_proj_weight`/`out_proj` and
        `nn.LayerNorm`'s weight/bias fall through untouched and can be left
        with uninitialized (effectively garbage) values, large enough to
        produce NaN/Inf inside attention. Confirmed via occ_vla's own
        end-to-end regression test: without this explicit, unconditional
        re-init call, `out_proj`/`film` alone being zero was NOT sufficient
        -- NaN propagates through `0 * NaN == NaN`, not 0.
        """
        std = 0.02  # matches this codebase's own PrismaticPreTrainedModel._init_weights default
        nn.init.normal_(self.proprio_proj[0].weight, mean=0.0, std=std)
        nn.init.zeros_(self.proprio_proj[0].bias)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)  # gamma=0, beta=0 at init -> FiLM starts as identity
        nn.init.normal_(self.in_proj.weight, mean=0.0, std=std)
        nn.init.zeros_(self.in_proj.bias)
        nn.init.normal_(self.cross_attn.in_proj_weight, mean=0.0, std=std)
        nn.init.zeros_(self.cross_attn.in_proj_bias)
        nn.init.normal_(self.cross_attn.out_proj.weight, mean=0.0, std=std)
        nn.init.zeros_(self.cross_attn.out_proj.bias)
        nn.init.ones_(self.norm.weight)
        nn.init.zeros_(self.norm.bias)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(
        self,
        f_current: torch.Tensor,  # [B, N, feature_dim] -- current (possibly-occluded) raw vision tokens
        past_latents: torch.Tensor,  # [B, N, feature_dim] -- previous timestep's resolved token set
        proprio: torch.Tensor,  # [B, proprio_dim]
    ) -> torch.Tensor:
        """Returns a residual correction of shape [B, N, feature_dim] -- ADD
        this (scaled by occlusion_mask) to `f_current`, don't replace it."""
        proprio_emb = self.proprio_proj(proprio)  # [B, hidden_dim]
        gamma, beta = self.film(proprio_emb).chunk(2, dim=-1)  # [B, feature_dim] each
        modulated_past = past_latents * (1.0 + gamma.unsqueeze(1)) + beta.unsqueeze(1)  # [B, N, feature_dim]

        query = self.in_proj(modulated_past)  # [B, N, hidden_dim]
        context = self.in_proj(f_current)  # [B, N, hidden_dim]
        attn_out, _ = self.cross_attn(query=query, key=context, value=context, need_weights=False)  # [B, N, hidden_dim]
        attn_out = self.norm(attn_out)

        residual = self.out_proj(attn_out)  # [B, N, feature_dim]; == 0 everywhere at init
        return residual
