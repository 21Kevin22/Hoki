"""MMaDA-8B (gen-verse/mmada: models/modeling_mmada.py::MMadaModelLM),
a LLaDA-based discrete-diffusion model over a shared image/text token
vocabulary.

`t2i_generate` iteratively unmasks whichever positions in `input_ids`
equal `mask_token_id`, using a cosine unmasking schedule
(`training.utils.get_mask_schedule`) over `timesteps` steps — the repo
only exposes this as full text-to-image generation (all
SUBGOAL_NUM_TOKENS image positions start masked), but the same call
works as region-conditioned inpainting if the caller pre-fills the
*non*-masked image token positions with real encoded content before
calling `denoise`, which is exactly what arm_free_subgoal.py does for
the arm region.
"""

import sys
from dataclasses import dataclass
from pathlib import Path

import torch

from occ_vla.world_model.tokenizer import CODEBOOK_SIZE, SUBGOAL_NUM_TOKENS, MagvitV2Tokenizer

MASK_TOKEN_ID = 126336
DEFAULT_TIMESTEPS = 18  # MMadaModelLM.t2i_generate default ("ideal" per MaskGIT paper)


def _compounding_temperature_schedule(timesteps: int, initial_temperature: float = 1.0) -> list[float]:
    """Reproduces MMadaModelLM.t2i_generate's temperature recurrence for
    mask_by_random_topk (third_party/mmada/models/modeling_mmada.py,
    `temperature = temperature * (1.0 - ratio)` at both line 203 and the
    stepwise variant's line 642): temperature is decayed from the
    *running* value each step, not recomputed fresh from
    initial_temperature every time -- temperature_i = temperature_{i-1}
    * (1 - ratio_i), ratio_i = (i+1)/timesteps.

    This compounds much faster than `initial_temperature * (1 -
    ratio_i)` evaluated fresh each step (which is what an earlier
    version of _denoise_impl below did, missing the compounding): e.g.
    for timesteps=18, step 8 is ~0.044 here vs ~0.5 non-compounding --
    over 10x more Gumbel-noise randomness in mask_by_random_topk's
    token-confidence ranking throughout the middle of the schedule,
    every one of the 70+ MMaDA generations logged before this fix was
    found (t08_mmada_log/) used the non-compounding version."""
    temperature = initial_temperature
    schedule = []
    for step in range(timesteps):
        ratio = 1.0 * (step + 1) / timesteps
        temperature = temperature * (1.0 - ratio)
        schedule.append(temperature)
    return schedule

_MMADA_ROOT = Path(__file__).resolve().parents[3] / "third_party" / "mmada"


@dataclass
class MaskedTokenBatch:
    input_ids: torch.LongTensor  # [task][sot][text][eot][soi][image tokens, some = MASK_TOKEN_ID][eoi]
    attention_mask: torch.Tensor


class MMaDAWorldModel:
    def __init__(self, checkpoint_path: str, tokenizer: MagvitV2Tokenizer, device: str = "cuda"):
        self.checkpoint_path = checkpoint_path
        self.tokenizer = tokenizer
        self.device = device
        self._model = None
        self._uni_prompting = None

    def load(self) -> None:
        if str(_MMADA_ROOT) not in sys.path:
            sys.path.insert(0, str(_MMADA_ROOT))
        from models import MMadaModelLM  # noqa: PLC0415
        from training.prompting_utils import UniversalPrompting  # noqa: PLC0415
        from transformers import AutoTokenizer  # noqa: PLC0415

        text_tokenizer = AutoTokenizer.from_pretrained(self.checkpoint_path, padding_side="left")
        # UniversalPrompting.__init__ defaults max_text_len=8000 — meant for
        # long CoT prompts, but t2i_gen_prompt left-pads the *entire* text
        # portion to max_text_len regardless of actual prompt length
        # (training/prompting_utils.py:150-151), so the default silently
        # built a ~8000+1024-token sequence per call. Confirmed against the
        # live model: that blew a 24GB GPU up during attention (tried to
        # allocate 9.71GB, OOM) even though the 8B model's own bf16 weights
        # only take ~18GB. 128 is generous for arm_free_subgoal.py's prompts.
        self._uni_prompting = UniversalPrompting(
            text_tokenizer, max_text_len=128, max_seq_len=SUBGOAL_NUM_TOKENS + 8
        )
        self._model = MMadaModelLM.from_pretrained(self.checkpoint_path, torch_dtype=torch.bfloat16).to(self.device).eval()

    @property
    def image_token_offset(self) -> int:
        """t2i_generate recovers raw MAGVIT-v2 codebook indices from
        `input_ids` via `input_ids - len(uni_prompting.text_tokenizer) -
        num_new_special_tokens` (num_new_special_tokens is hardcoded 0 in
        that function) for every non-mask position in the image region
        (models/modeling_mmada.py:149,579) — so real codebook values placed
        into `image_ids` before calling build_prompt() must already be
        offset by this amount, not passed as raw 0..codebook_size-1
        indices. Confirmed against the live model: passing raw indices
        produced an out-of-bounds embedding lookup (CUDA device-side
        assert) since they collided with the text-token id range instead
        of landing in the image-token range."""
        if self._uni_prompting is None:
            raise RuntimeError("call load() first")
        return len(self._uni_prompting.text_tokenizer)

    def build_prompt(self, text: str, image_ids: torch.LongTensor) -> MaskedTokenBatch:
        """image_ids: (1, SUBGOAL_NUM_TOKENS), MASK_TOKEN_ID at positions to
        be (re)generated, real MAGVIT-v2 codes + image_token_offset elsewhere
        (see image_token_offset's docstring)."""
        if self._uni_prompting is None:
            raise RuntimeError("call load() first")
        text_ids = [self._uni_prompting.text_tokenizer(text)["input_ids"]]
        # t2i_gen_prompt builds its output tensors on image_ids.device and
        # otherwise leaves everything on CPU by default; the model lives on
        # self.device, so without this the two mismatch and torch's
        # embedding lookup raises (confirmed against the live model: "cuda:0
        # and cpu" RuntimeError from nn.Embedding).
        image_ids = image_ids.to(self.device)
        input_ids, attention_mask = self._uni_prompting.t2i_gen_prompt(text_ids, image_ids)
        return MaskedTokenBatch(input_ids=input_ids.to(self.device), attention_mask=attention_mask.to(self.device))

    def denoise(self, batch: MaskedTokenBatch, timesteps: int = DEFAULT_TIMESTEPS) -> torch.LongTensor:
        """Iteratively unmask MASK_TOKEN_ID positions in batch.input_ids.
        Returns the resolved image-token ids, shape (1, SUBGOAL_NUM_TOKENS).

        This is NOT a call to `MMadaModelLM.t2i_generate` — it's a
        reimplementation of the same MaskGIT-style loop (reusing the
        vendored `cosine_schedule`/`mask_by_random_topk` helpers) with one
        deliberate change: the per-step reveal-count schedule is driven by
        the number of tokens actually masked in `batch.input_ids`, not by
        `SUBGOAL_NUM_TOKENS` (1024, the full image region).

        `t2i_generate` hardcodes `mask_len = floor(num_vq_tokens *
        mask_ratio)` with `num_vq_tokens = seq_len = 1024` — correct for
        its intended use (from-scratch generation, where all 1024 tokens
        start masked) but wrong for arm_free_subgoal.py's inpainting use,
        where only the arm's ~13% of tokens (~130/1024) start masked.
        Traced through the schedule: for the first ~16 of
        DEFAULT_TIMESTEPS=18 steps, `1024 * mask_ratio` stays far above
        the true remaining-masked count, so the
        `min(unknown_count - 1, mask_len)` clamp always picks the small
        side and only ONE token gets confidently committed per step — then
        the remaining ~88% of the masked region is dumped out in the last
        1-2 steps at near-zero temperature. That's close to a single
        unrefined resample, not the intended gradual coarse-to-fine
        denoising, and matches exactly what was observed: every one of 70
        logged t08 generations (t08_mmada_log/calibration_summary.json)
        showed the same garbled/blob-like held-object artifact regardless
        of PlausibilityChecker score — a generation-quality bug, not a
        metric-calibration issue. Driving the schedule off the actual
        initial masked-token count instead fixes the reveal-per-step
        arithmetic for any mask size without touching the vendored file."""
        if self._model is None:
            raise RuntimeError("call load() first")
        from models.sampling import cosine_schedule, mask_by_random_topk  # noqa: PLC0415 (vendored helper, reused not duplicated)

        return self._denoise_impl(batch, timesteps, cosine_schedule, mask_by_random_topk)

    @torch.no_grad()
    def _denoise_impl(self, batch, timesteps, cosine_schedule, mask_by_random_topk):
        input_ids = batch.input_ids.clone()
        attention_mask = batch.attention_mask
        text_vocab = len(self._uni_prompting.text_tokenizer)
        num_vq_tokens = SUBGOAL_NUM_TOKENS  # image-region slice width in the token *layout* — unrelated to the schedule fix above

        image_slice = input_ids[:, -(num_vq_tokens + 1) : -1].clone()
        ids_minus_vocab = torch.where(image_slice == MASK_TOKEN_ID, MASK_TOKEN_ID, image_slice - text_vocab)
        initial_mask_count = int((ids_minus_vocab == MASK_TOKEN_ID).sum().item())

        temperature_schedule = _compounding_temperature_schedule(timesteps)
        sampled_ids = ids_minus_vocab
        for step in range(timesteps):
            attention_bias = (attention_mask[:, :, None] & attention_mask[:, None, :]).bool().unsqueeze(1)
            logits = self._model(input_ids, attention_bias=attention_bias).logits
            logits = logits[:, -(num_vq_tokens + 1) : -1, text_vocab : text_vocab + CODEBOOK_SIZE]

            probs = logits.softmax(dim=-1)
            sampled = probs.reshape(-1, logits.size(-1))
            sampled_ids = torch.multinomial(sampled, 1)[:, 0].view(*logits.shape[:-1])

            unknown_map = ids_minus_vocab == MASK_TOKEN_ID
            sampled_ids = torch.where(unknown_map, sampled_ids, ids_minus_vocab)

            ratio = 1.0 * (step + 1) / timesteps
            mask_ratio = cosine_schedule(torch.tensor(ratio))
            selected_probs = torch.gather(probs, -1, sampled_ids.long()[..., None]).squeeze(-1)
            selected_probs = torch.where(unknown_map, selected_probs, torch.finfo(selected_probs.dtype).max)

            # the fix: initial_mask_count in place of t2i_generate's num_vq_tokens (1024)
            mask_len = (initial_mask_count * mask_ratio).floor().unsqueeze(0).to(logits.device)
            mask_len = torch.max(
                torch.tensor([1], device=logits.device),
                torch.min(unknown_map.sum(dim=-1, keepdim=True) - 1, mask_len),
            )

            masking = mask_by_random_topk(mask_len, selected_probs, temperature_schedule[step])
            input_ids[:, -(num_vq_tokens + 1) : -1] = torch.where(masking, MASK_TOKEN_ID, sampled_ids + text_vocab)
            ids_minus_vocab = torch.where(masking, MASK_TOKEN_ID, sampled_ids)

        return sampled_ids

    def generate_text(self, prompt: str, max_new_tokens: int = 128, steps: int = 128) -> str:
        """Block-wise diffusion text decoding via MMadaModelLM.mmu_generate
        (text-only, no image tokens) — used by cot_anchor.py."""
        if self._model is None or self._uni_prompting is None:
            raise RuntimeError("call load() first")
        idx = self._uni_prompting.text_tokenizer(prompt, return_tensors="pt")["input_ids"].to(self.device)
        out_ids = self._model.mmu_generate(idx=idx, max_new_tokens=max_new_tokens, steps=steps)
        new_tokens = out_ids[0, idx.shape[1] :]
        return self._uni_prompting.text_tokenizer.decode(new_tokens, skip_special_tokens=True)
