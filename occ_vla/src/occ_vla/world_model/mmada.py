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

from occ_vla.world_model.tokenizer import SUBGOAL_NUM_TOKENS, MagvitV2Tokenizer

MASK_TOKEN_ID = 126336
DEFAULT_TIMESTEPS = 18  # MMadaModelLM.t2i_generate default ("ideal" per MaskGIT paper)

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

        `t2i_generate` reads `uni_prompting` out of **kwargs (it's not a
        named parameter) — confirmed against the live model: omitting it
        raises AttributeError('NoneType' object has no attribute
        'text_tokenizer') since the function defaults it to None via
        `kwargs.get("uni_prompting", None)`. inference_t2i.py's own call
        site passes it the same way (models/modeling_mmada.py:146,576)."""
        if self._model is None:
            raise RuntimeError("call load() first")
        return self._model.t2i_generate(
            input_ids=batch.input_ids,
            attention_mask=batch.attention_mask,
            timesteps=timesteps,
            seq_len=SUBGOAL_NUM_TOKENS,
            mask_token_id=MASK_TOKEN_ID,
            uni_prompting=self._uni_prompting,
        )

    def generate_text(self, prompt: str, max_new_tokens: int = 128, steps: int = 128) -> str:
        """Block-wise diffusion text decoding via MMadaModelLM.mmu_generate
        (text-only, no image tokens) — used by cot_anchor.py."""
        if self._model is None or self._uni_prompting is None:
            raise RuntimeError("call load() first")
        idx = self._uni_prompting.text_tokenizer(prompt, return_tensors="pt")["input_ids"].to(self.device)
        out_ids = self._model.mmu_generate(idx=idx, max_new_tokens=max_new_tokens, steps=steps)
        new_tokens = out_ids[0, idx.shape[1] :]
        return self._uni_prompting.text_tokenizer.decode(new_tokens, skip_special_tokens=True)
