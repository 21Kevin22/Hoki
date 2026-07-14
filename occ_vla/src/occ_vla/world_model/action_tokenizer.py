"""FAST action tokenizer, extending MMaDA's shared image/text vocabulary
with a discrete action modality.

MMaDA itself has no action tokens — this is occ_vla's own extension,
not something in gen-verse/mmada. It reuses the same discretizer
openpi's Pi0FAST variant uses (`physical-intelligence/fast` on the HF
Hub, see third_party/openpi/src/openpi/models/tokenizer.py::FASTTokenizer),
but *not* openpi's FASTTokenizer class directly: that class immediately
remaps FAST's own token ids into PaliGemma's SentencePiece vocab space
(`_act_tokens_to_paligemma_tokens`), which is meaningless for MMaDA's
LLaDA vocab. We take the raw FAST token ids (BPE over quantized DCT
coefficients of the action chunk, see the FAST paper) and offset them
into a new block of ids appended after MMaDA's existing vocabulary,
the same pattern UniversalPrompting uses for `<|soi|>`/`<|eoi|>` image
boundaries (training/prompting_utils.py) — here `<|soa|>`/`<|eoa|>` for
actions.
"""

from dataclasses import dataclass

import numpy as np

ACTION_VOCAB_SIZE = 2048  # user-specified budget for the action token block


@dataclass
class ActionTokenizerConfig:
    fast_checkpoint: str = "physical-intelligence/fast"
    action_vocab_size: int = ACTION_VOCAB_SIZE
    base_vocab_size: int = 0  # MMaDA vocab size (incl. its own special tokens); set at wiring time


class FastActionTokenizer:
    def __init__(self, config: ActionTokenizerConfig):
        self.config = config
        self._processor = None

    def load(self) -> None:
        from transformers import AutoProcessor  # noqa: PLC0415

        self._processor = AutoProcessor.from_pretrained(self.config.fast_checkpoint, trust_remote_code=True)

    def encode(self, actions: np.ndarray) -> np.ndarray:
        """actions: (action_horizon, action_dim) -> action token ids,
        offset into [base_vocab_size, base_vocab_size + action_vocab_size)."""
        if self._processor is None:
            raise RuntimeError("call load() first")
        raw_tokens = np.asarray(self._processor(actions[None])[0])
        if raw_tokens.max(initial=0) >= self.config.action_vocab_size:
            raise ValueError(
                f"FAST token id {raw_tokens.max()} exceeds action_vocab_size={self.config.action_vocab_size}"
            )
        return raw_tokens + self.config.base_vocab_size

    def decode(self, action_token_ids: np.ndarray, action_horizon: int, action_dim: int) -> np.ndarray:
        if self._processor is None:
            raise RuntimeError("call load() first")
        raw_tokens = np.asarray(action_token_ids) - self.config.base_vocab_size
        return self._processor.decode([raw_tokens.tolist()], time_horizon=action_horizon, action_dim=action_dim)[0]
