"""MAGVIT-v2 visual tokenizer (gen-verse/mmada: models/modeling_magvitv2.py::MAGVITv2),
image <-> discrete token grid over the same vocabulary MMaDA's LLaDA
backbone consumes as "image tokens"."""

import sys
from pathlib import Path

import numpy as np
import torch

SUBGOAL_NUM_TOKENS = 1024  # seq_len used by MMadaModelLM.t2i_generate
SUBGOAL_IMAGE_SIDE = 512  # MAGVIT-v2 downsamples 16x spatially: 512/16=32, 32*32=1024
CODEBOOK_SIZE = 8192  # confirmed against the live model's startup log ("Look-up free quantizer with codebook size: 8192")

_MMADA_ROOT = Path(__file__).resolve().parents[3] / "third_party" / "mmada"


class MagvitV2Tokenizer:
    def __init__(self, checkpoint_path: str, device: str = "cuda"):
        self.checkpoint_path = checkpoint_path
        self.device = device
        self._model = None

    def load(self) -> None:
        if str(_MMADA_ROOT) not in sys.path:
            sys.path.insert(0, str(_MMADA_ROOT))
        from models import MAGVITv2  # noqa: PLC0415

        self._model = MAGVITv2.from_pretrained(self.checkpoint_path).to(self.device).eval()

    def _to_tensor(self, image: np.ndarray) -> torch.Tensor:
        # HWC uint8 -> 1CHW float in [-1, 1], matching training.utils.image_transform.
        # Verified against the live model (2026-07-14): a 256x256 input yields
        # 256 tokens (16x16), only a 512x512 input yields SUBGOAL_NUM_TOKENS=1024
        # (32x32) — the encoder downsamples 16x spatially, so token count is a
        # function of input resolution, not a fixed property of the model.
        if image.shape[0] != SUBGOAL_IMAGE_SIDE or image.shape[1] != SUBGOAL_IMAGE_SIDE:
            raise ValueError(
                f"MagvitV2Tokenizer requires a {SUBGOAL_IMAGE_SIDE}x{SUBGOAL_IMAGE_SIDE} "
                f"input to produce SUBGOAL_NUM_TOKENS={SUBGOAL_NUM_TOKENS} tokens; got {image.shape[:2]}. "
                "Resize before calling encode()."
            )
        t = torch.from_numpy(image).permute(2, 0, 1).float() / 127.5 - 1.0
        return t.unsqueeze(0).to(self.device)

    def encode(self, image: np.ndarray) -> np.ndarray:
        """image (HWC uint8, SUBGOAL_IMAGE_SIDE x SUBGOAL_IMAGE_SIDE) -> token
        ids, shape (SUBGOAL_NUM_TOKENS,)."""
        if self._model is None:
            raise RuntimeError("call load() first")
        with torch.no_grad():
            _, codebook_indices = self._model.encode(self._to_tensor(image))
        return codebook_indices[0].cpu().numpy()

    def decode(self, tokens: np.ndarray) -> np.ndarray:
        """token ids -> image (HWC uint8)."""
        if self._model is None:
            raise RuntimeError("call load() first")
        with torch.no_grad():
            ids = torch.from_numpy(tokens).long().to(self.device).unsqueeze(0)
            pixel_values = self._model.decode_code(ids)
        image = ((pixel_values[0].clamp(-1, 1) + 1.0) * 127.5).byte()
        return image.permute(1, 2, 0).cpu().numpy()
