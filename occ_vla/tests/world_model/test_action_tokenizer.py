import numpy as np
import pytest

from occ_vla.world_model.action_tokenizer import ActionTokenizerConfig, FastActionTokenizer


class _FakeProcessor:
    """Stands in for the physical-intelligence/fast AutoProcessor so the
    offset/bounds-check arithmetic is testable without network access."""

    def __call__(self, actions_batch):
        return [np.array([0, 5, 10])]

    def decode(self, token_batch, time_horizon, action_dim):
        return [np.zeros((time_horizon, action_dim), dtype=np.float32)]


def _tokenizer(base_vocab_size=100, action_vocab_size=2048):
    tok = FastActionTokenizer(ActionTokenizerConfig(base_vocab_size=base_vocab_size, action_vocab_size=action_vocab_size))
    tok._processor = _FakeProcessor()  # noqa: SLF001
    return tok


def test_encode_offsets_into_action_block():
    tok = _tokenizer(base_vocab_size=100)
    ids = tok.encode(np.zeros((10, 7)))
    np.testing.assert_array_equal(ids, [100, 105, 110])


def test_decode_removes_offset():
    tok = _tokenizer(base_vocab_size=100)
    out = tok.decode(np.array([100, 105, 110]), action_horizon=10, action_dim=7)
    assert out.shape == (10, 7)


def test_encode_raises_when_exceeding_action_vocab_size():
    tok = _tokenizer(base_vocab_size=100, action_vocab_size=8)  # raw token 10 won't fit
    with pytest.raises(ValueError):
        tok.encode(np.zeros((10, 7)))
