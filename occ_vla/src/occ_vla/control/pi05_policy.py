"""pi0.5 control backbone, wrapping the real openpi.policies.Policy.

Grounded against Physical-Intelligence/openpi (third_party/openpi, see
scripts/setup_third_party.sh):

- `openpi.training.config.get_config("pi05_libero")` (or `pi05_droid`,
  `pi05_aloha`, ...) returns a `TrainConfig` with `model=Pi0Config(pi05=True)`.
- `openpi.policies.policy_config.create_trained_policy(train_config, checkpoint_dir)`
  loads weights (JAX or, if `model.safetensors` is present, PyTorch) and
  returns an `openpi.policies.policy.Policy`.
- `Policy.infer(obs_dict) -> dict` where `obs_dict` has keys
  `observation/state`, `observation/image`, `observation/wrist_image`
  (uint8 HWC), and `prompt` (str); output has key `actions`, shape
  `(action_horizon, action_dim)` at 50Hz control.

The "hierarchical" high-level-subtask -> low-level-action split
described in the pi0.5 paper happens *inside* a single `infer()` call
(via the discrete language/state tokens consumed before flow-matching
action decoding) — openpi does not expose it as two separate calls, so
`HierarchicalPlanner` in this package is a wrapper for injecting
world-model/PKLP context between subtask and action, not a real
pi0.5-internal boundary.
"""

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# third_party/openpi/src is not pip-installed; see scripts/setup_third_party.sh
# and README.md "Setup" for why this is a path insert rather than a dependency.
_OPENPI_SRC = Path(__file__).resolve().parents[3] / "third_party" / "openpi" / "src"


@dataclass
class Pi05Observation:
    base_image: np.ndarray  # observation/image, HWC uint8
    wrist_image: np.ndarray  # observation/wrist_image, HWC uint8
    state: np.ndarray  # observation/state
    prompt: str  # language instruction
    subgoal_image: np.ndarray | None = None  # from world_model, when self-occluded
    cot_anchor: str | None = None  # from world_model, when vision unavailable

    def to_openpi_dict(self) -> dict:
        # observation/subgoal_image and cot_anchor are occ_vla additions,
        # not part of openpi's base LiberoInputs contract — they're only
        # meaningful if the policy was built with
        # control/observation_injection.py::OccVlaLiberoInputs as its
        # input transform (see that module's docstring for why a plain
        # LiberoInputs policy silently drops them instead of erroring).
        data = {
            "observation/image": self.base_image,
            "observation/wrist_image": self.wrist_image,
            "observation/state": self.state,
            "prompt": self.prompt,
        }
        if self.subgoal_image is not None:
            data["observation/subgoal_image"] = self.subgoal_image
        if self.cot_anchor is not None:
            data["cot_anchor"] = self.cot_anchor
        return data


class Pi05Policy:
    """Thin wrapper around openpi.policies.policy.Policy for a pi0.5 checkpoint."""

    def __init__(self, config_name: str, checkpoint_dir: str, pytorch_device: str | None = None):
        self.config_name = config_name
        self.checkpoint_dir = checkpoint_dir
        self.pytorch_device = pytorch_device
        self._policy = None

    def load(self) -> None:
        if str(_OPENPI_SRC) not in sys.path:
            sys.path.insert(0, str(_OPENPI_SRC))
        from openpi.policies import policy_config  # noqa: PLC0415
        from openpi.training import config as openpi_config  # noqa: PLC0415

        train_config = openpi_config.get_config(self.config_name)
        self._policy = policy_config.create_trained_policy(
            train_config, self.checkpoint_dir, pytorch_device=self.pytorch_device
        )

    def step(self, obs: Pi05Observation) -> np.ndarray:
        """Returns the action chunk, shape (action_horizon, action_dim)."""
        if self._policy is None:
            raise RuntimeError("call load() first")
        outputs = self._policy.infer(obs.to_openpi_dict())
        return outputs["actions"]
