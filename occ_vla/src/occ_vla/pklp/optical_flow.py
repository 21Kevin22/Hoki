"""RAFT optical flow (torchvision.models.optical_flow.raft_large /
raft_small), patchified to a 16x16-patch velocity grid for PKLP.

PKLP needs flow between three consecutive frames (o_{t-2}, o_{t-1},
o_t) to get two velocity samples (for the finite-difference
acceleration in kinematics.py::estimate_state), computed here as two
pairwise RAFT calls: (o_{t-2} -> o_{t-1}) and (o_{t-1} -> o_t).
"""

from dataclasses import dataclass

import numpy as np
import torch

DEFAULT_PATCH_SIZE = 16


@dataclass
class PatchFlow:
    patch_centers: np.ndarray  # (N, 2) xy, in pixels
    flow: np.ndarray  # (N, 2) per-patch flow vector, current frame
    grid_shape: tuple[int, int]  # (rows, cols), rows*cols == N


def patch_pool_flow(flow: np.ndarray, patch_size: int = DEFAULT_PATCH_SIZE) -> PatchFlow:
    """flow: (H, W, 2) dense pixel-space flow -> average-pooled per
    patch_size x patch_size patch. H, W must be multiples of patch_size."""
    h, w, c = flow.shape
    if c != 2:
        raise ValueError(f"expected flow with 2 channels, got {c}")
    if h % patch_size or w % patch_size:
        raise ValueError(f"flow shape {(h, w)} not divisible by patch_size={patch_size}")
    rows, cols = h // patch_size, w // patch_size

    pooled = flow.reshape(rows, patch_size, cols, patch_size, 2).mean(axis=(1, 3))  # (rows, cols, 2)

    ys, xs = np.meshgrid(
        np.arange(rows) * patch_size + patch_size / 2,
        np.arange(cols) * patch_size + patch_size / 2,
        indexing="ij",
    )
    centers = np.stack([xs, ys], axis=-1).reshape(-1, 2)  # (rows*cols, 2), xy order

    return PatchFlow(patch_centers=centers, flow=pooled.reshape(-1, 2), grid_shape=(rows, cols))


class RaftFlowEstimator:
    def __init__(self, patch_size: int = DEFAULT_PATCH_SIZE, variant: str = "raft_large", device: str = "cuda"):
        self.patch_size = patch_size
        self.variant = variant
        self.device = device
        self._model = None
        self._transforms = None

    def load(self) -> None:
        from torchvision.models.optical_flow import (  # noqa: PLC0415
            Raft_Large_Weights,
            Raft_Small_Weights,
            raft_large,
            raft_small,
        )

        if self.variant == "raft_large":
            weights = Raft_Large_Weights.DEFAULT
            self._model = raft_large(weights=weights)
        elif self.variant == "raft_small":
            weights = Raft_Small_Weights.DEFAULT
            self._model = raft_small(weights=weights)
        else:
            raise ValueError(f"unknown RAFT variant: {self.variant}")
        self._model = self._model.to(self.device).eval()
        self._transforms = weights.transforms()

    def _pair_flow(self, frame_a: np.ndarray, frame_b: np.ndarray) -> np.ndarray:
        """frame_{a,b}: HWC uint8, same shape. Returns dense flow (H, W, 2)
        at the original resolution."""
        if self._model is None:
            raise RuntimeError("call load() first")
        h, w = frame_a.shape[:2]
        a = torch.from_numpy(frame_a).permute(2, 0, 1).unsqueeze(0).to(self.device)
        b = torch.from_numpy(frame_b).permute(2, 0, 1).unsqueeze(0).to(self.device)
        a, b = self._transforms(a, b)
        with torch.no_grad():
            flow_predictions = self._model(a, b)
        flow = flow_predictions[-1][0]  # (2, H', W'), last (most refined) iteration
        # RAFT's transform resizes to a multiple of 8; resize flow back and
        # rescale magnitudes to the original pixel grid.
        flow = torch.nn.functional.interpolate(
            flow.unsqueeze(0), size=(h, w), mode="bilinear", align_corners=False
        )[0]
        scale_x, scale_y = w / a.shape[-1], h / a.shape[-2]
        flow[0] *= scale_x
        flow[1] *= scale_y
        return flow.permute(1, 2, 0).cpu().numpy()  # (H, W, 2)

    def three_frame_patch_flow(
        self, frame_t2: np.ndarray, frame_t1: np.ndarray, frame_t0: np.ndarray
    ) -> tuple[PatchFlow, PatchFlow]:
        """Returns (flow[t-2 -> t-1], flow[t-1 -> t]), both patch-pooled."""
        flow_a = self._pair_flow(frame_t2, frame_t1)
        flow_b = self._pair_flow(frame_t1, frame_t0)
        return patch_pool_flow(flow_a, self.patch_size), patch_pool_flow(flow_b, self.patch_size)
