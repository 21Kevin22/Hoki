"""MOKA-style visual prompting for PKLP's kinematic prediction.

Rather than asking a generative model to synthesize a new "predicted
view" image (MMaDA's arm-free subgoal path has an unresolved
generation-quality problem -- see occ_vla/CLAUDE.md), this draws the
constant-acceleration extrapolated position from
pklp/kinematics.py::KinematicExtrapolator directly onto the real
(already photorealistic) agentview frame as a marker, following MOKA's
core idea of marking the actionable point on the existing image instead
of generating a new one. Zero dependency on MAGVIT-v2/MMaDA, so it
carries none of that pipeline's failure modes.
"""

import cv2
import numpy as np

DEFAULT_MARKER_COLOR = (255, 0, 0)  # red, matches the image's own RGB channel order
DEFAULT_MARKER_RADIUS = 5
DEFAULT_LINE_THICKNESS = 2


def _clip_point(point: np.ndarray, image_shape: tuple[int, int]) -> tuple[int, int]:
    h, w = image_shape[:2]
    x = int(np.clip(round(float(point[0])), 0, w - 1))
    y = int(np.clip(round(float(point[1])), 0, h - 1))
    return x, y


def draw_kinematic_overlay(
    image: np.ndarray,
    current_position: np.ndarray,
    predicted_position: np.ndarray,
    color: tuple[int, int, int] = DEFAULT_MARKER_COLOR,
    marker_radius: int = DEFAULT_MARKER_RADIUS,
    line_thickness: int = DEFAULT_LINE_THICKNESS,
) -> np.ndarray:
    """Returns a copy of `image` with a line from `current_position` to
    `predicted_position` (both xy pixel coords, clipped to the frame)
    and a filled dot at the predicted point -- the "physically correct
    reach point" ground-truthed by PKLP's kinematic extrapolation, made
    visible even when the arm's own silhouette occludes that region."""
    overlaid = image.copy()
    current_xy = _clip_point(current_position, image.shape)
    predicted_xy = _clip_point(predicted_position, image.shape)
    cv2.line(overlaid, current_xy, predicted_xy, color, line_thickness)
    cv2.circle(overlaid, predicted_xy, marker_radius, color, thickness=-1)
    return overlaid
