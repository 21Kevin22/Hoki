"""Convert a PKLP pixel-space displacement (current gripper position ->
KinematicExtrapolator's predicted position, both from kinematics.py) into
a world-frame translation delta, via the local Jacobian of the camera's
pinhole projection at the robot's current end-effector position.

This deliberately does NOT hand-guess a pixel-axis -> robot-axis mapping
(screen-X is not any single robot axis in general, since the agentview
camera is not axis-aligned with the robot base frame). It numerically
differentiates CameraProjector.project -- the same formula already
validated by projecting the static moka_pot_1 body onto the rendered
agentview frame and checking it against the object's known on-screen
location (see CLAUDE.md, camera calibration note) -- to get the actual
local pixel<->world relationship, then inverts it.

Solving J @ world_delta = pixel_delta is underdetermined (2 equations, 3
unknowns: a pixel offset could be explained by moving along the camera's
viewing ray in infinitely many ways). Minimum-norm least squares handles
that, but blows up near projection singularities (J close to rank-deficient,
e.g. viewing near edge-on), so this uses damped least squares (Levenberg-
Marquardt style):

    world_delta = J^T (J J^T + lambda^2 I)^-1 pixel_delta

The Z (vertical) component is then zeroed: PKLP's job here is horizontal
reach correction toward a visually-predicted target, not commanding
vertical motion, so Z is left entirely to pi0.5's own output. Finally the
result is clipped to a max step norm, since J is only a local
linearization around the current eef position -- a large pixel_delta
(target far away in image space) must not be extrapolated into one large
world-frame jump; the caller should re-run this every control step so the
correction tracks as the arm moves.
"""

from dataclasses import dataclass

import numpy as np

DEFAULT_DAMPING = 1e-4  # lambda in the DLS formula; larger = more robust near singularities, less accurate far from them
DEFAULT_FD_EPS = 1e-3  # meters, finite-difference step for the numeric Jacobian
DEFAULT_MAX_STEP_M = 0.03  # meters, hard clip on |world_delta| per control step


@dataclass
class CameraProjector:
    """Pinhole projection: world xyz -> agentview pixel xy (origin
    top-left, y-down, matching the rendered image array). cam_mat is
    sim.data.cam_xmat reshaped (3, 3) -- MuJoCo's camera-frame-to-world
    rotation, so cam_mat.T rotates world vectors into the camera frame.

    IMPORTANT (validated empirically, see CLAUDE.md 2026-07-18 camera
    calibration note): this convention matches the *flipped* frame
    (`raw_agentview[::-1, ::-1]`, i.e. what `preprocess_image` produces
    and what `obs.base_image` / PKLP / pi0.5 actually see), NOT MuJoCo's
    raw `obs["agentview_image"]` buffer directly -- projecting onto the
    raw buffer will land points in the wrong place despite correct math.
    For the real control loop, construct with `resolution=224` to match
    `obs.base_image` (confirmed `resize_with_pad` 256->224 is a pure
    uniform scale, no letterbox offset, since both are square)."""

    cam_pos: np.ndarray  # (3,)
    cam_mat: np.ndarray  # (3, 3)
    fovy_deg: float
    resolution: int  # square render, pixels per side -- use 224 to match obs.base_image, not the raw 256 sim render

    def project(self, p_world: np.ndarray) -> np.ndarray:
        p_cam = self.cam_mat.T @ (p_world - self.cam_pos)
        x_screen = p_cam[0] / (-p_cam[2])
        y_screen = p_cam[1] / (-p_cam[2])
        f_px = (self.resolution / 2) / np.tan(np.radians(self.fovy_deg) / 2)
        px = self.resolution / 2 + x_screen * f_px
        py = self.resolution / 2 - y_screen * f_px
        return np.array([px, py])

    @classmethod
    def from_sim(cls, sim, camera_name: str, resolution: int) -> "CameraProjector":
        cam_id = sim.model.camera_name2id(camera_name)
        return cls(
            cam_pos=np.array(sim.data.cam_xpos[cam_id]),
            cam_mat=np.array(sim.data.cam_xmat[cam_id]).reshape(3, 3),
            fovy_deg=float(sim.model.cam_fovy[cam_id]),
            resolution=resolution,
        )


def projection_jacobian(projector: CameraProjector, p_world: np.ndarray, eps: float = DEFAULT_FD_EPS) -> np.ndarray:
    """Central-difference Jacobian d(pixel)/d(world) at p_world, (2, 3).
    Numeric rather than analytic so it directly reuses the already-
    validated `project`, instead of re-deriving (and risking a sign error
    in) a closed-form derivative."""
    j = np.zeros((2, 3))
    for axis in range(3):
        delta = np.zeros(3)
        delta[axis] = eps
        j[:, axis] = (projector.project(p_world + delta) - projector.project(p_world - delta)) / (2 * eps)
    return j


def pixel_delta_to_world_delta(
    jacobian: np.ndarray,
    pixel_delta: np.ndarray,
    damping: float = DEFAULT_DAMPING,
    zero_z: bool = True,
    max_step_m: float = DEFAULT_MAX_STEP_M,
) -> np.ndarray:
    """Damped-least-squares inverse of a (2, 3) projection Jacobian: the
    world-frame delta (3,) that best explains a desired pixel_delta (2,).
    See module docstring for the DLS formula, the zero_z rationale, and
    why the result is norm-clipped."""
    jjt = jacobian @ jacobian.T  # (2, 2)
    damped_inv = np.linalg.inv(jjt + (damping**2) * np.eye(2))
    world_delta = jacobian.T @ damped_inv @ pixel_delta  # (3,)

    if zero_z:
        world_delta = world_delta.copy()
        world_delta[2] = 0.0

    norm = np.linalg.norm(world_delta)
    if norm > max_step_m and norm > 0:
        world_delta = world_delta * (max_step_m / norm)

    return world_delta


def pklp_pixel_delta_to_world_delta(
    projector: CameraProjector,
    eef_pos_world: np.ndarray,
    current_pixel: np.ndarray,
    predicted_pixel: np.ndarray,
    damping: float = DEFAULT_DAMPING,
    max_step_m: float = DEFAULT_MAX_STEP_M,
) -> np.ndarray:
    """End-to-end convenience wrapper for the runtime blending hook:
    Jacobian at the current eef position, then DLS-invert the PKLP pixel
    displacement (predicted_pixel - current_pixel) into an XY world delta."""
    jacobian = projection_jacobian(projector, eef_pos_world)
    pixel_delta = np.asarray(predicted_pixel) - np.asarray(current_pixel)
    return pixel_delta_to_world_delta(jacobian, pixel_delta, damping=damping, max_step_m=max_step_m)
