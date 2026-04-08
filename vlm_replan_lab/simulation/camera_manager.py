import cv2
import numpy as np
from omni.isaac.sensor import Camera
from pxr import Gf


class CameraManager:
    def __init__(self, world):
        self.world = world
        self.cameras = {}
        self.main_camera_name = "front"
        self.camera_specs = {
            "front": {
                "prim_path": "/World/CameraFront",
                "position": np.array([1.35, 0.0, 1.05], dtype=float),
                "target": np.array([0.48, 0.0, 0.58], dtype=float),
            },
            "left": {
                "prim_path": "/World/CameraLeft",
                "position": np.array([0.95, 0.85, 0.95], dtype=float),
                "target": np.array([0.42, 0.0, 0.56], dtype=float),
            },
            "right": {
                "prim_path": "/World/CameraRight",
                "position": np.array([0.95, -0.85, 0.95], dtype=float),
                "target": np.array([0.42, 0.0, 0.56], dtype=float),
            },
            "top": {
                "prim_path": "/World/CameraTop",
                "position": np.array([0.48, 0.0, 1.85], dtype=float),
                "target": np.array([0.48, 0.0, 0.56], dtype=float),
            },
        }
        self.view_size = (640, 480)
        self.mug_colors = {
            "mug0": (80, 80, 220),
            "mug1": (80, 200, 80),
            "mug2": (220, 120, 80),
            "mug3": (80, 220, 220),
            "mug4": (220, 220, 80),
            "basket": (120, 200, 120),
        }

    def _look_at_orientation(self, position, target, forward_axis=(0, 0, -1)):
        position = Gf.Vec3d(*position)
        target = Gf.Vec3d(*target)
        forward = Gf.Vec3d(*forward_axis)
        direction = target - position
        if direction.GetLength() == 0:
            return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
        rotation = Gf.Rotation(forward, direction.GetNormalized())
        quat = Gf.Quatf(rotation.GetQuat())
        imag = quat.GetImaginary()
        return np.array([quat.GetReal(), imag[0], imag[1], imag[2]], dtype=float)

    def initialize(self):
        self.cameras = {}
        for name, spec in self.camera_specs.items():
            camera = Camera(
                prim_path=spec["prim_path"],
                position=spec["position"],
                orientation=self._look_at_orientation(spec["position"], spec["target"]),
                resolution=self.view_size,
                frequency=10,
            )
            camera.initialize()
            try:
                camera.add_rgb_to_frame()
            except Exception:
                pass
            self.cameras[name] = camera

        for _ in range(10):
            self.world.step(render=True)

    def _capture_camera(self, camera):
        image = camera.get_rgba()
        if image is None:
            return None

        image = np.asarray(image)
        if image.ndim != 3 or image.shape[0] == 0 or image.shape[1] == 0:
            return None

        return image[:, :, :3]

    def capture(self):
        camera = self.cameras.get(self.main_camera_name)
        if camera is None:
            return None
        return self._capture_camera(camera)

    def capture_all(self):
        frames = {}
        for name, camera in self.cameras.items():
            frame = self._capture_camera(camera)
            if frame is not None:
                frames[name] = frame
        if self.main_camera_name in frames:
            frames["main"] = frames[self.main_camera_name]
        return frames

    def _to_uint8(self, frame):
        if frame is None:
            return None
        frame = np.asarray(frame)
        if frame.dtype != np.uint8:
            if np.issubdtype(frame.dtype, np.floating):
                max_value = float(frame.max()) if frame.size else 0.0
                if max_value <= 1.0:
                    frame = frame * 255.0
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        return frame

    def _is_low_information(self, frame):
        frame = self._to_uint8(frame)
        if frame is None:
            return True
        return float(frame.std()) < 4.0

    def _blank_canvas(self, title):
        w, h = self.view_size
        canvas = np.full((h, w, 3), 28, dtype=np.uint8)
        cv2.putText(canvas, title, (18, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (235, 235, 235), 2, cv2.LINE_AA)
        return canvas

    def _project(self, value, src_min, src_max, dst_min, dst_max):
        if src_max == src_min:
            return int((dst_min + dst_max) * 0.5)
        t = (value - src_min) / (src_max - src_min)
        t = max(0.0, min(1.0, t))
        return int(dst_min + t * (dst_max - dst_min))

    def render_debug_views(self, objects, graph=None):
        top = self._blank_canvas("top debug")
        front = self._blank_canvas("front debug")
        left = self._blank_canvas("left debug")
        right = self._blank_canvas("right debug")

        self._draw_topdown(top)
        self._draw_front(front)
        self._draw_side(left, flip=False)
        self._draw_side(right, flip=True)

        for obj in objects:
            name = obj["name"]
            x, y, z = [float(v) for v in obj["position"]]
            color = self.mug_colors.get(name, (200, 200, 200))
            radius = 14 if obj["type"] == "basket" else 10

            tx = self._project(x, 0.0, 0.95, 40, 600)
            ty = self._project(y, -0.35, 0.35, 430, 60)
            cv2.circle(top, (tx, ty), radius, color, -1)
            cv2.putText(top, name, (tx + 10, ty - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

            fx = self._project(x, 0.0, 0.95, 40, 600)
            fz = self._project(z, 0.40, 0.95, 430, 60)
            cv2.circle(front, (fx, fz), radius, color, -1)
            cv2.putText(front, name, (fx + 10, fz - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

            ly = self._project(y, -0.35, 0.35, 40, 600)
            lz = self._project(z, 0.40, 0.95, 430, 60)
            cv2.circle(left, (ly, lz), radius, color, -1)
            cv2.putText(left, name, (ly + 10, lz - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

            ry = self._project(y, -0.35, 0.35, 600, 40)
            rz = self._project(z, 0.40, 0.95, 430, 60)
            cv2.circle(right, (ry, rz), radius, color, -1)
            cv2.putText(right, name, (ry + 10, rz - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

            if graph and name in graph and obj["type"] == "mug":
                status = []
                if graph[name].get("inside_basket"):
                    status.append("in")
                if graph[name].get("upright"):
                    status.append("up")
                if graph[name].get("fallen"):
                    status.append("fallen")
                if status:
                    cv2.putText(top, ",".join(status), (tx + 10, ty + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (230, 230, 230), 1, cv2.LINE_AA)

        return {
            "front": front,
            "left": left,
            "right": right,
            "top": top,
        }

    def _draw_topdown(self, canvas):
        cv2.rectangle(canvas, (90, 90), (550, 390), (90, 140, 190), 2)
        bx1 = self._project(0.60, 0.0, 0.95, 40, 600)
        bx2 = self._project(0.84, 0.0, 0.95, 40, 600)
        by1 = self._project(-0.12, -0.35, 0.35, 430, 60)
        by2 = self._project(0.12, -0.35, 0.35, 430, 60)
        cv2.rectangle(canvas, (bx1, by2), (bx2, by1), (120, 200, 120), 2)

    def _draw_front(self, canvas):
        table_z = self._project(0.50, 0.40, 0.95, 430, 60)
        cv2.line(canvas, (60, table_z), (610, table_z), (90, 140, 190), 2)
        bx1 = self._project(0.60, 0.0, 0.95, 40, 600)
        bx2 = self._project(0.84, 0.0, 0.95, 40, 600)
        bz1 = self._project(0.53, 0.40, 0.95, 430, 60)
        bz2 = self._project(0.65, 0.40, 0.95, 430, 60)
        cv2.rectangle(canvas, (bx1, bz2), (bx2, bz1), (120, 200, 120), 2)

    def _draw_side(self, canvas, flip=False):
        table_z = self._project(0.50, 0.40, 0.95, 430, 60)
        cv2.line(canvas, (60, table_z), (610, table_z), (90, 140, 190), 2)
        y1, y2 = (-0.12, 0.12) if not flip else (0.12, -0.12)
        by1 = self._project(y1, -0.35, 0.35, 40, 600)
        by2 = self._project(y2, -0.35, 0.35, 40, 600)
        bz1 = self._project(0.53, 0.40, 0.95, 430, 60)
        bz2 = self._project(0.65, 0.40, 0.95, 430, 60)
        cv2.rectangle(canvas, (min(by1, by2), bz2), (max(by1, by2), bz1), (120, 200, 120), 2)

    def compose_views(self, frames, debug_views=None):
        ordered_names = ["front", "left", "right", "top"]
        panels = []
        for name in ordered_names:
            frame = frames.get(name)
            debug_frame = None if debug_views is None else debug_views.get(name)
            use_debug = self._is_low_information(frame) and debug_frame is not None
            panel = debug_frame if use_debug else frame
            panel = self._to_uint8(panel)
            if panel is None:
                panel = np.zeros((self.view_size[1], self.view_size[0], 3), dtype=np.uint8)
            if panel.shape[:2] != (self.view_size[1], self.view_size[0]):
                panel = cv2.resize(panel, self.view_size)
            label = f"{name} debug" if use_debug else name
            panel = panel.copy()
            cv2.putText(panel, label, (20, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
            panels.append(panel)

        top_row = np.concatenate(panels[:2], axis=1)
        bottom_row = np.concatenate(panels[2:], axis=1)
        return np.concatenate([top_row, bottom_row], axis=0)
